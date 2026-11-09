# =============================================================================
# The log density and its closed-form gradient.
#
# Every model this package fits has the same structure: one LINEAR PREDICTOR
# per distributional parameter,
#
#     eta_d = offset + Intercept_d + Xc_d * b_d + sum_groups Z * r[group]
#     r     = diag(sd) * L * z          (z ~ N(0, 1), non-centred, as brms)
#
# and a likelihood that is a sum over observations of a function of
# (y_i, eta_mu_i, eta_sigma_i, ...). That structure is why no AD is needed:
#
#     d loglik / d eta_d[i]  = w_d[i]            (families.jl, per family)
#     d / d Intercept        = sum(w)
#     d / d b                = Xc' * w           (one BLAS call)
#     d / d r[k, j]          = sum over obs in group j of Z[i,k] * w[i]
#
# and r's own chain rule back to (sd, L, z) is a few small matrix products,
# including a closed-form reverse pass through the correlation-Cholesky
# transform (`_corr_value_grad!`).
#
# WHAT IS COMPUTED: the log posterior UP TO A CONSTANT, on the unconstrained
# scale, Jacobians included — Stan's `lp__`. Every term that cannot move with
# θ (prior normalisers, truncation constants, log y!, the z prior's 1/2
# log 2π) is left out of both paths below: the sampler cannot see them and
# some (Student-t's loggammas) are not free. `log_lik` and `lprior` add them
# back where a normalised number is the point.
#
# There are two implementations of the density and the tests require them to
# agree:
#
#   * `logdensity_and_gradient!` — the analytic path, in-place, into a
#     per-chain workspace; what the samplers call.
#   * `LogDensityProblems.logdensity` — a plain, allocating, number-type
#     generic evaluation. It is the reference the analytic gradient is
#     checked against, and it is what general AD backends differentiate, so
#     a comparison with them compares two gradients of the same function.
#
# Parameter vector layout (unconstrained), per distributional parameter in
# family order:  [Intercept] [b...] then per group term [log sd...]
# [corr (tanh-CPC)...] [z...]; a scalar auxiliary parameter is one entry
# u = log(x - lb).
# =============================================================================

"""
    GroupBlock

One group-level term `(terms | g)`: its observation-to-level index, its
per-observation design `Z` (N x q), and where its parameters live in θ.
`Zt` is `Z` transposed (contiguous per observation, which is how the hot
loops read it) and `unitZ` marks the intercept-only case `(1 | g)`, where
`Z` is a column of ones and the loops reduce to a gather and a scatter-add.
"""
struct GroupBlock{T}
    label::String
    idx::Vector{Int}
    levels::Vector{String}
    Z::Matrix{T}
    Zt::Matrix{T}
    unitZ::Bool
    coefs::Vector{String}
    sd::UnitRange{Int}
    corr::UnitRange{Int}     # empty unless correlated with q > 1
    z::UnitRange{Int}        # q * J entries, column-major q x J
    sd_priors::Vector{ScalarPrior}
    lkj::Float64
    # shared IDs `(1 | p | g)`: one joint block (sd, corr, z over every member's
    # columns) split across several predictors. A member uses `rows` of the
    # joint effects; the first member (in predictor order) computes them in the
    # forward pass, every member adds its d/d r rows, and the last member's
    # pullback runs the joint reverse pass — pullbacks run in predictor order.
    # role: 0 = an ordinary block, 1 = first, 2 = middle, 3 = last member.
    rows::UnitRange{Int}
    role::Int
    sid::Int                 # shared-block number (0 = none); members alias one GroupWS
    jlabels::Vector{String}  # joint coefficient labels with dpar prefixes (cor names)
    # gr(g, by = h): one block per level of h, covering that level's groups;
    # rows of other levels have Z = 0. brms names the sds sd_g__Intercept:hA.
    bylevel::String          # "" unless a gr(by =) block
    sdsfx::String            # ":hA" (appended to sd/cor coefficient labels)
    # mm(g1, g2): every observation belongs to several levels, with weights
    mmJ::Matrix{Int}         # N x M level indices (0 x 0 unless multi-membership)
    mmW::Matrix{T}           # N x M weights
    # gr(g, dist = "student") (brms): level j's effects are scaled by
    # dfm_j = sqrt(df * udf_j), udf_j ~ inv_chi_square(df), df ~ gamma(2, 0.1) on (1, Inf);
    # θ holds log(df - 1) and log(udf_j)
    df::Int                  # θ index of the df parameter (0 = gaussian effects)
    udf::UnitRange{Int}      # J entries
    df_prior::ScalarPrior
    # gr(g, cov = A) (brms): known covariance among the levels; r = diag(sd) L Z Lcov'
    # with Lcov = chol(A) (J x J; 0 x 0 = none), the Kronecker product brms forms
    Lcov::Matrix{T}
end
function GroupBlock{T}(label, idx, levels, Z, coefs, sd, corr, z, sd_priors, lkj;
                       rows=1:size(Z, 2), role=0, sid=0, jlabels=String[], bylevel="", sdsfx="",
                       mmJ=zeros(Int, 0, 0), mmW=zeros(T, 0, 0), df=0, udf=1:0, df_prior=flat(),
                       Lcov=zeros(T, 0, 0)) where {T}
    Zm = Matrix{T}(Z)
    unit = size(Zm, 2) == 1 && all(isone, Zm) && isempty(mmJ)
    return GroupBlock{T}(label, idx, levels, Zm, permutedims(Zm), unit, coefs, sd, corr, z, sd_priors, lkj,
                         rows, role, sid, jlabels, bylevel, sdsfx, mmJ, T.(mmW), df, udf, df_prior, Matrix{T}(Lcov))
end
_hascov(gb::GroupBlock) = !isempty(gb.Lcov)
_student(gb::GroupBlock) = gb.df > 0
# df and the per-level scales dfm (any number type)
function _dfm(gb::GroupBlock, θ)
    dfv = 1 + exp(θ[gb.df])
    return dfv, [sqrt(dfv * exp(θ[k])) for k in gb.udf]
end
# log inv_chi_square(udf | df) + log-Jacobian of udf = exp(v), summed over levels,
# plus df's prior and the Jacobian of df = 1 + exp(u) (normalisers of df's prior dropped)
function _udf_prior(gb::GroupBlock, θ)
    u = θ[gb.df]; dfv = 1 + exp(u)
    lp = prior_kernel(gb.df_prior, dfv) + u
    for k in gb.udf
        v = θ[k]
        lp += -dfv / 2 * _pv(dfv, _LOG2) - loggamma(dfv / 2) - dfv / 2 * v - exp(-v) / 2
    end
    return lp
end
_ismm(gb::GroupBlock) = !isempty(gb.mmJ)
_q(gb::GroupBlock) = length(gb.sd)        # columns of the (joint) block
_qm(gb::GroupBlock) = size(gb.Z, 2)       # this predictor's own columns
_owns(gb::GroupBlock) = gb.role <= 1      # forward pass / priors / names of the block
_closes(gb::GroupBlock) = gb.role == 0 || gb.role == 3   # runs the reverse pass
_J(gb::GroupBlock) = length(gb.levels)
_has_corr(gb::GroupBlock) = !isempty(gb.corr)

"""
    LinPred

A linear predictor for one distributional parameter. `Xc` is the
population-level design with the intercept column removed and every column
centred (`means` holds the column means); the intercept is then the
CENTRED intercept, as in brms, and the reported `b_Intercept` is
`Intercept - means' * b`.
"""
struct LinPred{T}
    dpar::Symbol
    intercept::Int           # θ index, 0 when the formula has no intercept
    Xc::Matrix{T}
    means::Vector{T}
    coefnames::Vector{String}
    b::UnitRange{Int}
    groups::Vector{GroupBlock{T}}
    offset::Vector{T}
    int_prior::ScalarPrior
    b_priors::Vector{ScalarPrior}
    any_b_prior::Bool        # false when every b prior is flat (the default)
    has_offset::Bool
    b_lb::Vector{Float64}    # bounds on each coefficient (prior(...; lb, ub))
    b_ub::Vector{Float64}
    bounded::Bool            # any finite bound: b is then a transform of θ
    mo::Vector{MoTerm}       # monotonic effects (special.jl)
    ar::ARTerm{T}            # autoregressive term (special.jl), p = 0 if none
    hs::HSTerm               # horseshoe prior on b (horseshoe.jl); hs.on = false if none
    r2d2::R2D2Term           # R2D2 prior on b (r2d2.jl)
    smooths::Vector{SmoothBlock{T}}   # penalised parts of s() terms (splines.jl)
    me::Vector{MeTerm{T}}             # measurement-error terms (special.jl)
    gp::Vector{GPTerm{T}}             # Gaussian processes (gp.jl)
    mi::Vector{MiTerm{T}}             # imputed predictors mi(x) (special.jl)
    car::Vector{CarTerm{T}}           # spatial CAR terms (special.jl)
    lac::Vector{LatentAC{T}}          # latent residual autocorrelation, non-gaussian (timecov.jl)
    mecor::Vector{MeCorTerm{T}}       # correlated me() latents (brms mecor, special.jl); its terms are in `me` too
end
function LinPred{T}(dpar, intercept, Xc, means, coefnames, b, groups, offset, int_prior, b_priors, any_b_prior;
                    b_lb=fill(-Inf, length(b)), b_ub=fill(Inf, length(b)), mo=MoTerm[], ar=ARTerm{T}(),
                    hs=HSTerm(), r2d2=R2D2Term(), smooths=SmoothBlock{T}[], me=MeTerm{T}[], gp=GPTerm{T}[],
                    mi=MiTerm{T}[], car=CarTerm{T}[], lac=LatentAC{T}[], mecor=MeCorTerm{T}[]) where {T}
    bounded = any(isfinite, b_lb) || any(isfinite, b_ub)
    return LinPred{T}(dpar, intercept, Xc, means, coefnames, b, groups, offset, int_prior, b_priors,
                      any_b_prior || bounded, any(!iszero, offset), b_lb, b_ub, bounded, mo, ar, hs, r2d2, smooths, me, gp, mi, car, lac,
                      mecor)
end

# b = transform(u) for a bounded coefficient: (value, d value/du, log-Jacobian,
# d log-Jacobian/du). Stan's lower/upper/both-bounded transforms.
@inline function _btrans(lb::Float64, ub::Float64, u)
    if lb == -Inf && ub == Inf
        return u, one(u), zero(u), zero(u)
    elseif ub == Inf
        e = exp(u)
        return oftype(e, lb) + e, e, u, one(u)
    elseif lb == -Inf
        e = exp(u)
        return oftype(e, ub) - e, -e, u, one(u)
    end
    s = logistic(u)
    w = oftype(s, ub - lb)
    return oftype(s, lb) + w * s, w * s * (1 - s), log(w) + log(s) + log1p(-s), 1 - 2s
end

# the coefficients themselves (transformed if bounded), any element type
function _bvalues(p::LinPred, θ)
    p.hs.on && return θ[p.b] .* _hs_scales(p.hs, θ)
    p.r2d2.on && return θ[p.b] .* _r2d2_scales(p.r2d2, θ, length(p.b))
    return p.bounded ? [first(_btrans(p.b_lb[j], p.b_ub[j], θ[k])) for (j, k) in enumerate(p.b)] : θ[p.b]
end

"""
    ScalarDpar

An auxiliary parameter without its own formula (e.g. `sigma` in a plain
gaussian model): one unconstrained entry `u`, equal to the parameter's
link-scale value. Natural value `lb + exp(u)` (`logit = false`: sigma,
shape, phi, nu) or `logistic(u)` (`logit = true`: zi, hu on (0, 1)).
"""
struct ScalarDpar
    dpar::Symbol
    idx::Int
    lb::Float64
    ub::Float64
    kind::Symbol           # :log (x = lb + exp(u)), :logit (x in (lb, ub)), :identity (x = u)
    prior::ScalarPrior
end

# natural value, d value / d u, log-Jacobian, d log-Jacobian / d u
@inline function _dpar_transform(p::ScalarDpar, u)
    if p.kind === :logit
        s = logistic(u)
        w = oftype(s, p.ub - p.lb)
        return oftype(s, p.lb) + w * s, w * s * (1 - s), log(w) - log1pexp(-u) - log1pexp(u), 1 - 2s
    elseif p.kind === :identity
        return u, one(u), zero(u), zero(u)
    end
    e = exp(u)
    return oftype(e, p.lb) + e, e, u, one(u)    # oftype: keep Float32 Float32
end
_natural(p::ScalarDpar, u) = first(_dpar_transform(p, u))

# --- per-chain scratch space -------------------------------------------------

struct GroupWS{T}
    M::Matrix{T}   # L * z           (q x J)
    R::Matrix{T}   # diag(sd) * M    (q x J): the group-level effects r
    G::Matrix{T}   # d loglik / d r  (q x J), then reused for d / d M
    L::Matrix{T}   # Cholesky factor of the correlation matrix (q x q)
    A::Matrix{T}   # d loglik / d L  (q x q)
    gc::Vector{T}  # gradient of the correlation parameters
    B::Matrix{T}   # q x J buffer for gr(cov =) products (0 x 0 otherwise)
end
function GroupWS{T}(gb::GroupBlock) where {T}
    q, J = _q(gb), _J(gb)
    return GroupWS{T}(zeros(T, q, J), zeros(T, q, J), zeros(T, q, J), Matrix{T}(I, q, q),
                      zeros(T, q, q), zeros(T, length(gb.corr)), _hascov(gb) ? zeros(T, q, J) : zeros(T, 0, 0))
end

# `w` is d loglik / d eta: per observation for a LinPred, but a single
# running SUM for a ScalarDpar — its gradient is that sum, so storing N
# values to add them up afterwards would be wasted traffic.
struct PredWS{T}
    eta::Vector{T}
    w::Vector{T}
    groups::Vector{GroupWS{T}}
    bvals::Vector{T}         # transformed coefficients (bounded b only)
    mo::Vector{MoWS{T}}      # monotonic-effect scratch
    base::Vector{T}          # eta before the AR term
    wbase::Vector{T}         # d loglik / d base
    sbuf::Vector{Vector{T}}  # smooth-term scratch, one per s()
    mecor::Vector{Matrix{T}} # correlated me() latents: L, W, dW, A, gc per block
end
PredWS{T}(eta, w, groups) where {T} = PredWS{T}(eta, w, groups, T[], MoWS{T}[], T[], T[], Vector{T}[], Matrix{T}[])
# (the positional form without the mecor workspace, for the non-LinPred predictors)
PredWS{T}(eta, w, groups, bvals, mo, base, wbase, sbuf) where {T} =
    PredWS{T}(eta, w, groups, bvals, mo, base, wbase, sbuf, Matrix{T}[])
function PredWS{T}(p::LinPred, N) where {T}
    nar = _has_ar(p.ar) ? N : 0
    return PredWS{T}(zeros(T, N), zeros(T, N), [GroupWS{T}(gb) for gb in p.groups], zeros(T, length(p.b)),
                     [MoWS{T}(m) for m in p.mo], zeros(T, nar), zeros(T, nar),
                     [zeros(T, length(sb.zs)) for sb in p.smooths],
                     reduce(vcat, [_mecor_ws(T, m, N) for m in p.mecor]; init=Matrix{T}[]))
end
PredWS{T}(p::ScalarDpar, N) where {T} = PredWS{T}(T[], zeros(T, 1), GroupWS{T}[])

"""
    GLMDensity

The posterior log density of a model (up to a constant), on the
unconstrained scale, implementing the LogDensityProblems interface with a
closed-form gradient.

Holds a scratch workspace, so one `GLMDensity` must not be shared between
threads: [`fresh_workspace`](@ref) gives an independent copy per chain.
"""
struct GLMDensity{F<:Family,P<:Tuple,W<:Tuple,T,X<:AbstractVector}
    family::F
    preds::P
    ws::W
    y::Vector{T}
    trials::Vector{Int}      # empty unless the family uses trials
    xdata::X                 # per-observation family data (`_xdata`), empty if none
    weights::Vector{T}       # empty when unweighted
    N::Int
    dim::Int
    lconst::Float64          # data-only log-likelihood constants (reporting only)
    prior_only::Bool         # brms sample_prior = "only": skip the likelihood
    ordered::Vector{Int}     # θ indices of intercepts kept in increasing order (mixtures), else empty
    obuf::Vector{T}          # per-chain θ as the model sees it (ordered / pinned)
    pinned::Vector{Int}      # θ indices fixed by constant() priors
    pinvals::Vector{T}       # their unconstrained values
    Jmi::Vector{Int}         # y | mi(): missing rows of the response ...
    ymi::UnitRange{Int}      # ... and the θ indices of their values (y is then per chain)
end

function GLMDensity(family, preds::Tuple, y::Vector{T}, trials, weights, dim, lconst; prior_only::Bool=false,
                    ordered=Int[], pinned=Int[], pinvals=T[], Jmi=Int[], ymi=1:0) where {T}
    N = length(y)
    family = _prec_family(T, family)
    ws = map(p -> PredWS{T}(p, N), preds)
    # members of a shared block work in ONE GroupWS
    shared = Dict{Int,Any}()
    for (p, pw) in zip(preds, ws)
        p isa LinPred || continue
        for (j, gb) in enumerate(p.groups)
            gb.sid > 0 || continue
            pw.groups[j] = get!(shared, gb.sid, pw.groups[j])
        end
    end
    isempty(Jmi) || (y = copy(y))           # imputed entries are written per gradient
    return GLMDensity(family, preds, ws, y, trials, _convert_x(T, _xdata(family, y, trials)), weights, N, dim,
                      lconst, prior_only, collect(Int, ordered),
                      zeros(T, isempty(ordered) && isempty(pinned) ? 0 : dim), collect(Int, pinned), T.(pinvals),
                      collect(Int, Jmi), ymi)
end

# y | mi() with the missing values as θ entries (brms), rather than integrated
# out: then ymi lies inside θ; generated draws (mimarg.jl) lie past it
_mi_sampled(ld::GLMDensity) = !isempty(ld.Jmi) && last(ld.ymi) <= ld.dim

# --- ordered intercepts (mixture identification, brms order = "mu") ------------
#
# The sampler's θ holds u at the ordered slots; the model sees
#     tau_1 = u_1,   tau_k = tau_{k-1} + exp(u_k)
# (Stan's `ordered` transform) with log-Jacobian sum_{k >= 2} u_k. Stored draws
# are put on the tau scale after sampling (`_order_draws!`), so everything
# downstream (constrain, predictions) reads tau directly.
function _order!(out, ordered, θ)
    copyto!(out, θ)
    @inbounds for k in 2:length(ordered)
        out[ordered[k]] = out[ordered[k - 1]] + exp(θ[ordered[k]])
    end
    return out
end
function _order_generic(ordered, θ)
    isempty(ordered) && return θ, zero(eltype(θ))
    out = Vector{eltype(θ)}(θ)
    lj = zero(eltype(θ))
    for k in 2:length(ordered)
        out[ordered[k]] = out[ordered[k - 1]] + exp(θ[ordered[k]])
        lj += θ[ordered[k]]
    end
    return out, lj
end
function _order_draws!(ld::GLMDensity, D::AbstractArray)
    isempty(ld.ordered) && isempty(ld.pinned) && return D
    buf = zeros(eltype(D), size(D, 1))
    for c in axes(D, 3), i in axes(D, 2)
        θ = view(D, :, i, c)
        _order!(buf, ld.ordered, θ)
        for (k, v) in zip(ld.pinned, ld.pinvals)
            buf[k] = v
        end
        θ .= buf
    end
    return D
end

# constant() priors: the model sees the fixed value at a pinned slot; the
# sampler's coordinate there is an independent standard normal, decoupled
# from everything (so sampling is unaffected), and stored draws are
# overwritten with the constant afterwards (`_order_draws!`).

_convert_x(::Type{T}, xd) where {T} = isempty(xd) ? T[] : map(v -> v isa Tuple ? map(T, v) : T(v), xd)

"""
    fresh_workspace(ld) -> GLMDensity

A copy of `ld` sharing all data but with its own scratch buffers, so chains
on different threads never write to the same memory.
"""
fresh_workspace(ld::GLMDensity) =
    GLMDensity(ld.family, ld.preds, ld.y, ld.trials, ld.weights, ld.dim, ld.lconst; prior_only=ld.prior_only,
               ordered=ld.ordered, pinned=ld.pinned, pinvals=ld.pinvals, Jmi=ld.Jmi, ymi=ld.ymi)

LogDensityProblems.capabilities(::Type{<:GLMDensity}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(ld::GLMDensity) = ld.dim

@inline _at(e::AbstractVector, i) = @inbounds e[i]
@inline _at(e::Real, i) = e
@inline _xd(ld, i) = isempty(ld.xdata) ? zero(eltype(ld.y)) : @inbounds ld.xdata[i]

# --- the correlation Cholesky transform --------------------------------------
#
# Stan's `cholesky_factor_corr` constraint, so that unconstrained values (and
# therefore initialisation and adaptation behaviour) mean the same thing as in
# brms: u -> z = tanh(u) (canonical partial correlations) -> rows of L.
# Returns L and the log-Jacobian of u -> L.
function _cholcorr(u::AbstractVector, q::Int)
    T = eltype(u)
    L = zeros(T, q, q)
    L[1, 1] = one(T)
    lj = zero(T)
    k = 1
    for i in 2:q
        z = tanh(u[k]); k += 1
        lj += log1p(-z * z)
        L[i, 1] = z
        ss = z * z
        for j in 2:(i - 1)
            z = tanh(u[k]); k += 1
            lj += log1p(-z * z)
            lj += log1p(-ss) / 2
            L[i, j] = z * sqrt(1 - ss)
            ss += L[i, j]^2
        end
        L[i, i] = sqrt(1 - ss)
    end
    return L, lj
end

# LKJ density of the correlation matrix, written on its Cholesky factor
# (Stan's lkj_corr_cholesky), WITHOUT the normalising constant.
function lkj_logpdf_chol(L, eta)
    q = size(L, 1)
    s = zero(eltype(L))
    for k in 2:q
        s += _pv(s, q - k + 2eta - 2) * log(L[k, k])
    end
    return s
end

# In-place version of `_cholcorr` for the analytic path (no allocation).
function _cholcorr!(L::AbstractMatrix{T}, u::AbstractVector{T}) where {T}
    q = size(L, 1)
    fill!(L, zero(T))
    L[1, 1] = one(T)
    k = 1
    @inbounds for i in 2:q
        t = one(T)                     # t_j = 1 - (sum of squares so far)
        for j in 1:(i - 1)
            z = tanh(u[k]); k += 1
            L[i, j] = z * sqrt(t)
            t *= 1 - z * z
        end
        L[i, i] = sqrt(t)
    end
    return L
end

# The three products of a correlated group block, with z the q x J matrix
# stored column-major in θ[zr]. Explicit loops, not mul! on
# reshape(view(θ, zr), q, J): that argument type sends LinearAlgebra to its
# generic matmul fallback, whose `wrap(A, tA)` returns a Union that
# juliac --trim=safe cannot resolve. q is the number of correlated terms
# (2-5 in practice), so BLAS bought nothing here; L is lower triangular.

# M = L * z
function _lz!(M::AbstractMatrix{T}, L::AbstractMatrix{T}, θ::AbstractVector{T}, zr::UnitRange{Int}) where {T}
    q, J = size(M)
    o = first(zr) - 1
    @inbounds for j in 1:J
        c = o + (j - 1) * q
        for k in 1:q
            s = zero(T)
            for i in 1:k
                s += L[k, i] * θ[c + i]
            end
            M[k, j] = s
        end
    end
    return M
end

# g[zr] += L' * G
function _ltg_add!(g::AbstractVector{T}, zr::UnitRange{Int}, L::AbstractMatrix{T}, G::AbstractMatrix{T}) where {T}
    q, J = size(G)
    o = first(zr) - 1
    @inbounds for j in 1:J
        c = o + (j - 1) * q
        for i in 1:q
            s = zero(T)
            for k in i:q
                s += L[k, i] * G[k, j]
            end
            g[c + i] += s
        end
    end
    return g
end

# A = G * z'
function _gzt!(A::AbstractMatrix{T}, G::AbstractMatrix{T}, θ::AbstractVector{T}, zr::UnitRange{Int}) where {T}
    q, J = size(G)
    o = first(zr) - 1
    fill!(A, zero(T))
    @inbounds for j in 1:J
        c = o + (j - 1) * q
        for i in 1:q
            zi = θ[c + i]
            for k in 1:q
                A[k, i] += G[k, j] * zi
            end
        end
    end
    return A
end

# Closed-form value and gradient of the correlation block. `A` is
# d loglik / d L (lower triangle used). Returns the value of
# logjac(u) + lkj(L(u)) and writes into `gc` the gradient w.r.t. u of
#     sum(A .* L(u)) + logjac(u) + lkj(L(u))
# i.e. the likelihood's chain rule through L plus the prior and Jacobian.
#
# Derivation, per row i (rows are independent; m = i - 1 entries):
#   z_j = tanh(u_j),  a_j = log(1 - z_j^2),  t_1 = 1,  t_{j+1} = t_j (1 - z_j^2)
#   L_ij = z_j sqrt(t_j),  L_ii = sqrt(t_{m+1}),  log t_j = sum_{l<j} a_l
#   F = sum_j A_ij L_ij + A_ii L_ii + sum_j a_j + 1/2 sum_{j=2}^m log t_j
#       + (c_i / 2) log t_{m+1},          c_i = q - i + 2 eta - 2
#   dF/da_l = 1 + (m - l)/2 + c_i/2 + 1/2 sum_{j>l} A_ij L_ij + A_ii L_ii / 2
#   dF/du_l = A_il sqrt(t_l) (1 - z_l^2) - 2 z_l dF/da_l
# The tests check this against ForwardDiff of the generic transform.
function _corr_value_grad!(gc::AbstractVector{T}, A::AbstractMatrix{T}, u::AbstractVector{T},
                           q::Int, eta::Float64) where {T}
    val = zero(T)
    k0 = 1
    @inbounds for i in 2:q
        m = i - 1
        c = T(q - i + 2eta - 2)
        # forward over the row, stashing t_j in gc for the backward sweep
        t = one(T)
        for j in 1:m
            gc[k0 + j - 1] = t
            z = tanh(u[k0 + j - 1])
            val += log1p(-z * z)                   # tanh Jacobian
            j >= 2 && (val += log(t) / 2)          # CPC Jacobian
            t *= 1 - z * z
        end
        Lii = sqrt(t)
        val += c * log(t) / 2                      # LKJ on the Cholesky factor
        # backward: suffix sum of A_ij L_ij
        suffix = A[i, i] * Lii / 2
        for l in m:-1:1
            kl = k0 + l - 1
            z = tanh(u[kl])
            st = sqrt(gc[kl])
            dFda = 1 + T(m - l) / 2 + c / 2 + suffix
            gc[kl] = A[i, l] * st * (1 - z * z) - 2z * dFda
            suffix += A[i, l] * z * st / 2         # add the L_il term for smaller l
        end
        k0 += m
    end
    return val
end

# Reference implementation by ForwardDiff, kept for the tests: the gradient
# w.r.t. u of sum(A .* L(u)) + lkj(L(u)) + logjac(u).
function _corr_grad(A::Matrix{Float64}, u::AbstractVector{Float64}, q::Int, eta::Float64)
    f = function (v)
        L, lj = _cholcorr(v, q)
        s = lj + lkj_logpdf_chol(L, eta)
        for j in 1:q, i in j:q
            s += A[i, j] * L[i, j]
        end
        return s
    end
    g = ForwardDiff.gradient(f, collect(u))
    L, lj = _cholcorr(u, q)
    return lj + lkj_logpdf_chol(L, eta), g
end

# =============================================================================
# Analytic path
# =============================================================================

function _group_forward!(gw::GroupWS, gb::GroupBlock, θ)
    _owns(gb) || return nothing
    gb.role == 1 && fill!(gw.G, zero(eltype(gw.G)))     # members accumulate into it
    q, J = _q(gb), _J(gb)
    if _has_corr(gb)
        _cholcorr!(gw.L, view(θ, gb.corr))
        _lz!(gw.M, gw.L, θ, gb.z)
    else
        copyto!(gw.M, view(θ, gb.z))
    end
    if _hascov(gb)
        mul!(gw.B, gw.M, transpose(gb.Lcov))
        copyto!(gw.M, gw.B)
    end
    @inbounds for k in 1:q
        s = exp(θ[gb.sd[k]])
        for j in 1:J
            gw.R[k, j] = s * gw.M[k, j]
        end
    end
    if _student(gb)
        dfv = 1 + exp(θ[gb.df])
        @inbounds for (j, kk) in enumerate(gb.udf)
            m = sqrt(dfv * exp(θ[kk]))
            for k in 1:q
                gw.R[k, j] *= m
            end
        end
    end
    return nothing
end

# eta += Z r[idx]: a plain gather for (1 | g)
function _group_eta!(η, gb::GroupBlock, gw::GroupWS)
    R, idx, r0 = gw.R, gb.idx, first(gb.rows) - 1
    if _ismm(gb)
        Zt, q = gb.Zt, _qm(gb)
        @inbounds for i in eachindex(η), m in axes(gb.mmJ, 2)
            j = gb.mmJ[i, m]; wt = gb.mmW[i, m]
            s = zero(eltype(η))
            for k in 1:q
                s += Zt[k, i] * R[r0 + k, j]
            end
            η[i] += wt * s
        end
        return η
    end
    if gb.unitZ
        @inbounds for i in eachindex(η)
            η[i] += R[r0 + 1, idx[i]]
        end
    else
        Zt = gb.Zt
        q = _qm(gb)
        @inbounds for i in eachindex(η)
            j = idx[i]
            s = zero(eltype(η))
            for k in 1:q
                s += Zt[k, i] * R[r0 + k, j]
            end
            η[i] += s
        end
    end
    return η
end

# X * b added into eta, and X' * w added into g, for the population-level
# design. BLAS for wide designs; for a handful of columns a plain SIMD loop,
# because the BLAS call overhead dominates there (profiled: gemv! was 22% of a
# K = 3, N = 1000 gaussian chain).
const _SMALL_K = 8
function _addmul!(η, X::Matrix{T}, b) where {T}
    K = size(X, 2)
    if K > _SMALL_K
        mul!(η, X, b, true, true)
        return η
    end
    @inbounds for k in 1:K
        bk = b[k]
        @simd for i in eachindex(η)
            η[i] += X[i, k] * bk
        end
    end
    return η
end
function _addmul_t!(g, X::Matrix{T}, w) where {T}
    K = size(X, 2)
    if K > _SMALL_K
        mul!(g, transpose(X), w, true, true)
        return g
    end
    @inbounds for k in 1:K
        s = zero(T)
        @simd for i in eachindex(w)
            s += X[i, k] * w[i]
        end
        g[k] += s
    end
    return g
end

function _eta!(pw::PredWS, p::LinPred, θ)
    η = pw.eta
    a = p.intercept > 0 ? θ[p.intercept] : zero(eltype(η))
    if p.has_offset
        η .= p.offset .+ a
    else
        fill!(η, a)
    end
    if p.hs.on
        _addmul!(η, p.Xc, _hs_bvals!(pw.bvals, p.hs, p.b, θ))
    elseif p.r2d2.on
        _addmul!(η, p.Xc, _r2d2_bvals!(pw.bvals, p.r2d2, p.b, θ))
    elseif p.bounded
        @inbounds for (j, k) in enumerate(p.b)
            pw.bvals[j] = first(_btrans(p.b_lb[j], p.b_ub[j], θ[k]))
        end
        _addmul!(η, p.Xc, pw.bvals)
    else
        isempty(p.b) || _addmul!(η, p.Xc, view(θ, p.b))
    end
    for (gb, gw) in zip(p.groups, pw.groups)
        _group_forward!(gw, gb, θ)
        _group_eta!(η, gb, gw)
    end
    for (m, mw) in zip(p.mo, pw.mo)
        _mo_forward!(η, m, mw, θ)
    end
    for m in p.me
        isempty(m.zme) || _me_forward!(η, m, θ)          # (empty: a correlated latent, below)
    end
    for (k, mc) in enumerate(p.mecor)
        _mecor_forward!(η, mc, pw.mecor, k, θ)
    end
    for t in p.gp
        _gp_forward!(η, t, θ)
    end
    for m in p.mi
        _mi_forward!(η, m, θ)
    end
    for t in p.lac
        _lac_forward!(η, t, θ)
    end
    for t in p.car
        _car_forward!(η, t, θ)
    end
    for (sb, buf) in zip(p.smooths, pw.sbuf)
        _smooth_forward!(η, sb, θ, buf)
    end
    _has_ar(p.ar) && _ar_forward!(η, p.ar, pw.base, θ)
    return η
end
_eta!(pw::PredWS, p::ScalarDpar, θ) = θ[p.idx]

# True (at compile time) when every auxiliary parameter is a scalar, so the
# family's aux-only work can be hoisted out of the observation loop.
@inline _aux_scalar(::Tuple{AbstractVector,Vararg{Real}}) = true
@inline _aux_scalar(::Tuple) = false

_reset!(pw::PredWS, ::LinPred) = nothing
_reset!(pw::PredWS, ::ScalarDpar) = (pw.w[1] = zero(eltype(pw.w)); nothing)

# write one observation's d/d eta into the workspace: per observation for a
# LinPred, accumulated for a ScalarDpar
@inline _put!(w, ::LinPred, i, v) = (@inbounds w[i] = v; nothing)
@inline _put!(w, ::ScalarDpar, i, v) = (@inbounds w[1] += v; nothing)
@inline _puts!(::Tuple{}, ::Tuple{}, i, ::Tuple{}, wt) = nothing
@inline function _puts!(ws::Tuple, ps::Tuple, i, g::Tuple, wt)
    _put!(ws[1], ps[1], i, wt * g[1])
    _puts!(Base.tail(ws), Base.tail(ps), i, Base.tail(g), wt)
end

function _obs_loop!(ld::GLMDensity{F,P,W,T}, etas) where {F,P,W,T}
    fam, y = ld.family, ld.y
    weighted = !isempty(ld.weights)
    foreach(_reset!, ld.ws, ld.preds)     # zero the accumulated (non per-obs) gradients
    if _aux_scalar(etas)
        # Common case: only mu varies by observation. Aux-only work is done
        # once (`_pre`), mu's derivative is stored per observation, and the
        # aux derivatives are summed in registers.
        pre = _pre(fam, map(e -> _at(e, 1), etas))
        mu, wmu = etas[1], ld.ws[1].w
        ll, gs = _obs_loop_scalar(fam, y, ld, mu, wmu, pre, weighted)
        _write_sums!(Base.tail(ld.ws), gs)
        return ll
    end
    wv = map(pw -> pw.w, ld.ws)
    ll = zero(T)
    @inbounds for i in 1:ld.N
        v, gi = _ll_grad(fam, y[i], _xd(ld, i), map(e -> _at(e, i), etas))
        wt = weighted ? ld.weights[i] : one(T)
        ll += wt * v
        _puts!(wv, ld.preds, i, gi, wt)
    end
    return ll
end

# zero tuple of the aux-gradient sums, typed by what the family returns
@inline _zero_aux(::Tuple{Any}, ::Type{T}) where {T} = ()
@inline _zero_aux(g::Tuple, ::Type{T}) where {T} = ntuple(_ -> zero(T), length(g) - 1)

# Stand-ins for "no per-observation family data" and "no weights", so the
# hot loop has no branches on either (`isempty` inside the loop was enough
# to stop it vectorising): indexing returns a constant the compiler folds.
struct _NoX{T} end
@inline Base.getindex(::_NoX{T}, i) where {T} = zero(T)
struct _Ones{T} end
@inline Base.getindex(::_Ones{T}, i) where {T} = one(T)

function _obs_loop_scalar(fam, y, ld::GLMDensity{F,P,W,T}, mu, wmu, pre, weighted) where {F,P,W,T}
    xd = isempty(ld.xdata) ? _NoX{T}() : ld.xdata
    wts = weighted ? ld.weights : _Ones{T}()
    return _scalar_kernel!(fam, y, xd, wts, mu, wmu, pre, ld.N)
end

function _scalar_kernel!(fam, y::AbstractVector{T}, xd, wts, mu, wmu, pre, N) where {T}
    _, g0 = _ll_grad_pre(fam, y[1], xd[1], mu[1], pre)
    gs = _zero_aux(g0, T)
    ll = zero(T)
    @inbounds @simd for i in 1:N
        wt = wts[i]
        v, gi = _ll_grad_pre(fam, y[i], xd[i], mu[i], pre)
        ll += wt * v
        wmu[i] = wt * gi[1]
        gs = map((a, b) -> a + wt * b, gs, Base.tail(gi))
    end
    return ll, gs
end

@inline _write_sums!(::Tuple{}, ::Tuple{}) = nothing
@inline function _write_sums!(ws::Tuple, gs::Tuple)
    ws[1].w[1] = gs[1]
    _write_sums!(Base.tail(ws), Base.tail(gs))
end

function _group_pullback!(g, gb::GroupBlock, gw::GroupWS{T}, w, θ) where {T}
    q, J = _q(gb), _J(gb)
    G, M, idx, r0 = gw.G, gw.M, gb.idx, first(gb.rows) - 1
    gb.role == 0 && fill!(G, zero(T))      # shared blocks were zeroed by their first member
    if _ismm(gb)
        Zt = gb.Zt
        @inbounds for i in eachindex(w), m in axes(gb.mmJ, 2)
            j = gb.mmJ[i, m]; wi = w[i] * gb.mmW[i, m]
            for k in 1:_qm(gb)
                G[r0 + k, j] += Zt[k, i] * wi
            end
        end
    elseif gb.unitZ
        @inbounds for i in eachindex(w)
            G[r0 + 1, idx[i]] += w[i]
        end
    else
        Zt = gb.Zt
        @inbounds for i in eachindex(w)
            j = idx[i]; wi = w[i]
            for k in 1:_qm(gb)
                G[r0 + k, j] += Zt[k, i] * wi
            end
        end
    end
    _closes(gb) || return zero(T)
    lp = zero(T)
    if _student(gb)
        # r[:, j] = dfm_j * r0[:, j]: d/d dfm_j = G[:, j]' r0[:, j] = G[:, j]' R[:, j] / dfm_j,
        # then G becomes d loglik / d r0. dfm_j = sqrt(df udf_j): d dfm / d log udf = dfm / 2,
        # d dfm / d df = dfm / (2 df), d df / d u = df - 1.
        u = θ[gb.df]; dfv = 1 + exp(u)
        gdf = zero(T)
        @inbounds for (j, kk) in enumerate(gb.udf)
            v = θ[kk]
            m = sqrt(dfv * exp(v))
            dm = zero(T)
            for k in 1:q
                dm += G[k, j] * gw.R[k, j]
                G[k, j] *= m
            end
            dm /= m
            # inv_chi_square(udf | df) on v = log udf, with its Jacobian
            g[kk] += dm * m / 2 - dfv / 2 + exp(-v) / 2
            gdf += dm * m / (2dfv) - T(_LOG2) / 2 - digamma(dfv / 2) / 2 - v / 2
            lp += -dfv / 2 * T(_LOG2) - loggamma(dfv / 2) - dfv / 2 * v - exp(-v) / 2
        end
        vd, dd = prior_kernel_grad(gb.df_prior, dfv)
        g[gb.df] += (gdf + dd) * (dfv - 1) + 1      # +1: log-Jacobian u
        lp += vd + u
    end
    @inbounds for k in 1:q
        # r[k,:] = sd_k * M[k,:], with sd_k = exp(u_k)
        u = θ[gb.sd[k]]
        s = exp(u)
        dsd = zero(T)
        for j in 1:J
            dsd += G[k, j] * M[k, j]
        end
        v, d = prior_kernel_grad(gb.sd_priors[k], s)
        g[gb.sd[k]] += s * (dsd + d) + 1          # +1: log-Jacobian of exp
        lp += v + u
        for j in 1:J
            G[k, j] *= s                          # G is now d loglik / d M
        end
    end
    if _hascov(gb)                                # M = M0 Lcov': d / d M0 = G Lcov
        mul!(gw.B, G, gb.Lcov)
        copyto!(G, gw.B)
    end
    if _has_corr(gb)
        _ltg_add!(g, gb.z, gw.L, G)               # d / d z = L' * dM
        _gzt!(gw.A, G, θ, gb.z)                   # d / d L = dM * z'
        lp += _corr_value_grad!(gw.gc, gw.A, view(θ, gb.corr), q, gb.lkj)
        @inbounds for (n, k) in enumerate(gb.corr)
            g[k] += gw.gc[n]
        end
    else
        o = first(gb.z) - 1
        @inbounds for i in eachindex(G)
            g[o + i] += G[i]
        end
    end
    # standard-normal prior on z (kernel only)
    @inbounds for k in gb.z
        zk = θ[k]
        g[k] -= zk
        lp -= zk * zk / 2
    end
    return lp
end

function _pullback!(g, p::LinPred, pw::PredWS{T}, θ) where {T}
    w = pw.w
    lp = zero(T)
    if _has_ar(p.ar)
        # the AR term is linear in the base predictor: pull w back through it
        lp += _ar_pullback!(g, p.ar, pw.base, w, pw.wbase, θ)
        w = pw.wbase
    end
    for (m, mw) in zip(p.mo, pw.mo)
        lp += _mo_pullback!(g, m, mw, w, θ)
    end
    for m in p.me
        isempty(m.zme) || (lp += _me_pullback!(g, m, w, θ))
    end
    for (k, mc) in enumerate(p.mecor)
        lp += _mecor_pullback!(g, mc, pw.mecor, k, w, θ)
    end
    for t in p.gp
        lp += _gp_pullback!(g, t, w, θ)
    end
    for m in p.mi
        lp += _mi_pullback!(g, m, w, θ)
    end
    for t in p.lac
        lp += _lac_pullback!(g, t, w, θ)
    end
    for t in p.car
        lp += _car_pullback!(g, t, w, θ)
    end
    for (sb, buf) in zip(p.smooths, pw.sbuf)
        lp += _smooth_pullback!(g, sb, w, θ, buf)
    end
    if p.intercept > 0
        k = p.intercept
        v, d = prior_kernel_grad(p.int_prior, θ[k])
        g[k] += sum(w) + d
        lp += v
    end
    if !isempty(p.b)
        _addmul_t!(view(g, p.b), p.Xc, w)                     # d loglik / d b
        if p.hs.on
            lp += _hs_pullback!(g, p.hs, p.b, θ)
        elseif p.r2d2.on
            lp += _r2d2_pullback!(g, p.r2d2, p.b, θ)
        elseif p.bounded
            @inbounds for (j, k) in enumerate(p.b)
                b, db, lj, dlj = _btrans(p.b_lb[j], p.b_ub[j], θ[k])
                v, d = prior_kernel_grad(p.b_priors[j], b)
                g[k] = (g[k] + d) * db + dlj
                lp += v + lj
            end
        elseif p.any_b_prior
            @inbounds for (j, k) in enumerate(p.b)
                v, d = prior_kernel_grad(p.b_priors[j], θ[k])
                g[k] += d
                lp += v
            end
        end
    end
    for (gb, gw) in zip(p.groups, pw.groups)
        lp += _group_pullback!(g, gb, gw, w, θ)
    end
    return lp
end

function _pullback!(g, p::ScalarDpar, pw::PredWS{T}, θ) where {T}
    u = θ[p.idx]
    x, dx, lj, dlj = _dpar_transform(p, u)
    v, d = prior_kernel_grad(p.prior, x)
    g[p.idx] += pw.w[1] + d * dx + dlj
    return v + lj
end

_pullbacks!(g, ::Tuple{}, ::Tuple{}, θ) = 0
_pullbacks!(g, ps::Tuple, ws::Tuple, θ) =
    _pullback!(g, ps[1], ws[1], θ) + _pullbacks!(g, Base.tail(ps), Base.tail(ws), θ)

"""
    logdensity_and_gradient!(g, ld, θ) -> logdensity

Closed-form log density (up to a constant) and gradient (written into `g`).
`θ` must have the element type `ld` was built with.
"""
function logdensity_and_gradient!(g::AbstractVector{T}, ld::GLMDensity{F,P,W,T}, θ::AbstractVector{T}) where {F,P,W,T}
    isempty(ld.ordered) && isempty(ld.pinned) && return _lg_body!(g, ld, θ)
    o = ld.ordered
    θm = _order!(ld.obuf, o, θ)
    for (k, c) in zip(ld.pinned, ld.pinvals)
        θm[k] = c
    end
    v = _lg_body!(g, ld, θm)
    for k in ld.pinned                      # the decoupled standard-normal coordinate
        g[k] = -θ[k]
        v -= θ[k]^2 / 2
    end
    # chain back to u: d tau_j / d u_1 = 1, d tau_j / d u_k = exp(u_k) for j >= k
    s = zero(T)
    @inbounds for k in length(o):-1:1
        s += g[o[k]]
        if k == 1
            g[o[k]] = s
        else
            g[o[k]] = exp(θ[o[k]]) * s + 1
            v += θ[o[k]]
        end
    end
    return v
end

function _lg_body!(g::AbstractVector{T}, ld::GLMDensity{F,P,W,T}, θ::AbstractVector{T}) where {F,P,W,T}
    _mi_sampled(ld) && return _lg_body_mi!(g, ld, θ)
    fill!(g, zero(T))
    etas = map((pw, p) -> _eta!(pw, p, θ), ld.ws, ld.preds)
    ll = if ld.prior_only
        foreach(pw -> fill!(pw.w, zero(T)), ld.ws)
        zero(T)
    else
        _obs_loop!(ld, etas)
    end
    lp = _pullbacks!(g, ld.preds, ld.ws, θ)
    return ll + lp
end

# y | mi(): the missing responses are θ[ymi]. For the (location) families this
# allows, d ll / d y = -d ll / d mu, read off mu's per-observation derivative.
function _lg_body_mi!(g::AbstractVector{T}, ld::GLMDensity{F,P,W,T}, θ::AbstractVector{T}) where {F,P,W,T}
    @inbounds for (k, n) in enumerate(ld.Jmi)
        ld.y[n] = θ[ld.ymi[k]]
    end
    fill!(g, zero(T))
    etas = map((pw, p) -> _eta!(pw, p, θ), ld.ws, ld.preds)
    ll = ld.prior_only ? (foreach(pw -> fill!(pw.w, zero(T)), ld.ws); zero(T)) : _obs_loop!(ld, etas)
    wmu = ld.ws[1].w
    gy = [-wmu[n] for n in ld.Jmi]
    lp = _pullbacks!(g, ld.preds, ld.ws, θ)
    @inbounds for (k, j) in enumerate(ld.ymi)
        g[j] += gy[k]
    end
    return ll + lp
end

function LogDensityProblems.logdensity_and_gradient(ld::GLMDensity{F,P,W,T}, θ::AbstractVector{T}) where {F,P,W,T}
    # a fresh gradient vector every call: samplers keep the returned vector
    # in their phase point, so handing back a shared buffer would corrupt it
    g = Vector{T}(undef, ld.dim)
    # outside the support (DomainError etc.): -Inf, a rejected proposal for any
    # sampler using this interface (AdvancedHMC), as Stan does
    v = _safe_grad!((gg, x) -> logdensity_and_gradient!(gg, ld, x), g, θ)
    return v, g
end
LogDensityProblems.logdensity_and_gradient(ld::GLMDensity{F,P,W,T}, θ::AbstractVector) where {F,P,W,T} =
    throw(ArgumentError("this density was built for element type $T but got a $(eltype(θ)) vector; rebuild with `T=$(eltype(θ))`"))

# =============================================================================
# Generic path: plain evaluation, any number type
# =============================================================================

function _group_R(gb::GroupBlock, θ)
    q, J = _q(gb), _J(gb)
    sd = exp.(θ[gb.sd])
    zmat = reshape(θ[gb.z], q, J)
    M = _has_corr(gb) ? first(_cholcorr(θ[gb.corr], q)) * zmat : zmat
    _hascov(gb) && (M = M * transpose(gb.Lcov))
    _student(gb) || return sd .* M
    return (sd .* M) .* last(_dfm(gb, θ))'
end

function _eta_generic(p::LinPred, θ, N)
    T = eltype(θ)
    η = Vector{T}(undef, N)
    η .= p.offset
    p.intercept > 0 && (η .+= θ[p.intercept])
    isempty(p.b) || (η .+= p.Xc * _bvalues(p, θ))
    for gb in p.groups
        R = _group_R(gb, θ)
        for i in 1:N
            if _ismm(gb)
                for m in axes(gb.mmJ, 2)
                    η[i] += gb.mmW[i, m] * sum(gb.Z[i, k] * R[gb.rows[k], gb.mmJ[i, m]] for k in 1:_qm(gb))
                end
                continue
            end
            j = gb.idx[i]
            s = zero(T)
            for k in 1:_qm(gb)
                s += gb.Z[i, k] * R[gb.rows[k], j]
            end
            η[i] += s
        end
    end
    for m in p.mo
        η .+= first(_mo_generic(m, θ))
    end
    for m in p.me
        isempty(m.zme) || (η .+= first(_me_generic(m, θ)))
    end
    for mc in p.mecor
        η .+= first(_mecor_generic(mc, θ, length(η)))
    end
    for t in p.gp
        η = η .+ first(_gp_generic(t, θ))
    end
    for m in p.mi
        η = η .+ first(_mi_generic(m, θ))
    end
    for t in p.lac
        η = η .+ first(_lac_generic(t, θ))
    end
    for t in p.car
        η = η .+ first(_car_generic(t, θ))
    end
    for sb in p.smooths
        η .+= first(_smooth_generic(sb, θ))
    end
    _has_ar(p.ar) && (η = _ar_generic(η, p.ar, θ))
    return η
end
_eta_generic(p::ScalarDpar, θ, N) = θ[p.idx]

function _prior_generic(p::LinPred, θ)
    T = eltype(θ)
    lp = zero(T)
    for m in p.mo
        lp += last(_mo_generic(m, θ))
    end
    for m in p.me
        isempty(m.zme) || (lp += last(_me_generic(m, θ)))
    end
    for mc in p.mecor
        lp += last(_mecor_generic(mc, θ, size(p.Xc, 1)))
    end
    for t in p.gp
        lp += last(_gp_generic(t, θ))
    end
    for m in p.mi
        lp += last(_mi_generic(m, θ))
    end
    for t in p.lac
        lp += last(_lac_generic(t, θ))
    end
    for t in p.car
        lp += last(_car_generic(t, θ))
    end
    for sb in p.smooths
        lp += last(_smooth_generic(sb, θ))
    end
    _has_ar(p.ar) && (lp += _ar_prior_generic(p.ar, θ))
    p.intercept > 0 && (lp += prior_kernel(p.int_prior, θ[p.intercept]))
    if p.hs.on
        lp += _hs_prior_generic(p.hs, p.b, θ)
    elseif p.r2d2.on
        lp += _r2d2_prior_generic(p.r2d2, p.b, θ)
    else
        for (j, k) in enumerate(p.b)
            b, _, lj, _ = _btrans(p.b_lb[j], p.b_ub[j], θ[k])
            lp += prior_kernel(p.b_priors[j], b) + lj
        end
    end
    for gb in p.groups
        _owns(gb) || continue            # a shared block's prior is counted once
        for (n, k) in enumerate(gb.sd)
            lp += prior_kernel(gb.sd_priors[n], exp(θ[k])) + θ[k]
        end
        if _has_corr(gb)
            L, lj = _cholcorr(θ[gb.corr], _q(gb))
            lp += lj + lkj_logpdf_chol(L, gb.lkj)
        end
        for k in gb.z
            lp -= θ[k]^2 / 2
        end
        _student(gb) && (lp += _udf_prior(gb, θ))
    end
    return lp
end
function _prior_generic(p::ScalarDpar, θ)
    x, _, lj, _ = _dpar_transform(p, θ[p.idx])
    return prior_kernel(p.prior, x) + lj
end

"""
    LogDensityProblems.logdensity(ld, θ)

The log density of [`logdensity_and_gradient!`](@ref) without the gradient,
written in generic Julia code that accepts any element type, so automatic
differentiation backends can differentiate it.
"""
function LogDensityProblems.logdensity(ld::GLMDensity, θraw::AbstractVector)
    θ, ljo = _order_generic(ld.ordered, θraw)
    if !isempty(ld.pinned)
        θ = Vector{promote_type(eltype(θ), eltype(ld.pinvals))}(θ)
        for (k, c) in zip(ld.pinned, ld.pinvals)
            θ[k] = c
            ljo -= θraw[k]^2 / 2
        end
    end
    etas = map(p -> _eta_generic(p, θ, ld.N), ld.preds)
    T = eltype(θ)
    ll = ljo
    yv = ld.y
    if _mi_sampled(ld)                       # y | mi(): the imputed responses
        yv = Vector{promote_type(eltype(θ), eltype(ld.y))}(ld.y)
        for (k, n) in enumerate(ld.Jmi)
            yv[n] = θ[ld.ymi[k]]
        end
    end
    for i in 1:(ld.prior_only ? 0 : ld.N)
        v = _ll(ld.family, yv[i], _xd(ld, i), map(e -> _at(e, i), etas))
        ll += isempty(ld.weights) ? v : ld.weights[i] * v
    end
    lp = ll
    for p in ld.preds
        lp += _prior_generic(p, θ)
    end
    return lp
end

# =============================================================================
# Names and constrained values (for the chain)
# =============================================================================

_pfx(d::Symbol) = d === :mu ? "" : string(d, "_")
# joint labels of a shared block ("Intercept", "sigma_Intercept"), with the
# response put after the dpar in multivariate names, as everywhere else
function _jlabels(gb::GroupBlock, resp)
    return map(gb.jlabels) do l
        # labels are stored as "<dpar>__<coef>", or "<resp>::<dpar>__<coef>" when
        # the ID spans responses (each label then carries its own response)
        rr = resp
        if occursin("::", l)
            rr, l = split(l, "::"; limit=2)
        end
        k = findfirst("__", l)
        d, c = l[1:(first(k) - 1)], l[(last(k) + 1):end]
        r = rr == "" ? "" : "$(rr)_"
        d == "mu" ? "$(r)$(c)" : "$(d)_$(r)$(c)"
    end
end

function _thresholds(ld::GLMDensity)
    k = findfirst(p -> p isa Union{Thresholds,ThresholdsCS}, ld.preds)
    k === nothing && return nothing
    return ld.preds[k] isa ThresholdsCS ? ld.preds[k].base : ld.preds[k]
end
_csblock(ld::GLMDensity) = (k = findfirst(p -> p isa ThresholdsCS, ld.preds); k === nothing ? nothing : ld.preds[k])

"""
    variables(ld) -> Vector{String}

Parameter names in brms's naming scheme (`b_Intercept`, `b_x`,
`sd_g__Intercept`, `cor_g__Intercept__x`, `sigma`, `Intercept`,
`r_g[a,Intercept]`), in the order [`constrain`](@ref) returns values.
"""
function variables(ld::GLMDensity; resp::AbstractString="")
    out = String[]
    # multivariate models put the response after the dpar, as brms:
    # b_y_x, b_sigma_y_x, sd_g__sigma_y_Intercept, sigma_y, r_g__sigma_y[...]
    px(d) = resp == "" ? _pfx(d) : d === :mu ? "$(resp)_" : "$(d)_$(resp)_"
    rs = resp == "" ? "" : "_" * resp
    lps = _linpreds(ld.preds)
    thr = _thresholds(ld)
    for p in lps
        x = px(p.dpar)
        if p.dpar === :mu && thr !== nothing
            if thr isa Thresholds{true}          # brms: b_Intercept[a,1] (level, threshold)
                append!(out, ["b_$(px(:mu))Intercept[$l,$k]" for (l, n) in zip(thr.labels, thr.len) for k in 1:n])
            else
                append!(out, ["b_$(px(:mu))Intercept[$k]" for k in eachindex(thr.idx)])
            end
        end
        p.intercept > 0 && push!(out, "b_$(x)Intercept")
        append!(out, ["b_$(x)$(c)" for c in p.coefnames])
        append!(out, ["bsp_$(x)$(m.label)" for m in p.mo])
        append!(out, ["bsp_$(x)$(m.label)" for m in p.me])
        append!(out, ["bsp_$(x)$(m.label)" for m in p.mi])
        if p.dpar === :mu && (cs = _csblock(ld)) !== nothing
            append!(out, ["bcs_$(x)$(l)[$k]" for l in cs.labels for k in 1:_K1(cs)])
        end
    end
    for p in lps, gb in p.groups
        x = px(p.dpar)
        append!(out, ["sd_$(gb.label)__$(x)$(c)$(gb.sdsfx)" for c in gb.coefs])
        _student(gb) && push!(out, "df_$(gb.label)")
    end
    for p in lps, sb in p.smooths
        push!(out, "sds_$(px(p.dpar))$(sb.label)")
    end
    for p in lps, gb in p.groups
        (_has_corr(gb) && _owns(gb)) || continue
        x = px(p.dpar); q = _q(gb)
        lab = gb.role == 0 ? ["$(x)$(c)$(gb.sdsfx)" for c in gb.coefs] : _jlabels(gb, resp)
        for j in 2:q, i in 1:(j - 1)
            push!(out, "cor_$(gb.label)__$(lab[i])__$(lab[j])")
        end
    end
    for p in ld.preds
        p isa ScalarDpar && push!(out, _scalar_label(p.dpar) * rs)
        p isa ThetaSimplex && append!(out, ["theta$k$rs" for k in 1:_K(p)])
        p isa BhazSimplex && append!(out, ["sbhaz$rs[$k]" for k in 1:_K(p)])
        if p isa LnCorBlock && p.dpar === :cortime            # unstr(): cortime__t1__t2 (brms)
            tl = ld.family.tlabels
            for k in 2:p.D, j in 1:(k - 1)
                push!(out, "cortime$(rs)__$(tl[j])__$(tl[k])")
            end
        elseif p isa LnCorBlock
            cats = ["mu" * c for c in ld.family.cats[2:end]]
            for k in 2:p.D, j in 1:(k - 1)
                push!(out, "lncor$(rs)__$(cats[j])__$(cats[k])")
            end
        end
    end
    for p in lps, m in p.mo, f in eachindex(m.Ds)        # brms simo_mox1[k], simo_mox1:mox22[k]
        append!(out, ["simo_$(px(p.dpar))$(m.label)$(f)[$k]" for k in 1:m.Ds[f]])
    end
    for p in lps, m in p.me
        append!(out, ["meanme_$(m.latent)", "sdme_$(m.latent)"])
    end
    for p in lps, mc in p.mecor
        append!(out, mc.names)
    end
    for p in lps, t in p.lac
        x = rstrip(px(p.dpar), '_'); sfx = x == "" ? "" : "_" * x
        push!(out, "sderr$sfx")
        append!(out, [replace(n, "[" => "$sfx[") for n in t.names])         # ar[1], ar_y[1]
    end
    for p in lps, t in p.car
        x = rstrip(px(p.dpar), '_')
        sfx = x == "" ? "" : "_" * x
        push!(out, "sdcar$sfx")
        t.type === :bym2 && push!(out, "rhocar$sfx")
        t.type === :escar && push!(out, "car$sfx")
        append!(out, ["rcar$sfx[$k]" for k in 1:t.Nloc])
    end
    for p in lps, t in p.gp
        append!(out, ["sdgp_$(px(p.dpar))$(t.label)", "lscale_$(px(p.dpar))$(t.label)"])
        append!(out, ["zgp_$(px(p.dpar))$(t.label)[$k]" for k in 1:_nz(t)])
    end
    for p in lps
        append!(out, ["ar$rs[$k]" for k in 1:p.ar.p])
        append!(out, ["ma$rs[$k]" for k in 1:p.ar.q])
    end
    for p in lps
        p.hs.on && append!(out, ["hs_$(px(p.dpar))global", "hs_$(px(p.dpar))slab"])
        p.r2d2.on && push!(out, "R2D2_$(px(p.dpar))R2")
    end
    thr === nothing || append!(out, ["Intercept$rs[$k]" for k in eachindex(thr.idx)])
    for p in lps
        p.intercept > 0 && push!(out, (p.dpar === :mu ? "Intercept" : "Intercept_$(p.dpar)") * rs)
    end
    for p in lps, gb in p.groups
        r = p.dpar === :mu ? (resp == "" ? "r_$(gb.label)" : "r_$(gb.label)__$(resp)") : "r_$(gb.label)__$(p.dpar)$rs"
        for c in gb.coefs, l in gb.levels
            push!(out, "$(r)[$(l),$(c)]")
        end
    end
    for p in lps, sb in p.smooths
        append!(out, ["s_$(px(p.dpar))$(sb.label)[$j]" for j in eachindex(sb.zs)])
    end
    append!(out, ["Ymi$rs[$k]" for k in eachindex(ld.ymi)])
    return out
end

"""
    constrain(ld, θ) -> Vector{Float64}

The values of [`variables(ld)`](@ref) at unconstrained point `θ`.
"""
function constrain(ld::GLMDensity, θ::AbstractVector)
    out = Float64[]
    lps = _linpreds(ld.preds)
    thr = _thresholds(ld)
    tau = thr === nothing ? Float64[] : _eta_generic(thr, θ, 0).tau
    for p in lps
        b = _bvalues(p, θ)
        # ordinal thresholds on the uncentred scale: eta enters as tau - eta
        p.dpar === :mu && append!(out, tau .+ dot(p.means, b))
        p.intercept > 0 && push!(out, θ[p.intercept] - dot(p.means, b))
        append!(out, b)
        append!(out, [θ[m.bsp] for m in p.mo])
        append!(out, [θ[m.bsp] for m in p.me])
        append!(out, [θ[m.bsp] for m in p.mi])
        if p.dpar === :mu && (cs = _csblock(ld)) !== nothing
            B = reshape(θ[cs.bcs], _Kcs(cs), _K1(cs))
            append!(out, [B[j, k] for j in 1:_Kcs(cs) for k in 1:_K1(cs)])
        end
    end
    for p in lps, gb in p.groups
        append!(out, exp.(θ[gb.sd[gb.rows]]))
        _student(gb) && push!(out, 1 + exp(θ[gb.df]))              # df_g (brms)
    end
    for p in lps, sb in p.smooths
        push!(out, exp(θ[sb.sds]))
    end
    for p in lps, gb in p.groups
        (_has_corr(gb) && _owns(gb)) || continue
        L, _ = _cholcorr(θ[gb.corr], _q(gb))
        Ω = L * L'
        q = _q(gb)
        for j in 2:q, i in 1:(j - 1)
            push!(out, Ω[i, j])
        end
    end
    for p in ld.preds
        p isa ScalarDpar && push!(out, _natural(p, θ[p.idx]))
        p isa ThetaSimplex && append!(out, first(_stick(θ[p.idx], _K(p))))
        p isa BhazSimplex && append!(out, first(_stick(θ[p.idx], _K(p))))
        if p isa LnCorBlock
            L, _ = _cholcorr(θ[p.idx], p.D); R = L * L'
            for k in 2:p.D, j in 1:(k - 1)
                push!(out, R[j, k])
            end
        end
    end
    for p in lps, m in p.mo, f in eachindex(m.Ds)
        append!(out, first(_stick(θ[m.simos[f]], m.Ds[f])))
    end
    for p in lps, m in p.me
        append!(out, [θ[m.meanme], exp(θ[m.sdme])])
    end
    for p in lps, mc in p.mecor
        K = length(mc.terms)
        L, _ = _cholcorr(θ[mc.cor], K); R = L * L'
        for k in 2:K, j in 1:(k - 1)
            push!(out, R[j, k])
        end
    end
    for p in lps, t in p.lac
        push!(out, exp(θ[t.sd]))
        append!(out, collect(_lac_rho(t, θ)))
    end
    for p in lps, t in p.car
        push!(out, exp(θ[t.sd]))
        t.type in (:bym2, :escar) && push!(out, logistic(θ[t.rho]))
        append!(out, _car_r(t, θ))
    end
    for p in lps, t in p.gp
        append!(out, [exp(θ[t.sdgp]), exp(θ[t.lscale])])
        append!(out, θ[t.z])
    end
    for p in lps
        append!(out, θ[p.ar.idx])
        append!(out, θ[p.ar.ma_idx])
    end
    for p in lps
        p.hs.on && append!(out, [exp(θ[p.hs.glob]), exp(θ[p.hs.slab])])
        p.r2d2.on && push!(out, logistic(θ[p.r2d2.r2]))
    end
    append!(out, tau)
    for p in lps
        p.intercept > 0 && push!(out, θ[p.intercept])
    end
    for p in lps, gb in p.groups
        R = _group_R(gb, θ)
        for k in gb.rows, j in 1:_J(gb)
            push!(out, R[k, j])
        end
    end
    for p in lps, sb in p.smooths
        append!(out, exp(θ[sb.sds]) .* θ[sb.zs])
    end
    append!(out, θ[ld.ymi])
    return out
end
