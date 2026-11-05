# =============================================================================
# Multi-category responses (brms): `multinomial()` for counts and
# `dirichlet()` for proportions, with the response a matrix — written here as
# `cbind(a, b, c) ~ x`. The first category is the reference; every other
# category k has its own predictor `mu<k>` named after the category (brms:
# b_mub_x, b_muc_x), from the main formula unless `mu<k> ~ ...` is given.
#
#   multinomial:  log P = sum_k y_k log p_k + log n! - sum_k log y_k!,
#                 p = softmax(0, eta_2..eta_K);  d/d eta_k = y_k - n p_k
#   dirichlet:    y ~ Dirichlet(phi * softmax(0, eta)), phi > 0 (gamma(0.01, 0.01));
#                 differentiated by dual numbers
#
# The family carries the training response matrix (each observation's row
# reaches `_ll` as its per-observation data tuple), as shifted_lognormal
# carries min(y). K is a type parameter, set from the data at build time.
# =============================================================================

Multinomial{L}() where {L} = Multinomial{L,0}(String[], zeros(0, 0))
DirichletFam{L}() where {L} = DirichletFam{L,0}(String[], zeros(0, 0))
const _MultiCat = Union{Multinomial,DirichletFam,LogisticNormal}
_with_matrix(::Multinomial{L}, cats, Y) where {L} = Multinomial{L,length(cats)}(cats, Y)
_with_matrix(::DirichletFam{L}, cats, Y) where {L} = DirichletFam{L,length(cats)}(cats, Y)
# one method per family, no `_ncat(::_MultiCat)` fallback: Julia 1.12 picked such a
# fallback over the parametric methods below, so every family reported 0 categories
_ncat(::Multinomial{L,K}) where {L,K} = K
_ncat(::DirichletFam{L,K}) where {L,K} = K

dpars(f::Multinomial) = Tuple(Symbol("mu", c) for c in f.cats[2:end])
dpars(f::DirichletFam) = (Tuple(Symbol("mu", c) for c in f.cats[2:end])..., :phi)
dpar_link_name(::_MultiCat, d::Symbol) = d === :phi ? "log" : "logit"
_is_cat_mu(f::_MultiCat, d::Symbol) = d !== :phi

# per-observation data: the counts (multinomial) or log proportions (dirichlet)
_xdata(f::Multinomial{L,K}, y, trials) where {L,K} = [ntuple(k -> f.Y[i, k], Val(K)) for i in axes(f.Y, 1)]
_xdata(f::DirichletFam{L,K}, y, trials) where {L,K} = [ntuple(k -> log(f.Y[i, k]), Val(K)) for i in axes(f.Y, 1)]

@inline function _ll(::Multinomial, y, x, eta)
    lse, _ = _cat_lse(eta)
    v = -x[1] * lse
    for k in eachindex(eta)
        v += x[k + 1] * (eta[k] - lse)
    end
    return v
end
@inline function _ll_grad(f::Multinomial, y, x, eta)
    lse, p = _cat_lse(eta)
    n = sum(x)
    return _ll(f, y, x, eta), ntuple(j -> x[j + 1] - n * p[j], length(eta))
end
_ll_const(::Multinomial, y, t) = 0.0
@inline _ll_grad_pre(f::Multinomial, y, x, m, p) = _ll_grad(f, y, x, (m,))     # K = 2: the scalar fast path
_ll_const_obs(::Multinomial, y, t, x) = loggamma(sum(x) + 1) - sum(v -> loggamma(v + 1), x)

@inline function _ll(f::DirichletFam{L,K}, y, x, eta) where {L,K}
    mus = ntuple(j -> eta[j], Val(K - 1))
    lse, p = _cat_lse(mus)
    phi = exp(eta[K])
    a1 = phi * (1 - sum(p))
    v = loggamma(phi) - loggamma(a1) + (a1 - 1) * x[1]
    for k in 1:(K - 1)
        a = phi * p[k]
        v += -loggamma(a) + (a - 1) * x[k + 1]
    end
    return v
end
_ll_const(::DirichletFam, y, t) = 0.0

function _check_response(f::_MultiCat, y, t)
    isempty(f.Y) && return true
    f isa Multinomial && _chk(all(v -> isinteger(v) && v >= 0, f.Y), "multinomial responses must be non-negative counts")
    f isa DirichletFam && _chk(all(>(0), f.Y) && all(r -> isapprox(sum(r), 1; atol=1e-6), eachrow(f.Y)),
                               "dirichlet responses must be positive proportions summing to one per row")
    return true
end

# category probabilities of one observation (epred: times the trials for multinomial)
function _mc_probs(f::_MultiCat, eta)
    p = last(_cat_lse(Tuple(eta[1:(_ncat(f) - 1)])))
    return [1 - sum(p); collect(p)]
end
# logistic_normal: one draw of the proportions, and their mean by Monte Carlo
function _ln_draw(rng, f::LogisticNormal, eta)
    D = _ncat(f) - 1
    mu, ls, L = _ln_split(f, eta)
    x = collect(mu) .+ exp.(collect(ls)) .* (L * randn(rng, D))
    e = exp.(vcat(0.0, x)); return e ./ sum(e)
end
function _ln_mean(f::LogisticNormal, eta; n=200)
    rng = Random.Xoshiro(1)
    return sum(_ln_draw(rng, f, eta) for _ in 1:n) ./ n
end
_mean(f::_MultiCat, eta, t) = NaN        # vector-valued: see posterior_epred
_rand(rng, f::_MultiCat, eta, t) = NaN

"""
    multinomial(; link = :logit)

Multinomial distribution for counts in `K` categories, with the response
given as columns: `cbind(a, b, c) ~ x`. The first column is the reference
category; each other category `c` gets its own predictor `muc`, which shares
the main formula unless given its own (`muc ~ z`).
"""
multinomial(; link=nothing) = _make_family(Multinomial, link)

"""
    dirichlet(; link = :logit)

Dirichlet distribution for proportions in `K` categories that sum to one,
with the response given as columns: `cbind(a, b, c) ~ x`. As in
[`multinomial`](@ref), every category but the first gets a predictor; `phi` is
the precision.
"""
dirichlet(; link=nothing) = _make_family(DirichletFam, link)

# =============================================================================
# logistic_normal (brms): x = log(y_k / y_1), k = 2..K, is multivariate normal
#     x ~ MVN(mu, diag(sigma) Lcor),   ll = mvn(x) - sum_k log y_k
# with one predictor and one sigma per non-reference category and an LKJ
# correlation `lncor` (a block, like residual correlations): each observation
# returns, with s = (x - mu) ./ sigma, v = L \ s, a = L' \ v,
#     d/d mu_k = a_k / sigma_k,  d/d log sigma_k = a_k s_k - 1,  d/d L = a v' - diag(1 ./ L)
# and the block accumulates d/d L for the closed-form LKJ reverse pass.
# =============================================================================
LogisticNormal{L}() where {L} = LogisticNormal{L,0}(String[], zeros(0, 0))
_with_matrix(::LogisticNormal{L}, cats, Y) where {L} = LogisticNormal{L,length(cats)}(cats, Y)
_ncat(::LogisticNormal{L,K}) where {L,K} = K
dpars(f::LogisticNormal) = (Tuple(Symbol("mu", c) for c in f.cats[2:end])...,
                            Tuple(Symbol("sigma", c) for c in f.cats[2:end])..., :lncor)
dpar_link_name(::LogisticNormal, d::Symbol) = startswith(string(d), "mu") ? "identity" : d === :lncor ? "cholesky" : "log"
_is_cat_mu(f::LogisticNormal, d::Symbol) = startswith(string(d), "mu")
# per observation: (log y_2/y_1, ..., log y_K/y_1, sum log y)
_xdata(f::LogisticNormal{L,K}, y, trials) where {L,K} =
    [(ntuple(k -> log(f.Y[i, k + 1]) - log(f.Y[i, 1]), Val(K - 1))..., sum(log, view(f.Y, i, :))) for i in axes(f.Y, 1)]
_ll_const(::LogisticNormal, y, t) = 0.0
_ll_const_obs(f::LogisticNormal{L,K}, y, t, x) where {L,K} = -(K - 1) * log(2π) / 2
_check_response(f::LogisticNormal, y, t) = isempty(f.Y) ||
    _chk(all(>(0), f.Y) && all(r -> isapprox(sum(r), 1; atol=1e-6), eachrow(f.Y)),
         "logistic_normal responses must be positive proportions summing to one per row")

struct LnCorBlock
    dpar::Symbol
    idx::UnitRange{Int}
    D::Int
    eta::Float64                 # LKJ concentration
end
struct LnCorL{M}
    L::M
end
@inline _at(c::LnCorL, i) = c
struct LnCorGrad{D,T}
    a::NTuple{D,T}
    v::NTuple{D,T}
    invd::NTuple{D,T}            # 1 / L_kk: the log-determinant's share
    s::T                         # observation weight
end
@inline Base.:*(w::Real, g::LnCorGrad) = LnCorGrad(g.a, g.v, g.invd, oftype(g.s, w * g.s))

PredWS{T}(p::LnCorBlock, N) where {T} =
    PredWS{T}(zeros(T, p.D * p.D), zeros(T, p.D * p.D), GroupWS{T}[], zeros(T, max(length(p.idx), 1)), MoWS{T}[],
              T[], T[], Vector{T}[])
function _eta!(pw::PredWS, p::LnCorBlock, θ)
    L = reshape(pw.eta, p.D, p.D)
    _cholcorr!(L, view(θ, p.idx))
    return LnCorL(L)
end
_reset!(pw::PredWS, ::LnCorBlock) = fill!(pw.w, zero(eltype(pw.w)))
@inline function _put!(w, p::LnCorBlock, i, g::LnCorGrad)
    D = p.D
    @inbounds for j in 1:D, k in j:D           # lower triangle, column-major
        w[(j - 1) * D + k] += g.s * g.a[k] * g.v[j]
    end
    @inbounds for k in 1:D
        w[(k - 1) * D + k] -= g.s * g.invd[k]
    end
    return nothing
end
function _pullback!(g, p::LnCorBlock, pw::PredWS{T}, θ) where {T}
    A = reshape(pw.w, p.D, p.D)
    gc = pw.bvals
    val = _corr_value_grad!(view(gc, 1:length(p.idx)), A, view(θ, p.idx), p.D, p.eta)
    @inbounds for (j, k) in enumerate(p.idx)
        g[k] += gc[j]
    end
    return val
end
_eta_generic(p::LnCorBlock, θ, N) = LnCorL(first(_cholcorr(θ[p.idx], p.D)))
function _prior_generic(p::LnCorBlock, θ)
    L, lj = _cholcorr(θ[p.idx], p.D)
    return lj + lkj_logpdf_chol(L, p.eta)
end

@inline function _ln_split(f::LogisticNormal{Lk,K}, eta) where {Lk,K}
    D = K - 1
    return ntuple(j -> eta[j], Val(D)), ntuple(j -> eta[D + j], Val(D)), eta[2D + 1].L
end
@inline function _ll_grad(f::LogisticNormal{Lk,K}, y, x, eta) where {Lk,K}
    D = K - 1
    mu, ls, L = _ln_split(f, eta)
    T = promote_type(typeof(mu[1]), typeof(ls[1]), eltype(L))
    s = ntuple(k -> (x[k] - mu[k]) * exp(-ls[k]), Val(D))
    # v = L \ s, a = L' \ v (forward / back substitution on tuples)
    v = _fwdsub(L, s, Val(D)); a = _backsub(L, v, Val(D))
    ll = -sum(abs2, v) / 2 - sum(ls) - sum(k -> log(L[k, k]), 1:D) - x[K]
    gmu = ntuple(k -> a[k] * exp(-ls[k]), Val(D))
    gls = ntuple(k -> a[k] * s[k] - 1, Val(D))
    # the diagonal part of d/d L: -1/L_kk per observation, carried in a v' by adding
    # the term to the k = j entries in the block (see _put!): fold it into v here
    return T(ll), (gmu..., gls..., LnCorGrad(a, ntuple(k -> v[k], Val(D)), ntuple(k -> 1 / L[k, k], Val(D)), one(T)))
end
@inline _ll(f::LogisticNormal, y, x, eta) = first(_ll_grad(f, y, x, eta))
@inline function _fwdsub(L, s, ::Val{D}) where {D}
    v = ntuple(_ -> zero(promote_type(eltype(L), eltype(s))), Val(D))
    for i in 1:D
        t = s[i]
        for j in 1:(i - 1)
            t -= L[i, j] * v[j]
        end
        v = Base.setindex(v, t / L[i, i], i)
    end
    return v
end
@inline function _backsub(L, v, ::Val{D}) where {D}
    a = ntuple(_ -> zero(promote_type(eltype(L), eltype(v))), Val(D))
    for i in D:-1:1
        t = v[i]
        for j in (i + 1):D
            t -= L[j, i] * a[j]
        end
        a = Base.setindex(a, t / L[i, i], i)
    end
    return a
end
"""
    logistic_normal(; link = :identity)

Logistic-normal distribution for proportions in `K` categories, with the
response given as columns: `cbind(a, b, c) ~ x`. The log ratios of each
category to the first are multivariate normal, with a mean `muc` and scale
`sigmac` per non-reference category `c` and a correlation matrix `lncor`.
"""
logistic_normal(; link=nothing) = _make_family(LogisticNormal, link)
