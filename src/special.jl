# =============================================================================
# Special linear-predictor terms: monotonic effects `mo(x)` and
# autoregressive terms `ar(time, gr, p)`, as brms writes them.
#
# mo(x): x an ordinal predictor coded 0..D. brms (Buerkner & Charpentier 2020):
#     eta[i] += bsp * D * sum(simo[1:x[i]])
# with simo a D-simplex (Dirichlet prior, brms default dirichlet(1)) under
# Stan's stick-breaking transform, and bsp an ordinary coefficient (class b,
# coef "mo<x>"). Gradient: with S[j] = sum of w over observations at level j,
#     d/d bsp     = D * sum_j S[j] * C[j],        C = cumsum(simo), C[0] = 0
#     d/d simo[l] = bsp * D * sum_{j >= l} S[j]
# and a closed-form reverse pass through the stick-breaking transform.
#
# ar(time, gr, p), ma(time, gr, q), arma(time, gr, p, q): brms's formulation
# (cov = FALSE), read off its Stan code. With base the predictor without the
# ARMA part and lags only within the same `gr` group, in `time` order:
#     m[n]   = base[n] + sum_i ma[i] * e[n-i]        (i <= q)
#     e[n]   = y[n] - m[n]                           (the residual brms lags)
#     eta[n] = m[n]    + sum_j ar[j] * e[n-j]        (j <= p)
# so AR acts on the residuals INCLUDING the MA part (for pure AR, e = y - base).
# MA makes e recursive; the gradient is the matching reverse recursion, run
# in reverse series order, for the adjoint ebar = d ll / d e:
#     ebar[n] = -w[n] + sum_j ar[j] w[n+j] - sum_i ma[i] ebar[n+i]
#     d/d base[n] = -ebar[n]
#     d/d ma[i]   = -sum_n ebar[n] e[n-i],    d/d ar[j] = sum_n w[n] e[n-j]
# For pure AR this reduces to the plain adjoint w[m] - sum ar[k] w[m+k].
# =============================================================================

# A monotonic TERM (brms): bsp * prod_f mo(simo_f, x_f) * prod(numeric covariates),
# e.g. mo(x) (one factor), mo(x):z (one factor times z), mo(x1):mo(x2) (two
# factors). Every term has its own coefficient and its own simplex per factor,
# even when it repeats a variable (brms simo_1, simo_2, ...).
struct MoTerm
    label::String                    # brms coefficient name: "mox", "mox:z", "mox1:mox2"
    xs::Vector{Vector{Int}}          # per factor: level of each observation, 0..D
    Ds::Vector{Int}
    bsp::Int                         # θ index of the coefficient
    simos::Vector{UnitRange{Int}}    # per factor: D - 1 unconstrained simplex parameters
    bsp_prior::ScalarPrior
    alphas::Vector{Float64}          # per factor: Dirichlet concentration
    csp::Vector{Float64}             # product of the term's numeric covariates (empty: 1)
    moatoms::Vector{Atom}            # the mo() variables, for new data
    cspatoms::Vector{Atom}           # the numeric covariates, for new data
    mins::Vector{Int}                # per factor: the training minimum of a numeric mo() variable (level 0)
    cspfac::Vector{Tuple{Atom,String}}   # mo(x):f -- (factor, level): csp includes the indicator f == level
end
MoTerm(label, x, D, bsp, simo, bsp_prior, alpha, lo=0) =
    MoTerm(label, [x], [D], bsp, [simo], bsp_prior, [Float64(alpha)], Float64[], Atom[], Atom[], [lo],
           Tuple{Atom,String}[])
# the covariate product of an interaction term: numeric covariates times the
# indicators of its factor levels (empty = 1)
function _mo_csp(vals, nums, facs)
    isempty(nums) && isempty(facs) && return Float64[]
    parts = Vector{Float64}[Float64.(vals[a]) for a in nums]
    for (a, lev) in facs
        push!(parts, [_level_name(v) == lev ? 1.0 : 0.0 for v in vals[a]])
    end
    return reduce((u, v) -> u .* v, parts)
end

struct ARTerm{T}
    p::Int                 # AR order (0 = none)
    idx::UnitRange{Int}    # θ indices of ar[1:p]
    q::Int                 # MA order (0 = none)
    ma_idx::UnitRange{Int} # θ indices of ma[1:q]
    prev::Matrix{Int}      # prev[n, k] = observation k steps before n in its group, 0 if none
    order::Vector{Int}     # all observations, each after its predecessors
    y::Vector{T}
    priors::Vector{ScalarPrior}
    ma_priors::Vector{ScalarPrior}
end
ARTerm{T}() where {T} = ARTerm{T}(0, 1:0, 0, 1:0, zeros(Int, 0, 0), Int[], T[], ScalarPrior[], ScalarPrior[])
_has_ar(a::ARTerm) = a.p > 0 || a.q > 0

struct MoWS{T}               # per factor of the term:
    simo::Vector{Vector{T}}  # the simplex
    C::Vector{Vector{T}}     # C[j + 1] = sum(simo[1:j])
    S::Vector{Vector{T}}     # S[j + 1] = sum over level-j observations of w x (the other factors' values)
    rem::Vector{Vector{T}}   # stick length before each break
    z::Vector{Vector{T}}
end
MoWS{T}(m::MoTerm) where {T} = MoWS{T}([zeros(T, D) for D in m.Ds], [zeros(T, D + 1) for D in m.Ds],
                                       [zeros(T, D + 1) for D in m.Ds], [zeros(T, D) for D in m.Ds],
                                       [zeros(T, max(D - 1, 0)) for D in m.Ds])

# --- Stan's stick-breaking simplex transform ------------------------------------

# fills simo (and rem, z), returns the log-Jacobian
function _stick!(simo, rem, z, u)
    K = length(simo)
    T = eltype(simo)
    stick = one(T)
    lj = zero(T)
    @inbounds for k in 1:(K - 1)
        a = u[k] - log(K - k)
        zk = logistic(a)
        rem[k] = stick
        z[k] = zk
        simo[k] = stick * zk
        lj += log(stick) - log1pexp(-a) - log1pexp(a)
        stick -= simo[k]
    end
    simo[K] = stick
    return lj
end
function _stick(u, K)
    T = eltype(u)
    simo = zeros(T, K); rem = zeros(T, K); z = zeros(T, max(K - 1, 0))
    lj = _stick!(simo, rem, z, u)
    return simo, lj
end

# value of logJ + Dirichlet kernel, gradient w.r.t. u of
#     sum(a .* simo) + logJ + sum((alpha - 1) .* log.(simo))
function _stick_grad!(gu, a, simo, rem, z, alpha)
    K = length(simo)
    T = eltype(simo)
    val = zero(T)
    for k in 1:K
        alpha == 1 || (val += (alpha - 1) * log(simo[k]))
    end
    xbar(k) = a[k] + (alpha == 1 ? zero(T) : (alpha - 1) / simo[k])
    rbar = xbar(K)                 # adjoint of the stick left after break K-1
    @inbounds for k in (K - 1):-1:1
        r, zk = rem[k], z[k]
        val += log(r) + log(zk) + log1p(-zk)
        xb = xbar(k)
        gu[k] = (xb - rbar) * r * zk * (1 - zk) + (1 - 2zk)
        rbar = xb * zk + rbar * (1 - zk) + 1 / r
    end
    return val
end

# --- analytic path hooks (called from density.jl) ----------------------------------

function _mo_forward!(η, m::MoTerm, mw::MoWS{T}, θ) where {T}
    K = length(m.Ds)
    for f in 1:K
        _stick!(mw.simo[f], mw.rem[f], mw.z[f], view(θ, m.simos[f]))
        C = mw.C[f]
        C[1] = 0
        for j in 1:m.Ds[f]
            C[j + 1] = C[j] + mw.simo[f][j]
        end
    end
    b = θ[m.bsp]
    if K == 1 && isempty(m.csp)                       # plain mo(x): one gather
        c = b * m.Ds[1]; x = m.xs[1]; C = mw.C[1]
        @inbounds for i in eachindex(η)
            η[i] += c * C[x[i] + 1]
        end
        return η
    end
    @inbounds for i in eachindex(η)
        v = isempty(m.csp) ? b : b * T(m.csp[i])
        for f in 1:K
            v *= m.Ds[f] * mw.C[f][m.xs[f][i] + 1]
        end
        η[i] += v
    end
    return η
end

# S_f[j + 1] = sum over observations at level j of factor f of w x csp x the
# other factors' mo values; then d/d bsp = D_1 sum_j S_1[j + 1] C_1[j + 1] and,
# per factor, a[l] = bsp D_f sum_{j >= l} S_f[j] into the simplex reverse pass.
function _mo_pullback!(g, m::MoTerm, mw::MoWS{T}, w, θ) where {T}
    K = length(m.Ds)
    foreach(S -> fill!(S, zero(T)), mw.S)
    @inbounds for i in eachindex(w)
        c = isempty(m.csp) ? w[i] : w[i] * T(m.csp[i])
        if K == 1
            mw.S[1][m.xs[1][i] + 1] += c
        else
            for f in 1:K
                o = c
                for h in 1:K
                    h == f || (o *= m.Ds[h] * mw.C[h][m.xs[h][i] + 1])
                end
                mw.S[f][m.xs[f][i] + 1] += o
            end
        end
    end
    bsp = θ[m.bsp]
    db = zero(T)
    for j in 1:m.Ds[1]
        db += mw.S[1][j + 1] * mw.C[1][j + 1]
    end
    v, d = prior_kernel_grad(m.bsp_prior, bsp)
    g[m.bsp] += m.Ds[1] * db + d
    lp = v
    for f in 1:K
        D = m.Ds[f]
        # a[l] = bsp * D * sum_{j >= l} S[j], reusing C's slots (C_f is no longer needed)
        a = mw.C[f]
        acc = zero(T)
        for l in D:-1:1
            acc += mw.S[f][l + 1]
            a[l] = bsp * D * acc
        end
        tmp = zeros(T, length(m.simos[f]))      # tiny (D - 1), simplex reverse pass
        lp += _stick_grad!(tmp, a, mw.simo[f], mw.rem[f], mw.z[f], m.alphas[f])
        g[m.simos[f]] .+= tmp
    end
    return lp
end

# η holds base on entry and eta on exit; `e` (a per-chain N buffer) gets the
# residuals the pullback needs
function _ar_forward!(η, a::ARTerm, e, θ)
    @inbounds for n in a.order
        m = η[n]
        for i in 1:a.q
            k = a.prev[n, i]
            k > 0 && (m += θ[a.ma_idx[i]] * e[k])
        end
        e[n] = a.y[n] - m
        for j in 1:a.p
            k = a.prev[n, j]
            k > 0 && (m += θ[a.idx[j]] * e[k])
        end
        η[n] = m
    end
    return η
end

# w: d/d eta (final); writes d/d base into wb and the ar/ma gradient into g
function _ar_pullback!(g, a::ARTerm, e, w, wb, θ)
    ebar = wb                               # adjoint of e, negated at the end
    @inbounds for n in eachindex(w)
        ebar[n] = -w[n]
    end
    @inbounds for n in Iterators.reverse(a.order)
        for j in 1:a.p
            k = a.prev[n, j]
            k > 0 || continue
            ebar[k] += θ[a.idx[j]] * w[n]
            g[a.idx[j]] += w[n] * e[k]
        end
        # ebar[n] is final here: every observation lagging n comes later in order
        for i in 1:a.q
            k = a.prev[n, i]
            k > 0 || continue
            ebar[k] -= θ[a.ma_idx[i]] * ebar[n]
            g[a.ma_idx[i]] -= ebar[n] * e[k]
        end
    end
    @inbounds for n in eachindex(wb)
        wb[n] = -ebar[n]
    end
    lp = zero(eltype(w))
    for (pr, j) in Iterators.flatten((zip(a.priors, a.idx), zip(a.ma_priors, a.ma_idx)))
        v, d = prior_kernel_grad(pr, θ[j])
        g[j] += d
        lp += v
    end
    return lp
end

# --- generic path ------------------------------------------------------------------

# the term's contribution at levels xs (training: m.xs) and covariates csp
function _mo_generic(m::MoTerm, θ, xs=m.xs, csp=m.csp)
    lp = prior_kernel(m.bsp_prior, θ[m.bsp])
    v = isempty(csp) ? fill(θ[m.bsp], length(xs[1])) : θ[m.bsp] .* csp
    for f in eachindex(m.Ds)
        simo, lj = _stick(θ[m.simos[f]], m.Ds[f])
        C = vcat(zero(eltype(simo)), cumsum(simo))
        lp += lj
        m.alphas[f] == 1 || (lp += sum((m.alphas[f] - 1) .* log.(simo)))
        v = v .* (m.Ds[f] .* C[xs[f] .+ 1])
    end
    return v, lp
end

function _ar_generic(η, a::ARTerm, θ)
    S = promote_type(eltype(η), eltype(θ))
    out = Vector{S}(η)
    e = zeros(S, length(η))
    for n in a.order
        m = out[n]
        for i in 1:a.q
            k = a.prev[n, i]
            k > 0 && (m += θ[a.ma_idx[i]] * e[k])
        end
        e[n] = a.y[n] - m
        for j in 1:a.p
            k = a.prev[n, j]
            k > 0 && (m += θ[a.idx[j]] * e[k])
        end
        out[n] = m
    end
    return out
end
_ar_prior_generic(a::ARTerm, θ) =
    sum((prior_kernel(a.priors[k], θ[j]) for (k, j) in enumerate(a.idx)); init=zero(eltype(θ))) +
    sum((prior_kernel(a.ma_priors[k], θ[j]) for (k, j) in enumerate(a.ma_idx)); init=zero(eltype(θ)))

# --- building -----------------------------------------------------------------------

# lag structure: within each group (or the whole data), observations in time
# order; prev[n, k] is the k-th predecessor of n
function _ar_prev(time, gr, p)
    N = length(time)
    prev = zeros(Int, N, p)
    order = Int[]
    keys_ = gr === nothing ? fill("", N) : string.(gr)
    for gk in unique(keys_)
        rows = findall(==(gk), keys_)
        ord = rows[sortperm(time[rows])]
        append!(order, ord)
        for (pos, n) in enumerate(ord), k in 1:p
            pos - k >= 1 && (prev[n, k] = ord[pos - k])
        end
    end
    return prev, order
end

# levels 0..D of a mo() variable: a factor by its level order, a numeric one
# relative to its TRAINING minimum `lo` (new data must not recode it to its own)
function _mo_levels(v, info::AtomInfo, lo=nothing, D=nothing)
    if iscategorical(info)
        lv = info.levels
        return [findfirst(isequal(x), lv) - 1 for x in v]
    end
    all(isinteger, v) || throw(ArgumentError("mo() needs an integer or ordered-factor predictor"))
    x = Int.(v)
    x = x .- (lo === nothing ? minimum(x) : lo)
    D === nothing || all(k -> 0 <= k <= D, x) ||
        throw(ArgumentError("mo(): new data outside the training range of the variable"))
    return x
end
_mo_min(v, info::AtomInfo) = iscategorical(info) ? 0 : Int(minimum(v))

# =============================================================================
# Measurement error `me(x, sdx)` (brms): the true covariate is latent,
#     Xme_n = meanme + sdme * zme_n,  zme ~ N(0, 1),
#     x_n ~ N(Xme_n, sdx_n)            (the measurement model, in the density)
#     eta_n += bsp * Xme_n
# with bsp a special-effect coefficient (class b, coef "mex<sdx>" as brms) and
# meanme, sdme flat by default. Gradient, with w = d ll / d eta:
#     dXme_n = bsp w_n + (x_n - Xme_n) / sdx_n^2
#     d bsp = sum w_n Xme_n,  d meanme = sum dXme_n,  d sdme = sum dXme_n zme_n,
#     d zme_n = sdme dXme_n - zme_n
# =============================================================================
struct MeTerm{T}
    label::String            # brms coefficient name, e.g. "mexsdx"
    latent::String           # latent-variable name, e.g. "mex"
    x::Vector{T}
    sdx::Vector{T}
    bsp::Int
    meanme::Int
    sdme::Int                # log sdme
    zme::UnitRange{Int}
    bsp_prior::ScalarPrior
    mean_prior::ScalarPrior
    sd_prior::ScalarPrior
end

function _me_forward!(η, m::MeTerm, θ)
    b, mu, s = θ[m.bsp], θ[m.meanme], exp(θ[m.sdme])
    @inbounds for n in eachindex(η)
        η[n] += b * (mu + s * θ[m.zme[n]])
    end
    return η
end
function _me_pullback!(g, m::MeTerm, w, θ)
    T = eltype(w)
    b, mu, u = θ[m.bsp], θ[m.meanme], θ[m.sdme]
    s = exp(u)
    gb = zero(T); gmu = zero(T); gs = zero(T); lp = zero(T)
    @inbounds for n in eachindex(w)
        z = θ[m.zme[n]]
        xme = mu + s * z
        r = (m.x[n] - xme) / m.sdx[n]
        dx = b * w[n] + r / m.sdx[n]
        gb += w[n] * xme
        gmu += dx
        gs += dx * z
        g[m.zme[n]] += s * dx - z
        lp -= (r * r + z * z) / 2
    end
    v1, d1 = prior_kernel_grad(m.bsp_prior, b)
    v2, d2 = prior_kernel_grad(m.mean_prior, mu)
    v3, d3 = prior_kernel_grad(m.sd_prior, s)
    g[m.bsp] += gb + d1
    g[m.meanme] += gmu + d2
    g[m.sdme] += s * (gs + d3) + 1                # +1: log-Jacobian of sdme = exp(u)
    return lp + v1 + v2 + v3 + u
end
function _me_generic(m::MeTerm, θ)
    s = exp(θ[m.sdme])
    xme = θ[m.meanme] .+ s .* θ[m.zme]
    lp = -sum(abs2, (m.x .- xme) ./ m.sdx) / 2 - sum(abs2, θ[m.zme]) / 2 +
         prior_kernel(m.bsp_prior, θ[m.bsp]) + prior_kernel(m.mean_prior, θ[m.meanme]) +
         prior_kernel(m.sd_prior, s) + θ[m.sdme]
    return θ[m.bsp] .* xme, lp
end

# =============================================================================
# Correlated latents: brms's mecor = TRUE (its default) when a formula has two
# or more me() terms,
#     Xme_n = meanme + diag(sdme) Lme z_n,   z_n ~ N(0, I_K),   Lme ~ LKJ(eta)
# (brms: Xme1 = rep_matrix(meanme', N) + (diag_pre_multiply(sdme, Lme) * zme)').
# z is one K x N block, column-major (brms's zme_1); with W = Lme z, term k
# uses W[k, :] where an independent term uses its own z. The terms keep their
# MeTerm (bsp, meanme, sdme, priors, names) with an empty zme. Gradient:
#     dW[k, n] = sdme_k dXme_kn,  d z = Lme' dW - z,  d Lme = dW z'
# (the group-effect helpers _ltg_add! / _gzt!), then Lme's CPC / LKJ reverse
# pass (_corr_value_grad!). Workspace (per chain, PredWS.mecor): L, W, dW, A, gc.
# =============================================================================
struct MeCorTerm{T}
    terms::Vector{MeTerm{T}}
    cor::UnitRange{Int}      # K(K-1)/2 CPC parameters of Lme
    z::UnitRange{Int}        # K x N, column-major
    eta::Float64             # LKJ shape
    names::Vector{String}    # corme__mex1__mex2, ...
end
_mecor_ws(::Type{T}, m::MeCorTerm, N) where {T} =
    (K = length(m.terms); Matrix{T}[zeros(T, K, K), zeros(T, K, N), zeros(T, K, N), zeros(T, K, K),
                                    zeros(T, length(m.cor), 1)])

function _mecor_forward!(η, mc::MeCorTerm, ws, k, θ)
    L, W = ws[5k - 4], ws[5k - 3]
    _cholcorr!(L, view(θ, mc.cor))
    _lz!(W, L, θ, mc.z)
    for (j, m) in enumerate(mc.terms)
        b, mu, s = θ[m.bsp], θ[m.meanme], exp(θ[m.sdme])
        @inbounds for n in eachindex(η)
            η[n] += b * (mu + s * W[j, n])
        end
    end
    return η
end
function _mecor_pullback!(g, mc::MeCorTerm, ws, k, w, θ)
    T = eltype(w)
    L, W, dW, A, gcm = ws[5k - 4], ws[5k - 3], ws[5k - 2], ws[5k - 1], ws[5k]
    lp = zero(T)
    for (j, m) in enumerate(mc.terms)
        b, mu, u = θ[m.bsp], θ[m.meanme], θ[m.sdme]
        s = exp(u)
        gb = zero(T); gmu = zero(T); gs = zero(T)
        @inbounds for n in eachindex(w)
            wz = W[j, n]
            xme = mu + s * wz
            r = (m.x[n] - xme) / m.sdx[n]
            dx = b * w[n] + r / m.sdx[n]
            gb += w[n] * xme
            gmu += dx
            gs += dx * wz
            dW[j, n] = s * dx
            lp -= r * r / 2
        end
        v1, d1 = prior_kernel_grad(m.bsp_prior, b)
        v2, d2 = prior_kernel_grad(m.mean_prior, mu)
        v3, d3 = prior_kernel_grad(m.sd_prior, s)
        g[m.bsp] += gb + d1
        g[m.meanme] += gmu + d2
        g[m.sdme] += s * (gs + d3) + 1            # +1: log-Jacobian of sdme = exp(u)
        lp += v1 + v2 + v3 + u
    end
    _ltg_add!(g, mc.z, L, dW)                     # d z = Lme' dW ...
    @inbounds for i in mc.z
        z = θ[i]
        g[i] -= z                                 # ... - z (the std normal)
        lp -= z * z / 2
    end
    _gzt!(A, dW, θ, mc.z)                         # d Lme = dW z'
    gc = view(gcm, :, 1)
    lp += _corr_value_grad!(gc, A, view(θ, mc.cor), length(mc.terms), mc.eta)
    @inbounds for (q, i) in enumerate(mc.cor)
        g[i] += gc[q]
    end
    return lp
end
# generic path (any number type): eta contribution and log density
function _mecor_generic(mc::MeCorTerm, θ, N)
    K = length(mc.terms)
    L, lj = _cholcorr(θ[mc.cor], K)
    Z = reshape(θ[mc.z], K, N)
    W = L * Z
    η = zeros(eltype(θ), N)
    lp = lj + lkj_logpdf_chol(L, mc.eta) - sum(abs2, Z) / 2
    for (j, m) in enumerate(mc.terms)
        s = exp(θ[m.sdme])
        xme = θ[m.meanme] .+ s .* W[j, :]
        η .+= θ[m.bsp] .* xme
        lp += -sum(abs2, (m.x .- xme) ./ m.sdx) / 2 + prior_kernel(m.bsp_prior, θ[m.bsp]) +
              prior_kernel(m.mean_prior, θ[m.meanme]) + prior_kernel(m.sd_prior, s) + θ[m.sdme]
    end
    return η, lp
end

# =============================================================================
# mi(x) (brms): x as a predictor, its missing entries the parameters Ymi of
# x's own model (`x | mi() ~ ...`, a response of the same multivariate model).
#     eta_n += bsp * xl_n,   xl_n = x_n, or Ymi_k for the k-th missing row
# Gradient: d bsp = sum w_n xl_n;  d Ymi_k += bsp * w_{Jmi_k}. (x's own model
# adds its share of d Ymi in its density, see `_lg_body!`.)
# =============================================================================
struct MiTerm{T}
    label::String             # "mi" * x, brms's coefficient name
    var::String               # the imputed variable (its response name)
    x::Vector{T}              # observed values, NaN where missing
    Jmi::Vector{Int}          # missing rows
    ymi::Base.RefValue{UnitRange{Int}}   # θ indices of the imputed values (set by the multivariate build)
    bsp::Int
    bsp_prior::ScalarPrior
    # integrated out (mimarg.jl): the missing rows add nothing here; their
    # likelihood is the marginal one, and ymi[] then points at generated draws
    # beyond the sampled θ (filled after sampling, used by predictions)
    marg::Bool
end
@inline function _mi_value(m::MiTerm, θ, n, k)
    return k > 0 ? θ[m.ymi[][k]] : m.x[n]
end
function _mi_forward!(η, m::MiTerm, θ)
    b = θ[m.bsp]; r = m.ymi[]
    @inbounds for n in eachindex(η)
        isnan(m.x[n]) || (η[n] += b * m.x[n])        # missing rows: below, from θ
    end
    m.marg && return η
    @inbounds for (k, n) in enumerate(m.Jmi)
        η[n] += b * θ[r[k]]
    end
    return η
end
function _mi_pullback!(g, m::MiTerm, w, θ)
    T = eltype(w)
    b = θ[m.bsp]; r = m.ymi[]
    gb = zero(T)
    @inbounds for n in eachindex(w)
        isnan(m.x[n]) || (gb += w[n] * m.x[n])
    end
    m.marg || @inbounds for (k, n) in enumerate(m.Jmi)
        gb += w[n] * θ[r[k]]
        g[r[k]] += b * w[n]
    end
    v, d = prior_kernel_grad(m.bsp_prior, b)
    g[m.bsp] += gb + d
    return v
end
function _mi_generic(m::MiTerm, θ)
    xl = Vector{promote_type(eltype(θ), eltype(m.x))}(m.x)
    # integrated out: no value inside the sampled θ (0 here); predictions get
    # θ extended by the generated draws, which do hold the imputed values
    gen = m.marg && (isempty(m.ymi[]) || last(m.ymi[]) > length(θ))
    for (k, n) in enumerate(m.Jmi)
        xl[n] = gen ? zero(eltype(xl)) : θ[m.ymi[][k]]
    end
    return θ[m.bsp] .* xl, prior_kernel(m.bsp_prior, θ[m.bsp])
end

# =============================================================================
# Spatial conditional autoregressive terms `car(W, gr = loc, type = )` (brms),
# with W the adjacency matrix of the locations and rcar the spatial effects:
#   icar:   rcar = sdcar * zcar,  -1/2 sum_edges (z_i - z_j)^2 + N(sum z | 0, 0.001 Nloc)
#   bym2:   rcar = sdcar (sqrt(1 - rho) nsz + sqrt(rho / scale) z), the icar prior on z,
#           nsz ~ N(0, 1), rhocar ~ beta(1, 1), scale = geometric mean of the
#           constrained ICAR marginal variances (Riebler et al. 2016)
#   escar:  rcar free, proper CAR: 1/2 (Nloc log tau + sum log(1 - car lambda) - tau Q),
#           Q = phi' D phi - car phi' W phi, tau = 1/sdcar^2, lambda = eig(D^-1/2 W D^-1/2)
#   esicar: rcar = [z; -sum z], 1/2 ((Nloc - 1) log tau - tau (phi' D phi - phi' W phi))
# Gradients are closed form; rbar = per-location sums of w = d ll / d eta.
# =============================================================================
struct CarTerm{T}
    type::Symbol
    J::Vector{Int}                 # observation -> location
    Nloc::Int
    e1::Vector{Int}; e2::Vector{Int}
    nneigh::Vector{Float64}
    eig::Vector{Float64}
    scale::Float64                 # bym2
    sd::Int                        # log sdcar
    z::UnitRange{Int}              # icar/bym2: Nloc; escar: Nloc (rcar itself); esicar: Nloc - 1
    nsz::UnitRange{Int}            # bym2
    rho::Int                       # bym2 rhocar / escar car (logit scale), 0 otherwise
    sd_prior::ScalarPrior
    rho_prior::ScalarPrior
end

function _car_edges(W)
    e1 = Int[]; e2 = Int[]
    for j in axes(W, 2), i in 1:(j - 1)
        W[i, j] != 0 && (push!(e1, i); push!(e2, j))
    end
    return e1, e2
end
function _car_scale(W)
    n = size(W, 1)
    Q = Diagonal(vec(sum(W; dims=2))) - W
    Qp = Q + I * maximum(diag(Q)) * sqrt(eps())
    Qi = inv(Matrix(Qp))
    v = Qi * ones(n)
    Qc = Qi - v * v' / sum(v)                  # the sum-to-zero constrained inverse
    return exp(mean(log, diag(Qc)))
end

# the spatial effects (generic in the number type)
function _car_r(t::CarTerm, θ)
    s = exp(θ[t.sd])
    if t.type === :icar
        return s .* θ[t.z]
    elseif t.type === :bym2
        ρ = logistic(θ[t.rho])
        return s .* (sqrt(1 - ρ) .* θ[t.nsz] .+ sqrt(ρ / t.scale) .* θ[t.z])
    elseif t.type === :escar
        return θ[t.z]
    end
    z = θ[t.z]
    return vcat(z, -sum(z))
end
function _car_prior(t::CarTerm, θ)
    T = eltype(θ)
    u = θ[t.sd]
    lp = prior_kernel(t.sd_prior, exp(u)) + u
    if t.type in (:icar, :bym2)
        z = θ[t.z]
        lp += -sum(abs2, z[t.e1] .- z[t.e2]) / 2 - (sum(z) / (0.001 * t.Nloc))^2 / 2
        if t.type === :bym2
            ρ = logistic(θ[t.rho])
            lp += -sum(abs2, θ[t.nsz]) / 2 + prior_kernel(t.rho_prior, ρ) + log(ρ) + log1p(-ρ)
        end
        return lp
    end
    φ = _car_r(t, θ)
    τ = exp(-2u)
    Wφ = zeros(T, t.Nloc)
    for (i, j) in zip(t.e1, t.e2)
        Wφ[i] += φ[j]; Wφ[j] += φ[i]
    end
    if t.type === :escar
        c = logistic(θ[t.rho])
        Q = sum(t.nneigh .* φ .^ 2) - c * sum(φ .* Wφ)
        return lp + (t.Nloc * log(τ) + sum(log1p.(-c .* t.eig)) - τ * Q) / 2 + prior_kernel(t.rho_prior, c) +
               log(c) + log1p(-c)
    end
    Q = sum(t.nneigh .* φ .^ 2) - sum(φ .* Wφ)
    return lp + ((t.Nloc - 1) * log(τ) - τ * Q) / 2
end
_car_generic(t::CarTerm, θ) = (_car_r(t, θ)[t.J], _car_prior(t, θ))

function _car_forward!(η, t::CarTerm, θ)
    r = _car_r(t, θ)
    @inbounds for n in eachindex(η)
        η[n] += r[t.J[n]]
    end
    return η
end
function _car_pullback!(g, t::CarTerm, w, θ)
    T = eltype(w)
    rbar = zeros(T, t.Nloc)
    @inbounds for n in eachindex(w)
        rbar[t.J[n]] += w[n]
    end
    u = θ[t.sd]; s = exp(u)
    v, d = prior_kernel_grad(t.sd_prior, s)
    lp = v + u
    gsd = s * d + 1
    if t.type in (:icar, :bym2)
        z = θ[t.z]
        gz = zeros(T, t.Nloc)
        if t.type === :icar
            gz .+= s .* rbar
            gsd += s * sum(rbar .* z)
        else
            ρ = logistic(θ[t.rho]); a = sqrt(1 - ρ); c = sqrt(ρ / t.scale)
            nsz = θ[t.nsz]
            r = s .* (a .* nsz .+ c .* z)
            gsd += sum(rbar .* r)
            gz .+= s * c .* rbar
            for (k, j) in enumerate(t.nsz)
                g[j] += s * a * rbar[k] - nsz[k]
                lp -= nsz[k]^2 / 2
            end
            # d/d rho through a and c, then rho = logistic(u_rho) (with its Jacobian)
            dρ = sum(rbar .* s .* (-nsz ./ (2a) .+ z ./ (2 * c * t.scale)))
            vr, dr = prior_kernel_grad(t.rho_prior, ρ)
            g[t.rho] += (dρ + dr) * ρ * (1 - ρ) + (1 - 2ρ)
            lp += vr + log(ρ) + log1p(-ρ)
        end
        for (i, j) in zip(t.e1, t.e2)
            dz = z[i] - z[j]
            gz[i] -= dz; gz[j] += dz
            lp -= dz^2 / 2
        end
        sz = sum(z); k0 = 0.001 * t.Nloc
        gz .-= sz / k0^2
        lp -= (sz / k0)^2 / 2
        for (k, j) in enumerate(t.z)
            g[j] += gz[k]
        end
        g[t.sd] += gsd
        return lp
    end
    φ = _car_r(t, θ)
    τ = exp(-2u)
    Wφ = zeros(T, t.Nloc)
    for (i, j) in zip(t.e1, t.e2)
        Wφ[i] += φ[j]; Wφ[j] += φ[i]
    end
    c = t.type === :escar ? logistic(θ[t.rho]) : one(T)
    Q = sum(t.nneigh .* φ .^ 2) - c * sum(φ .* Wφ)
    gφ = rbar .- τ .* (t.nneigh .* φ .- c .* Wφ)        # likelihood + CAR prior, d/d phi
    nfree = t.type === :escar ? t.Nloc : t.Nloc - 1
    gsd += -nfree + τ * Q                                # d/d log sdcar of the CAR prior
    lp += (nfree * log(τ) - τ * Q) / 2
    if t.type === :escar
        lp += sum(log1p.(-c .* t.eig)) / 2
        dc = sum(-t.eig ./ (1 .- c .* t.eig)) / 2 + τ * sum(φ .* Wφ) / 2
        vr, dr = prior_kernel_grad(t.rho_prior, c)
        g[t.rho] += (dc + dr) * c * (1 - c) + (1 - 2c)
        lp += vr + log(c) + log1p(-c)
        for (k, j) in enumerate(t.z)
            g[j] += gφ[k]
        end
    else
        for (k, j) in enumerate(t.z)              # phi_k = z_k, phi_N = -sum z
            g[j] += gφ[k] - gφ[end]
        end
    end
    g[t.sd] += gsd
    return lp
end
