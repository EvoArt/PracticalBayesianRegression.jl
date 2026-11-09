# =============================================================================
# Gaussian processes `gp(x, ...)`, as brms (squared-exponential kernel,
# isotropic, `gr = TRUE`, `scale = TRUE` defaults), read off brms 2.23's
# .data_gp / def_lscale_prior / Stan code:
#
#   * the GP lives on the unique covariate rows (Nsub of them, J maps
#     observations to rows); covariates are divided by the largest distance
#     between unique rows (so it is 1);
#   * exact:  f = chol(K) z,  K_ij = sdgp^2 exp(-|x_i - x_j|^2 / (2 lscale^2)) + 1e-12 [i == j]
#   * approximate (`k` basis functions per dimension, boundary factor `c`):
#     centred covariates, L = c * max(1, range), Laplacian eigenfunctions
#         phi_m(x) = prod_d sin(m_d pi (x_d + L_d) / (2 L_d)) / sqrt(L_d)
#     sqrt(lambda_m) = m pi / (2 L), and
#         f = Phi (sqrt(spd(lambda)) .* z),
#         spd = sdgp^2 sqrt(2 pi)^D lscale^D exp(-lscale^2 |sqrt(lambda_m)|^2 / 2)
#   * eta_n += f[J_n]; z ~ N(0, 1); sdgp: class sdgp (as sd); lscale: class
#     lscale, inv_gamma(a, b) tuned so 1% of its mass lies below the smallest
#     and 1% above the largest distance (brms's def_lscale_prior).
#
# Gradients (closed form): fbar = per-row sums of w = d ll / d eta.
#   approximate: rbar = Phi' fbar; d z = sqrt(spd) .* rbar - z;
#     d log sdgp = sum rbar .* r;  d log lscale = sum rbar .* r .* (D/2 - lscale^2 lambda / 2)
#   exact: d z = L' fbar - z; Lbar = tril(fbar z'); Kbar = the Cholesky
#     reverse pass (Murray 2016): P = Phi(L' Lbar), Kbar = sym(L^-T P L^-1)/1,
#     then d log sdgp = 2 sum Kbar .* K0,  d log lscale = sum Kbar .* K0 .* D2 / lscale^2
#     with K0 = K - jitter, D2 the squared distances.
# =============================================================================

struct GPTerm{T}
    label::String             # brms suffix, e.g. "gpx"
    J::Vector{Int}            # observation -> unique row
    Xs::Matrix{Float64}       # scaled unique rows (Nsub x D)
    D2::Matrix{T}             # squared distances between unique rows (exact GP)
    approx::Bool
    Phi::Matrix{T}            # basis (Nsub x M) for the approximate GP
    lam::Vector{T}            # |sqrt(lambda_m)|^2
    dmax::Float64
    cmeans::Vector{Float64}
    Lb::Vector{Float64}       # boundary per dimension (approximate GP)
    ks::Matrix{Int}           # basis index combinations (M x D)
    sdgp::Int                 # θ index of log sdgp
    lscale::Int               # θ index of log lscale
    z::UnitRange{Int}
    sd_prior::ScalarPrior
    ls_prior::ScalarPrior
    cov::Symbol               # :exp_quad, :matern52, :matern32, :exponential (brms 2.23)
end
_nz(t::GPTerm) = length(t.z)

# --- covariance functions (brms 2.23: cov = exp_quad / matern52 / matern32 / exponential) ---
# exact: the kernel at squared distance d2, and its derivative in log lscale
#   exp_quad     k = s2 exp(-d2 / 2l^2)             dk/dlog l = k d2 / l^2
#   matern52     a = sqrt(5) r / l: k = s2 (1 + a + a^2/3) e^-a,  dk/dlog l = s2 e^-a a^2 (1 + a) / 3
#   matern32     a = sqrt(3) r / l: k = s2 (1 + a) e^-a,          dk/dlog l = s2 a^2 e^-a
#   exponential  a = r / l:         k = s2 e^-a,                  dk/dlog l = s2 a e^-a
@inline function _gp_k(cov::Symbol, d2, s2, l)
    if cov === :exp_quad
        k = s2 * exp(-d2 / (2l^2))
        return k, k * d2 / l^2
    end
    r = sqrt(d2)
    if cov === :matern52
        a = sqrt(_pv(r, 5)) * r / l; e = exp(-a)
        return s2 * (1 + a + a^2 / 3) * e, s2 * e * a^2 * (1 + a) / 3
    elseif cov === :matern32
        a = sqrt(_pv(r, 3)) * r / l; e = exp(-a)
        return s2 * (1 + a) * e, s2 * a^2 * e
    end
    a = r / l; e = exp(-a)
    return s2 * e, s2 * a * e
end
# approximate: log sqrt(spd) - log sdgp at |sqrt(lambda)|^2 = lam, and its derivative
# in log lscale (brms spd_gp_*):  spd = s^2 C_D l^D (nu' + l^2 lam)^-(D + nu')/2 for the
# Matern family (nu' = 5, 3, 1), spd = s^2 sqrt(2 pi)^D l^D exp(-l^2 lam / 2) for exp_quad
@inline function _gp_logsq(cov::Symbol, D, l, lam)
    if cov === :exp_quad
        return (_pv(l, D * log(sqrt(2π))) + D * log(l) - l^2 * lam / 2) / 2, (D - l^2 * lam) / 2
    end
    # the constants in Float64, then in l's precision
    ν0 = cov === :matern52 ? 5.0 : cov === :matern32 ? 3.0 : 1.0
    logC = _pv(l, D * log(2) + D / 2 * log(π) + loggamma((D + ν0) / 2) - log(sqrt(π)) +
                  (cov === :matern52 ? 2.5 * log(5) - log(0.75) : cov === :matern32 ? 1.5 * log(3) - log(0.5) : 0.0))
    ν = _pv(l, ν0)
    expo = -(D + ν) / 2
    u = ν + l^2 * lam
    return (logC + D * log(l) + expo * log(u)) / 2, (D + expo * 2 * l^2 * lam / u) / 2
end
const _GP_JITTER = 1e-12

_sqdist(X) = [sum(abs2, view(X, i, :) .- view(X, j, :)) for i in axes(X, 1), j in axes(X, 1)]

# brms's def_lscale_prior: inverse-gamma (a, b) with P(l < lb) = P(l > ub) = 0.01
function _lscale_prior(Xs)
    D2 = _sqdist(Xs)
    ub = sqrt(maximum(D2))
    pos = D2[D2 .> 0]
    lb = isempty(pos) ? 0.01ub : max(sqrt(minimum(pos)), 0.01ub)
    # Newton on (log a, log b) for the two log-probability conditions
    f(v) = (a = exp(v[1]); b = exp(v[2]); IG = Distributions.InverseGamma(a, b);
            [Distributions.logcdf(IG, lb) - log(0.01), Distributions.logccdf(IG, ub) - log(0.01)])
    v = [0.0, 0.0]
    for _ in 1:200
        F = f(v)
        maximum(abs, F) < 1e-10 && break
        # central differences: the inverse-gamma cdf (gamma_inc) has no dual rule in its shape
        Jm = reduce(hcat, [(f(v .+ 1e-6 .* (1:2 .== j)) .- f(v .- 1e-6 .* (1:2 .== j))) ./ 2e-6 for j in 1:2])
        step = Jm \ F
        t = 1.0
        while t > 1e-6 && maximum(abs, f(v .- t .* step)) > maximum(abs, F)
            t /= 2
        end
        v .-= t .* step
    end
    maximum(abs, f(v)) < 1e-6 || return normal(0, 0.5)
    return inv_gamma(round(exp(v[1]); digits=6), round(exp(v[2]); digits=6))
end

function _gp_basis(Xc, ks, Lb)
    M, D = size(ks)
    Phi = ones(size(Xc, 1), M)
    lam = zeros(M)
    for m in 1:M, d in 1:D
        w = ks[m, d] * π / (2Lb[d])
        Phi[:, m] .*= sin.(w .* (Xc[:, d] .+ Lb[d])) ./ sqrt(Lb[d])
        lam[m] += w^2
    end
    return Phi, lam
end

# build-time data: unique rows, scaling, basis
function _gp_data(X::Matrix{Float64}, k, c, scale)
    rows = unique(eachrow(X))
    U = reduce(vcat, (permutedims(collect(r)) for r in rows))
    lookup = Dict(collect(r) => i for (i, r) in enumerate(rows))
    J = [lookup[collect(r)] for r in eachrow(X)]
    dmax = scale ? sqrt(maximum(_sqdist(U))) : 1.0
    dmax > 0 || throw(ArgumentError("gp(): cannot scale the covariates (all rows equal); use scale = false"))
    Xs = U ./ dmax
    D = size(Xs, 2)
    if k === nothing
        return J, Xs, dmax, zeros(D), zeros(D), zeros(Int, 0, D)
    end
    cmeans = vec(mean(Xs; dims=1))
    Xc = Xs .- cmeans'
    rng = max(1.0, maximum(Xc) - minimum(Xc))
    cv = c === nothing ? fill(5 / 4, D) : (length(c) == 1 ? fill(Float64(c[1]), D) : Float64.(c))
    Lb = cv .* rng
    ks = reduce(vcat, (permutedims(collect(t)) for t in Iterators.product(ntuple(_ -> 1:k, D)...)))
    return J, Xs, dmax, cmeans, Lb, ks
end

@inline _spd_logc(D) = D * log(sqrt(2π))

# f at the unique rows (generic in the number type) and the z prior
function _gp_f(t::GPTerm, θ)
    s, l = exp(θ[t.sdgp]), exp(θ[t.lscale])
    z = θ[t.z]
    if t.approx
        D = size(t.ks, 2)
        sq = [s * exp(first(_gp_logsq(t.cov, D, l, λ))) for λ in t.lam]
        return t.Phi * (sq .* z)
    end
    K = [first(_gp_k(t.cov, d2, s^2, l)) for d2 in t.D2]
    K += _pv(s, _GP_JITTER) * I
    # A long length-scale makes K numerically singular despite the jitter.
    # Stan rejects such a proposal; NaN does the same here (the samplers treat
    # a non-finite density as divergent) instead of throwing mid-chain.
    C = cholesky(Symmetric(K); check=false)
    issuccess(C) || return fill(oftype(s, NaN), size(K, 1))
    return C.L * z
end
function _gp_generic(t::GPTerm, θ)
    f = _gp_f(t, θ)
    lp = -sum(abs2, θ[t.z]) / 2 + prior_kernel(t.sd_prior, exp(θ[t.sdgp])) + θ[t.sdgp] +
         prior_kernel(t.ls_prior, exp(θ[t.lscale])) + θ[t.lscale]
    return f[t.J], lp
end

function _gp_forward!(η, t::GPTerm, θ)
    f = _gp_f(t, θ)
    @inbounds for n in eachindex(η)
        η[n] += f[t.J[n]]
    end
    return η
end

function _gp_pullback!(g, t::GPTerm{T}, w, θ) where {T}
    u_s, u_l = θ[t.sdgp], θ[t.lscale]
    s, l = exp(u_s), exp(u_l)
    z = view(θ, t.z)
    fbar = zeros(T, size(t.approx ? t.Phi : t.D2, 1))
    @inbounds for n in eachindex(w)
        fbar[t.J[n]] += w[n]
    end
    gz = view(g, t.z)
    gs = zero(T); gl = zero(T)
    if t.approx
        D = size(t.ks, 2)
        rbar = t.Phi' * fbar
        @inbounds for m in eachindex(t.lam)
            lq, dlq = _gp_logsq(t.cov, D, l, t.lam[m])
            sqm = s * exp(lq)
            r = sqm * z[m]
            gz[m] += sqm * rbar[m]
            gs += rbar[m] * r
            gl += rbar[m] * r * dlq
        end
    else
        KD = [_gp_k(t.cov, d2, s^2, l) for d2 in t.D2]
        K0 = first.(KD); dK = last.(KD)
        C = cholesky(Symmetric(K0 + T(_GP_JITTER) * I); check=false)
        issuccess(C) || return T(NaN)              # rejected proposal, as in _gp_f
        L = Matrix(C.L)
        gz .+= L' * fbar
        Lbar = LowerTriangular(fbar * transpose(z))
        Kbar = _chol_rev(L, Matrix(Lbar))
        @inbounds for j in axes(K0, 2), i in axes(K0, 1)
            gs += 2 * Kbar[i, j] * K0[i, j]
            gl += Kbar[i, j] * dK[i, j]
        end
    end
    lp = zero(T)
    @inbounds for k in t.z
        gz_k = θ[k]
        g[k] -= gz_k
        lp -= gz_k^2 / 2
    end
    v1, d1 = prior_kernel_grad(t.sd_prior, s)
    v2, d2 = prior_kernel_grad(t.ls_prior, l)
    g[t.sdgp] += gs + s * d1 + 1
    g[t.lscale] += gl + l * d2 + 1
    return lp + v1 + v2 + u_s + u_l
end

# reverse pass of A = L L' (Murray 2016): given Lbar (lower), the symmetric
# Abar with sum(Abar .* dA) = sum(Lbar .* dL)
function _chol_rev(L, Lbar)
    Φ(X) = (Y = tril(X); for i in axes(Y, 1); Y[i, i] /= 2; end; Y)
    P = Φ(L' * Lbar)
    Lt = LowerTriangular(L)
    S = Lt' \ (Lt' \ P')'           # L^-T P L^-1
    return (S + S') / 2
end

# GP values at new covariate rows for one draw: the approximate GP evaluates
# its basis; the exact GP draws from the conditional given the training f
function _gp_new(t::GPTerm, θ, Xnew::Matrix{Float64}, rng)
    s, l = exp(θ[t.sdgp]), exp(θ[t.lscale])
    Xs = Xnew ./ t.dmax
    if t.approx
        Phi, _ = _gp_basis(Xs .- t.cmeans', t.ks, t.Lb)
        D = size(t.ks, 2)
        sq = [s * exp(first(_gp_logsq(t.cov, D, l, λ))) for λ in t.lam]
        return Phi * (sq .* θ[t.z])
    end
    k(a, b) = first(_gp_k(t.cov, sum(abs2, a .- b), s^2, l))
    Koo = [k(view(t.Xs, i, :), view(t.Xs, j, :)) for i in axes(t.Xs, 1), j in axes(t.Xs, 1)] + _GP_JITTER * I
    Kno = [k(view(Xs, i, :), view(t.Xs, j, :)) for i in axes(Xs, 1), j in axes(t.Xs, 1)]
    Knn = [k(view(Xs, i, :), view(Xs, j, :)) for i in axes(Xs, 1), j in axes(Xs, 1)]
    C = cholesky(Symmetric(Koo))
    f = C.L * θ[t.z]
    A = Kno / C
    m = A * f
    V = Symmetric(Knn - A * Kno' + 1e-8 * I)
    return m .+ cholesky(V; check=false).L * randn(rng, size(Xs, 1))
end
