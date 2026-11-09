# =============================================================================
# Families whose per-observation derivative comes from dual numbers.
#
# A family is a scalar function of (y, eta_mu, eta_aux...), so its gradient
# with respect to those few link-scale values can be had from ONE dual-number
# evaluation per observation — no tape, no allocation, cost a small multiple
# of the log-likelihood itself. The chain back through X, Z and the group
# effects is the same closed form as for every other family. So a family here
# is just its log density (`_ll`), and the tests check that against
# Distributions.jl like the hand-derived ones.
#
# The hand-derived families (families.jl) stay faster for the common cases;
# this is how the long tail of brms families gets in without a derivation each.
# =============================================================================


@inline function _ll_grad(fam::DualFamily, y, x, eta::NTuple{D,T}) where {D,T}
    duals = ntuple(k -> ForwardDiff.Dual{Nothing}(eta[k], ntuple(j -> T(j == k), Val(D))), Val(D))
    r = _ll(fam, y, x, duals)
    return T(ForwardDiff.value(r)), ntuple(k -> T(ForwardDiff.partials(r)[k]), Val(D))
end
# scalar-aux fast path: the aux etas themselves are the "pre" quantities
@inline _pre(::DualFamily, eta) = Base.tail(eta)
@inline _ll_grad_pre(fam::DualFamily, y, x, m, p) = _ll_grad(fam, y, x, (m, p...))
_ll_const(::DualFamily, y, t) = 0.0

const _SQRT_2_OVER_PI = sqrt(2 / π)

# --- skew_normal: brms's mean/sd parameterisation (mu, sigma, alpha) ------------
dpars(::SkewNormal) = (:mu, :sigma, :alpha)
aux_bounds(::SkewNormal, d::Symbol) = d === :alpha ? (:identity, -Inf) : (:log, 0.0)
dpar_link_name(::SkewNormal, d::Symbol) = d === :mu ? "identity" : d === :alpha ? "identity" : "log"
function _sn_params(mu, ls, alpha)
    delta = alpha / sqrt(1 + alpha^2)
    omega = exp(ls) / sqrt(1 - _pv(delta, 2 / π) * delta^2)
    xi = mu - omega * delta * _pv(delta, _SQRT_2_OVER_PI)
    return xi, omega, delta
end
@inline function _ll(::SkewNormal, y, x, eta)
    xi, omega, _ = _sn_params(eta[1], eta[2], eta[3])
    z = (y - xi) / omega
    return _pv(omega, _LOG2) - log(omega) + normlogpdf(z) + normlogcdf(eta[3] * z)
end
_mean(::SkewNormal, eta, t) = eta[1]
function _rand(rng, ::SkewNormal, eta, t)
    xi, omega, delta = _sn_params(eta[1], eta[2], eta[3])
    return xi + omega * (delta * abs(randn(rng)) + sqrt(1 - delta^2) * randn(rng))
end

# --- exgaussian: brms's (mu = mean, sigma, beta = exponential scale) -------------
dpars(::ExGaussian) = (:mu, :sigma, :beta)
@inline function _ll(::ExGaussian, y, x, eta)
    mu, s, b = eta[1], exp(eta[2]), exp(eta[3])
    lam = 1 / b
    m = mu - b                              # mean of the gaussian component
    return log(lam / 2) + lam / 2 * (2m + lam * s^2 - 2y) + logerfc((m + lam * s^2 - y) / (sqrt(2) * s))
end
# Hand-derived gradient: the dual-number route is slow for this family's
# log-erfc term. With lambda = 1/b,
#   ll = -log b - log 2 - 1 + (mu - y)/b + s^2/(2 b^2) + logerfc(z),
#   z  = (mu - y - b + s^2/b) / (sqrt(2) s),
# and d logerfc(z)/dz = -r with r = (2/sqrt(pi)) / erfcx(z) (finite in both
# tails). Derivatives are on the link scale (mu, log s, log b). `_pre` holds
# the per-draw quantities so the scalar-aux path computes them once.
@inline function _pre(::ExGaussian, eta)
    s, b = exp(eta[2]), exp(eta[3])
    T = typeof(s)
    return (s, b, inv(b), s * s, inv(T(sqrt(2)) * s), log(b) + T(log(2)) + 1)
end
@inline function _ll_grad_pre(::ExGaussian, y, x, mu, p)
    s, b, ib, s2, k, lc = p
    T = typeof(s)
    d = mu - y
    z = (d - b + s2 * ib) * k
    e = erfcx(z)
    # log erfc(z) = log erfcx(z) - z^2; erfcx overflows for very negative z
    lerfc = z < 0 ? logerfc(z) : log(e) - z * z
    r = T(2 / sqrt(π)) / e
    v = -lc + d * ib + s2 * ib * ib / 2 + lerfc
    gmu = ib - r * k
    gs = s2 * ib * ib - r * (T(sqrt(2)) * s * ib - z)
    gb = -1 - d * ib - s2 * ib * ib + r * (b + s2 * ib) * k
    return v, (gmu, gs, gb)
end
@inline _ll_grad(fam::ExGaussian, y, x, eta::NTuple{3,T}) where {T} = _ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta))
_mean(::ExGaussian, eta, t) = eta[1]
_rand(rng, ::ExGaussian, eta, t) = eta[1] - exp(eta[3]) + exp(eta[2]) * randn(rng) + exp(eta[3]) * randexp(rng)

# --- inverse.gaussian: mean mu, shape (brms default link 1/mu^2) -------------------
dpars(::InverseGaussian) = (:mu, :shape)
@inline function _ll(f::InverseGaussian, y, x, eta)
    mu = linkinv(link(f), eta[1])
    sh = exp(eta[2])
    return (log(sh) - _pv(sh, _LOG_2PI) - 3log(y)) / 2 - sh * (y - mu)^2 / (2 * mu^2 * y)
end
_mean(f::InverseGaussian, eta, t) = linkinv(link(f), eta[1])
_rand(rng, f::InverseGaussian, eta, t) = rand(rng, D_.InverseGaussian(linkinv(link(f), eta[1]), exp(eta[2])))
_check_response(::InverseGaussian, y, t) = _chk(all(>(0), y), "responses must be positive")

# --- frechet: mean mu (log), shape nu > 1 (logm1); scale = mu / Gamma(1 - 1/nu) ----
dpars(::Frechet) = (:mu, :nu)
@inline function _ll(::Frechet, y, x, eta)
    nu = 1 + exp(eta[2])
    ls = eta[1] - loggamma(1 - 1 / nu)     # log scale
    lz = log(y) - ls
    return log(nu) - ls - (1 + nu) * lz - exp(-nu * lz)
end
_mean(::Frechet, eta, t) = exp(eta[1])
function _rand(rng, ::Frechet, eta, t)
    nu = 1 + exp(eta[2])
    return rand(rng, D_.Frechet(nu, exp(eta[1] - loggamma(1 - 1 / nu))))
end
_check_response(::Frechet, y, t) = _chk(all(>(0), y), "responses must be positive")

# --- shifted_lognormal: lognormal of y - ndt, ndt in (0, min(y)) --------------------
dpars(::ShiftedLognormal) = (:mu, :sigma, :ndt)
aux_bounds(::ShiftedLognormal, d::Symbol) = d === :ndt ? (:logit, 0.0) : (:log, 0.0)
aux_ub(f::ShiftedLognormal, d::Symbol) = d === :ndt ? f.miny : 1.0
dpar_link_name(::ShiftedLognormal, d::Symbol) = d === :mu ? "identity" : d === :ndt ? "logit (0, min_Y)" : "log"
prior_link(::ShiftedLognormal) = :log
@inline function _ll(f::ShiftedLognormal, y, x, eta)
    ndt = _pv(eta[3], f.miny) * logistic(eta[3])
    ly = log(y - ndt)
    r = (ly - eta[1]) * exp(-eta[2])
    return -r * r / 2 - eta[2] - ly - _pv(r, _LOG_2PI_HALF)
end
_mean(f::ShiftedLognormal, eta, t) = f.miny * logistic(eta[3]) + exp(eta[1] + exp(2eta[2]) / 2)
_rand(rng, f::ShiftedLognormal, eta, t) = f.miny * logistic(eta[3]) + exp(eta[1] + exp(eta[2]) * randn(rng))
_check_response(::ShiftedLognormal, y, t) = _chk(all(>(0), y), "responses must be positive")

# the default upper bound of a logit-scale aux parameter
aux_ub(::Family, d::Symbol) = 1.0

# =============================================================================
# More brms families, each transcribed from brms 2.23's Stan functions;
# densities are normalised as there.
# =============================================================================

# --- asym_laplace: quantile regression; mu, sigma, quantile in (0, 1) -------------
dpars(::AsymLaplace) = (:mu, :sigma, :quantile)
@inline _rho_q(u, q) = u < 0 ? u * (q - 1) : u * q
@inline function _ll(f::AsymLaplace, y, x, eta)
    mu = linkinv(link(f), eta[1]); s = exp(eta[2]); q = logistic(eta[3])
    return log(q) + log1p(-q) - eta[2] - _rho_q((y - mu) / s, q)
end
function _mean(f::AsymLaplace, eta, t)
    mu = linkinv(link(f), eta[1]); s = exp(eta[2]); q = logistic(eta[3])
    return mu + s * (1 - 2q) / (q * (1 - q))
end
function _rand(rng, f::AsymLaplace, eta, t)
    mu = linkinv(link(f), eta[1]); s = exp(eta[2]); q = logistic(eta[3])
    u = rand(rng)
    return u < q ? mu + s / (1 - q) * log(u / q) : mu - s / q * log((1 - u) / (1 - q))
end

# --- von_mises: mu = 2 atan(eta) (tan_half link), precision kappa -------------------
# log I0(kappa) through the scaled Bessel function (no overflow); ForwardDiff has
# no rule for besselix, so the Dual method supplies d/dk log I0(k) = I1(k) / I0(k)
# Above _I0_ASYM the Hankel expansion is exact to double precision, and it is
# needed: AMOS's besselix throws (ierr 4, "argument magnitude too large") for
# kappa ~ 1e9, which warm-up reaches when exp(eta_kappa) runs away.
#   log I0(k) = k - log(2 pi k) / 2 + log(1 + 1/(8k) + 9/(128k^2) + 225/(3072k^3))
#   I1/I0(k)  = 1 - 1/(2k) - 1/(8k^2) - 1/(8k^3)
const _I0_ASYM = 1e4
_logI0(k::Real) = k > _pv(k, _I0_ASYM) ? k - log(_pv(k, 2π) * k) / 2 + log1p(1 / (8k) + 9 / (128k^2) + 225 / (3072k^3)) :
                  log(besselix(0, k)) + k
_I1_over_I0(k::Real) = k > _pv(k, _I0_ASYM) ? 1 - 1 / (2k) - 1 / (8k^2) - 1 / (8k^3) : besselix(1, k) / besselix(0, k)
function _logI0(k::ForwardDiff.Dual{T}) where {T}
    v = ForwardDiff.value(k)
    return ForwardDiff.Dual{T}(_logI0(v), _I1_over_I0(v) * ForwardDiff.partials(k))
end
dpars(::VonMises) = (:mu, :kappa)
@inline function _ll(f::VonMises, y, x, eta)
    mu = linkinv(link(f), eta[1]); k = exp(eta[2])
    return k * cos(y - mu) - _pv(k, _LOG_2PI) - _logI0(k)
end
_mean(f::VonMises, eta, t) = linkinv(link(f), eta[1])
_rand(rng, f::VonMises, eta, t) = rand(rng, D_.VonMises(linkinv(link(f), eta[1]), exp(eta[2])))
_check_response(::VonMises, y, t) = _chk(all(v -> -π <= v <= π, y), "von_mises responses must be angles in [-pi, pi]")

# --- zero_inflated_beta / zero_one_inflated_beta (mean-precision beta) ------------
@inline _beta_mp(y, mu, phi) = (mu * phi - 1) * log(y) + ((1 - mu) * phi - 1) * log1p(-y) - logbeta(mu * phi, (1 - mu) * phi)
dpars(::ZeroInflatedBeta) = (:mu, :phi, :zi)
@inline function _ll(f::ZeroInflatedBeta, y, x, eta)
    y == 0 && return -log1pexp(-eta[3])                   # log zi
    return -log1pexp(eta[3]) + _beta_mp(y, linkinv(link(f), eta[1]), exp(eta[2]))
end
_mean(f::ZeroInflatedBeta, eta, t) = (1 - logistic(eta[3])) * linkinv(link(f), eta[1])
function _rand(rng, f::ZeroInflatedBeta, eta, t)
    rand(rng) < logistic(eta[3]) && return 0.0
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2])
    return rand(rng, D_.Beta(mu * phi, (1 - mu) * phi))
end
_check_response(::ZeroInflatedBeta, y, t) = _chk(all(v -> 0 <= v < 1, y), "zero_inflated_beta responses must be in [0, 1)")

dpars(::ZeroOneInflatedBeta) = (:mu, :phi, :zoi, :coi)
@inline function _ll(f::ZeroOneInflatedBeta, y, x, eta)
    lzoi = -log1pexp(-eta[3])
    y == 0 && return lzoi - log1pexp(eta[4])              # zoi * (1 - coi)
    y == 1 && return lzoi - log1pexp(-eta[4])             # zoi * coi
    return -log1pexp(eta[3]) + _beta_mp(y, linkinv(link(f), eta[1]), exp(eta[2]))
end
function _mean(f::ZeroOneInflatedBeta, eta, t)
    zoi, coi = logistic(eta[3]), logistic(eta[4])
    return zoi * coi + (1 - zoi) * linkinv(link(f), eta[1])
end
function _rand(rng, f::ZeroOneInflatedBeta, eta, t)
    rand(rng) < logistic(eta[3]) && return Float64(rand(rng) < logistic(eta[4]))
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2])
    return rand(rng, D_.Beta(mu * phi, (1 - mu) * phi))
end
_check_response(::ZeroOneInflatedBeta, y, t) = _chk(all(v -> 0 <= v <= 1, y), "zero_one_inflated_beta responses must be in [0, 1]")

# --- discrete_weibull: P(Y >= y) = mu^(y^shape), mu in (0, 1) ---------------------
dpars(::DiscreteWeibull) = (:mu, :shape)
@inline function _ll(f::DiscreteWeibull, y, x, eta)
    lmu = -log1pexp(-eta[1])                              # log mu, logit link
    k = exp(eta[2])
    a = (y == 0 ? zero(k) : y^k) * lmu                    # log mu^(y^k)
    b = (y + 1)^k * lmu
    return a + log1mexp(b - a)                            # log(mu^(y^k) - mu^((y+1)^k))
end
function _mean(f::DiscreteWeibull, eta, t)
    mu = logistic(eta[1]); k = exp(eta[2])
    # E[Y] = sum_{y >= 1} P(Y >= y)
    s = 0.0
    for y in 1:100_000
        p = mu^(y^k)
        s += p
        p < 1e-12 && break
    end
    return s
end
_rand(rng, f::DiscreteWeibull, eta, t) = floor((log(rand(rng)) / log(logistic(eta[1])))^(1 / exp(eta[2])))
_check_response(::DiscreteWeibull, y, t) = _chk(all(v -> isinteger(v) && v >= 0, y), "count responses must be non-negative integers")

# --- negbinomial2: negative binomial with sigma = 1 / shape ------------------------
dpars(::NegBinomial2) = (:mu, :sigma)
@inline function _ll(f::NegBinomial2, y, x, eta)
    mu = linkinv(link(f), eta[1]); phi = exp(-eta[2])
    return loggamma(y + phi) - loggamma(phi) - loggamma(y + 1) + phi * (log(phi) - log(phi + mu)) +
           y * (log(mu) - log(phi + mu))
end
_mean(f::NegBinomial2, eta, t) = linkinv(link(f), eta[1])
function _rand(rng, f::NegBinomial2, eta, t)
    mu = linkinv(link(f), eta[1]); phi = exp(-eta[2])
    return Float64(rand(rng, D_.NegativeBinomial(phi, phi / (phi + mu))))
end
_check_response(::NegBinomial2, y, t) = _chk(all(v -> isinteger(v) && v >= 0, y), "count responses must be non-negative integers")

# --- com_poisson: Conway-Maxwell-Poisson (brms: mu = rate on the log scale, nu = shape)
#     log P(y) = y log mu - nu lgamma(y + 1) - log Z(log mu, nu)
# log Z as brms: the asymptotic expansion when log mu >= log 1.5 and
# nu log mu >= log 1.5, else the series summed until the remainder bound is
# below 2^-23 (checked every 50 terms), at most 10000 terms.
dpars(::ComPoisson) = (:mu, :shape)
function _logZ_cmp_approx(lmu, nu)
    nu2 = nu^2
    lcommon = log(nu) + lmu / nu
    lcte = nu * exp(lmu / nu) - ((nu - 1) / (2nu) * lmu + (nu - 1) / 2 * _pv(nu, _LOG_2PI) + log(nu) / 2)
    c1 = (nu2 - 1) / 24
    c2 = (nu2 - 1) / 1152 * (nu2 + 23)
    c3 = (nu2 - 1) / 414720 * (5 * nu2^2 - 298nu2 + 11237)
    return lcte + log(1 + c1 * exp(-lcommon) + c2 * exp(-2lcommon) + c3 * exp(-3lcommon))
end
_lkterm(lmu, nu, k) = (k - 1) * lmu - nu * loggamma(_pv(lmu, k))
function _logZ_cmp(lmu, nu)
    nu == 1 && return exp(lmu)
    (lmu * nu >= _pv(lmu, log(1.5)) && lmu >= _pv(lmu, log(1.5))) && return _logZ_cmp_approx(lmu, nu)
    leps = -23 * log(2)
    acc = _lkterm(lmu, nu, 1)
    prev = acc; k = 1
    while k < 10_000
        k += 1
        cur = _lkterm(lmu, nu, k)
        acc = logaddexp(acc, cur)
        # stop once terms decrease and the geometric bound on the rest is tiny
        if cur < prev && k % 50 == 0 && cur - log(-expm1(cur - prev)) < leps
            break
        end
        prev = cur
    end
    return acc
end
@inline function _ll(f::ComPoisson, y, x, eta)
    nu = exp(eta[2])
    return y * eta[1] - nu * loggamma(y + 1) - _logZ_cmp(eta[1], nu)
end
function _cmp_pmf(eta; kmax=10_000)
    lz = _logZ_cmp(Float64(eta[1]), exp(Float64(eta[2])))
    nu = exp(Float64(eta[2]))
    ps = Float64[]
    for k in 0:kmax
        p = exp(k * eta[1] - nu * loggamma(k + 1) - lz)
        push!(ps, p)
        k > 5 && p < 1e-14 && sum(ps) > 0.999999 && break
    end
    return ps
end
_mean(f::ComPoisson, eta, t) = (p = _cmp_pmf(eta); sum((k - 1) * p[k] for k in eachindex(p)) / sum(p))
function _rand(rng, f::ComPoisson, eta, t)
    p = _cmp_pmf(eta); u = rand(rng) * sum(p); c = 0.0
    for k in eachindex(p)
        c += p[k]
        u <= c && return Float64(k - 1)
    end
    return Float64(length(p) - 1)
end
_check_response(::ComPoisson, y, t) = _chk(all(v -> isinteger(v) && v >= 0, y), "count responses must be non-negative integers")

# --- xbeta (brms): a beta on the widened interval (-kappa, 1 + kappa), clipped to [0, 1]
#     y = 0:   log P(Z <= kappa / (1 + 2 kappa))
#     y = 1:   log P(Z >= (1 + kappa) / (1 + 2 kappa))
#     else:    beta_proportion_lpdf((y + kappa) / (1 + 2 kappa) | mu, phi) - log(1 + 2 kappa)
# with Z ~ Beta(mu phi, (1 - mu) phi). The incomplete beta has no dual rule in
# its shape parameters: the Dual method below gives d/dx in closed form and
# d/da, d/db by central differences (accurate to ~1e-8, ample for HMC).
dpars(::XBeta) = (:mu, :phi, :kappa)
# AbstractFloat, not Real: a ForwardDiff.Dual is a Real, and must reach the
# Dual method below rather than SpecialFunctions.beta_inc (which has no rule)
function _logbetainc(x::AbstractFloat, a::AbstractFloat, b::AbstractFloat, upper::Bool)
    p, q = SpecialFunctions.beta_inc(a, b, x)
    return log(upper ? q : p)
end
function _logbetainc(x, a, b, upper::Bool)
    T = typeof(ForwardDiff.value(x isa ForwardDiff.Dual ? x : a isa ForwardDiff.Dual ? a : b))
    D = x isa ForwardDiff.Dual ? typeof(x) : a isa ForwardDiff.Dual ? typeof(a) : typeof(b)
    xv, av, bv = ForwardDiff.value(x), ForwardDiff.value(a), ForwardDiff.value(b)
    v = _logbetainc(xv, av, bv, upper)
    # d/dx log I = x^(a-1)(1-x)^(b-1) / (B(a,b) I)   (negated for the upper tail)
    dx = exp((av - 1) * log(xv) + (bv - 1) * log1p(-xv) - logbeta(av, bv) - v) * (upper ? -1 : 1)
    # the step never crosses zero: with a = mu phi < 1e-5 (reachable in warm-up)
    # an absolute 1e-5 step evaluated beta_inc at a negative shape (DomainError)
    # (relative step 1e-5 in Float64; cbrt(eps) in lower precision, where 1e-5 would
    # leave the difference quotient dominated by rounding, eps / h ~ 1e-2 in Float32)
    hr = T === Float64 ? 1e-5 : cbrt(eps(T))
    ha = min(hr * max(one(T), av), av / 2); hb = min(hr * max(one(T), bv), bv / 2)
    da = (_logbetainc(xv, av + ha, bv, upper) - _logbetainc(xv, av - ha, bv, upper)) / 2ha
    db = (_logbetainc(xv, av, bv + hb, upper) - _logbetainc(xv, av, bv - hb, upper)) / 2hb
    part(z) = z isa ForwardDiff.Dual ? ForwardDiff.partials(z) : zero(ForwardDiff.partials(zero(D)))
    return D(v, dx * part(x) + da * part(a) + db * part(b))
end
@inline function _ll(f::XBeta, y, x, eta)
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2]); kap = exp(eta[3])
    den = 1 + 2kap
    a, b = mu * phi, (1 - mu) * phi
    # mu rounds to exactly 0 or 1 for extreme eta: outside the support, so -Inf
    # (a divergence for NUTS, as Stan's reject) rather than a DomainError
    (a > 0 && b > 0) || return oftype(a, -Inf)
    y <= 0 && return _logbetainc(kap / den, a, b, false)
    y >= 1 && return _logbetainc((1 + kap) / den, a, b, true)
    return _beta_mp((y + kap) / den, mu, phi) - log(den)
end
function _mean(f::XBeta, eta, t)
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2]); kap = exp(eta[3])
    den = 1 + 2kap; Bd = Distributions.Beta(mu * phi, (1 - mu) * phi)
    lo, hi = kap / den, (1 + kap) / den
    zs = range(lo, hi; length=801); h = step(zs)
    inner = sum((z * den - kap) * Distributions.pdf(Bd, z) for z in zs) * h      # middle part
    return inner + Distributions.ccdf(Bd, hi)
end
function _rand(rng, f::XBeta, eta, t)
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2]); kap = exp(eta[3])
    z = rand(rng, Distributions.Beta(mu * phi, (1 - mu) * phi))
    return clamp(z * (1 + 2kap) - kap, 0.0, 1.0)
end
_check_response(::XBeta, y, t) = _chk(all(v -> 0 <= v <= 1, y), "xbeta responses must be in [0, 1]")
"""
    xbeta(; link = :logit)

Extended-support beta distribution for proportions in [0, 1], exact zeros and
ones included: a beta distribution on (-kappa, 1 + kappa) with the values
outside [0, 1] moved to 0 and 1. Parameters: `mu`, `phi` (as in
[`betafamily`](@ref)) and `kappa`.

Links for `mu`: $(_links_doc(XBeta)).
"""
xbeta(; link=nothing) = _make_family(XBeta, link)
