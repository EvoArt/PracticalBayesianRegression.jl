# =============================================================================
# Log-cdfs of the continuous families, for `cens()` and `trunc()` (brms allows
# both for every continuous family; its Stan code calls <family>_lcdf /
# _lccdf). Each `_cc_lcdf(fam, y, eta)` / `_cc_lccdf(fam, y, eta)` takes the
# link-scale tuple and is generic in the number type: the censored and
# truncated observations get their derivatives from one dual-number pass
# (censoring.jl), the generic density path evaluates the same functions, and
# uncensored observations keep the family's own fast path.
#
# Closed forms where they exist; incomplete beta / gamma functions (student,
# beta, gamma) with a dual rule (closed form in x, central differences in the
# shape parameters, as `_logbetainc`); Owen's T (skew_normal) by Gauss-Legendre
# quadrature.
# =============================================================================

# --- incomplete gamma, as _logbetainc (dualfamilies.jl) ----------------------------
function _loggammainc(a::AbstractFloat, x::AbstractFloat, upper::Bool)
    x <= 0 && return upper ? zero(x) : oftype(x, -Inf)
    p, q = SpecialFunctions.gamma_inc(a, x)
    return log(upper ? q : p)
end
function _loggammainc(a, x, upper::Bool)
    T = typeof(ForwardDiff.value(x isa ForwardDiff.Dual ? x : a))
    D = x isa ForwardDiff.Dual ? typeof(x) : typeof(a)
    av, xv = ForwardDiff.value(a), ForwardDiff.value(x)
    v = _loggammainc(av, xv, upper)
    # d/dx log P = x^(a-1) e^-x / (Gamma(a) P)   (negated for the upper tail)
    dx = xv > 0 ? exp((av - 1) * log(xv) - xv - loggamma(av) - v) * (upper ? -1 : 1) : zero(T)
    hr = T === Float64 ? 1e-5 : cbrt(eps(T))
    ha = min(hr * max(one(T), av), av / 2)
    da = (_loggammainc(av + ha, xv, upper) - _loggammainc(av - ha, xv, upper)) / 2ha
    part(z) = z isa ForwardDiff.Dual ? ForwardDiff.partials(z) : zero(ForwardDiff.partials(zero(D)))
    return D(v, dx * part(x) + da * part(a))
end

# --- Owen's T function, for the skew-normal cdf -----------------------------------
# Gauss-Legendre nodes / weights on [0, 1] (Golub-Welsch), computed once
function _gauss_legendre01(n)
    J = LinearAlgebra.SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:(n - 1)])
    x, V = LinearAlgebra.eigen(J)
    return (x .+ 1) ./ 2, V[1, :] .^ 2
end
const _GL01 = _gauss_legendre01(48)
# T(h, a) = (1 / 2pi) int_0^a exp(-h^2 (1 + t^2) / 2) / (1 + t^2) dt. For |a| <= 1
# the integrand is smooth on [0, a]; otherwise Owen's identity (even in h, a >= 0)
#   T(h, a) + T(a h, 1 / a) = Q(h) / 2 + Q(a h) / 2 - Q(h) Q(a h),  Q(x) = Phi(-|x|)
function _owens_t(h, a)
    a < 0 && return -_owens_t(h, -a)
    if a <= 1
        s = zero(h * a)
        xs, ws = _GL01
        for k in eachindex(xs)
            t = a * _pv(a, xs[k]); q = 1 + t * t
            s += _pv(a, ws[k]) * exp(-h * h * q / 2) / q
        end
        return a * s / _pv(a, 2π)
    end
    hh = abs(h); ah = a * hh
    q1, q2 = normcdf(-hh), normcdf(-ah)
    return (q1 + q2) / 2 - q1 * q2 - _owens_t(ah, 1 / a)
end

# --- student: P(T <= z) = I_{nu / (nu + z^2)}(nu / 2, 1 / 2) / 2 for z <= 0 -----------
function _student_lcdf(z, nu)
    lt = _logbetainc(nu / (nu + z * z), nu / 2, _pv(nu, 0.5) + zero(nu), false) - _pv(nu, _LOG2)   # log P(T <= -|z|)
    return z <= 0 ? lt : log1mexp(lt)
end
function _cc_lcdf(f::Student, y, eta)
    nu = 1 + exp(eta[3])
    return _student_lcdf((y - linkinv(link(f), eta[1])) * exp(-eta[2]), nu)
end
function _cc_lccdf(f::Student, y, eta)
    nu = 1 + exp(eta[3])
    return _student_lcdf((linkinv(link(f), eta[1]) - y) * exp(-eta[2]), nu)
end

# --- skew_normal (brms's mean / sd parameterisation): F = Phi(z) - 2 T(z, alpha) ------
# That difference cancels where F is tiny (z < 0 with alpha > 0: the thin tail;
# Stan's skew_normal_lcdf loses ~1e-4 there, as a 256-bit check showed). There
# F = 2 int_{-Inf}^z phi(t) Phi(alpha t) dt directly: the integrand decreases away
# from t = z at rate lam = -z + alpha Phi'/Phi(alpha z), so [z - L, z] with
# lam L + L^2 / 2 = 40 holds all of it; two Gauss-Legendre panels, in log space.
# The upper tail by reflection: 1 - F(z; alpha) = F(-z; -alpha).
function _sn_lcdf_std(z, a)
    if z < 0 && a > 0
        lg0 = normlogpdf(z) + normlogcdf(a * z)
        lam = -z + a * exp(normlogpdf(a * z) - normlogcdf(a * z))
        L = -lam + sqrt(lam * lam + _pv(lam, 80))
        xs, ws = _GL01
        s = zero(lg0)
        for panel in 0:1, k in eachindex(xs)
            t = z - (panel + _pv(z, xs[k])) * L / 2
            s += _pv(z, ws[k]) * exp(normlogpdf(t) + normlogcdf(a * t) - lg0)
        end
        return _pv(z, _LOG2) + lg0 + log(s * L / 2)
    end
    return log(normcdf(z) - 2 * _owens_t(z, a))
end
function _sn_z(f::SkewNormal, y, eta)
    xi, omega, _ = _sn_params(linkinv(link(f), eta[1]), eta[2], eta[3])
    return (y - xi) / omega
end
_cc_lcdf(f::SkewNormal, y, eta) = _sn_lcdf_std(_sn_z(f, y, eta), eta[3])
_cc_lccdf(f::SkewNormal, y, eta) = _sn_lcdf_std(-_sn_z(f, y, eta), -eta[3])

# --- shifted_lognormal: lognormal of y - ndt ------------------------------------------
function _sln_z(f::ShiftedLognormal, y, eta)
    d = y - _pv(eta[3], f.miny) * logistic(eta[3])
    d > 0 || return oftype(d, -Inf)
    return (log(d) - linkinv(link(f), eta[1])) * exp(-eta[2])
end
_cc_lcdf(f::ShiftedLognormal, y, eta) = normlogcdf(_sln_z(f, y, eta))
_cc_lccdf(f::ShiftedLognormal, y, eta) = normlogcdf(-_sln_z(f, y, eta))

# --- exgaussian: Stan's exp_mod_normal cdf, gaussian mean m = mu - beta, rate 1 / beta --
#   F(y) = Phi(u) - exp(c) Phi(u - s / beta),  u = (y - m) / s,  c = -(y - m) / beta + s^2 / (2 beta^2)
function _exg_parts(f::ExGaussian, y, eta)
    s, b = exp(eta[2]), exp(eta[3])
    m = linkinv(link(f), eta[1]) - b
    u = (y - m) / s
    return normlogcdf(u), normlogcdf(-u), -(y - m) / b + s * s / (2 * b * b) + normlogcdf(u - s / b)
end
_cc_lcdf(f::ExGaussian, y, eta) = ((lu, _, lc) = _exg_parts(f, y, eta); lu + log1mexp(lc - lu))
_cc_lccdf(f::ExGaussian, y, eta) = ((_, lnu, lc) = _exg_parts(f, y, eta); logaddexp(lnu, lc))

# --- inverse.gaussian: F = Phi(r (y/mu - 1)) + exp(2 lambda / mu) Phi(-r (y/mu + 1)), r = sqrt(lambda / y)
function _ig_parts(f::InverseGaussian, y, eta)
    mu = linkinv(link(f), eta[1]); lam = exp(eta[2])
    r = sqrt(lam / y)
    return r * (y / mu - 1), 2lam / mu + normlogcdf(-r * (y / mu + 1)), mu
end
function _cc_lcdf(f::InverseGaussian, y, eta)
    a, lb, mu = _ig_parts(f, y, eta)
    mu > 0 || return oftype(a, -Inf)
    return logaddexp(normlogcdf(a), lb)
end
function _cc_lccdf(f::InverseGaussian, y, eta)
    a, lb, mu = _ig_parts(f, y, eta)
    mu > 0 || return oftype(a, -Inf)
    la = normlogcdf(-a)
    return la + log1mexp(lb - la)
end

# --- gamma (mean mu, shape a): P(a, y a / mu) ---------------------------------------
function _cc_gamma(f::GammaFamily, y, eta, upper)
    a = exp(eta[2]); mu = linkinv(link(f), eta[1])
    mu > 0 || return oftype(a * mu, -Inf)
    return _loggammainc(a, y * a / mu, upper)
end
_cc_lcdf(f::GammaFamily, y, eta) = _cc_gamma(f, y, eta, false)
_cc_lccdf(f::GammaFamily, y, eta) = _cc_gamma(f, y, eta, true)

# --- frechet: F = exp(-(y / s)^-nu), s = mu / Gamma(1 - 1/nu) ---------------------------
function _cc_lcdf(f::Frechet, y, eta)
    nu = 1 + exp(eta[2])
    ls = _logmu(link(f), eta[1]) - loggamma(1 - 1 / nu)
    return -exp(-nu * (log(y) - ls))
end

# --- asym_laplace (brms's asym_laplace_lcdf / _lccdf) ---------------------------------
function _cc_lcdf(f::AsymLaplace, y, eta)
    mu = linkinv(link(f), eta[1]); s = exp(eta[2]); q = logistic(eta[3])
    y < mu && return log(q) + (1 - q) * (y - mu) / s
    return log1p(-(1 - q) * exp(-q * (y - mu) / s))
end
function _cc_lccdf(f::AsymLaplace, y, eta)
    mu = linkinv(link(f), eta[1]); s = exp(eta[2]); q = logistic(eta[3])
    y < mu && return log1p(-q * exp((1 - q) * (y - mu) / s))
    return log1p(-q) - q * (y - mu) / s
end

# --- beta (mean mu, precision phi): the regularised incomplete beta -------------------
function _cc_beta(f::BetaFamily, y, eta, upper)
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2])
    a, b = mu * phi, (1 - mu) * phi
    (a > 0 && b > 0) || return oftype(a, -Inf)
    y <= 0 && return upper ? zero(a) : oftype(a, -Inf)
    y >= 1 && return upper ? oftype(a, -Inf) : zero(a)
    return _logbetainc(y + zero(a), a, b, upper)
end
_cc_lcdf(f::BetaFamily, y, eta) = _cc_beta(f, y, eta, false)
_cc_lccdf(f::BetaFamily, y, eta) = _cc_beta(f, y, eta, true)

# --- zero_inflated_beta / hurdle_gamma / hurdle_lognormal (brms's Stan functions):
#     lccdf(y) = log(1 - p) + base lccdf(y),  lcdf = log(1 - exp(lccdf))
_zh_base(f::ZeroInflatedBeta{L}) where {L} = BetaFamily{L}()
_zh_base(f::HurdleGamma{L}) where {L} = GammaFamily{L}()
_zh_base(f::HurdleLognormal{L}) where {L} = Lognormal{L}()
_cc_lccdf(f::Union{ZeroInflatedBeta,HurdleGamma,HurdleLognormal}, y, eta) =
    -log1pexp(eta[3]) + _cc_lccdf(_zh_base(f), y, (eta[1], eta[2]))
_cc_lcdf(f::Union{ZeroInflatedBeta,HurdleGamma,HurdleLognormal}, y, eta) = log1mexp(_cc_lccdf(f, y, eta))

# --- the closed-form censoring families (censoring.jl) as plain log-cdfs, for
# cens() together with trunc() and as bases above
_cvx(::Family, v) = v
_cvx(::Union{Lognormal,Weibull}, v) = v > 0 ? log(v) : oftype(float(v), -Inf)
_cc_lcdf(f::_CLOSED_CENS, y, eta) = first(_lcdf_grad(f, y, _cvx(f, y), eta[1], _pre(f, eta)))
_cc_lccdf(f::_CLOSED_CENS, y, eta) = first(_lccdf_grad(f, y, _cvx(f, y), eta[1], _pre(f, eta)))

# --- zero_inflated_asym_laplace (brms's _lcdf / _lccdf: the zero mass enters at y >= 0)
function _cc_lcdf(f::ZeroInflatedAsymLaplace, y, eta)
    b = -log1pexp(eta[4]) + _cc_lcdf(_zial_base(f), y, (eta[1], eta[2], eta[3]))
    return y < 0 ? b : logaddexp(-log1pexp(-eta[4]), b)
end
function _cc_lccdf(f::ZeroInflatedAsymLaplace, y, eta)
    b = -log1pexp(eta[4]) + _cc_lccdf(_zial_base(f), y, (eta[1], eta[2], eta[3]))
    return y > 0 ? b : logaddexp(-log1pexp(-eta[4]), b)
end

# --- von_mises: Stan's von_mises_cdf (brms calls von_mises_lcdf / _lccdf) -------------
# centred cdf: the series (Hill 1977, as Stan) for kappa < 49, the normal
# approximation for kappa >= 50, linear interpolation between; the mean and x are
# moved to (-pi, pi) and the cdf taken from the cut opposite the mean, as Stan.
_vm_val(x::ForwardDiff.Dual) = _vm_val(ForwardDiff.value(x))
_vm_val(x) = x
function _vm_cdf_series(x, k)
    p = trunc(Int, _vm_val(28 + k / 2 - 100 / (k + 5) + 1))
    s, c = sin(x), cos(x)
    sn, cn = sin(p * x), cos(p * x)
    R = zero(x * k); V = zero(x * k)
    for n in (p - 1):-1:1
        sn, cn = sn * c - cn * s, cn * c + sn * s
        R = 1 / (2n / k + R)
        V = R * (sn / n + V)
    end
    return 1 / 2 + x / _pv(x, 2π) + V / _pv(x, π)
end
function _vm_cdf_normalapprox(x, k)
    b = sqrt(_pv(k, 2 / π)) * exp(k - _logI0(k))
    return normcdf(b * sin(x / 2))
end
function _vm_cdf_centered(x, k)
    if k < 49
        f = _vm_cdf_series(x, k)
        return f < 0 ? zero(f) : f > 1 ? one(f) : f
    elseif k < 50
        return (50 - k) * _vm_cdf_series(x, oftype(k, 49)) + (k - 49) * _vm_cdf_normalapprox(x, oftype(k, 50))
    end
    return _vm_cdf_normalapprox(x, k)
end
function _vm_cdf(y, mu, k)
    tp = _pv(mu, 2π); p = _pv(mu, π)
    x2 = y - mu + p
    x2 = x2 - floor(_vm_val(x2) / _vm_val(tp)) * tp - p
    mu2 = mu + p
    mu2 = mu2 - floor(_vm_val(mu2) / _vm_val(tp)) * tp - p
    # the centred position of -pi (where the cdf starts) and the point past which
    # y - mu wraps around the circle
    cut, bound = mu2 < 0 ? (mu2 + p, -p - mu2) : (mu2 - p, p - mu2)
    fb = _vm_cdf_centered(bound + zero(x2), k)
    f = y <= cut ? _vm_cdf_centered(x2, k) - fb : _vm_cdf_centered(x2, k) + 1 - fb
    return f < 0 ? zero(f) : f
end
_cc_lcdf(f::VonMises, y, eta) = log(_vm_cdf(y, linkinv(link(f), eta[1]), exp(eta[2])))
_cc_lccdf(f::VonMises, y, eta) = log1p(-_vm_cdf(y, linkinv(link(f), eta[1]), exp(eta[2])))
