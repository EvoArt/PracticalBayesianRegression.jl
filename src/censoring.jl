# =============================================================================
# Censored responses: brms `y | cens(c)` and `y | cens(c, y2)`.
#
# Per observation, with c = -1 (left), 0 (none), 1 (right), 2 (interval):
#     none      log f(y)
#     right     log P(Y > y)   = log S(y)
#     left      log P(Y <= y)  = log F(y)
#     interval  log P(y < Y <= y2) = log(F(y2) - F(y))
#
# `Censored{F}` wraps a family and dispatches on the code, so the density
# needs no changes: the code and y2 travel in the per-observation family data
# `x = (code, x(y2), x(y))`. Closed-form log-cdf/log-ccdf derivatives exist
# for the location-scale and Weibull-type families here; others raise at build
# time rather than silently ignoring the censoring.
# =============================================================================

struct Censored{F<:Family} <: Family
    fam::F
    codes::Vector{Int}       # per observation: -1 left, 0 none, 1 right, 2 interval
    y2::Vector{Float64}      # upper bound for interval-censored observations
end

# count families: cdfs as sums of the family's own pmf (end of file), any link
const _DISC_CENS = Union{Poisson,NegBinomial,Geometric,Binomial,BetaBinomial,NegBinomial2,DiscreteWeibull,ComPoisson,
                         ZeroInflatedPoisson,ZeroInflatedNegBinomial,ZeroInflatedBinomial,ZeroInflatedBetaBinomial,
                         HurdlePoisson,HurdleNegBinomial}
# continuous families with a generic log-cdf / log-ccdf (contcens.jl), dual-number
# derivatives for the censored observations only
const _CONT_CENS = Union{Student,SkewNormal,ShiftedLognormal,ExGaussian,InverseGaussian,GammaFamily,Frechet,
                         AsymLaplace,ZeroInflatedAsymLaplace,BetaFamily,ZeroInflatedBeta,HurdleGamma,HurdleLognormal,VonMises}
const _GEN_CENS = Union{_DISC_CENS,_CONT_CENS}
# closed-form log-cdf derivatives below (their own fast censoring path)
const _CLOSED_CENS = Union{Gaussian,Lognormal,Weibull,Exponential}
const _CENSORABLE = Union{_CLOSED_CENS,_GEN_CENS}

# delegation: a censored family is its base family for everything but the
# likelihood of the censored observations
link(c::Censored) = link(c.fam)
dpars(c::Censored) = dpars(c.fam)
aux_bounds(c::Censored, d::Symbol) = aux_bounds(c.fam, d)
dpar_link_name(c::Censored, d::Symbol) = dpar_link_name(c.fam, d)
prior_link(c::Censored) = prior_link(c.fam)
mu_link(c::Censored) = mu_link(c.fam)
uses_trials(c::Censored) = uses_trials(c.fam)
family_name(c::Censored) = family_name(c.fam)
@inline _pre(c::Censored, eta) = _pre(c.fam, eta)
_mean(c::Censored, eta, t) = _mean(c.fam, eta, t)
_rand(rng, c::Censored, eta, t) = _rand(rng, c.fam, eta, t)
_check_response(c::Censored, y, t) = _check_response(c.fam, y, t)
_ll_const(c::Censored, y, t) = _ll_const(c.fam, y, t)

# per-observation constant: only uncensored observations have a density
_ll_const_obs(fam::Family, y, t, x) = _ll_const(fam, y, t)
_ll_const_obs(c::Censored, y, t, x) = x[1] == 0 ? _ll_const(c.fam, y, t) : 0.0

"""
    cens_code(v)

brms's censoring codes: "left"/-1, "none"/0/false, "right"/1/true, "interval"/2.
"""
cens_code(v::Integer) = (v in (-1, 0, 1, 2) || throw(ArgumentError("censoring code must be -1, 0, 1 or 2, got $v")); Int(v))
cens_code(v::Bool) = v ? 1 : 0
cens_code(v::Real) = cens_code(Int(v))
function cens_code(v::AbstractString)
    s = lowercase(v)
    s == "left" && return -1
    s == "none" && return 0
    s == "right" && return 1
    s == "interval" && return 2
    throw(ArgumentError("censoring value must be left/none/right/interval, got `$v`"))
end
cens_code(v::Symbol) = cens_code(string(v))

# x data of the base family for a single value
_xone(fam::Family, v, t=0) = (xv = _xdata(fam, [v], [t]); isempty(xv) ? 0.0 : xv[1])
_tri(trials, i) = isempty(trials) ? 0 : trials[i]
# the interval's upper bound on the scale the cdf formulas use
_cens_x2(::Union{Gaussian,Exponential}, v) = Float64(v)
_cens_x2(::Union{Lognormal,Weibull}, v) = log(v)

_xdata(c::Censored, y, trials) =
    [(Float64(c.codes[i]), c.codes[i] == 2 ? _cens_x2(c.fam, c.y2[i]) : 0.0, _xone(c.fam, y[i], _tri(trials, i)))
     for i in eachindex(y)]

@inline function _ll_grad_pre(c::Censored, y, x, m, p)
    code = x[1]
    code == 0 && return _ll_grad_pre(c.fam, y, x[3], m, p)
    code == 1 && return _lccdf_grad(c.fam, y, x[3], m, p)
    code == -1 && return _lcdf_grad(c.fam, y, x[3], m, p)
    return _lint_grad(c.fam, y, x[3], x[2], m, p)
end
@inline _ll(c::Censored, y, x, eta) = first(_ll_grad_pre(c, y, x, eta[1], _pre(c, eta)))

# --- gaussian / lognormal: z = (v - mu) / sigma, v = y or log y -------------------
# d/d mu = dz * (-1/sigma),  d/d log sigma = dz * (-z)
_cv(::Gaussian, y, x) = y
_cv(::Lognormal, y, x) = x                 # x = log y

@inline function _lccdf_mu(f::Union{Gaussian,Lognormal}, y, x, mu, p)
    z = (_cv(f, y, x) - mu) * p.is
    r = mills(-z)                          # -d/dz log Phi(-z)
    return normlogcdf(-z), (r * p.is, r * z)
end
@inline function _lcdf_mu(f::Union{Gaussian,Lognormal}, y, x, mu, p)
    z = (_cv(f, y, x) - mu) * p.is
    r = mills(z)
    return normlogcdf(z), (-r * p.is, -r * z)
end
@inline function _lint_mu(f::Union{Gaussian,Lognormal}, y, x, x2, mu, p)
    z1 = (_cv(f, y, x) - mu) * p.is
    z2 = (_cv(f, x2, x2) - mu) * p.is      # for gaussian x2 = y2; for lognormal x2 = log y2
    v, d2, d1 = _cum_mid(ProbitLink(), z2, z1)
    return v, (-(d1 + d2) * p.is, -(d1 * z1 + d2 * z2))
end
# any mean link: the formulas above at mu = linkinv(eta), d/d eta = d/d mu * mu_eta
@inline _chain_mu(L, e, (v, g)) = (v, (g[1] * mu_eta(L, e), Base.tail(g)...))
@inline _lccdf_grad(f::Union{Gaussian,Lognormal}, y, x, e, p) =
    _chain_mu(link(f), e, _lccdf_mu(f, y, x, linkinv(link(f), e), p))
@inline _lcdf_grad(f::Union{Gaussian,Lognormal}, y, x, e, p) =
    _chain_mu(link(f), e, _lcdf_mu(f, y, x, linkinv(link(f), e), p))
@inline _lint_grad(f::Union{Gaussian,Lognormal}, y, x, x2, e, p) =
    _chain_mu(link(f), e, _lint_mu(f, y, x, x2, linkinv(link(f), e), p))

# --- weibull: z = (y / lambda)^k, S(y) = exp(-z), log lambda = eta - lgamma(1 + 1/k) ---
# dz/d eta = -k z;  dz/dk = z (log y - log lambda - digamma(1 + 1/k) / k)
@inline function _weibull_z(y, ly, e, p)
    k = p.k
    llam = e - p.lg1
    d = ly - llam
    z = exp(k * d)
    return z, -k * z, z * (d - p.dg1 / k)
end
@inline function _lccdf_lm(::Weibull, y, ly, e, p)
    z, dze, dzk = _weibull_z(y, ly, e, p)
    return -z, (-dze, -dzk * p.k)
end
@inline function _lcdf_lm(::Weibull, y, ly, e, p)
    z, dze, dzk = _weibull_z(y, ly, e, p)
    r = 1 / expm1(z)                       # d/dz log(1 - exp(-z))
    return log1mexp(-z), (r * dze, r * dzk * p.k)
end
@inline function _lint_lm(::Weibull, y, ly, ly2, e, p)
    z1, d1e, d1k = _weibull_z(y, ly, e, p)
    z2, d2e, d2k = _weibull_z(exp(ly2), ly2, e, p)
    # log(exp(-z1) - exp(-z2)); d/dz1 = -w1, d/dz2 = w1 - 1
    w1 = -1 / expm1(z1 - z2)
    v = -z1 + log1mexp(z1 - z2)
    return v, (-w1 * d1e + (w1 - 1) * d2e, (-w1 * d1k + (w1 - 1) * d2k) * p.k)
end

# --- exponential: z = y exp(-eta), S = exp(-z) --------------------------------------
@inline function _lccdf_lm(::Exponential, y, x, e, p)
    z = y * exp(-e)
    return -z, (z,)
end
@inline function _lcdf_lm(::Exponential, y, x, e, p)
    z = y * exp(-e)
    return log1mexp(-z), (-z / expm1(z),)
end
@inline function _lint_lm(::Exponential, y, x, x2, e, p)
    z1 = y * exp(-e); z2 = x2 * exp(-e)   # x2 = y2 (exponential has no x data)
    w1 = -1 / expm1(z1 - z2)
    return -z1 + log1mexp(z1 - z2), (w1 * z1 - (w1 - 1) * z2,)
end

# weibull / exponential under any link: the formulas above take e = log mu;
# chain d/d log mu by d log mu / d eta (exactly 1 for the log link)
@inline function _chain_lm(f, e, g)
    lm, c = _logmu_d(link(f), e)
    v, gr = g(lm)
    return v, (gr[1] * c, Base.tail(gr)...)
end
@inline _lccdf_grad(f::Union{Weibull,Exponential}, y, x, e, p) = _chain_lm(f, e, lm -> _lccdf_lm(f, y, x, lm, p))
@inline _lcdf_grad(f::Union{Weibull,Exponential}, y, x, e, p) = _chain_lm(f, e, lm -> _lcdf_lm(f, y, x, lm, p))
@inline _lint_grad(f::Union{Weibull,Exponential}, y, x, x2, e, p) = _chain_lm(f, e, lm -> _lint_lm(f, y, x, x2, lm, p))


# --- count families: the cdf as a sum of the pmf --------------------------------------
#
# brms's Stan code calls <family>_lcdf / _lccdf; here P(a <= Y <= b) is a
# streaming log-sum of the family's own normalised log pmf (`_ll` + `_ll_const`),
# so every link and every count family works without a derivation. Censored or
# truncated observations are the only ones that pay for it. Their derivatives
# come from one dual-number pass over (eta_mu, eta_aux...); the generic (AD
# reference) path evaluates the same sums directly in its own number type.
# `_pre` keeps the raw aux etas next to the base family's pre-computed values,
# so uncensored observations keep the closed-form fast path.

@inline _disc_t(fam, xb) = uses_trials(fam) ? round(Int, xb) : 0
_disc_ub(fam, t) = uses_trials(fam) ? t : typemax(Int)
# the count in eta's precision (exact in Float32 below 2^24), so a Float32 fit's
# pmf sums stay Float32
@inline function _disc_lpmf(fam, j, xb, eta, t)
    y = _pv(eta[1], Float64(j))
    return _ll(fam, y, xb, eta) + _pv(eta[1], _ll_const(fam, y, t))
end

# log sum_{j = a}^{b} p(j); b = typemax(Int): until the terms are negligible
function _disc_lsum(fam, a::Int, b::Int, xb, eta, t)
    acc = oftype(_disc_lpmf(fam, max(a, 0), xb, eta, t), -Inf)
    b = min(b, _disc_ub(fam, t))
    prev = acc
    j = max(a, 0)
    while j <= b
        l = _disc_lpmf(fam, j, xb, eta, t)
        acc = logaddexp(acc, l)
        # an open-ended tail: stop once the pmf decreases and adds < e^-40 of the sum
        if b == typemax(Int) && j > a + 2 && l < prev && l < acc - 40
            break
        end
        j - a > 10_000_000 && break
        prev = l
        j += 1
    end
    return acc
end
_disc_lcdf(fam, y, xb, eta, t) = y < 0 ? oftype(_disc_lpmf(fam, 0, xb, eta, t), -Inf) :
                                 _disc_lsum(fam, 0, round(Int, y), xb, eta, t)
function _disc_lccdf(fam, y, xb, eta, t)
    y < 0 && return zero(_disc_lpmf(fam, 0, xb, eta, t))
    lF = _disc_lcdf(fam, y, xb, eta, t)
    # 1 - F loses everything when F is near 1: sum the upper tail directly then
    return lF < -_pv(lF, _LOG2) ? log1mexp(lF) : _disc_lsum(fam, round(Int, y) + 1, typemax(Int), xb, eta, t)
end
_disc_lint(fam, y, y2, xb, eta, t) = _disc_lsum(fam, round(Int, y) + 1, round(Int, y2), xb, eta, t)

# value and d/d eta of a scalar function of the eta tuple, by one dual pass
@inline function _dual_vg(f, eta::NTuple{D,T}) where {D,T}
    duals = ntuple(k -> ForwardDiff.Dual{Nothing}(eta[k], ntuple(j -> T(j == k), Val(D))), Val(D))
    r = f(duals)
    return T(ForwardDiff.value(r)), ntuple(k -> T(ForwardDiff.partials(r)[k]), Val(D))
end

_cens_x2(::_GEN_CENS, v) = Float64(v)
_disc_cens_val(fam, code, y, y2, xb, eta, t) =
    code == 1 ? _disc_lccdf(fam, y, xb, eta, t) :
    code == -1 ? _disc_lcdf(fam, y, xb, eta, t) : _disc_lint(fam, y, y2, xb, eta, t)
# continuous: right / left / interval from the family's log-cdf and log-ccdf
_cont_cens_val(fam, code, y, y2, eta) =
    code == 1 ? _cc_lccdf(fam, y, eta) : code == -1 ? _cc_lcdf(fam, y, eta) : _cc_lint(fam, y, y2, eta)
_gen_cens_val(fam::_DISC_CENS, code, y, y2, xb, eta) = _disc_cens_val(fam, code, y, y2, xb, eta, _disc_t(fam, xb))
_gen_cens_val(fam::Union{_CONT_CENS,_CLOSED_CENS}, code, y, y2, xb, eta) = _cont_cens_val(fam, code, y, y2, eta)
# log(F(b) - F(a)) from whichever tail keeps the precision
function _cc_lint(fam, a, b, eta)
    lb = _cc_lcdf(fam, b, eta)
    if lb < -_pv(lb, _LOG2)
        return lb + log1mexp(_cc_lcdf(fam, a, eta) - lb)
    end
    ca = _cc_lccdf(fam, a, eta)
    return ca + log1mexp(_cc_lccdf(fam, b, eta) - ca)
end
_cc_lccdf(fam, y, eta) = log1mexp(_cc_lcdf(fam, y, eta))

@inline _pre(c::Censored{<:_GEN_CENS}, eta) = (Base.tail(eta), _pre(c.fam, eta))
@inline function _ll_grad_pre(c::Censored{<:_GEN_CENS}, y, x, m, p)
    aux, pb = p
    code = x[1]
    code == 0 && return _ll_grad_pre(c.fam, y, x[3], m, pb)
    return _dual_vg(e -> _gen_cens_val(c.fam, code, y, x[2], x[3], e), (m, aux...))
end
@inline _ll(c::Censored{<:_GEN_CENS}, y, x, eta) =
    x[1] == 0 ? _ll(c.fam, y, x[3], eta) : _gen_cens_val(c.fam, x[1], y, x[2], x[3], eta)
