# =============================================================================
# Response addition terms beyond trials / weights / cens:
#
#   y | se(s)                known standard errors (meta-analysis); no sigma
#   y | se(s, sigma = TRUE)  known standard errors plus a residual sigma:
#                            total variance s^2 + sigma^2
#   y | rate(r)              exposure for count models: log(r) offset on mu
#   y | trunc(lb = , ub = )  truncated likelihood
# =============================================================================

# --- se(): gaussian with known measurement error --------------------------------

"""
    GaussianSE{S}

brms's `gaussian()` with `y | se(s)`: per-observation known standard errors
`s` (the family data). `S = false`: no residual sigma, sd = s. `S = true`:
sd = sqrt(s^2 + sigma^2).
"""
struct GaussianSE{L<:Link,S} <: Family
    se::Vector{Float64}
end

link(::GaussianSE{L}) where {L} = L()
family_name(::GaussianSE) = "gaussian"
dpars(::GaussianSE{L,false}) where {L} = (:mu,)
dpars(::GaussianSE{L,true}) where {L} = (:mu, :sigma)
prior_link(::GaussianSE) = :identity
_xdata(f::GaussianSE, y, trials) = copy(f.se)

@inline function _ll(::GaussianSE{L,false}, y, s, eta) where {L}
    r = (y - eta[1]) / s
    return -r * r / 2
end
@inline function _ll_grad_pre(::GaussianSE{L,false}, y, s, mu, p) where {L}
    r = (y - mu) / s
    return -r * r / 2, (r / s,)
end
@inline _pre(::GaussianSE{L,true}, eta) where {L} = (s2=exp(2eta[2]),)
@inline function _ll(f::GaussianSE{L,true}, y, s, eta) where {L}
    return first(_ll_grad_pre(f, y, s, eta[1], _pre(f, eta)))
end
@inline function _ll_grad_pre(::GaussianSE{L,true}, y, s, mu, p) where {L}
    v2 = s * s + p.s2
    e = y - mu
    q = e * e / v2
    return -(q + log(v2)) / 2, (e / v2, (q - 1) * p.s2 / v2)
end
# known-sd part of the constant (the se(s) without sigma case has its log s here)
_ll_const(::GaussianSE, y, t) = -_LOG_2PI_HALF
_ll_const_obs(::GaussianSE{L,false}, y, t, s) where {L} = -_LOG_2PI_HALF - log(s)
_mean(::GaussianSE, eta, t) = eta[1]
_rand(rng, ::GaussianSE{L,false}, eta, t) where {L} = eta[1]     # no residual draw without sigma or s
_rand(rng, ::GaussianSE{L,true}, eta, t) where {L} = eta[1] + exp(eta[2]) * randn(rng)

# --- trunc(): truncated likelihood ------------------------------------------------

"""
    Truncated{F}

A family truncated to [lb, ub] per observation: log f(y) - log(F(ub) - F(lb)),
with the normaliser's derivatives from the censoring module's interval /
one-sided log-cdf formulas.
"""
struct Truncated{F<:Family} <: Family
    fam::F
    lb::Vector{Float64}
    ub::Vector{Float64}
end

link(c::Truncated) = link(c.fam)
dpars(c::Truncated) = dpars(c.fam)
aux_bounds(c::Truncated, d::Symbol) = aux_bounds(c.fam, d)
dpar_link_name(c::Truncated, d::Symbol) = dpar_link_name(c.fam, d)
prior_link(c::Truncated) = prior_link(c.fam)
uses_trials(c::Truncated) = uses_trials(c.fam)
family_name(c::Truncated) = family_name(c.fam)
@inline _pre(c::Truncated, eta) = _pre(c.fam, eta)
_mean(c::Truncated, eta, t) = _mean(c.fam, eta, t)        # untruncated mean (brms warns likewise)
function _rand(rng, c::Truncated, eta, t)
    # rejection from the base family; the bounds are per observation, so
    # posterior_predict on a truncated fit uses the widest training bounds
    lo, hi = minimum(c.lb), maximum(c.ub)
    for _ in 1:10_000
        v = _rand(rng, c.fam, eta, t)
        lo <= v <= hi && return v
    end
    return NaN
end
_check_response(c::Truncated, y, t) = (_check_response(c.fam, y, t);
    _chk(all(i -> c.lb[i] <= y[i] <= c.ub[i], eachindex(y)), "responses must lie within their trunc() bounds"))
_ll_const(c::Truncated, y, t) = _ll_const(c.fam, y, t)

# x = (code, x(lb), x(ub), x(y)); code 1: lb only, -1: ub only, 2: both
function _xdata(c::Truncated, y, trials)
    return map(eachindex(y)) do i
        lb, ub = c.lb[i], c.ub[i]
        code = isfinite(lb) && isfinite(ub) ? 2.0 : isfinite(lb) ? 1.0 : -1.0
        (code, isfinite(lb) ? _cens_x2(c.fam, _tlb(c.fam, lb)) : 0.0, isfinite(ub) ? _cens_x2(c.fam, ub) : 0.0,
         _xone(c.fam, y[i], _tri(trials, i)))
    end
end

@inline function _ll_grad_pre(c::Truncated, y, x, m, p)
    code, xl, xu, xy = x
    v, g = _ll_grad_pre(c.fam, y, xy, m, p)
    # log P(lb < Y <= ub) and its gradient
    vn, gn = if code == 2
        _lint_grad(c.fam, _cens_y(c.fam, xl), xl, xu, m, p)
    elseif code == 1
        _lccdf_grad(c.fam, _cens_y(c.fam, xl), xl, m, p)
    else
        _lcdf_grad(c.fam, _cens_y(c.fam, xu), xu, m, p)
    end
    return v - vn, map(-, g, gn)
end
@inline _ll(c::Truncated, y, x, eta) = first(_ll_grad_pre(c, y, x, eta[1], _pre(c, eta)))

# the raw value back from the stored x (log scale for lognormal/weibull)
_cens_y(::Union{Gaussian,Exponential}, x) = x
_cens_y(::Union{Lognormal,Weibull}, x) = exp(x)
# a count family truncated to [lb, ub] keeps lb itself (Stan's T[lb, ub] for
# integers): the normaliser is P(lb - 1 < Y <= ub), so lb is stored as lb - 1
_tlb(::Family, lb) = lb
_tlb(::_DISC_CENS, lb) = lb - 1

@inline _pre(c::Truncated{<:_DISC_CENS}, eta) = (Base.tail(eta), _pre(c.fam, eta))
@inline function _ll_grad_pre(c::Truncated{<:_DISC_CENS}, y, x, m, p)
    aux, pb = p
    code, xl, xu, xy = x
    v, g = _ll_grad_pre(c.fam, y, xy, m, pb)
    t = _disc_t(c.fam, xy)
    vn, gn = _dual_vg(e -> _disc_trunc_norm(c.fam, code, xl, xu, xy, e, t), (m, aux...))
    return v - vn, map(-, g, gn)
end
@inline _ll(c::Truncated{<:_DISC_CENS}, y, x, eta) =
    _ll(c.fam, y, x[4], eta) - _disc_trunc_norm(c.fam, x[1], x[2], x[3], x[4], eta, _disc_t(c.fam, x[4]))
_disc_trunc_norm(fam, code, xl, xu, xb, eta, t) =
    code == 2 ? _disc_lint(fam, xl, xu, xb, eta, t) :
    code == 1 ? _disc_lccdf(fam, xl, xb, eta, t) : _disc_lcdf(fam, xu, xb, eta, t)
