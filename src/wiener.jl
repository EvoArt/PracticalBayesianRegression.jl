# =============================================================================
# brms `wiener()`: the four-parameter drift-diffusion model of response times,
# `rt | dec(d) ~ ...` with d = 1 for the upper boundary, 0 for the lower.
# dpars: mu (drift, identity), bs (boundary separation, log), ndt (non-decision
# time, log), bias (a-priori bias, logit). Stan's wiener_lpdf is the UPPER
# boundary density; brms uses it with (bias, mu) for d = 1 and (1 - bias, -mu)
# for d = 0. Here the lower-boundary density of Navarro & Fuss (2009):
#     f(t | v, a, w) = a^-2 exp(-v a w - v^2 (t - tau) / 2) f0(u | w),  u = (t - tau) / a^2
#     f0 small-time:  (2 pi u^3)^-1/2 sum_k (w + 2k) exp(-(w + 2k)^2 / (2u))
#     f0 large-time:  pi sum_{k >= 1} k exp(-k^2 pi^2 u / 2) sin(k pi w)
# with the number of terms from their error bounds (eps = 1e-10), whichever
# series is shorter; the upper boundary is the lower one at (-v, 1 - w).
# The term counts come from plain values, so dual numbers flow through the
# sums (a DualFamily: gradients by one dual evaluation per observation).
# =============================================================================

struct Wiener{L<:Link} <: DualFamily
    dec::Vector{Float64}       # 1 upper / 0 lower boundary (from dec())
    miny::Float64              # ndt < min(y)
end
Wiener{L}() where {L} = Wiener{L}(Float64[], Inf)

"""
    wiener(; link = :identity)

brms's `wiener()` family (drift-diffusion); the response needs `| dec(d)`.
Links for the drift `mu`: `identity` (default), `log`, `softplus`, `squareplus`.
"""
function wiener(; link=nothing)
    l = link === nothing ? :identity : Symbol(link)
    l in (:identity, :log, :softplus, :squareplus) ||
        throw(ArgumentError("family wiener supports links identity, log, softplus, squareplus, not `$l`"))
    return Wiener{typeof(_link(l))}()
end
link(::Wiener{L}) where {L} = L()
family_name(::Wiener) = "wiener"
dpars(::Wiener) = (:mu, :bs, :ndt, :bias)
aux_bounds(::Wiener, d::Symbol) = d === :bias ? (:logit, 0.0) : d === :ndt ? (:logit, 0.0) : (:log, 0.0)
aux_ub(f::Wiener, d::Symbol) = d === :ndt ? f.miny : 1.0
dpar_link_name(f::Wiener, d::Symbol) = d === :mu ? link_name(link(f)) : d === :bias ? "logit" : "log"
_with_dec(f::Wiener{L}, dec, y) where {L} = Wiener{L}(Float64.(dec), minimum(y))
_xdata(f::Wiener, y, trials) = f.dec
function _check_response(f::Wiener, y, t)
    isempty(f.dec) && throw(ArgumentError("wiener needs the boundary of each response: rt | dec(d) ~ ..."))
    _chk(all(>(0), y), "wiener response times must be positive")
    _chk(all(d -> d == 0 || d == 1, f.dec), "dec() must be 0 (lower) or 1 (upper)")
end

# log f0(u | w): the standardised (v = 0, a = 1) lower-boundary density
function _wiener_logf0(u, w)
    uv, wv = ForwardDiff.value(u), ForwardDiff.value(w)
    # u -> 0 (a warm-up proposal with ndt at the response time): density 0, and the
    # term counts below would be Inf (ceil(Int, Inf) throws)
    (uv > 0 && isfinite(uv)) || return oftype(u * w, -Inf)
    ϵ = 1e-10
    kl = π * uv * ϵ < 1 ? max(sqrt(-2log(π * uv * ϵ) / (π^2 * uv)), 1 / (π * sqrt(uv))) : 1 / (π * sqrt(uv))
    ks = 2sqrt(2π * uv) * ϵ < 1 ? max(2 + sqrt(-2uv * log(2sqrt(2π * uv) * ϵ)), sqrt(uv) + 1) : 2.0
    kl = min(kl, 1e4); ks = min(ks, 1e4)            # bounded work at extreme u
    # each series relative to its leading exponential (k = 0 / k = 1), so a
    # far-tail density does not underflow to log(0) = -Inf
    if ks < kl
        K = ceil(Int, ks)
        s = zero(u * w)
        for k in (-floor(Int, (K - 1) / 2)):ceil(Int, (K - 1) / 2)
            s += (w + 2k) * exp(-((w + 2k)^2 - w^2) / (2u))
        end
        return log(s) - w^2 / (2u) - (_pv(u, _LOG_2PI) + 3log(u)) / 2
    end
    K = ceil(Int, kl)
    s = zero(u * w)
    for k in 1:K
        s += k * exp(-(k^2 - 1) * _pv(u, π^2) * u / 2) * sin(k * _pv(w, π) * w)
    end
    return log(s) - _pv(u, π^2) * u / 2 + _pv(u, log(π))
end

# log density of response time t at the lower boundary
function _wiener_lower(t, v, a, τ, w)
    tt = t - τ
    tt > 0 || return oftype(v * a * w * tt, -Inf)
    return -v * a * w - v^2 * tt / 2 - 2log(a) + _wiener_logf0(tt / a^2, w)
end

function _ll(f::Wiener, y, d, eta)
    v = linkinv(link(f), eta[1]); a = exp(eta[2]); τ = _pv(eta[3], f.miny) * logistic(eta[3]); w = logistic(eta[4])
    return d == 1 ? _wiener_lower(y, -v, a, τ, 1 - w) : _wiener_lower(y, v, a, τ, w)
end
_ll_const(::Wiener, y, t) = 0.0
# brms get_prior: bs gamma(1, 1), ndt uniform(0, min_Y) (flat on its bounded scale), bias beta(1, 1)
function _default_prior(f::Wiener, class::Symbol, dpar::Symbol, y)
    class === :bs && return gamma(1, 1)
    class === :bias && return beta_prior(1, 1)
    class === :ndt && return flat()
    return invoke(_default_prior, Tuple{Family,Symbol,Symbol,Any}, f, class, dpar, y)
end
_mean(::Wiener, eta, t) = throw(ArgumentError("posterior_epred is not defined for wiener (as brms); use posterior_predict"))

# a draw (rt, boundary) by simulating the diffusion (Euler, dt = 1e-4); brms
# returns the response time with the boundary sign in its `negative_rt` option --
# here the time, as posterior_predict's default
function _rand(rng, f::Wiener, eta, t)
    v = linkinv(link(f), eta[1]); a = exp(eta[2]); τ = _pv(eta[3], f.miny) * logistic(eta[3]); w = logistic(eta[4])
    x = a * w; s = 0.0; dt = 1e-4; sq = sqrt(dt)
    while 0 < x < a
        x += v * dt + sq * randn(rng); s += dt
    end
    return τ + s
end
