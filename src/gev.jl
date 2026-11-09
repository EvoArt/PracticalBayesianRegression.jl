# =============================================================================
# Generalised extreme value (brms `gen_extreme_value()`): mu (identity),
# sigma (log), xi. brms samples an unbounded `tmp_xi` and rescales it so the
# density is positive at every observation:
#     x_n = (y_n - mu_n) / sigma_n,   b1 = -1 / min(x),  b2 = -1 / max(x)
#     xi  = inv_logit(tmp_xi) * (ub - lb) + lb,   lb = min(b1, b2), ub = max(b1, b2)
#     ll_n = -log sigma_n - (1 + 1/xi) log(1 + xi x_n) - (1 + xi x_n)^(-1/xi)   (xi != 0)
# tmp_xi: normal(0, 2.5). Because xi depends on min and max of x over all
# observations, this family has its own observation loop: a pre-pass finds
# xi, the per-observation derivatives (dual numbers in mu, log sigma, xi) are
# summed for xi, and d xi flows back through the bounds into the mu and sigma
# of the observations attaining min(x) and max(x).
# =============================================================================

dpars(::GenExtremeValue) = (:mu, :sigma, :xi)
aux_bounds(::GenExtremeValue, d::Symbol) = d === :xi ? (:identity, -Inf) : (:log, 0.0)
dpar_link_name(::GenExtremeValue, d::Symbol) = d === :mu ? "identity" : d === :xi ? "identity (scaled)" : "log"
_ll_const(::GenExtremeValue, y, t) = 0.0
"""
    gen_extreme_value(; link = :identity)

Generalised extreme value distribution, for maxima or minima. Parameters:
`mu` (location), `sigma` (scale) and `xi` (shape). As in brms, `xi` is
bounded so that every observation lies inside the distribution's support.
"""
gen_extreme_value(; link=nothing) = _make_family(GenExtremeValue, link)

function _gev_ll(y, mu, ls, xi)
    x = (y - mu) * exp(-ls)
    abs(xi) < _pv(xi, 1e-12) && return -ls - x - exp(-x)
    t = 1 + xi * x
    t <= 0 && return oftype(t, -Inf)          # outside the support: density 0 (Stan: -Inf, rejected)
    return -ls - (1 + 1 / xi) * log(t) - t^(-1 / xi)
end

# xi from tmp_xi and the extreme standardised residuals, with the pieces the
# gradient needs: (xi, d xi/d tmp, d xi/d xmin, d xi/d xmax)
function _gev_xi(tmp, xmin, xmax)
    s = logistic(tmp)
    b1, b2 = -1 / xmin, -1 / xmax
    lb, ub = min(b1, b2), max(b1, b2)
    xi = s * (ub - lb) + lb
    dtmp = s * (1 - s) * (ub - lb)
    # d xi / d b1 and d b2 (whichever is the upper bound gets weight s)
    db1 = b1 >= b2 ? s : 1 - s
    db2 = b1 >= b2 ? 1 - s : s
    return xi, dtmp, db1 / xmin^2, db2 / xmax^2        # d b1/d xmin = 1/xmin^2, d b2/d xmax = 1/xmax^2
end

_gev_mu(etas, i) = _at(etas[1], i)
_gev_ls(etas, i) = _at(etas[2], i)

function _obs_loop!(ld::GLMDensity{F,P,W,T}, etas) where {F<:GenExtremeValue,P,W,T}
    y, N = ld.y, ld.N
    foreach(_reset!, ld.ws, ld.preds)
    xmin, imin, xmax, imax = T(Inf), 0, T(-Inf), 0
    @inbounds for i in 1:N
        x = (y[i] - _gev_mu(etas, i)) * exp(-_gev_ls(etas, i))
        x < xmin && (xmin = x; imin = i)
        x > xmax && (xmax = x; imax = i)
    end
    tmp = _at(etas[3], 1)
    xi, dtmp, dxmin, dxmax = _gev_xi(tmp, xmin, xmax)
    weighted = !isempty(ld.weights)
    wv = map(pw -> pw.w, ld.ws)
    ll = zero(T); gxi = zero(T)
    @inbounds for i in 1:N
        mu, ls = _gev_mu(etas, i), _gev_ls(etas, i)
        d = ForwardDiff.Dual{Nothing}
        r = _gev_ll(y[i], d(mu, (one(T), zero(T), zero(T))), d(ls, (zero(T), one(T), zero(T))),
                    d(xi, (zero(T), zero(T), one(T))))
        v = ForwardDiff.value(r); p = ForwardDiff.partials(r)
        wt = weighted ? ld.weights[i] : one(T)
        ll += wt * v
        gxi += wt * p[3]
        _puts!(wv, ld.preds, i, (T(p[1]), T(p[2]), zero(T)), wt)
    end
    # xi -> tmp_xi, and the extremes' share through the bounds
    _put!(wv[3], ld.preds[3], 1, gxi * dtmp)
    for (i, dx) in ((imin, dxmin), (imax, dxmax))
        g = gxi * dx
        s = exp(-_gev_ls(etas, i)); x = (y[i] - _gev_mu(etas, i)) * s
        _gev_addw!(wv[1], ld.preds[1], i, -g * s)         # d x / d mu = -1/sigma
        _gev_addw!(wv[2], ld.preds[2], i, -g * x)         # d x / d log sigma = -x
    end
    return ll
end
# add (rather than set) an observation's derivative, for either a per-observation
# predictor or a scalar
_gev_addw!(w, ::LinPred, i, v) = (@inbounds w[i] += v; nothing)
_gev_addw!(w, ::ScalarDpar, i, v) = (@inbounds w[1] += v; nothing)

function _gev_xi_generic(ld, etas)
    xs = [(ld.y[i] - _at(etas[1], i)) * exp(-_at(etas[2], i)) for i in 1:ld.N]
    return first(_gev_xi(_at(etas[3], 1), minimum(xs), maximum(xs)))
end
function LogDensityProblems.logdensity(ld::GLMDensity{F}, θraw::AbstractVector) where {F<:GenExtremeValue}
    θ, ljo = _order_generic(ld.ordered, θraw)
    etas = map(p -> _eta_generic(p, θ, ld.N), ld.preds)
    xi = _gev_xi_generic(ld, etas)
    lp = ljo
    for i in 1:(ld.prior_only ? 0 : ld.N)
        v = _gev_ll(ld.y[i], _at(etas[1], i), _at(etas[2], i), xi)
        lp += isempty(ld.weights) ? v : ld.weights[i] * v
    end
    for p in ld.preds
        lp += _prior_generic(p, θ)
    end
    return lp
end

# per-observation functions take xi itself in the third slot (predict.jl
# substitutes the scaled xi of each draw)
_ll(::GenExtremeValue, y, x, eta) = _gev_ll(y, eta[1], eta[2], eta[3])
function _mean(::GenExtremeValue, eta, t)
    mu, s, xi = eta[1], exp(eta[2]), eta[3]
    abs(xi) < 1e-12 && return mu + s * Base.MathConstants.eulergamma
    return xi < 1 ? mu + s * (exp(loggamma(1 - xi)) - 1) / xi : Inf
end
function _rand(rng, ::GenExtremeValue, eta, t)
    mu, s, xi = eta[1], exp(eta[2]), eta[3]
    u = rand(rng)
    abs(xi) < 1e-12 && return mu - s * log(-log(u))
    return mu + s * ((-log(u))^(-xi) - 1) / xi
end
