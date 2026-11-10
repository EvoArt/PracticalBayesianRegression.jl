# =============================================================================
# Cox proportional hazards (brms `cox()`), as brms 2.23 builds it: the
# baseline hazard is an M-spline basis in the response (splines2::mSpline,
# df = 5, intercept = TRUE, cubic, boundary knots at the response range
# widened by 1/50 on each side and clamped at 0, interior knots at quantiles),
# the cumulative baseline hazard the matching I-spline (its integral), and the
# basis weights a simplex `sbhaz` with a dirichlet(1) prior:
#     h0(t) = Zbhaz(t) . sbhaz,   H0(t) = Zcbhaz(t) . sbhaz,   mu = exp(eta) (log link)
#     event:          log h0 + eta - H0 e^eta
#     right-censored: -H0 e^eta
#     left-censored:  log(1 - exp(-H0 e^eta))
# Censoring as brms's `cens()` (0 event, 1 right, -1 left).
#
# The simplex is a block (like mixture weights) that hands each observation
# (h0_i, H0_i); the family returns d/d h0_i and d/d H0_i, which the block's
# pullback turns into d/d sbhaz = Zbhaz' dh0 + Zcbhaz' dH0, then the
# stick-breaking reverse pass with the Dirichlet prior.
# =============================================================================

Cox{L}() where {L} = Cox{L}(Float64[])
Cox{L}(codes::Vector{Float64}) where {L} = Cox{L}(codes, Float64[], Float64[])
_cox_trunc(f::Cox) = !isempty(f.lb)
# trunc(): which bounds an observation has (1 lb only, -1 ub only, 2 both); a lower
# bound at or below 0 is no bound for event times
_cox_tcode(lb, ub) = (l = lb > 0; u = isfinite(ub); l && u ? 2.0 : l ? 1.0 : u ? -1.0 : 0.0)
dpars(::Cox) = (:mu, :sbhaz)
dpar_link_name(f::Cox, d::Symbol) = d === :mu ? link_name(link(f)) : "simplex"
_xdata(f::Cox, y, trials) = (c = isempty(f.codes) ? zeros(length(y)) : f.codes;
                             _cox_trunc(f) ? [(c[i], _cox_tcode(f.lb[i], f.ub[i])) for i in eachindex(y)] : c)
_ll_const(::Cox, y, t) = 0.0
_check_response(::Cox, y, t) = _chk(all(>(0), y), "cox responses (times) must be positive")

# --- M- and I-spline bases (splines2 conventions) -----------------------------------

struct BhazBasis
    knots::Vector{Float64}       # full knot sequence for the cubic M-splines
    lo::Float64
    hi::Float64
end
# bhaz(df = , knots = , Boundary.knots = ): given interior knots override df
function BhazBasis(y; df=5, knots=nothing, bknots=nothing)
    lo0, hi0 = extrema(y); dy = hi0 - lo0
    lo, hi = bknots === nothing ? (max(lo0 - dy / 50, 0.0), hi0 + dy / 50) : (Float64(bknots[1]), Float64(bknots[2]))
    if knots !== nothing
        inner = sort(Float64.(knots))
        all(k -> lo < k < hi, inner) || throw(ArgumentError("bhaz(): knots must lie inside the boundary knots"))
    else
        nik = df - 4                             # cubic, with intercept
        nik >= 0 || throw(ArgumentError("cox(): df must be at least 4"))
        probs = range(0, 1; length=nik + 2)[2:(end - 1)]
        inner = isempty(probs) ? Float64[] : quantile(y, collect(probs))
    end
    return BhazBasis(vcat(fill(lo, 4), inner, fill(hi, 4)), lo, hi)
end
# M_i = 4 B_i / (t_{i+4} - t_i);  I_i(x) = sum_{j > i} of the order-5 B-splines
# on the knots with one more boundary repeat (the integral of M_i from lo)
function _bhaz_rows(b::BhazBasis, x)
    t = b.knots; n = length(t) - 4
    xc = clamp(x, b.lo, b.hi)
    B4 = _bsplines(t, xc, 4)
    M = [4 * B4[i] / (t[i + 4] - t[i]) for i in 1:n]
    t5 = vcat(b.lo, t, b.hi)
    B5 = _bsplines(t5, xc, 5)
    I = [sum(B5[(i + 1):end]) for i in 1:n]
    return M, I
end
function _bhaz_matrices(b::BhazBasis, y)
    rows = [_bhaz_rows(b, v) for v in y]
    return reduce(vcat, (permutedims(r[1]) for r in rows)), reduce(vcat, (permutedims(r[2]) for r in rows))
end

# --- the simplex block --------------------------------------------------------------

struct BhazSimplex{T}
    dpar::Symbol
    idx::UnitRange{Int}          # K - 1 stick-breaking parameters
    alpha::Float64
    Zb::Matrix{T}                # N x K
    Zc::Matrix{T}
    basis::BhazBasis
    # trunc(): the I-spline (cumulative hazard) rows at each observation's lower /
    # upper bound (0 x 0 without truncation; a missing bound's row is 0)
    Zl::Matrix{T}
    Zu::Matrix{T}
end
BhazSimplex{T}(dpar, idx, alpha, Zb, Zc, basis) where {T} =
    BhazSimplex{T}(dpar, idx, alpha, Zb, Zc, basis, zeros(T, 0, 0), zeros(T, 0, 0))
_K(p::BhazSimplex) = length(p.idx) + 1
_btrunc(p::BhazSimplex) = !isempty(p.Zl)

# per observation: hazard h0, cumulative hazard H0, and (trunc) H0 at the bounds
struct BhazVals{V}
    h::V
    H::V
    Hl::V
    Hu::V
end
BhazVals(h, H) = BhazVals(h, H, similar(h, 0), similar(h, 0))
# (always four values, so the observation loop stays type-stable; 0 without truncation)
@inline _at(b::BhazVals, i) = (b.h[i], b.H[i], isempty(b.Hl) ? zero(b.h[i]) : b.Hl[i], isempty(b.Hu) ? zero(b.h[i]) : b.Hu[i])
struct BhazGrad{T}
    dh::T
    dH::T
    dHl::T
    dHu::T
end
BhazGrad(dh, dH) = BhazGrad(dh, dH, zero(dh), zero(dh))
@inline Base.:*(w::Real, g::BhazGrad) =
    BhazGrad(oftype(g.dh, w * g.dh), oftype(g.dH, w * g.dH), oftype(g.dH, w * g.dHl), oftype(g.dH, w * g.dHu))

# workspace: eta = [h; H (; Hl; Hu)], w the same layout, bvals = simplex, base = rem, wbase = z
function PredWS{T}(p::BhazSimplex, N) where {T}
    K = _K(p); m = _btrunc(p) ? 4 : 2
    return PredWS{T}(zeros(T, m * N), zeros(T, m * N), GroupWS{T}[], zeros(T, K), MoWS{T}[], zeros(T, K),
                     zeros(T, K - 1), Vector{T}[])
end
function _eta!(pw::PredWS, p::BhazSimplex, θ)
    _stick!(pw.bvals, pw.base, pw.wbase, view(θ, p.idx))
    N = size(p.Zb, 1)
    h = view(pw.eta, 1:N); H = view(pw.eta, (N + 1):(2N))
    mul!(h, p.Zb, pw.bvals); mul!(H, p.Zc, pw.bvals)
    _btrunc(p) || return BhazVals(h, H, view(pw.eta, 1:0), view(pw.eta, 1:0))
    Hl = view(pw.eta, (2N + 1):(3N)); Hu = view(pw.eta, (3N + 1):(4N))
    mul!(Hl, p.Zl, pw.bvals); mul!(Hu, p.Zu, pw.bvals)
    return BhazVals(h, H, Hl, Hu)
end
_reset!(pw::PredWS, ::BhazSimplex) = nothing
@inline function _put!(w, p::BhazSimplex, i, g::BhazGrad)
    N = size(p.Zb, 1)
    @inbounds w[i] = g.dh
    @inbounds w[N + i] = g.dH
    if _btrunc(p)
        @inbounds w[2N + i] = g.dHl
        @inbounds w[3N + i] = g.dHu
    end
    return nothing
end
function _pullback!(g, p::BhazSimplex, pw::PredWS{T}, θ) where {T}
    N = size(p.Zb, 1); K = _K(p)
    a = p.Zb' * view(pw.w, 1:N) .+ p.Zc' * view(pw.w, (N + 1):(2N))     # d ll / d sbhaz
    _btrunc(p) && (a .+= p.Zl' * view(pw.w, (2N + 1):(3N)) .+ p.Zu' * view(pw.w, (3N + 1):(4N)))
    gu = zeros(T, K)
    val = _stick_grad!(gu, a, pw.bvals, pw.base, pw.wbase, p.alpha)
    @inbounds for (j, k) in enumerate(p.idx)
        g[k] += gu[j]
    end
    return val
end
function _eta_generic(p::BhazSimplex, θ, N)
    s, _ = _stick(θ[p.idx], _K(p))
    _btrunc(p) || return BhazVals(p.Zb * s, p.Zc * s)
    return BhazVals(p.Zb * s, p.Zc * s, p.Zl * s, p.Zu * s)
end
function _prior_generic(p::BhazSimplex, θ)
    s, lj = _stick(θ[p.idx], _K(p))
    return lj + (p.alpha == 1 ? zero(lj) : (p.alpha - 1) * sum(log, s))
end

# --- the family -----------------------------------------------------------------------

@inline _pre(::Cox, eta) = eta[2]
# mu = linkinv(eta) is the hazard multiplier (exp(eta) for the log link, as
# brms's cox_log_lpdf; other links as its cox_lpdf). Derivatives are taken in
# log mu and chained by c = d log mu / d eta (1 for the log link).
# (value, d/d log mu, d/d h0, d/d H0) of one observation
@inline function _cox_obs(code, h, H, lm, em)
    if code == 0                       # event
        return log(h) + lm - H * em, 1 - H * em, 1 / h, -em
    elseif code == 1                   # right-censored
        return -H * em, -H * em, zero(h), -em
    end
    # left-censored: log(1 - exp(-H mu))
    q = H * em
    v = log1mexp(-q)
    r = exp(-q - v)                    # exp(-q) / (1 - exp(-q))
    return v, r * q, zero(h), r * em
end
@inline function _ll_grad_pre(f::Cox, y, code::Real, m, hb)
    h, H = hb
    lm, c = _logmu_d(link(f), m)
    isfinite(lm) || return oftype(lm * h, -Inf), (zero(lm * h), BhazGrad(zero(h), zero(h)))
    v, dlm, dh, dH = _cox_obs(code, h, H, lm, exp(lm))
    return v, (dlm * c, BhazGrad(dh, dH))
end
# trunc(): minus log P(lb < T <= ub) = log(S(lb) - S(ub)), S(t) = exp(-H0(t) mu), with
# H0 at the observation's own bounds. (brms 2.23 normalises with the cumulative
# hazard at the RESPONSE instead of at the bounds -- its cox_log_lcdf ignores its
# y argument -- so its truncated cox likelihood is not a truncated density.)
@inline function _ll_grad_pre(f::Cox, y, x::Tuple, m, hb)
    code, tcode = x
    h, H, Hl, Hu = hb
    lm, c = _logmu_d(link(f), m)
    z = zero(lm * h)
    isfinite(lm) || return oftype(z, -Inf), (z, BhazGrad(z, z, z, z))
    em = exp(lm)
    v, dlm, dh, dH = _cox_obs(code, h, H, lm, em)
    ql = Hl * em; qu = Hu * em
    # n = log normaliser and its derivatives in ql, qu
    if tcode == 1                      # lb only: log S(lb) = -ql
        n, dql, dqu = -ql, -one(z), z
    elseif tcode == -1                 # ub only: log(1 - S(ub))
        n = log1mexp(-qu); r = exp(-qu - n)
        dql, dqu = z, r
    elseif tcode == 2                  # both: -ql + log(1 - exp(-(qu - ql)))
        dd = qu - ql
        L = log1mexp(-dd); r = exp(-dd - L)
        n, dql, dqu = -ql + L, -1 - r, r
    else
        n, dql, dqu = z, z, z
    end
    return v - n, ((dlm - dql * ql - dqu * qu) * c, BhazGrad(dh, dH, -dql * em, -dqu * em))
end
@inline _ll_grad(f::Cox, y, x, eta) = _ll_grad_pre(f, y, x, eta[1], eta[2])
@inline _ll(f::Cox, y, x, eta) = first(_ll_grad_pre(f, y, x, eta[1], eta[2]))
_mean(f::Cox, eta, t) = linkinv(link(f), eta[1])     # brms's epred for cox: the relative hazard mu
_rand(rng, ::Cox, eta, t) = NaN

"""
    cox(; link = :log)

Cox proportional hazards model for positive event times, as brms's `cox()`:
the baseline hazard is a cubic M-spline in time with simplex weights
`sbhaz`. Mark right-censored times with `time | cens(c) ~ x`, where `c` is 0
for an event and 1 for a right-censored time.
"""
cox(; link=nothing) = _make_family(Cox, link)
