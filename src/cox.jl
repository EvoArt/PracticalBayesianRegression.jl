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
dpars(::Cox) = (:mu, :sbhaz)
dpar_link_name(::Cox, d::Symbol) = d === :mu ? "log" : "simplex"
_xdata(f::Cox, y, trials) = isempty(f.codes) ? zeros(length(y)) : f.codes
_ll_const(::Cox, y, t) = 0.0
_check_response(::Cox, y, t) = _chk(all(>(0), y), "cox responses (times) must be positive")

# --- M- and I-spline bases (splines2 conventions) -----------------------------------

struct BhazBasis
    knots::Vector{Float64}       # full knot sequence for the cubic M-splines
    lo::Float64
    hi::Float64
end
function BhazBasis(y; df=5)
    lo0, hi0 = extrema(y); dy = hi0 - lo0
    lo, hi = max(lo0 - dy / 50, 0.0), hi0 + dy / 50
    nik = df - 4                                 # cubic, with intercept
    nik >= 0 || throw(ArgumentError("cox(): df must be at least 4"))
    probs = range(0, 1; length=nik + 2)[2:(end - 1)]
    inner = isempty(probs) ? Float64[] : quantile(y, collect(probs))
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
end
_K(p::BhazSimplex) = length(p.idx) + 1

struct BhazVals{V}
    h::V
    H::V
end
@inline _at(b::BhazVals, i) = (b.h[i], b.H[i])
struct BhazGrad{T}
    dh::T
    dH::T
end
@inline Base.:*(w::Real, g::BhazGrad) = BhazGrad(oftype(g.dh, w * g.dh), oftype(g.dH, w * g.dH))

# workspace: eta = [h; H] (2N), w = [dh; dH] (2N), bvals = simplex, base = rem, wbase = z
function PredWS{T}(p::BhazSimplex, N) where {T}
    K = _K(p)
    return PredWS{T}(zeros(T, 2N), zeros(T, 2N), GroupWS{T}[], zeros(T, K), MoWS{T}[], zeros(T, K),
                     zeros(T, K - 1), Vector{T}[])
end
function _eta!(pw::PredWS, p::BhazSimplex, θ)
    _stick!(pw.bvals, pw.base, pw.wbase, view(θ, p.idx))
    N = size(p.Zb, 1)
    h = view(pw.eta, 1:N); H = view(pw.eta, (N + 1):(2N))
    mul!(h, p.Zb, pw.bvals); mul!(H, p.Zc, pw.bvals)
    return BhazVals(h, H)
end
_reset!(pw::PredWS, ::BhazSimplex) = nothing
@inline function _put!(w, p::BhazSimplex, i, g::BhazGrad)
    N = size(p.Zb, 1)
    @inbounds w[i] = g.dh
    @inbounds w[N + i] = g.dH
    return nothing
end
function _pullback!(g, p::BhazSimplex, pw::PredWS{T}, θ) where {T}
    N = size(p.Zb, 1); K = _K(p)
    a = p.Zb' * view(pw.w, 1:N) .+ p.Zc' * view(pw.w, (N + 1):(2N))     # d ll / d sbhaz
    gu = zeros(T, K)
    val = _stick_grad!(gu, a, pw.bvals, pw.base, pw.wbase, p.alpha)
    @inbounds for (j, k) in enumerate(p.idx)
        g[k] += gu[j]
    end
    return val
end
function _eta_generic(p::BhazSimplex, θ, N)
    s, _ = _stick(θ[p.idx], _K(p))
    return BhazVals(p.Zb * s, p.Zc * s)
end
function _prior_generic(p::BhazSimplex, θ)
    s, lj = _stick(θ[p.idx], _K(p))
    return lj + (p.alpha == 1 ? zero(lj) : (p.alpha - 1) * sum(log, s))
end

# --- the family -----------------------------------------------------------------------

@inline _pre(::Cox, eta) = eta[2]
@inline function _ll_grad_pre(::Cox, y, code, m, hb)
    h, H = hb
    em = exp(m)
    if code == 0                       # event
        v = log(h) + m - H * em
        return v, (1 - H * em, BhazGrad(1 / h, -em))
    elseif code == 1                   # right-censored
        return -H * em, (-H * em, BhazGrad(zero(h), -em))
    end
    # left-censored: log(1 - exp(-H e^m))
    q = H * em
    v = log1mexp(-q)
    r = exp(-q - v)                    # exp(-q) / (1 - exp(-q))
    return v, (r * q, BhazGrad(zero(h), r * em))
end
@inline _ll_grad(f::Cox, y, x, eta) = _ll_grad_pre(f, y, x, eta[1], eta[2])
@inline _ll(f::Cox, y, x, eta) = first(_ll_grad_pre(f, y, x, eta[1], eta[2]))
_mean(::Cox, eta, t) = exp(eta[1])     # brms's epred for cox: the relative hazard exp(eta)
_rand(rng, ::Cox, eta, t) = NaN

"""
    cox(; link = :log)

Cox proportional hazards model for positive event times, as brms's `cox()`:
the baseline hazard is a cubic M-spline in time with simplex weights
`sbhaz`. Mark right-censored times with `time | cens(c) ~ x`, where `c` is 0
for an event and 1 for a right-censored time.
"""
cox(; link=nothing) = _make_family(Cox, link)
