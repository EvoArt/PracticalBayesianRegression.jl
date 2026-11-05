# =============================================================================
# Ordinal regression: brms `cumulative(link)`, and (at the end of the file)
# the sequential `sratio` / `cratio` and adjacent-category `acat` families.
#
#     P(y <= k) = F(tau_k - eta),   k = 1..K-1,   eta = Xc * b  (no intercept)
#     log P(y = k) = log(F(tau_k - eta) - F(tau_{k-1} - eta))
#
# with ordered thresholds tau (brms's `ordered[nthres] Intercept`, Stan's
# ordered transform: tau_1 = u_1, tau_k = tau_{k-1} + exp(u_k)). The population
# design is centred as everywhere else, and the reported thresholds are
# `b_Intercept[k] = Intercept[k] + means_X' * b` (note the plus: eta enters
# with a minus sign).
#
# The thresholds are one more block in the density's parameter tuple
# (`Thresholds`), handed to the family as `Thr(tau)`. Each observation's
# derivative touches only tau_y and tau_{y-1}, returned as a `ThrGrad` that
# `_put!` scatters into the block's accumulator; the pullback then chains
# through the ordered transform with a suffix sum.
# =============================================================================

"""
    Thresholds

The ordered cut-points of an ordinal model: `K - 1` unconstrained entries
`u`, with `tau_1 = u_1` and `tau_k = tau_{k-1} + exp(u_k)`.
"""
struct Thresholds{G}     # G: grouped (brms thres(gr = g))
    idx::UnitRange{Int}
    priors::Vector{ScalarPrior}
    ordered::Bool        # cumulative only; sratio/cratio/acat use a plain vector
    # grouped thresholds: one block per level of g in one merged vector (brms's
    # merged_Intercept); each block is ordered on its own
    grp::Vector{Int}     # observation -> block
    start::Vector{Int}   # first entry of each block
    len::Vector{Int}     # entries per block
    labels::Vector{String}
    gatom::Any           # the grouping variable (an Atom), for new data
end
Thresholds(idx, priors, ordered::Bool=true) =
    Thresholds{false}(idx, priors, ordered, Int[], [1], [length(idx)], String[], nothing)
Thresholds(idx, priors, ordered, grp, start, len, labels, gatom) =
    Thresholds{true}(idx, priors, ordered, grp, start, len, labels, gatom)
_blocks(p::Thresholds) = zip(p.start, p.len)
_with_grp(p::Thresholds{true}, grp) = Thresholds{true}(p.idx, p.priors, p.ordered, grp, p.start, p.len, p.labels, p.gatom)

struct Thr{V<:AbstractVector}
    tau::V
end
@inline _at(t::Thr, i) = t
# grouped: each observation sees its own block of the merged vector
struct ThrG{V<:AbstractVector}
    tau::V
    grp::Vector{Int}
    start::Vector{Int}
    len::Vector{Int}
end
@inline function _at(t::ThrG, i)
    g = @inbounds t.grp[i]
    s = @inbounds t.start[g]
    return Thr(view(t.tau, s:(s + t.len[g] - 1)))
end
_thr(p::Thresholds{false}, tau) = Thr(tau)
_thr(p::Thresholds{true}, tau) = ThrG(tau, p.grp, p.start, p.len)
# the accumulator window of observation i (the whole vector when ungrouped)
@inline _thr_w(w, ::Thresholds{false}, i) = w
@inline _thr_w(w, p::Thresholds{true}, i) = (g = @inbounds p.grp[i]; view(w, p.start[g]:(p.start[g] + p.len[g] - 1)))

struct ThrGrad{T}
    y::Int
    lo::T     # d / d tau_{y-1}
    hi::T     # d / d tau_y
end
@inline Base.:*(w::Real, g::ThrGrad) = ThrGrad(g.y, w * g.lo, w * g.hi)

PredWS{T}(p::Thresholds, N) where {T} = PredWS{T}(zeros(T, length(p.idx)), zeros(T, length(p.idx)), GroupWS{T}[])

function _ordered!(tau, u)
    tau[1] = u[1]
    @inbounds for k in 2:length(u)
        tau[k] = tau[k - 1] + exp(u[k])
    end
    return tau
end

function _eta!(pw::PredWS, p::Thresholds, θ)
    u = view(θ, p.idx)
    if p.ordered
        for (s, n) in _blocks(p)
            r = s:(s + n - 1)
            _ordered!(view(pw.eta, r), view(u, r))
        end
    else
        copyto!(pw.eta, u)
    end
    return _thr(p, pw.eta)
end
function _eta_generic(p::Thresholds, θ, N)
    u = θ[p.idx]
    p.ordered || return _thr(p, u)
    tau = similar(u)
    for (s, n) in _blocks(p)
        r = s:(s + n - 1)
        _ordered!(view(tau, r), view(u, r))
    end
    return _thr(p, tau)
end

@inline function _put!(w0, p::Thresholds, i, g::ThrGrad)
    w = _thr_w(w0, p, i)
    K1 = length(w)
    @inbounds begin
        g.y > 1 && (w[g.y - 1] += g.lo)
        g.y <= K1 && (w[g.y] += g.hi)
    end
    return nothing
end

function _pullback!(g, p::Thresholds, pw::PredWS{T}, θ) where {T}
    w, tau = pw.w, pw.eta
    lp = zero(T)
    @inbounds for (k, pr) in enumerate(p.priors)
        v, d = prior_kernel_grad(pr, tau[k])
        w[k] += d
        lp += v
    end
    if !p.ordered
        @inbounds for k in eachindex(w)
            g[p.idx[k]] += w[k]
        end
        return lp
    end
    # tau_k depends on u_1 and on u_j for 2 <= j <= k, the latter through exp
    # (within each block for grouped thresholds)
    for (b0, n) in _blocks(p)
        s = zero(T)
        @inbounds for k in (b0 + n - 1):-1:b0
            s += w[k]
            u = θ[p.idx[k]]
            if k == b0
                g[p.idx[k]] += s
            else
                g[p.idx[k]] += exp(u) * s + 1      # +1: log-Jacobian u_k
                lp += u
            end
        end
    end
    return lp
end

function _prior_generic(p::Thresholds, θ)
    tau = _eta_generic(p, θ, 0).tau
    lp = zero(eltype(θ))
    for (k, pr) in enumerate(p.priors)
        lp += prior_kernel(pr, tau[k])
    end
    if p.ordered
        for (s, n) in _blocks(p), k in (s + 1):(s + n - 1)
            lp += θ[p.idx[k]]
        end
    end
    return lp
end

_reset!(pw::PredWS, ::Thresholds) = fill!(pw.w, zero(eltype(pw.w)))

# --- the family ------------------------------------------------------------------

@inline _pre(::Cumulative, eta) = eta[2].tau

# log(F(a) - F(b)) for a > b, and the two partial derivatives, per link
@inline function _cum_mid(::LogitLink, a, b)
    # F(a) - F(b) = e^a (1 - e^(b-a)) / ((1 + e^a)(1 + e^b))
    v = a + log1mexp(b - a) - log1pexp(a) - log1pexp(b)
    la = -log1pexp(-a) - log1pexp(a)        # log f(a)
    lb = -log1pexp(-b) - log1pexp(b)
    return v, exp(la - v), -exp(lb - v)
end
@inline _cum_lo(::LogitLink, a) = (-log1pexp(-a), logistic(-a))          # log F(a), d/da
@inline _cum_hi(::LogitLink, b) = (-log1pexp(b), -logistic(b))           # log(1 - F(b)), d/db
@inline function _cum_mid(::ProbitLink, a, b)
    # both in the upper tail: use the symmetric form Phi(-b) - Phi(-a)
    v = if b > 0
        lb_, la_ = normlogcdf(-b), normlogcdf(-a)
        lb_ + log1mexp(la_ - lb_)
    else
        la_, lb_ = normlogcdf(a), normlogcdf(b)
        la_ + log1mexp(lb_ - la_)
    end
    return v, exp(normlogpdf(a) - v), -exp(normlogpdf(b) - v)
end
@inline _cum_lo(::ProbitLink, a) = (normlogcdf(a), mills(a))
@inline _cum_hi(::ProbitLink, b) = (normlogcdf(-b), -mills(-b))
# cloglog: F(a) = 1 - exp(-e^a), so log(1 - F) = -e^a exactly
@inline function _cum_lo(::CloglogLink, a)
    ea = exp(a)
    lF = log1mexp(-ea)
    return lF, exp(a - ea - lF)
end
@inline _cum_hi(::CloglogLink, b) = (ea = exp(b); (-ea, -ea))
# F(a) - F(b) = exp(-e^b) - exp(-e^a) = exp(-e^b) (1 - exp(e^b - e^a))
@inline function _cum_mid(::CloglogLink, a, b)
    ea, eb = exp(a), exp(b)
    v = -eb + log1mexp(eb - ea)
    return v, exp(a - ea - v), -exp(b - eb - v)
end
# any other link (brms's probit_approx, cauchit, softit): F = linkinv, f = mu_eta.
# Tail cancellation is not guarded as above, but these CDFs (bar probit_approx)
# have heavy tails, so 1 - F does not round to zero in the range NUTS visits.
@inline function _cum_lo(L::Link, a)
    F = linkinv(L, a)
    return _slog(F), mu_eta(L, a) / F
end
@inline function _cum_hi(L::Link, b)
    F1 = 1 - linkinv(L, b)
    return _slog(F1), -mu_eta(L, b) / F1
end
@inline function _cum_mid(L::Link, a, b)
    d = linkinv(L, a) - linkinv(L, b)
    return _slog(d), mu_eta(L, a) / d, -mu_eta(L, b) / d
end

@inline function _ll_grad_pre(fam::Cumulative, y, x, m, tau)
    L = link(fam)
    K1 = length(tau)
    k = Int(y)
    if k == 1
        v, da = _cum_lo(L, tau[1] - m)
        return v, (-da, ThrGrad(1, zero(da), da))
    elseif k == K1 + 1
        v, db = _cum_hi(L, tau[K1] - m)
        return v, (-db, ThrGrad(k, db, zero(db)))
    end
    v, da, db = _cum_mid(L, tau[k] - m, tau[k - 1] - m)
    return v, (-(da + db), ThrGrad(k, db, da))
end
@inline _ll(fam::Cumulative, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], eta[2].tau))
_ll_const(::Cumulative, y, t) = 0.0
_check_response(::Cumulative, y, t) =
    _chk(all(v -> isinteger(v) && v >= 1, y), "cumulative responses must be integer categories 1, 2, ...")

# category probabilities for one observation
function _cum_probs(fam::Cumulative, m, tau)
    F = x -> linkinv(link(fam), x)
    K1 = length(tau)
    p = Vector{Float64}(undef, K1 + 1)
    prev = 0.0
    for k in 1:K1
        c = F(tau[k] - m)
        p[k] = c - prev
        prev = c
    end
    p[K1 + 1] = 1 - prev
    return p
end
# expected category (posterior_epred for ordinal returns probabilities; see
# predict.jl). The mean category is what `_mean` reports for residuals.
_mean(fam::Cumulative, eta, t) = sum(k * p for (k, p) in enumerate(_cum_probs(fam, eta[1], eta[2].tau)))
function _rand(rng, fam::Cumulative, eta, t)
    p = _cum_probs(fam, eta[1], eta[2].tau)
    u = rand(rng); c = 0.0
    for k in eachindex(p)
        c += p[k]
        u <= c && return Float64(k)
    end
    return Float64(length(p))
end

# =============================================================================
# sratio, cratio, acat (brms's Stan functions, disc = 1). Thresholds are a plain
# vector here (brms: `vector[nthres] Intercept`), and as for cumulative the
# reported thresholds are b_Intercept[k] = Intercept[k] + means_X' b.
#
#   sratio: log P(y = k) = sum_{j<k} log(1 - F(tau_j - eta)) + [k <= K1] log F(tau_k - eta)
#   cratio: log P(y = k) = sum_{j<k} log F(eta - tau_j)      + [k <= K1] log(1 - F(eta - tau_k))
#   acat:   P(y = k) propto prod_{j<k} F(a_j) prod_{j>=k} (1 - F(a_j)),  a_j = eta - tau_j
#           i.e. log P(y = k) = L_k - logsumexp(L), L_k = sum_{j<k} logit-F(a_j)
#           (the sum over j >= k of log(1 - F) is common to every k and cancels;
#           for the logit link logit-F(a) = a, brms's cumulative_sum form)
#
# Unlike cumulative, an observation's threshold derivative touches tau_1..tau_y
# (all of them for acat), so the family returns a `SeqGrad` carrying what is
# needed to recompute them, and `_put!` loops over the thresholds. The
# recomputation costs one more pass of link evaluations but allocates nothing.
# =============================================================================

@inline _pre(::Union{Sratio,Cratio,Acat}, eta) = eta[2].tau

struct SeqGrad{F,V,T}
    fam::F
    y::Int
    m::T
    tau::V
    s::T          # observation weight (scales every threshold derivative)
    lse::T        # acat only: logsumexp(L) and L_{K1+1}
    cend::T
end
@inline Base.:*(w::Real, g::SeqGrad) = SeqGrad(g.fam, g.y, g.m, g.tau, oftype(g.s, w * g.s), g.lse, g.cend)

# term j of sratio / cratio for response category k: (log value, d / d tau_j)
@inline function _seq_term(::Sratio, L, j, k, m, tau)
    @inbounds return j < k ? _cum_hi(L, tau[j] - m) : _cum_lo(L, tau[j] - m)
end
@inline function _seq_term(::Cratio, L, j, k, m, tau)
    @inbounds (v, d) = j < k ? _cum_lo(L, m - tau[j]) : _cum_hi(L, m - tau[j])
    return v, -d
end

@inline function _ll_grad_pre(fam::Union{Sratio,Cratio}, y, x, m0, tau)
    m = m0 + zero(eltype(tau))      # one number type for SeqGrad (eta Float, tau Dual in `y ~ 1`)
    L = link(fam)
    k = Int(y)
    v = zero(m); gs = zero(m)
    for j in 1:min(k, length(tau))
        lv, d = _seq_term(fam, L, j, k, m, tau)
        v += lv; gs += d
    end
    # every term depends on tau_j - eta (or its negative): d/d eta = -sum d/d tau
    return v, (-gs, SeqGrad(fam, k, m, tau, one(m), zero(m), zero(m)))
end

@inline function _put!(w0, p::Thresholds, i, g::SeqGrad{<:Union{Sratio,Cratio}})
    w = _thr_w(w0, p, i)
    L = link(g.fam)
    @inbounds for j in 1:min(g.y, length(w))
        w[j] += g.s * last(_seq_term(g.fam, L, j, g.y, g.m, g.tau))
    end
    return nothing
end

# logit-F(a) and its derivative
@inline _acat_delta(::LogitLink, a) = (a, one(a))
@inline function _acat_delta(L::Link, a)
    lq, dq = _cum_lo(L, a)
    lr, dr = _cum_hi(L, a)
    return lq - lr, dq - dr
end

@inline function _ll_grad_pre(fam::Acat, y, x, m0, tau)
    m = m0 + zero(eltype(tau))
    L = link(fam)
    K1 = length(tau)
    k = Int(y)
    # forward: L_1 = 0, L_{j+1} = L_j + delta_j, with a running logsumexp
    c = zero(m); mx = zero(m); s = one(m); ly = zero(m)
    @inbounds for j in 1:K1
        c += first(_acat_delta(L, m - tau[j]))
        j + 1 == k && (ly = c)
        if c > mx
            s = s * exp(mx - c) + 1; mx = c
        else
            s += exp(c - mx)
        end
    end
    lse = mx + log(s)
    # backward: T_j = P(category > j); d ll / d a_j = ([j < y] - T_j) delta_j'
    T = zero(m); gm = zero(m); cend = c
    @inbounds for j in K1:-1:1
        T += exp(c - lse)
        δ, dδ = _acat_delta(L, m - tau[j])
        gm += ((j < k) - T) * dδ
        c -= δ
    end
    return ly - lse, (gm, SeqGrad(fam, k, m, tau, one(m), lse, cend))
end

@inline function _put!(w0, p::Thresholds, i, g::SeqGrad{<:Acat})
    w = _thr_w(w0, p, i)
    L = link(g.fam)
    c = g.cend; T = zero(g.m)
    @inbounds for j in length(w):-1:1
        T += exp(c - g.lse)
        δ, dδ = _acat_delta(L, g.m - g.tau[j])
        w[j] -= g.s * ((j < g.y) - T) * dδ      # a_j = eta - tau_j
        c -= δ
    end
    return nothing
end

@inline _ll(fam::Union{Sratio,Cratio,Acat}, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], eta[2].tau))
_ll_const(::Union{Sratio,Cratio,Acat}, y, t) = 0.0
_check_response(fam::Union{Sratio,Cratio,Acat}, y, t) =
    _chk(all(v -> isinteger(v) && v >= 1, y), "$(family_name(fam)) responses must be integer categories 1, 2, ...")

_cum_probs(fam::Union{Sratio,Cratio,Acat}, m, tau) =
    [exp(first(_ll_grad_pre(fam, k, 0.0, Float64(m), Float64.(tau)))) for k in 1:(length(tau) + 1)]
_mean(fam::Union{Sratio,Cratio,Acat}, eta, t) = sum(k * p for (k, p) in enumerate(_cum_probs(fam, eta[1], eta[2].tau)))
function _rand(rng, fam::Union{Sratio,Cratio,Acat}, eta, t)
    p = _cum_probs(fam, eta[1], eta[2].tau)
    u = rand(rng); c = 0.0
    for k in eachindex(p)
        c += p[k]
        u <= c && return Float64(k)
    end
    return Float64(length(p))
end

# =============================================================================
# Discrimination `disc` (brms, ordinal families): every link argument is
# multiplied by disc, e.g. cumulative P(y <= k) = F(disc * (tau_k - mu)), with
# log(disc) a linear predictor. brms fixes disc = 1 unless it has a formula,
# so the wrapper exists only for `disc ~ ...` models (no cost otherwise).
#
# The inner family is evaluated at the scaled arguments mu' = disc mu and
# tau' = disc tau (a lazily scaled view), so its value and gradients are
# reused unchanged; by the chain rule
#     d/d mu = disc d/d mu',  d/d tau = disc d/d tau',
#     d/d log disc = mu' d/d mu' + sum_j tau'_j d/d tau'_j
# (the threshold gradient objects scale by disc through their `*` methods).
# =============================================================================

struct Disc{F<:Family} <: Family end
@inline _inner(::Disc{F}) where {F} = F()
dpars(f::Disc) = (dpars(_inner(f))..., :disc)
link(f::Disc) = link(_inner(f))
family_name(f::Disc) = family_name(_inner(f))
_check_response(f::Disc, y, t) = _check_response(_inner(f), y, t)
_ll_const(::Disc, y, t) = 0.0
const _Ord = Union{OrdinalFamily,Disc{<:OrdinalFamily}}     # (not hurdle_cumulative's)
_ord_core(f) = f
_ord_core(f::Disc) = _inner(f)

struct _ScaledTau{T,V<:AbstractVector} <: AbstractVector{T}
    tau::V
    d::T
end
# (a named builder: an outer `_ScaledTau(tau, d)` would overwrite the default constructor)
_scaled_tau(tau::AbstractVector, d) =
    (T = promote_type(eltype(tau), typeof(d)); _ScaledTau{T,typeof(tau)}(tau, convert(T, d)))
Base.size(s::_ScaledTau) = size(s.tau)
Base.@propagate_inbounds Base.getindex(s::_ScaledTau, i::Int) = s.d * s.tau[i]

# sum_j tau_j d/d tau_j for each threshold-gradient type (mirrors its `_put!`)
@inline function _thr_dot(g::ThrGrad, tau)
    K1 = length(tau)
    v = zero(g.lo)
    @inbounds begin
        g.y > 1 && (v += g.lo * tau[g.y - 1])
        g.y <= K1 && (v += g.hi * tau[g.y])
    end
    return v
end
@inline function _thr_dot(g::SeqGrad{<:Union{Sratio,Cratio}}, tau)
    L = link(g.fam)
    v = zero(g.m)
    @inbounds for j in 1:min(g.y, length(tau))
        v += g.s * last(_seq_term(g.fam, L, j, g.y, g.m, g.tau)) * tau[j]
    end
    return v
end
@inline function _thr_dot(g::SeqGrad{<:Acat}, tau)
    L = link(g.fam)
    c = g.cend; T = zero(g.m); v = zero(g.m)
    @inbounds for j in length(tau):-1:1
        T += exp(c - g.lse)
        δ, dδ = _acat_delta(L, g.m - g.tau[j])
        v -= g.s * ((j < g.y) - T) * dδ * tau[j]
        c -= δ
    end
    return v
end

@inline _pre(f::Disc, eta) = (eta[2].tau, exp(eta[3]))
@inline function _ll_grad_pre(f::Disc, y, x, m, p)
    tau, d = p
    ts = _scaled_tau(tau, d)
    md = d * m
    v, g = _ll_grad_pre(_inner(f), y, x, md, ts)
    dm, tg = g
    return v, (d * dm, d * tg, dm * md + _thr_dot(tg, ts))
end
@inline _ll_grad(f::Disc, y, x, eta) = _ll_grad_pre(f, y, x, eta[1], _pre(f, eta))
@inline _ll(f::Disc, y, x, eta) = first(_ll_grad(f, y, x, eta))
# post-processing on the scaled arguments
_disc_eta(::Disc, eta) = (d = exp(eta[3]); (d * eta[1], Thr(d .* eta[2].tau)))
_cum_probs(f::Disc, m, tau, d) = _cum_probs(_inner(f), d * m, d .* tau)
_mean(f::Disc, eta, t) = _mean(_inner(f), _disc_eta(f, eta), t)
_rand(rng, f::Disc, eta, t) = _rand(rng, _inner(f), _disc_eta(f, eta), t)

# =============================================================================
# hurdle_cumulative (brms): y = 0 with probability hu, otherwise a cumulative
# model over categories 1..K (ordered thresholds, as cumulative).
#     ll(0) = log hu,   ll(k) = log(1 - hu) + cumulative ll(k)
# =============================================================================
dpars(::HurdleCumulative) = (:mu, :hu, :thres)
@inline _pre(::HurdleCumulative, eta) = eta[2:3]
@inline _cum_of(::HurdleCumulative{L}) where {L} = Cumulative{L}()
@inline function _ll_grad_pre(fam::HurdleCumulative, y, x, m, p)
    h, thr = p
    if y == 0
        return -log1pexp(-h), (zero(m), logistic(-h), ThrGrad(1, zero(m), zero(m)))
    end
    v, (dm, tg) = _ll_grad_pre(_cum_of(fam), y, x, m, thr.tau)
    return v - log1pexp(h), (dm, -logistic(h), tg)
end
@inline _ll_grad(fam::HurdleCumulative, y, x, eta) = _ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta))
@inline _ll(fam::HurdleCumulative, y, x, eta) = first(_ll_grad(fam, y, x, eta))
_ll_const(::HurdleCumulative, y, t) = 0.0
_check_response(::HurdleCumulative, y, t) =
    _chk(all(v -> isinteger(v) && v >= 0, y), "hurdle_cumulative responses must be 0 (the hurdle) or categories 1, 2, ...")
function _hc_probs(fam::HurdleCumulative, eta)
    hu = logistic(eta[2])
    return vcat(hu, (1 - hu) .* _cum_probs(_cum_of(fam), eta[1], eta[3].tau))
end
_mean(fam::HurdleCumulative, eta, t) = sum((k - 1) * p for (k, p) in enumerate(_hc_probs(fam, eta)))
function _rand(rng, fam::HurdleCumulative, eta, t)
    p = _hc_probs(fam, eta); u = rand(rng); c = 0.0
    for k in eachindex(p)
        c += p[k]
        u <= c && return Float64(k - 1)
    end
    return Float64(length(p) - 1)
end

# hurdle_cumulative with disc ~ ...: (mu, hu, thres, disc); the cumulative part
# goes through Disc{Cumulative}
@inline _pre(::Disc{<:HurdleCumulative}, eta) = (eta[2], eta[3].tau, exp(eta[4]))
@inline function _ll_grad_pre(f::Disc{<:HurdleCumulative}, y, x, m, p)
    h, tau, d = p
    if y == 0
        z = zero(m * d)
        return -log1pexp(-h), (z, logistic(-h), ThrGrad(1, z, z), z)
    end
    v, (dm, tg, dl) = _ll_grad_pre(Disc{typeof(_cum_of(_inner(f)))}(), y, x, m, (tau, d))
    return v - log1pexp(h), (dm, -logistic(h), tg, dl)
end
_disc_eta(::Disc{<:HurdleCumulative}, eta) = (d = exp(eta[4]); (d * eta[1], eta[2], Thr(d .* eta[3].tau)))

# =============================================================================
# Category-specific effects `cs(x)` (brms; sratio / cratio / acat only, whose
# thresholds are unordered): per-observation thresholds
#     tau_n = tau - Bcs' xcs_n          (Bcs: Kcs x K1, brms's `bcs`)
# The families' gradients are unchanged — they scatter d/d tau_n into a
# per-observation adjoint W (K1 x N), and the pullback forms
#     d/d tau = sum_n W[:, n],   d/d Bcs = -Xcs' W'.
# brms does not centre the design of a model with cs() terms; nor do we.
# =============================================================================

struct ThresholdsCS{T}
    base::Thresholds
    Xcs::Matrix{T}            # N x Kcs, uncentred
    Xcst::Matrix{T}
    bcs::UnitRange{Int}       # Kcs * K1 parameters, column-major Kcs x K1
    priors::Vector{ScalarPrior}   # one per cs column (brms: class b)
    labels::Vector{String}
    spec::DesignSpec          # to code new data
end
ThresholdsCS{T}(base, Xcs, bcs, priors, labels, spec) where {T} =
    ThresholdsCS{T}(base, Matrix{T}(Xcs), permutedims(Matrix{T}(Xcs)), bcs, priors, labels, spec)
_K1(p::ThresholdsCS) = length(p.base.idx)
_Kcs(p::ThresholdsCS) = size(p.Xcs, 2)

# per-observation thresholds, handed to the family one column at a time
struct ThrMat{M<:AbstractMatrix}
    TAU::M
end
@inline _at(t::ThrMat, i) = Thr(view(t.TAU, :, i))

PredWS{T}(p::ThresholdsCS, N) where {T} =
    PredWS{T}(zeros(T, _K1(p) * N), zeros(T, _K1(p) * N), GroupWS{T}[], zeros(T, _K1(p)), MoWS{T}[],
              T[], T[], Vector{T}[])

function _eta!(pw::PredWS, p::ThresholdsCS, θ)
    K1, N = _K1(p), size(p.Xcs, 1)
    TAU = reshape(pw.eta, K1, N)
    # TAU[k, n] = tau_k - sum_c bcs[c, k] Xcs[n, c]. An explicit loop, not mul! on
    # reshape(view(θ)): that hits LinearAlgebra's generic matmul, whose wrapper
    # Union juliac --trim cannot resolve (and Kc is small)
    Kc = _Kcs(p); b0 = first(p.bcs) - 1
    @inbounds for n in 1:N, k in 1:K1
        s = zero(eltype(TAU))
        for c in 1:Kc
            s += θ[b0 + (k - 1) * Kc + c] * p.Xcst[c, n]
        end
        TAU[k, n] = θ[p.base.idx[k]] - s
    end
    return ThrMat(TAU)
end
_reset!(pw::PredWS, ::ThresholdsCS) = fill!(pw.w, zero(eltype(pw.w)))
@inline function _put!(w, p::ThresholdsCS, i, g)
    K1 = _K1(p)
    _put!(view(w, ((i - 1) * K1 + 1):(i * K1)), p.base, i, g)
    return nothing
end
function _pullback!(g, p::ThresholdsCS, pw::PredWS{T}, θ) where {T}
    K1, N, Kc = _K1(p), size(p.Xcs, 1), _Kcs(p)
    W = reshape(pw.w, K1, N)
    lp = zero(T)
    @inbounds for k in 1:K1
        s = zero(T)
        for n in 1:N
            s += W[k, n]
        end
        u = θ[p.base.idx[k]]
        v, d = prior_kernel_grad(p.base.priors[k], u)
        g[p.base.idx[k]] += s + d
        lp += v
    end
    @inbounds for k in 1:K1, j in 1:Kc
        s = zero(T)
        for n in 1:N
            s += p.Xcst[j, n] * W[k, n]
        end
        idx = p.bcs[(k - 1) * Kc + j]
        v, d = prior_kernel_grad(p.priors[j], θ[idx])
        g[idx] += -s + d
        lp += v
    end
    return lp
end
function _eta_generic(p::ThresholdsCS, θ, N)
    B = reshape(θ[p.bcs], _Kcs(p), _K1(p))
    tau = θ[p.base.idx]
    return ThrMat(tau .- transpose(B) * transpose(p.Xcs))
end
function _prior_generic(p::ThresholdsCS, θ)
    lp = sum(prior_kernel(p.base.priors[k], θ[p.base.idx[k]]) for k in 1:_K1(p))
    Kc = _Kcs(p)
    for k in 1:_K1(p), j in 1:Kc
        lp += prior_kernel(p.priors[j], θ[p.bcs[(k - 1) * Kc + j]])
    end
    return lp
end
