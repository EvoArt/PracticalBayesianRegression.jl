# =============================================================================
# Finite mixtures: brms's `mixture(gaussian, gaussian, ...)`.
#
#     ll = logsumexp_k( log theta_k + ll_k(y | mu_k, aux_k) )
#
# read off brms's Stan code. Each component k gets its own distributional
# parameters, named with the component number (mu1, sigma1, mu2, ...); the
# main formula is every mu_k's unless mu_k has its own. Mixing proportions:
#   * no theta formulas -> `simplex[K] theta` with a dirichlet(1) prior
#     (Stan's stick-breaking transform, the `ThetaSimplex` block below);
#   * theta2 ~ ..., ..., thetaK ~ ... -> logit-scale predictors with theta1 the
#     reference (log theta_k = eta_k - logsumexp(0, eta_2, ..., eta_K)),
#     intercept prior logistic(0, 1).
# For mixtures of one family brms identifies the components by ordering the
# mu intercepts (`order = "mu"`): Intercept_mu1 < Intercept_mu2 < ...; here
# the density's `ordered` intercept transform (density.jl).
#
# Gradient: with responsibilities r_k = softmax_k(log theta_k + ll_k), every
# component's own derivatives are scaled by r_k, and d ll / d log theta_k = r_k.
# Component normalising constants stay INSIDE the logsumexp (they do not
# factor out of a mixture), so `_ll` here is fully normalised.
# =============================================================================

struct Mixture{C<:Tuple} <: Family
    comps::C
    theta_formulas::Bool       # set at build from the formula
    order::Bool                # order the mu intercepts (brms order = "mu")
end

"""
    mixture(fams...; order=nothing)

A finite mixture of families, as brms's `mixture()`: `mixture(gaussian(),
gaussian())`, or with family names, `mixture(:gaussian, :student)`. `order`
defaults to ordering the mean intercepts when every component is the same
family (brms's `order = "mu"`), which identifies the components.
"""
function mixture(fams...; order=nothing)
    length(fams) >= 2 || throw(ArgumentError("a mixture needs at least two components"))
    comps = map(f -> f isa Family ? f : brmsfamily(f), fams)
    for c in comps
        c isa Union{Mixture,OrdinalFamily,Categorical,Censored,Truncated} &&
            throw(ArgumentError("$(family_name(c)) cannot be a mixture component"))
    end
    o = order === nothing ? all(c -> typeof(c) == typeof(comps[1]), comps) : (order in (true, :mu, "mu"))
    return Mixture(Tuple(comps), false, o)
end
_with_theta_formulas(m::Mixture, tf::Bool) = Mixture(m.comps, tf, m.order)

_ncomp(m::Mixture) = length(m.comps)
family_name(m::Mixture) = "mixture(" * join((family_name(c) for c in m.comps), ", ") * ")"
link(m::Mixture) = link(m.comps[1])
prior_link(m::Mixture) = prior_link(m.comps[1])
uses_trials(m::Mixture) = any(uses_trials, m.comps)

function dpars(m::Mixture)
    out = Symbol[]
    for (k, c) in enumerate(m.comps), d in dpars(c)
        push!(out, Symbol(d, k))
    end
    if m.theta_formulas
        append!(out, [Symbol(:theta, k) for k in 2:_ncomp(m)])
    else
        push!(out, :theta)
    end
    return Tuple(out)
end

# a component dpar name (:sigma2) -> (component index, the component's own name)
function _mix_split(d::Symbol)
    s = string(d)
    m = match(r"^(.*?)(\d+)$", s)
    m === nothing && throw(ArgumentError("`$d` is not a mixture component parameter"))
    return parse(Int, m.captures[2]), Symbol(m.captures[1])
end
_is_mix_mu(d::Symbol) = occursin(r"^mu\d+$", string(d))
_is_mix_theta(d::Symbol) = occursin(r"^theta\d*$", string(d))
aux_bounds(m::Mixture, d::Symbol) = (k = _mix_split(d); aux_bounds(m.comps[k[1]], k[2]))
aux_ub(m::Mixture, d::Symbol) = (k = _mix_split(d); aux_ub(m.comps[k[1]], k[2]))
function dpar_link_name(m::Mixture, d::Symbol)
    d === :theta && return "simplex"
    _is_mix_theta(d) && return "logit"
    k, b = _mix_split(d)
    return dpar_link_name(m.comps[k], b)
end
_check_response(m::Mixture, y, t) = foreach(c -> _check_response(c, y, t), m.comps)

# per-observation family data: a tuple of every component's
function _xdata(m::Mixture, y, trials)
    xs = map(c -> _xdata(c, y, trials), m.comps)
    any(!isempty, xs) || return Float64[]
    return [map(x -> isempty(x) ? 0.0 : x[i], xs) for i in eachindex(y)]
end
@inline _mix_x(x::Tuple, k) = x[k]
@inline _mix_x(x::Real, k) = x
_ll_const(::Mixture, y, t) = 0.0
_ll_const_obs(::Mixture, y, t, x) = 0.0

# the simplex theta, handed to the family as its log
struct LogTheta{V<:AbstractVector}
    lt::V
end
@inline _at(t::LogTheta, i) = t

struct ThetaGrad{K,T}
    r::NTuple{K,T}
end
@inline Base.:*(w::Real, g::ThetaGrad) = ThetaGrad(map(v -> oftype(v, w * v), g.r))

# split a per-observation eta tuple into the components' tuples
@inline _take(t::Tuple, ::Val{0}) = ((), t)
@inline function _take(t::Tuple, ::Val{n}) where {n}
    h, r = _take(Base.tail(t), Val(n - 1))
    return ((first(t), h...), r)
end
@inline _mix_split_eta(::Tuple{}, eta) = ((), eta)
@inline function _mix_split_eta(comps::Tuple, eta)
    ek, rest = _take(eta, Val(length(dpars(first(comps)))))
    others, theta = _mix_split_eta(Base.tail(comps), rest)
    return ((ek, others...), theta)
end

# log theta_k from the theta part of the eta tuple
@inline _mix_logtheta(m::Mixture, th::Tuple{LogTheta}) = ntuple(k -> th[1].lt[k], Val(length(m.comps)))
@inline function _mix_logtheta(m::Mixture, th::Tuple)
    z = zero(first(th))
    lse = logsumexp((z, th...))
    return (z - lse, map(e -> e - lse, th)...)
end

@inline function _ll(m::Mixture, y, x, eta)
    parts, th = _mix_split_eta(m.comps, eta)
    lt = _mix_logtheta(m, th)
    ps = ntuple(Val(length(m.comps))) do k
        c = m.comps[k]; xk = _mix_x(x, k)
        lt[k] + _ll(c, y, xk, parts[k]) + _ll_const_obs(c, y, uses_trials(c) ? round(Int, xk) : 0, xk)
    end
    return logsumexp(ps)
end

@inline function _ll_grad(m::Mixture, y, x, eta)
    parts, th = _mix_split_eta(m.comps, eta)
    lt = _mix_logtheta(m, th)
    vg = ntuple(Val(length(m.comps))) do k
        c = m.comps[k]; xk = _mix_x(x, k)
        v, g = _ll_grad(c, y, xk, parts[k])
        (v + oftype(v, _ll_const_obs(c, y, uses_trials(c) ? round(Int, xk) : 0, xk)), g)
    end
    ps = ntuple(k -> lt[k] + vg[k][1], Val(length(m.comps)))
    L = logsumexp(ps)
    r = map(p -> exp(p - L), ps)
    gcomp = _mix_flatten(ntuple(k -> map(gi -> r[k] * gi, vg[k][2]), Val(length(m.comps))))
    gtheta = if th isa Tuple{LogTheta}
        (ThetaGrad(r),)
    else
        # d/d eta_theta_k (k >= 2) = r_k - softmax_k
        ntuple(j -> r[j + 1] - exp(lt[j + 1]), Val(length(m.comps) - 1))
    end
    return L, (gcomp..., gtheta...)
end
@inline _mix_flatten(::Tuple{}) = ()
@inline _mix_flatten(t::Tuple) = (first(t)..., _mix_flatten(Base.tail(t))...)

function _mean(m::Mixture, eta, t)
    parts, th = _mix_split_eta(m.comps, eta)
    lt = _mix_logtheta(m, th)
    return sum(exp(lt[k]) * _mean(m.comps[k], parts[k], t) for k in 1:_ncomp(m))
end
function _rand(rng, m::Mixture, eta, t)
    parts, th = _mix_split_eta(m.comps, eta)
    lt = _mix_logtheta(m, th)
    u = rand(rng); c = 0.0
    for k in 1:_ncomp(m)
        c += exp(lt[k])
        (u <= c || k == _ncomp(m)) && return _rand(rng, m.comps[k], parts[k], t)
    end
end

# --- the simplex block ------------------------------------------------------------
#
# K - 1 unconstrained parameters -> theta (Stan's stick-breaking simplex, shared
# with mo()) with a symmetric dirichlet(alpha) prior. Its PredWS reuses fields:
# eta = log theta (then the adjoint d ll / d theta in the pullback), w = d ll /
# d log theta, bvals = theta, base = stick remainders, wbase = break fractions.

struct ThetaSimplex
    dpar::Symbol
    idx::UnitRange{Int}
    alpha::Float64
end
_K(p::ThetaSimplex) = length(p.idx) + 1

PredWS{T}(p::ThetaSimplex, N) where {T} =
    PredWS{T}(zeros(T, _K(p)), zeros(T, _K(p)), GroupWS{T}[], zeros(T, _K(p)), MoWS{T}[],
              zeros(T, _K(p)), zeros(T, _K(p) - 1), Vector{T}[])

function _eta!(pw::PredWS, p::ThetaSimplex, θ)
    _stick!(pw.bvals, pw.base, pw.wbase, view(θ, p.idx))
    @inbounds for k in eachindex(pw.eta)
        pw.eta[k] = log(pw.bvals[k])
    end
    return LogTheta(pw.eta)
end
_reset!(pw::PredWS, ::ThetaSimplex) = fill!(pw.w, zero(eltype(pw.w)))
@inline function _put!(w, ::ThetaSimplex, i, g::ThetaGrad)
    @inbounds for k in eachindex(g.r)
        w[k] += g.r[k]
    end
    return nothing
end
function _pullback!(g, p::ThetaSimplex, pw::PredWS{T}, θ) where {T}
    a = pw.eta                          # log theta no longer needed: reuse as d ll / d theta
    @inbounds for k in eachindex(a)
        a[k] = pw.w[k] / pw.bvals[k]
    end
    val = _stick_grad!(pw.w, a, pw.bvals, pw.base, pw.wbase, p.alpha)
    @inbounds for (j, k) in enumerate(p.idx)
        g[k] += pw.w[j]
    end
    return val
end
function _eta_generic(p::ThetaSimplex, θ, N)
    simo, _ = _stick(θ[p.idx], _K(p))
    return LogTheta(log.(simo))
end
function _prior_generic(p::ThetaSimplex, θ)
    simo, lj = _stick(θ[p.idx], _K(p))
    return lj + (p.alpha == 1 ? zero(lj) : (p.alpha - 1) * sum(log, simo))
end
