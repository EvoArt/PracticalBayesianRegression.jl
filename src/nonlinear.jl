# =============================================================================
# Non-linear models: brms `bf(y ~ a * exp(-b * x), a ~ 1 + (1 | g), b ~ 1, nl = TRUE)`.
#
# The mean's linear predictor is replaced by a user function of "non-linear
# parameters" (a, b, ...), each of which has its own linear predictor, and of
# data columns:
#
#     eta_mu[i] = f(a[i], b[i], ..., x[i], ...)
#
# There is no closed form for an arbitrary f, but f is a SCALAR function of a
# handful of values per observation, so its partials come from one small
# dual-number evaluation per observation (no tape, no allocation): with
# w[i] = d loglik / d eta_mu[i] from the family,
#
#     d loglik / d a[i] = w[i] * df/da (i)
#
# and each parameter's own pullback (density.jl) does the rest — group
# effects, priors, everything, unchanged.
#
# As in brms, non-linear parameters are NOT centred (their intercept is an
# ordinary `Intercept` coefficient, prior class b) and their coefficients
# need proper priors.
# =============================================================================

"""
    NLPred

The mean parameter of a non-linear model: the compiled function `f`, the
non-linear parameters' linear predictors (in `f`'s argument order), and the
data columns `f` reads (after them).
"""
struct NLPred{F,P<:Tuple,D<:Tuple}
    dpar::Symbol
    f::F
    pars::P
    names::Vector{Symbol}
    data::D
end

struct NLWS{T,W<:Tuple}
    eta::Vector{T}
    w::Vector{T}
    inner::W               # workspaces of the non-linear parameters
    D::Matrix{T}           # D[i, k] = df/d(parameter k) at observation i
end
function PredWS{T}(p::NLPred, N) where {T}
    inner = map(q -> PredWS{T}(q, N), p.pars)
    return NLWS{T,typeof(inner)}(zeros(T, N), zeros(T, N), inner, zeros(T, N, length(p.pars)))
end

# value and partials of f with respect to its first n arguments
@inline function _nl_value_grad(f, args::NTuple{n,T}, xs) where {n,T}
    duals = ntuple(k -> ForwardDiff.Dual{Nothing}(args[k], ntuple(j -> T(j == k), Val(n))), Val(n))
    r = f(duals..., xs...)
    r isa ForwardDiff.Dual || return T(r), ntuple(_ -> zero(T), Val(n))
    return T(ForwardDiff.value(r)), ntuple(k -> T(ForwardDiff.partials(r)[k]), Val(n))
end

function _eta!(pw::NLWS, p::NLPred, θ)
    inner = map((w, q) -> _eta!(w, q, θ), pw.inner, p.pars)
    η, D = pw.eta, pw.D
    @inbounds for i in eachindex(η)
        v, gr = _nl_value_grad(p.f, map(e -> e[i], inner), map(d -> d[i], p.data))
        η[i] = v
        for k in eachindex(gr)
            D[i, k] = gr[k]
        end
    end
    return η
end

function _pullback!(g, p::NLPred, pw::NLWS, θ)
    lp = zero(eltype(pw.w))
    for k in eachindex(p.pars)
        w = pw.inner[k]
        w.w .= pw.w .* view(pw.D, :, k)
        lp += _pullback!(g, p.pars[k], w, θ)
    end
    return lp
end

_reset!(pw::NLWS, ::NLPred) = nothing
@inline _put!(w, ::NLPred, i, v) = (@inbounds w[i] = v; nothing)

function _eta_generic(p::NLPred, θ, N)
    inner = map(q -> _eta_generic(q, θ, N), p.pars)
    return [p.f(map(e -> e[i], inner)..., map(d -> d[i], p.data)...) for i in 1:N]
end
_prior_generic(p::NLPred, θ) = sum(q -> _prior_generic(q, θ), p.pars)

# every LinPred of a model, including the non-linear parameters' (for naming)
_linpreds(preds) = reduce(vcat, [p isa LinPred ? [p] : p isa NLPred ? collect(p.pars) : [] for p in preds]; init=Any[])
