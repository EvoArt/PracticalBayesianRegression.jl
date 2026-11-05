# =============================================================================
# hypothesis() and conditional_effects(), as in brms.
# =============================================================================

"""
    HypothesisResult

One tested hypothesis: the posterior of `lhs - rhs` summarised (estimate,
error, central interval), and for one-sided hypotheses the evidence ratio
P(H) / P(not H) and the posterior probability P(H).
"""
struct HypothesisResult
    hypothesis::String
    estimate::Float64
    est_error::Float64
    lower::Float64
    upper::Float64
    evid_ratio::Float64
    post_prob::Float64
    draws::Vector{Float64}
end

function Base.show(io::IO, ::MIME"text/plain", hs::Vector{HypothesisResult})
    println(io, "Hypothesis Tests:")
    data = Any[[h.hypothesis, h.estimate, h.est_error, h.lower, h.upper, h.evid_ratio, h.post_prob] for h in hs]
    M = permutedims(reduce(hcat, data))
    pretty_table(io, M; column_labels=["Hypothesis", "Estimate", "Est.Error", "CI.Lower", "CI.Upper", "Evid.Ratio", "Post.Prob"],
                 formatters=[fmt__printf("%.2f", [2, 3, 4, 5, 6, 7])], table_format=_TABLE_FORMAT,
                 alignment=[:l, :r, :r, :r, :r, :r, :r])
    print(io, "Note: 90% CI for one-sided and 95% for two-sided hypotheses.")
end
Base.show(io::IO, m::MIME"text/plain", h::HypothesisResult) = show(io, m, [h])

# split `a > b`, `a < b`, `a = b` (brms's `=`), returning (lhs, op, rhs)
function _split_hypothesis(s::AbstractString)
    for op in (">=", "<=", ">", "<", "=")
        k = findfirst(op, s)
        k === nothing && continue
        return strip(s[1:first(k)-1]), op, strip(s[last(k)+1:end])
    end
    throw(ArgumentError("a hypothesis needs one of >, <, = (got `$s`)"))
end

# evaluate an expression over draws; bare names resolve to variables(fit),
# trying `b_name` for population-level effects as brms does
function _hyp_eval(ex, fit::BrmsFit, vars; prior::Bool=false)
    ex isa Number && return fill(Float64(ex), ndraws(fit))
    if ex isa Symbol
        s = string(ex)
        v = s in vars ? s : "b_" * s in vars ? "b_" * s :
            throw(ArgumentError("`$s` is not a parameter of this model (see `variables(fit)`)"))
        prior || return vec(draws(fit, v))
        # its prior draws: prior_b_x, else the class's prior_b (brms)
        toks = split(replace(v, "__" => "_"), "_")
        for n in length(toks):-1:1
            c = "prior_" * join(toks[1:n], "_")
            FlexiChains.Parameter(c) in keys(fit.chain) && return vec(Matrix{Float64}(fit.chain[FlexiChains.Parameter(c)]))
        end
        throw(ArgumentError("no prior draws for `$v`: fit with sample_prior = :yes and a proper prior"))
    end
    if ex isa Expr && ex.head == :call
        f = ex.args[1]
        args = map(a -> _hyp_eval(a, fit, vars; prior), ex.args[2:end])
        fn = f === :+ ? (+) : f === :- ? (-) : f === :* ? (*) : f === :/ ? (/) : f === :^ ? (^) :
             getfield(Base, f)
        return broadcast(fn, args...)
    end
    if ex isa Expr && ex.head == :ref       # r_g[a,Intercept] style names
        return _hyp_eval(Symbol(replace(string(ex), " " => "")), fit, vars; prior)
    end
    throw(ArgumentError("unsupported expression `$ex` in a hypothesis"))
end

"""
    hypothesis(fit, h; alpha=0.05)

Test hypotheses about the parameters, as brms's `hypothesis`: `h` is a string
such as `"x > 0"`, `"x - z = 0"`, `"2 * x = Intercept"` (a vector of strings
tests several). Population-level effects may be named without their `b_`
prefix. For one-sided hypotheses the evidence ratio is P(H)/P(not H) under the
posterior and the interval is the one-sided 90% (alpha = 0.05 each side) as
in brms.
"""
function hypothesis(fit::BrmsFit, h::AbstractString; alpha=0.05)
    vars = Set(variables(fit))
    lhs, op, rhs = _split_hypothesis(h)
    d = _hyp_eval(Meta.parse(lhs), fit, vars) .- _hyp_eval(Meta.parse(rhs), fit, vars)
    if op == "="
        lo, hi = quantile(d, alpha / 2), quantile(d, 1 - alpha / 2)
        # Savage-Dickey: posterior / prior density of lhs - rhs at 0, when
        # the fit has prior draws (brms uses logspline densities; here a
        # Gaussian kernel density, so values differ slightly)
        er = NaN
        if any(k -> k isa FlexiChains.Parameter && startswith(string(k.name), "prior_"), keys(fit.chain))
            dp = _hyp_eval(Meta.parse(lhs), fit, vars; prior=true) .- _hyp_eval(Meta.parse(rhs), fit, vars; prior=true)
            er = _kde_at(d, 0.0) / _kde_at(dp, 0.0)
        end
        return HypothesisResult(h, mean(d), std(d), lo, hi, er, NaN, d)
    end
    pos = op in (">", ">=")
    p = mean(pos ? d .> 0 : d .< 0)
    lo, hi = quantile(d, 2alpha), quantile(d, 1 - 2alpha)
    lo, hi = pos ? (lo, Inf) : (-Inf, hi)
    return HypothesisResult(h, mean(d), std(d), lo, hi, p / (1 - p), p, d)
end
hypothesis(fit::BrmsFit, hs::AbstractVector; kw...) = [hypothesis(fit, h; kw...) for h in hs]

# Gaussian kernel density at x, Silverman's bandwidth
function _kde_at(v, x)
    n = length(v)
    s = min(std(v), (quantile(v, 0.75) - quantile(v, 0.25)) / 1.34)
    h = 0.9 * (s > 0 ? s : std(v)) * n^(-1 / 5)
    return sum(exp(-((x - vi) / h)^2 / 2) for vi in v) / (n * h * sqrt(2π))
end

"""
    conditional_effects(fit, effect; resolution=100, prob=0.95, method=:epred,
                        conditions=nothing, re_formula=:none)

The data behind brms's `conditional_effects` plots: the response (or linear
predictor, `method=:linpred`) as a function of `effect` with every other
predictor held at its mean (numeric) or reference level (factor), group-level
effects excluded by default (brms's `re_formula = NA`). `effect` is a column
name (`:x`), or a pair `:x => :g` for an interaction (one curve per level of
`g`, or at mean ± sd for a numeric `g`).

Returns a NamedTuple of vectors: the grid columns, `estimate`, `lower`,
`upper`. The plotting extensions draw it.
"""
function conditional_effects(fit::BrmsFit, effect; resolution::Int=100, prob=0.95,
                             method::Symbol=:epred, conditions=nothing, re_formula=:none)
    info = fit.info
    x, by = effect isa Pair ? (Symbol(first(effect)), Symbol(last(effect))) : (Symbol(effect), nothing)
    base = _reference_row(fit)
    conditions === nothing || (base = merge(base, NamedTuple(conditions)))
    haskey(base, x) || throw(ArgumentError("`$x` is not a predictor of this model"))
    xs = _effect_grid(fit, x, resolution)
    bys = by === nothing ? [nothing] : _effect_grid(fit, by, 3)
    rows = [(xv, bv) for bv in bys for xv in xs]
    n = length(rows)
    nd = Dict{Symbol,Any}(k => fill(v, n) for (k, v) in pairs(base))
    nd[x] = [r[1] for r in rows]
    by === nothing || (nd[by] = [r[2] for r in rows])
    newdata = NamedTuple(nd)
    P = method === :linpred ? posterior_linpred(fit; newdata=newdata, re_formula=re_formula) :
        posterior_epred(fit; newdata=newdata, re_formula=re_formula)
    a = (1 - prob) / 2
    est = vec(mean(P; dims=1))
    lo = [quantile(view(P, :, i), a) for i in 1:n]
    hi = [quantile(view(P, :, i), 1 - a) for i in 1:n]
    cols = (; Symbol(x) => nd[x])
    by === nothing || (cols = merge(cols, (; Symbol(by) => nd[by])))
    return merge(cols, (; estimate=est, lower=lo, upper=hi))
end

# every predictor column of the model at its mean / reference level
function _reference_row(fit::BrmsFit)
    info = fit.info
    f = info.formula
    out = Dict{Symbol,Any}()
    for (a, inf) in info.infos
        a.ex isa Symbol || continue
        a == f.response && continue
        out[a.ex] = inf.levels === nothing ? _training_mean(fit, a) : inf.levels[1]
    end
    # group columns: any existing level (their effects are excluded by default)
    for di in info.dinfo, (gt, gi, _) in di.groups, ga in gt.group
        ga.ex isa Symbol || continue
        haskey(out, ga.ex) || (out[ga.ex] = gi.levels[1])
    end
    if uses_trials(info.family) && f.trials isa Atom && f.trials.ex isa Symbol
        out[f.trials.ex] = round(Int, mean(fit.density.trials))
    end
    return NamedTuple(out)
end

# training mean of a numeric atom, recomputed from the training rows
function _training_mean(fit::BrmsFit, a::Atom)
    return mean(Float64.(_atom_values(fit, a)))
end
_atom_values(fit::BrmsFit, a::Atom) = fit.info.atom_values[a]

function _effect_grid(fit::BrmsFit, s::Symbol, resolution)
    a = Atom(s)
    inf = get(fit.info.infos, a, nothing)
    inf === nothing && throw(ArgumentError("`$s` is not a predictor of this model"))
    inf.levels === nothing || return inf.levels
    v = Float64.(_atom_values(fit, a))
    resolution == 3 && return [mean(v) - std(v), mean(v), mean(v) + std(v)]
    return collect(range(minimum(v), maximum(v); length=resolution))
end
