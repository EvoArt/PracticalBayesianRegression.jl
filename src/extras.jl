# =============================================================================
# brms helpers that sit on top of the rest of post-processing:
#   get_prior / default_prior / validate_prior   the prior table before fitting
#   VarCorr                                       group-level sd / cor / cov
#   loo_predict / loo_epred / loo_linpred         PSIS-LOO weighted predictions
#   pp_mixture                                    mixture membership probabilities
#   prior_draws, nuts_params, log_posterior, rhat, neff_ratio, niterations
# =============================================================================

# --- the prior table -----------------------------------------------------------

"""
    PriorTable

The rows of a model's prior table ([`PriorSpec`](@ref)); prints in brms's
layout and indexes like a vector.
"""
struct PriorTable <: AbstractVector{PriorSpec}
    rows::Vector{PriorSpec}
end
Base.size(t::PriorTable) = size(t.rows)
Base.getindex(t::PriorTable, i::Int) = t.rows[i]
function Base.show(io::IO, ::MIME"text/plain", t::PriorTable)
    println(io, rpad("prior", 30), rpad("class", 11), rpad("coef", 16), rpad("group", 10), rpad("resp", 8),
            rpad("dpar", 8), "source")
    for r in t.rows
        print(io, rpad(sprint(show, r.prior), 30), rpad(string(r.class), 11), rpad(r.coef, 16),
              rpad(r.group, 10), rpad(r.resp, 8), rpad(r.dpar, 8), r.source)
        r.class === :b && (isfinite(r.lb) || isfinite(r.ub)) && print(io, "   [", r.lb, ", ", r.ub, "]")
        println(io)
    end
end

function _prior_rows(formula, data; family, prior, data2)
    f = formula isa MVFormula ? formula : bf(formula)
    if f isa MVFormula || length(_mv_split(f)) > 1
        _, infos = build_mv_model(f, data; family=family, prior=prior)
        rows = PriorSpec[]
        resp = [i.formula.response.label for i in infos]
        for (r, info) in zip(resp, infos), p in info.priors
            push!(rows, PriorSpec(p.prior, p.class, p.coef, p.group, p.dpar, p.source, p.lb, p.ub,
                                  p.resp == "" ? r : p.resp))
        end
        return rows
    end
    _, info = build_model(f, data; family=family, prior=prior, data2=data2)
    return copy(info.priors)
end

"""
    get_prior(formula, data; family=gaussian(), data2=nothing) -> PriorTable

The parameter classes of a model and the default prior of each, before
fitting — brms's `get_prior()` / `default_prior()`. Set priors on any row with
[`prior`](@ref) and pass them to [`brm`](@ref).
"""
get_prior(formula, data; family=gaussian(), data2=nothing) =
    PriorTable(_prior_rows(formula, data; family, prior=PriorSpec[], data2))

"""
    default_prior(formula, data; kwargs...)

The same as [`get_prior`](@ref) (brms 2.21+'s name).
"""
default_prior(formula, data; kw...) = get_prior(formula, data; kw...)

"""
    validate_prior(formula, data; prior, family=gaussian(), data2=nothing) -> PriorTable

The full prior table the model would be fitted with, as brms's
`validate_prior()`: your priors where they apply, defaults elsewhere. A
prior whose `coef` or `group` matches nothing in the model does not appear.
"""
validate_prior(formula, data; prior=PriorSpec[], family=gaussian(), data2=nothing) =
    PriorTable(_prior_rows(formula, data; family, prior, data2))

# --- VarCorr ----------------------------------------------------------------------

"""
    VarCorr(fit; prob=0.95) -> Dict{String, NamedTuple}

Group-level standard deviations, correlations and covariances, as brms's
`VarCorr()`: `VarCorr(fit)["g"].sd` (one row per coefficient), `.cor` and
`.cov` (one row per pair, from the draws of sd_a sd_b cor_ab). A model with a
scalar `sigma` also gets `"residual__"` (its sd).
"""
function VarCorr(fit::BrmsFit; prob=0.95)
    vars = variables(fit)
    out = Dict{String,NamedTuple{(:sd, :cor, :cov),NTuple{3,Vector{SummaryRow}}}}()
    labels = String[]
    for p in _linpreds(fit.density.preds), gb in p.groups
        gb.label in labels || push!(labels, gb.label)
    end
    for g in labels
        sdn = filter(n -> startswith(n, "sd_$(g)__"), vars)
        corn = filter(n -> startswith(n, "cor_$(g)__"), vars)
        sdrows = [_relabel(_summarise(n, draws(fit, n), prob), n[(length("sd_$(g)__") + 1):end]) for n in sdn]
        corrows = SummaryRow[]; covrows = SummaryRow[]
        for n in corn
            ab = n[(length("cor_$(g)__") + 1):end]
            push!(corrows, _relabel(_summarise(n, draws(fit, n), prob), ab))
            # cor_g__a__b: find the split that names two existing sd's
            sa = findfirst(k -> "sd_$(g)__" * ab[1:(k - 1)] in sdn && "sd_$(g)__" * ab[(k + 2):end] in sdn,
                           [i for i in 1:(length(ab) - 1) if ab[i:(i + 1)] == "__"])
            sa === nothing && continue
            k = [i for i in 1:(length(ab) - 1) if ab[i:(i + 1)] == "__"][sa]
            a, b = ab[1:(k - 1)], ab[(k + 2):end]
            c = draws(fit, n) .* draws(fit, "sd_$(g)__$a") .* draws(fit, "sd_$(g)__$b")
            push!(covrows, _relabel(_summarise("cov", c, prob), ab))
        end
        out[g] = (sd=sdrows, cor=corrows, cov=covrows)
    end
    "sigma" in vars && (out["residual__"] = (sd=[_relabel(_summarise("sigma", draws(fit, "sigma"), prob), "Intercept")],
                                             cor=SummaryRow[], cov=SummaryRow[]))
    return out
end

# --- PSIS-LOO weighted predictions --------------------------------------------------

# normalised PSIS weights (draws x observations), as loo() computes them
function _psis_weights(fit::BrmsFit)
    LL = log_lik(fit)
    S, N = size(LL)
    niter, nch = size(fit.draws, 2), size(fit.draws, 3)
    W = similar(LL)
    for i in 1:N
        ll = view(LL, :, i)
        reff = clamp(only(MCMCDiagnosticTools.ess(reshape(exp.(ll .- maximum(ll)), niter, nch, 1); kind=:basic)) / S,
                     1e-3, 1.0)
        W[:, i] .= vec(PSIS.psis(-collect(ll), reff; warn=false).weights)
    end
    return W
end

# the loo package's weighted quantile (E_loo, type = "quantile")
function _wquant(x, w, p)
    o = sortperm(x); xs = x[o]; ws = cumsum(w[o]); ws ./= ws[end]
    j = findfirst(>=(p), ws)
    j == 1 && return xs[1]
    return xs[j - 1] + (xs[j] - xs[j - 1]) * (p - ws[j - 1]) / (ws[j] - ws[j - 1])
end

function _e_loo(X::AbstractMatrix, W::AbstractMatrix; type::Symbol, probs)
    size(X) == size(W) || throw(DimensionMismatch("draws and PSIS weights differ in shape"))
    N = size(X, 2)
    if type === :mean
        return [sum(view(W, :, i) .* view(X, :, i)) for i in 1:N]
    elseif type === :var
        return map(1:N) do i
            w, x = view(W, :, i), view(X, :, i)
            m = sum(w .* x)
            sum(w .* (x .- m) .^ 2)
        end
    elseif type === :quantile
        return [_wquant(X[:, i], W[:, i], p) for i in 1:N, p in probs]
    end
    throw(ArgumentError("type must be :mean, :var or :quantile"))
end

"""
    loo_predict(fit; type = :mean, probs = [0.025, 0.975], rng = Random.default_rng())

Leave-one-out posterior predictive summaries by Pareto-smoothed importance
sampling, as brms's `loo_predict`: each observation's predictive
distribution, weighted as if that observation had been left out of the fit.

`type = :mean` or `:var` returns one value per observation; `type =
:quantile` returns an observations x `probs` matrix.

See also [`loo_epred`](@ref), [`loo_linpred`](@ref), [`loo`](@ref).
"""
function loo_predict(fit::BrmsFit; type::Symbol=:mean, probs=[0.025, 0.975], rng=Random.default_rng())
    return _e_loo(posterior_predict(fit; rng=rng), _psis_weights(fit); type, probs)
end

"""
    loo_epred(fit; type = :mean, probs = [0.025, 0.975])

Leave-one-out summaries of the expected response ([`posterior_epred`](@ref)),
weighted by PSIS as in [`loo_predict`](@ref). Not available for ordinal and
categorical models.
"""
function loo_epred(fit::BrmsFit; type::Symbol=:mean, probs=[0.025, 0.975])
    E = posterior_epred(fit)
    ndims(E) == 2 || throw(ArgumentError("loo_epred: not for ordinal / categorical models (category probabilities)"))
    return _e_loo(E, _psis_weights(fit); type, probs)
end

"""
    loo_linpred(fit; type = :mean, probs = [0.025, 0.975], dpar = :mu, transform = false)

Leave-one-out summaries of the linear predictor of `dpar`
([`posterior_linpred`](@ref)), weighted by PSIS as in [`loo_predict`](@ref).
`transform = true` applies the inverse link.
"""
function loo_linpred(fit::BrmsFit; type::Symbol=:mean, probs=[0.025, 0.975], dpar::Symbol=:mu, transform::Bool=false)
    return _e_loo(posterior_linpred(fit; dpar=dpar, transform=transform), _psis_weights(fit); type, probs)
end

# --- mixtures ---------------------------------------------------------------------------

"""
    pp_mixture(fit; summary=false)

Posterior probabilities that each observation belongs to each mixture
component (brms's `pp_mixture`): a draws x observations x components array,
or with `summary = true` the observations x components posterior means.
Training data only.
"""
function pp_mixture(fit::BrmsFit; summary::Bool=false)
    m = fit.info.family
    m isa Mixture || throw(ArgumentError("pp_mixture needs a mixture model"))
    ld = fit.density
    K = _ncomp(m)
    preds, N, _ = _prediction_preds(fit, nothing; re_formula=nothing, allow_new_levels=false)
    opts = _PredOpts(Random.default_rng(), :uncertainty)
    D = reshape(fit.draws, size(fit.draws, 1), :)
    out = zeros(size(D, 2), N, K)
    for s in axes(D, 2)
        θ = view(D, :, s)
        etas = map(p -> _eta_pred(p, θ, N, opts), preds)
        for i in 1:N
            eta = map(e -> _at(e, i), etas)
            parts, th = _mix_split_eta(m.comps, eta)
            lt = _mix_logtheta(m, th)
            x = _xd(ld, i); y = ld.y[i]
            ps = [lt[k] + _ll(m.comps[k], y, _mix_x(x, k), parts[k]) +
                  _ll_const_obs(m.comps[k], y, uses_trials(m.comps[k]) ? round(Int, _mix_x(x, k)) : 0, _mix_x(x, k))
                  for k in 1:K]
            L = logsumexp(ps)
            out[s, i, :] .= exp.(ps .- L)
        end
    end
    return summary ? dropdims(mean(out; dims=1); dims=1) : out
end

# --- small accessors ---------------------------------------------------------------------

"""
    prior_draws(fit) -> Dict{String, Matrix{Float64}}

The draws from the priors of a fit with `sample_prior = :yes` (brms's
`prior_draws`), by variable name (`prior_b`, `prior_sigma`, ...).
"""
function prior_draws(fit::Union{BrmsFit,BrmsMVFit})
    names = filter(n -> startswith(n, "prior_"),
                   [string(k.name) for k in keys(fit.chain) if k isa FlexiChains.Parameter])
    isempty(names) && throw(ArgumentError("no prior draws: fit with sample_prior = :yes"))
    return Dict(n => draws(fit, n) for n in names)
end

"""
    nuts_params(fit) -> Dict{String, Matrix{Float64}}

The sampler's per-draw diagnostics (iterations x chains), Stan's names:
`accept_stat__`, `stepsize__`, `treedepth__`, `n_leapfrog__`, `divergent__`,
`energy__` (where the sampler reports it).
"""
nuts_params(fit::Union{BrmsFit,BrmsMVFit}) =
    Dict(string(k.name) => Matrix{Float64}(fit.chain[k]) for k in keys(fit.chain)
         if k isa FlexiChains.Extra && string(k.name) != "lp__")

"""
    log_posterior(fit) -> Matrix{Float64}

The unnormalised log posterior density of each draw (`lp__`, Stan's
convention: constants dropped, Jacobians included), iterations x chains.
"""
log_posterior(fit::Union{BrmsFit,BrmsMVFit}) = Matrix{Float64}(fit.chain[FlexiChains.Extra("lp__")])

"""
    niterations(fit) -> Int

The number of post-warm-up draws per chain (after thinning).
"""
niterations(fit::Union{BrmsFit,BrmsMVFit}) = size(fit.draws, 2)

# a method of MCMCDiagnosticTools.rhat, so `using` both packages gives one `rhat`
"""
    rhat(fit) -> Dict{String, Float64}

Rank-normalised split R-hat of every variable, as brms's `rhat`. Values above
about 1.01 indicate that the chains have not mixed.
"""
MCMCDiagnosticTools.rhat(fit::Union{BrmsFit,BrmsMVFit}) = Dict(r.variable => r.rhat for r in posterior_summary(fit))

"""
    neff_ratio(fit) -> Dict{String, Float64}

Bulk effective sample size divided by the number of draws, per variable, as
brms's `neff_ratio`.
"""
neff_ratio(fit::Union{BrmsFit,BrmsMVFit}) =
    Dict(r.variable => r.bulk_ess / (size(fit.draws, 2) * size(fit.draws, 3)) for r in posterior_summary(fit))
