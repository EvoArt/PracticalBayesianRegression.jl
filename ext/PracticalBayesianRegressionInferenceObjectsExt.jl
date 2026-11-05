module PracticalBayesianRegressionInferenceObjectsExt

# `to_inferencedata(fit)`: hand a fit to ArviZ.jl (diagnostics, LOO, plots)
# as an InferenceObjects.InferenceData.

using PracticalBayesianRegression
using PracticalBayesianRegression: BrmsFit, variables, draws, posterior_predict, log_lik, FlexiChains
import PracticalBayesianRegression: to_inferencedata
using InferenceObjects
using Random

# FlexiChain extras -> ArviZ's sample_stats names
const _STATS = ("lp__" => :lp, "accept_stat__" => :acceptance_rate, "stepsize__" => :step_size,
                "treedepth__" => :tree_depth, "n_leapfrog__" => :n_steps, "divergent__" => :diverging)

function to_inferencedata(fit::BrmsFit; posterior_predictive::Bool=true, log_likelihood::Bool=true,
                          rng=Random.default_rng())
    niter, nch = size(fit.draws, 2), size(fit.draws, 3)
    post = NamedTuple(Symbol(v) => Matrix{Float64}(draws(fit, v)) for v in variables(fit))
    yname = Symbol(string(fit.info.formula.response))
    y = Float64.(fit.density.y)
    N = length(y)
    # (draws x obs) matrices are ordered chain by chain -> (draw, chain, obs)
    to3(M) = reshape(M, niter, nch, N)
    stats = Dict{Symbol,Any}()
    for (k, s) in _STATS
        key = FlexiChains.Extra(k)
        try
            v = Matrix{Float64}(fit.chain[key])
            stats[s] = s === :diverging ? Bool.(v .> 0) : v
        catch
        end
    end
    kw = Dict{Symbol,Any}(:observed_data => NamedTuple{(yname,)}((y,)), :library => "PracticalBayesianRegression")
    isempty(stats) || (kw[:sample_stats] = NamedTuple(stats))
    posterior_predictive && (kw[:posterior_predictive] = NamedTuple{(yname,)}((to3(posterior_predict(fit; rng=rng)),)))
    log_likelihood && (kw[:log_likelihood] = NamedTuple{(yname,)}((to3(log_lik(fit)),)))
    return from_namedtuple(post; kw...)
end

end
