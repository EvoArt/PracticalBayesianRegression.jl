# Compare models

## Cross-validation

[`loo`](@ref) estimates how well a model predicts observations it was not
fitted to, by Pareto-smoothed importance sampling (PSIS-LOO), from the draws
of one fit:

```julia
loo(fit)
```

The output gives `elpd_loo` (the expected log predictive density; higher is
better), `p_loo` (the effective number of parameters) and `looic`
(`-2 elpd_loo`), with standard errors, and a count of observations whose
Pareto k diagnostic is high. For those, the estimate is unreliable. Two
remedies:

```julia
loo(fit; moment_match = true)     # adapt the draws for the problematic observations
kfold(fit; K = 10)                # refit the model K times
```

[`waic`](@ref) computes the widely applicable information criterion from the
same pointwise log-likelihoods.

## Ranking models

[`loo_compare`](@ref) ranks fits by `elpd_loo` and gives each difference from
the best model with its standard error. Name the fits with pairs:

```julia
loo_compare("linear" => fit1, "quadratic" => fit2, "spline" => fit3)
```

A difference smaller than about two standard errors is weak evidence.

## Model weights and averaging

Rather than choosing one model, combine their predictions.
[`model_weights`](@ref) computes stacking weights (the default), pseudo-BMA+
or LOO weights:

```julia
w = model_weights(fit1, fit2, fit3)
w = model_weights(fit1, fit2, fit3; weights = :pseudobma)
```

[`pp_average`](@ref) draws from the weighted mixture of the models'
posterior predictive distributions, and [`posterior_average`](@ref) mixes
the draws of parameters the models share:

```julia
Y = pp_average(fit1, fit2; newdata = nd)
posterior_average(fit1, fit2; variables = ["b_x"])
```

## Bayes factors

[`bridge_sampler`](@ref) estimates a fit's log marginal likelihood by bridge
sampling. [`bayes_factor`](@ref) and [`post_prob`](@ref) compare models with
it:

```julia
bayes_factor(fit1, fit2)          # (bf, log_bf)
post_prob(fit1, fit2, fit3)       # posterior model probabilities, equal prior odds
```

Marginal likelihoods depend strongly on the priors and are undefined with
flat priors, so set proper priors on every parameter, and run more draws
than usual (for example `iter = 10_000`) for a stable estimate. Some model
structures are not supported by `bridge_sampler` yet; it raises an error
naming them.

For a point hypothesis about one parameter, [`hypothesis`](@ref) gives the
Savage-Dickey Bayes factor of a fit with `sample_prior = :yes`:

```julia
fit = brm(f, df; prior = priors, sample_prior = :yes)
hypothesis(fit, "x = 0")
```
