# Predict for new data

All prediction functions return draws: a matrix with one row per posterior
draw and one column per observation. Without `newdata` they predict for the
data the model was fitted to.

| Function | Draws of |
|:---------|:---------|
| [`posterior_epred`](@ref) | the expected value of the response (`mu`, or `n * mu` for binomial) |
| [`posterior_predict`](@ref) | new responses, including the observation noise |
| [`posterior_linpred`](@ref) | the linear predictor, on the link scale |
| [`log_lik`](@ref) | the log-likelihood of each observation |

For ordinal and categorical models, `posterior_epred` returns the category
probabilities, an array of draws x observations x categories.

## New data

`newdata` is a table with the predictors the model uses (the response is not
needed, except by `log_lik`). Factors are coded as in the training data:

```julia
nd = DataFrame(x = [0.0, 1.0, 2.0], group = ["a", "a", "b"])
E = posterior_epred(fit; newdata = nd)
Y = posterior_predict(fit; newdata = nd)
```

Summarise draws with the usual functions, or use the summary helpers:

```julia
fitted_summary(fit; newdata = nd)        # mean, sd and 95% interval of E[y]
predict_summary(fit; newdata = nd)       # the same for new observations
predictive_interval(fit; newdata = nd, prob = 0.9)
```

## Group-level effects

`re_formula` chooses which group-level effects to include:

- `nothing` (the default): all of them;
- `:none`: none, which predicts for an average group (brms's
  `re_formula = NA`);
- a grouping factor's name, or a vector of names: only those.

```julia
posterior_epred(fit; newdata = nd, re_formula = :none)
posterior_epred(fit; newdata = nd, re_formula = "site")
```

## New groups

A level of a grouping factor that was not in the training data raises an
error unless `allow_new_levels = true`. `sample_new_levels` then decides how
its effects are drawn, as in brms:

- `:uncertainty` (default): for each draw and new level, the effects of a
  randomly chosen existing level;
- `:gaussian`: from the estimated normal distribution of the effects;
- `:old_levels`: the effects of a randomly chosen existing level for each
  observation.

```julia
nd = DataFrame(x = [0.0], site = ["new site"])
posterior_predict(fit; newdata = nd, allow_new_levels = true, sample_new_levels = :gaussian)
```

## Other distributional parameters

`posterior_linpred` takes `dpar` to predict a distributional parameter on its
link scale:

```julia
log_sigma = posterior_linpred(fit; newdata = nd, dpar = :sigma)
```

## Effects along one predictor

[`conditional_effects`](@ref) predicts along a grid of one predictor, with
the others at their means or reference levels, and returns the estimate and
interval. With a pair `:x => :g` it does so for each level of `g`.
[`plot_effects`](@ref) draws the result.

```julia
ce = conditional_effects(fit, :x => :g)
plot_effects(ce)
```

## Not supported with new data

Predictions with `newdata` are not available for models with `car()`,
`sar()`, `fcor()` or latent (non-gaussian) autocorrelation terms.
