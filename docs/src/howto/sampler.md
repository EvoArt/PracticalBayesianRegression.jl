# Choose sampler settings

The defaults are Stan's and brms's: 4 chains of 2000 iterations, half of them
warm-up, a target acceptance rate of 0.8, a maximum tree depth of 10 and a
diagonal mass matrix. They suit most models. This page covers the settings to
change when a fit reports problems or is slow.

## Run chains in parallel

Chains run on separate threads. Start Julia with as many threads as chains:

```
julia -t 4
```

With one thread the chains run one after another. Check with
`Threads.nthreads()`.

## Divergent transitions

A warning such as "There were 12 divergent transitions after warmup" means
the sampler met regions of the posterior it could not follow, and the draws
may be biased. Raise the target acceptance rate, which makes the sampler
take smaller steps:

```julia
fit = brm(f, df; delta = 0.95)
```

If divergences remain, the model is usually the cause: weakly identified
parameters, or a funnel between a group-level standard deviation and its
effects when there are few groups or little data per group. Tighter priors
on the standard deviations often help.

## Maximum tree depth

If many iterations hit the maximum tree depth (see
`nuts_params(fit)["treedepth__"]`), sampling is inefficient but not biased.
Raise the limit with `max_depth = 12`; often reparameterising or rescaling
the predictors helps more.

## Dense mass matrix

During warm-up the sampler learns the scale of each parameter (a diagonal
mass matrix). With `metric = :dense` it also learns their correlations, at
the cost of more work per step:

```julia
fit = brm(f, df; metric = :dense)
```

Whether that pays off depends on the model. In comparisons of effective
sample size per second over 19 models with up to about 100 parameters,
five runs each:

| Model | dense / diagonal |
|:------|:-----------------|
| gaussian, random intercepts (50 groups) | 3.6 |
| gaussian, random intercepts and slopes | 1.9 |
| gaussian, crossed random intercepts | 2.3 |
| Poisson, random intercepts | 3.0 |
| binomial, random intercepts | 4.0 |
| beta, random intercepts | 5.2 |
| Student-t, distributional, zero-inflated, hurdle | 1.2 to 1.6 |
| fixed effects only, few parameters | about 1 |
| Bernoulli, random intercepts | 0.55 |
| 50 population-level coefficients | 0.59 |
| ordinal, random intercepts | about 1 |

So:

- Try `metric = :dense` for multilevel models with a moderate number of
  parameters and for models with correlated parameters (distributional
  models, families with several parameters).
- Keep the default for models with many coefficients (with hundreds or
  thousands of parameters, the warm-up cannot estimate a dense matrix well and
  each step costs more), and for Bernoulli multilevel models.
- When it matters, compare: fit both and look at the effective sample sizes
  and the sampling time printed with the summary.

## Warm-up and number of draws

The defaults give 4000 draws, plenty for means and 95% intervals. For tail
quantities or Bayes factors, run more iterations (`iter = 4000`). Thinning
(`thin`) only discards information; use it to save memory.

## Initial values

Chains start at values drawn uniformly from (-2, 2) on the unconstrained
scale. If initialisation fails ("no finite log density in 100 random
initialisations"), narrow the range with `init = 0.5`, or check the priors
and data for impossible values.

## Reproducibility

`seed = 1` makes a fit reproducible: the same seed, data and settings give
the same draws, with or without threads.

## Other samplers

`sampler = :advancedhmc` uses AdvancedHMC.jl's NUTS on the same log density,
and any `AdvancedHMC.AbstractHMCSampler` can be passed. The package's own
NUTS is the default because it was faster in every comparison.
