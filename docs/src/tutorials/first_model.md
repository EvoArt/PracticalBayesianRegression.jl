# A first model

This tutorial fits a regression model to count data, checks it, finds a
problem, and fixes it by changing the response distribution. Along the way it
covers the steps of most analyses: writing a formula, fitting, reading the
summary, checking the fit against the data, and comparing models.

You need PracticalBayesianRegression, DataFrames, Distributions (to simulate
the data) and CairoMakie (for the plots):

```julia
using Pkg
Pkg.add(["PracticalBayesianRegression", "DataFrames", "Distributions", "CairoMakie"])
```

Start Julia with several threads, for example `julia -t 4`, so that the four
chains run in parallel.

## The data

We simulate the number of seedlings counted in 120 plots, which depends on
soil moisture. Simulated data has the advantage that we know the true values:
an intercept of 1.5 and a slope of 0.4 on the log scale. The counts are more
variable than a Poisson distribution allows (a negative binomial with shape
3), as real counts often are.

```@example first
using PracticalBayesianRegression, DataFrames, Distributions, Random

rng = Xoshiro(2024)
n = 120
moisture = randn(rng, n)
mu = exp.(1.5 .+ 0.4 .* moisture)
seedlings = [rand(rng, NegativeBinomial(3, 3 / (3 + m))) for m in mu]
plots = DataFrame(; seedlings, moisture)
first(plots, 5)
```

## Priors

Every parameter gets a prior. Without your input the package uses the same
default priors as brms, which depend on the data. List them before fitting
with [`get_prior`](@ref):

```@example first
get_prior(@bf(seedlings ~ moisture), plots; family = poisson())
```

The slope `b` has a flat prior and the intercept a Student-t prior on the
link (log) scale. [Set priors](@ref "Set priors") shows how to change them.

## Fitting

A Poisson regression of the counts on moisture:

```@example first
fit_pois = brm(@bf(seedlings ~ moisture), plots; family = poisson(), seed = 1)
```

[`@bf`](@ref) writes the formula in brms syntax, [`brm`](@ref) fits it.
The fit ran four chains of 2000 iterations, the first 1000 of each being
warm-up, and kept 4000 draws from the posterior.

The table has one row per regression coefficient. `Estimate` is the
posterior mean and `Est.Error` the posterior standard deviation; the next two
columns bound the central 95% credible interval. `Rhat` near 1.00 and large
effective sample sizes (`Bulk_ESS`, `Tail_ESS`) show that the chains agree
and the estimates are precise.

The draws of any parameter are available by name:

```@example first
variables(fit_pois)
```

```@example first
slope = vec(draws(fit_pois, "b_moisture"))
mean(slope .> 0)
```

Every draw of the slope is positive. [`hypothesis`](@ref) gives the same
answer in brms's format:

```@example first
hypothesis(fit_pois, "moisture > 0")
```

## Checking the model

A posterior predictive check compares the data with data simulated from the
fitted model. If the model describes the data well, the observed counts look
like the simulated ones. Plots need a plotting package; here CairoMakie.

```@example first
using CairoMakie
pp_check(fit_pois; type = :bars)
```

The bars are the observed number of plots with each count; the points and
lines are the median and 90% interval of the same numbers in data simulated
from the model. Eleven plots had no seedlings, but the model expects about
two, and it gives almost no chance to the plot with more than 20 seedlings:
the counts vary more than a Poisson distribution allows.

## Changing the family

A negative binomial distribution adds a `shape` parameter for this extra
variation:

```@example first
fit_nb = brm(@bf(seedlings ~ moisture), plots; family = negbinomial(), seed = 1)
```

The table now has a `shape` row under "Further Distributional Parameters".
The estimates of the intercept and slope barely change, but their standard
errors grow: the Poisson model was overconfident. The check looks better:

```@example first
pp_check(fit_nb; type = :bars)
```

## Comparing the models

Leave-one-out cross-validation estimates how well each model predicts
observations it has not seen. [`loo`](@ref) approximates it from the
existing draws:

```@example first
loo(fit_nb)
```

`elpd_loo` is the expected log predictive density; higher is better.
[`loo_compare`](@ref) ranks the models and gives the standard error of each
difference:

```@example first
loo_compare("negbinomial" => fit_nb, "poisson" => fit_pois)
```

The Poisson model's `elpd_loo` is lower by about 32, more than two and a
half times the standard error of the difference, so the negative binomial
model predicts new plots better. Its estimates are also close to the values
used in the simulation: an intercept of 1.5, a slope of 0.4 and a shape of 3.

## Next steps

- [A multilevel model](@ref) adds group-level effects.
- [A distributional model](@ref) lets the variance depend on predictors.
- [Coming from brms](@ref) lists the differences in syntax from R.
