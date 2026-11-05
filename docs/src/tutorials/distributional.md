# A distributional model

An ordinary regression models the mean of the response and assumes that the
other parameters of its distribution, such as the residual standard
deviation, are the same for every observation. A distributional model gives
those parameters their own formulas. This tutorial models data whose spread
grows with the predictor.

## The data

We simulate crop yields at 150 fertiliser doses. The mean yield rises with
the dose, and so does the variation between plots.

```@example dist
using PracticalBayesianRegression, DataFrames, Distributions, Random

rng = Xoshiro(7)
dose = 10 .* rand(rng, 150)
yield = 2 .+ 0.5 .* dose .+ exp.(-1 .+ 0.25 .* dose) .* randn(rng, 150)
crops = DataFrame(; yield, dose)
first(crops, 3)
```

## A model with constant spread

```@example dist
fit_const = brm(@bf(yield ~ dose), crops; seed = 1)
```

The residuals of this fit fan out as the fitted values grow, which a constant
`sigma` cannot describe:

```@example dist
using CairoMakie
residual_plot(fit_const)
```

## Modelling sigma

Add a formula for `sigma`, the residual standard deviation of the gaussian
family:

```@example dist
fit_dist = brm(@bf(yield ~ dose, sigma ~ dose), crops; seed = 1)
```

The coefficients of `sigma` are on the log scale, the default link for
`sigma`. `sigma_dose` is the change in `log(sigma)` per unit of dose, so the
standard deviation is multiplied by `exp(sigma_dose)` for each unit:

```@example dist
exp(mean(draws(fit_dist, "b_sigma_dose")))
```

The estimate is close to `exp(0.25)`, the value used in the simulation.

## Comparing the models

```@example dist
loo_compare("sigma ~ dose" => fit_dist, "constant sigma" => fit_const)
```

The distributional model predicts held-out observations clearly better.

## What it changes

Both models estimate much the same mean line, but their predictive intervals
differ. With a constant `sigma` the 90% predictive interval has the same
width at every dose; with `sigma ~ dose` it widens with the dose, as the data
do:

```@example dist
grid = DataFrame(dose = [1.0, 5.0, 9.0])
widths(fit) = [hi - lo for (lo, hi) in eachrow(predictive_interval(fit; newdata = grid))]
DataFrame(dose = grid.dose, constant = widths(fit_const), distributional = widths(fit_dist))
```

Any distributional parameter of a family can be modelled this way, with
population- and group-level terms: `shape` of the negative binomial, `phi` of
the beta, `zi` of zero-inflated families, and so on. The family's reference
entry lists its parameters ([Families](@ref)).
