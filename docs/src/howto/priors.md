# Set priors

Every parameter has a prior. Those you do not set get brms's default, which
depends on the family and the data. The workflow is the same as in brms:
list the parameters and their defaults, set the ones you want, check the
result.

## List the parameters and their default priors

```julia
using PracticalBayesianRegression

f = @bf(y ~ x + z + (1 + x | g))
get_prior(f, df; family = gaussian())
```

[`get_prior`](@ref) (or its brms 2.21 name, [`default_prior`](@ref)) prints
one row per parameter class, and one per coefficient or group within it. The
`class`, `coef`, `group` and `dpar` columns are what you pass to
[`prior`](@ref) to target a row.

## Give priors to the model

Pass a vector of [`prior`](@ref) rows to `brm`. A prior can be a brms prior
string or a Distributions.jl distribution:

```julia
using Distributions

priors = [
    prior("normal(0, 1)"; class = :b),                   # all slopes
    prior("normal(0, 0.5)"; class = :b, coef = "z"),     # the slope of z only
    prior("student_t(3, 0, 10)"; class = :Intercept),
    prior(Exponential(1); class = :sigma),
    prior("exponential(2)"; class = :sd, group = "g"),   # standard deviations of g's effects
    prior(lkj(2); class = :cor),                         # their correlations
]
fit = brm(f, df; prior = priors)
```

The most specific row wins: here `z` gets `normal(0, 0.5)` and `x` gets
`normal(0, 1)`. Standard deviations and other positive parameters are
automatically restricted to positive values, so `normal(0, 1)` on `sigma` is a
half-normal prior.

## Check the full prior table

[`validate_prior`](@ref) shows the table the model would be fitted with,
user priors and defaults together. Check that each of your priors appears
where you meant it to; a `coef` or `group` that matches nothing in the model
is ignored.

```julia
validate_prior(f, df; prior = priors)
```

After fitting, [`prior_summary`](@ref) shows the same table for the fit.

## Priors on distributional parameters

When a distributional parameter has its own formula, its coefficients are a
`:b` class with `dpar` set, and its intercept an `:Intercept` class:

```julia
f = @bf(y ~ x, sigma ~ x)
priors = [prior("normal(0, 0.5)"; class = :b, dpar = "sigma"),
          prior("normal(0, 1)"; class = :Intercept, dpar = "sigma")]
```

These are on the link scale (log for `sigma`).

## Bounded coefficients

`lb` and `ub` restrict a coefficient, as in brms:

```julia
prior("normal(0, 1)"; class = :b, coef = "dose", lb = 0)    # a positive effect of dose
```

## Fixed values

`constant(v)` fixes a parameter to `v` instead of estimating it:

```julia
prior("constant(0.9)"; class = :quantile)   # asym_laplace: model the 0.9 quantile
```

## Shrinkage priors

`horseshoe(...)` and `R2D2(...)` are priors on all population-level
coefficients together, with brms's arguments:

```julia
prior("horseshoe(1)"; class = :b)
prior("R2D2(0.5, 2)"; class = :b)
```

## Sampling from the prior

`sample_prior = :only` ignores the data and samples the priors alone, for
prior predictive checks. It needs a proper prior on every parameter, so
replace brms's flat default on `b` first:

```julia
fit0 = brm(f, df; prior = [prior("normal(0, 1)"; class = :b)], sample_prior = :only)
pp_check(fit0)     # data simulated from the priors alone
```

`sample_prior = :yes` fits the model and also draws from the priors, which
[`hypothesis`](@ref) uses for Savage-Dickey Bayes factors of point
hypotheses such as `"x = 0"`.
