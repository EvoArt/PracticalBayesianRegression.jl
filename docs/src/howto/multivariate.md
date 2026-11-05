# Multivariate models

A multivariate model fits several responses at once. They can share
parameters, have correlated residuals, or have correlated group-level
effects.

## Several responses

Add one formula per response:

```julia
f = @bf(height ~ age + (1 | p | family)) + @bf(weight ~ age + (1 | p | family))
fit = brm(f, df)
```

When all responses share the same predictors, `mvbind` is shorter:

```julia
fit = brm(@bf(mvbind(height, weight) ~ age + (1 | p | family)), df)
```

The shared ID `p` in `(1 | p | family)` correlates the `family` effects
across the two responses.

## Residual correlations

When every response is gaussian, the residuals are correlated by default, as
in brms (`rescor__height__weight`); student responses get residual
correlations with `set_rescor(true)`. Turn them off with
`set_rescor(false)`:

```julia
f = @bf(height ~ age) + @bf(weight ~ age) + set_rescor(false)
```

## Different families

Give one family per response, in the order of the formulas:

```julia
f = @bf(count ~ x) + @bf(size ~ x) + set_rescor(false)
fit = brm(f, df; family = [poisson(), gaussian()])
```

## Priors per response

Target one response with `resp`:

```julia
priors = [prior("normal(0, 1)"; class = :b, resp = "weight")]
```

## Results

Parameter names include the response: `b_height_age`, `sigma_weight`.
Index a fit by response name to get that response's own fit, on which every
function for univariate fits works:

```julia
fit["height"]                    # summary of the height model
posterior_predict(fit["weight"])
loo(fit)                         # LOO of the joint model
```

## Different rows per response

`subset(s)` fits a response on only the rows where `s` is true:

```julia
f = @bf(y1 | subset(has_y1) ~ x) + @bf(y2 | subset(has_y2) ~ x)
```

Residual correlations are then turned off.
