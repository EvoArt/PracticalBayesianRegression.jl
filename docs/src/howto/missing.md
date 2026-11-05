# Missing data and measurement error

By default, rows with a missing value in any variable the model uses are
dropped, with a warning. To keep them, model the missing values.

## Missing predictors

Mark the predictor with `mi(x)` and give `x` its own model with `x | mi()`.
The models are fitted together as a multivariate model, without residual
correlation:

```julia
f = @bf(y ~ mi(x) + z) + @bf(x | mi() ~ z) + set_rescor(false)
fit = brm(f, df)
```

Here `x` may contain `missing` values; they are imputed from its model given
`z`, and the uncertainty about them carries into the estimates for `y`.
The model for `x` must be `gaussian` or `student`.

### Integrated or sampled

brms treats every missing value as a parameter and samples it. By default
(`mi = :auto`) this package instead integrates a missing value out of the
likelihood where that is exact (a gaussian `x` entering a gaussian `y`) or
accurate by Gauss-Hermite quadrature (other families for `y`). That removes
parameters the sampler would otherwise have to explore. Where neither applies
(for example an `x` that enters several responses), it samples them as brms
does.

The posterior is the same either way. To sample every missing value, as
brms does:

```julia
fit = brm(f, df; mi = :sample)
```

In both cases the imputed values are available as draws named `Ymi_x[k]`.

## Missing responses

`y | mi()` keeps rows with a missing response. They do not contribute to the
likelihood, and their values are predicted from the model:

```julia
brm(@bf(y | mi() ~ x), df)
```

## Measurement error in predictors

`me(x, sdx)` treats the observed `x` as a noisy measurement, with known
standard error `sdx`, of a latent true value:

```julia
brm(@bf(y ~ me(x, sdx) + z), df)
```

With two or more `me()` terms the latent values are correlated, as in brms;
add `set_mecor(false)` to the formula to make them independent.

## Measurement error in the response

For a response with known standard errors, as in a meta-analysis, use
`se()`; see [Censored, truncated and weighted data](@ref).

## Multiple imputation

If you have already imputed the data, fit the model to each imputed data set
and pool the draws with [`brm_multiple`](@ref):

```julia
fit = brm_multiple(@bf(y ~ x + z), [df1, df2, df3, df4, df5])
```
