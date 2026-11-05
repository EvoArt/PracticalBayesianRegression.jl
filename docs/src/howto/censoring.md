# Censored, truncated and weighted data

Information about the response beyond its value goes on the left-hand side
of the formula, after `|`. Several addition terms combine with `+`:
`y | trials(n) + weights(w) ~ x`.

## Censoring

A censored observation is known only to lie beyond a limit, or within an
interval. Give a censoring indicator with `cens()`:

```julia
brm(@bf(time | cens(censored) ~ treatment), df; family = weibull())
```

`censored` holds, for each row, `"none"` (or 0, or `false`) for an exact
value, `"right"` (1, `true`) when the true value is above `time`, `"left"`
(-1) when it is below, and `"interval"` (2) when it lies between `time` and a
second variable given as `cens(censored, time_upper)`:

```julia
brm(@bf(y | cens(censored, y_upper) ~ x), df)
```

Censoring works for `gaussian`, `lognormal`, `weibull` and `exponential`
responses and for the count families `poisson`, `negbinomial`,
`negbinomial2`, `geometric`, `binomial`, `beta_binomial`, `discrete_weibull`
and `com_poisson`. For survival times, [`cox`](@ref) also takes
`cens()`.

## Truncation

A truncated response cannot be observed outside known bounds:

```julia
brm(@bf(y | trunc(lb = 0) ~ x), df)                       # y > 0
brm(@bf(count | trunc(lb = 1) ~ x), df; family = poisson())   # zero-truncated Poisson
brm(@bf(y | trunc(lb = 0, ub = 100) ~ x), df)
```

For count families the bounds are inclusive, as in Stan: `trunc(lb = 1)`
allows 1, 2, ... Bounds may also be data columns: `trunc(ub = ymax)`.

## Weights

`weights(w)` multiplies each observation's log-likelihood by `w`:

```julia
brm(@bf(y | weights(w) ~ x), df)
```

## Known standard errors

`se(s)` gives each observation a known standard error, as in a
meta-analysis. With `se(s, sigma = true)` a residual standard deviation is
estimated as well:

```julia
brm(@bf(effect | se(std_err) ~ 1 + (1 | study)), meta)
```

## Trials and exposure

`trials(n)` is the number of trials of a binomial-type response; `rate(t)`
gives a count model an exposure `t` (an offset of `log(t)` on the mean):

```julia
brm(@bf(successes | trials(n) ~ x), df; family = binomial())
brm(@bf(cases | rate(person_years) ~ x), df; family = poisson())
```
