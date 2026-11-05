# Non-linear models

When the mean is a known non-linear function of parameters, write the
function in the main formula and give each parameter its own formula, with
`nl = true`:

```julia
f = @bf(y ~ a * exp(-b * x), a ~ 1, b ~ 1 + (1 | g), nl = true)
fit = brm(f, df; prior = [prior("normal(10, 5)"; class = :b, nlpar = "a"),
                          prior("normal(0.5, 0.5)"; class = :b, nlpar = "b", lb = 0)])
```

- The right-hand side of the main formula is an ordinary Julia expression in
  the parameters and in data columns (`x` here). Any Julia function can
  appear in it.
- Each parameter formula (`a ~ 1`, `b ~ 1 + (1 | g)`) is a linear model with
  the usual terms, including group-level effects.
- Every non-linear parameter needs a proper prior, as in brms: the defaults
  are flat. Target them with `nlpar`.

The parameters are named after their formulas: `b_a_Intercept`,
`b_b_Intercept`, `sd_g__b_Intercept`, ...

## Formulas given as strings

A non-linear formula given as a string defines a Julia function when it is
parsed:

```julia
f = bf("y ~ a * exp(-b * x)", "a ~ 1", "b ~ 1"; nl = true)
```

Inside a function, a newly defined function cannot be called until control
returns to the top level, so call `brm` through `Base.invokelatest`:

```julia
function fit_decay(df)
    f = bf("y ~ a * exp(-b * x)", "a ~ 1", "b ~ 1"; nl = true)
    return Base.invokelatest(brm, f, df; prior = priors)
end
```

The `@bf` form has no such restriction. Inside a package, pass
`mod = @__MODULE__` to `bf`.

## Distributional parameters

A non-linear model can also model the family's other parameters; put their
formulas in the same `@bf`:

```julia
@bf(y ~ a * exp(-b * x), a ~ 1, b ~ 1, sigma ~ x, nl = true)
```
