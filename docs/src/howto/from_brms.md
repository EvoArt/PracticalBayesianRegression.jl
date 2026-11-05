# Coming from brms

Most brms code carries over with Julia's syntax for function calls. This page
lists what is written differently and the few places where results can
differ.

## Calls

| brms (R) | PracticalBayesianRegression (Julia) |
|:---------|:------------------------------------|
| `brm(y ~ x, data = d)` | `brm(@bf(y ~ x), d)` or `brm("y ~ x", d)` |
| `bf(y ~ x, sigma ~ x)` | `@bf(y ~ x, sigma ~ x)` or `bf("y ~ x", "sigma ~ x")` |
| `bf(y ~ x) + lf(sigma ~ x)` | `@bf(y ~ x) + @lf(sigma ~ x)` |
| `bf(y ~ a * exp(b * x), a + b ~ 1, nl = TRUE)` | `@bf(y ~ a * exp(b * x), a ~ 1, b ~ 1, nl = true)` |
| `family = poisson()` | `family = poisson()` |
| `family = poisson("sqrt")` | `family = poisson(link = :sqrt)` |
| `family = Gamma()` | `family = gammafamily(link = :inverse)` |
| `family = Beta()` | `family = betafamily()` |
| `family = inverse.gaussian()` | `family = inverse_gaussian()` |
| `prior(normal(0, 1), class = b)` | `prior("normal(0, 1)"; class = :b)` |
| `set_prior("normal(0, 1)", class = "b", coef = "x")` | `prior("normal(0, 1)"; class = :b, coef = "x")` |
| `c(prior(...), prior(...))` | `[prior(...), prior(...)]` |
| `control = list(adapt_delta = 0.95, max_treedepth = 12)` | `delta = 0.95, max_depth = 12` |
| `control = list(metric = "dense_e")` | `metric = :dense` |
| `cores = 4` | start Julia with `-t 4`; chains use the available threads |
| `sample_prior = "yes"` | `sample_prior = :yes` |
| `algorithm = "meanfield"` | `algorithm = :meanfield` |
| `data2 = list(W = W)` | `data2 = (; W = W)` |
| `summary(fit)`, `print(fit)` | `fit` (display) |
| `fixef(fit)`, `ranef(fit)`, `coef(fit)` | `fixef(fit)`, `ranef(fit)`, `group_coef(fit)` |
| `fitted(fit)`, `predict(fit)` | `fitted_summary(fit)`, `predict_summary(fit)` |
| `posterior_epred(fit, newdata = nd, re_formula = NA)` | `posterior_epred(fit; newdata = nd, re_formula = :none)` |
| `as_draws_df(fit)`, `variables(fit)` | `draws(fit)`, `variables(fit)` |
| `stancode(fit)`, `standata(fit)` | `turing_code(fit)`, `model_data(fit)` (Julia code instead of Stan) |
| `get_prior(...)`, `default_prior(...)` | `get_prior(...)`, `default_prior(...)` |
| `loo(fit1, fit2)` | `loo_compare(fit1, fit2)` |

`TRUE` and `FALSE` inside a formula may be written as in R or as `true` and
`false`: `ar(time, g, cov = TRUE)`, `se(s, sigma = TRUE)`.

Families with the same name in R and Julia are called the same way. Two
brms names belong to Distributions.jl in Julia, hence `gammafamily()` and
`betafamily()`. `binomial()` works as in R: it is a method of Julia's
`binomial` function, so it needs no import.

## Formulas

- Julia parses `:` more loosely than `+`. Write `(a + b):c` out as
  `a:c + b:c`; terms such as `x + a:b` are read as in R. `&` is accepted as a
  synonym for `:`.
- Functions in a formula (`log(x)`, `myfun(x)`) are Julia functions. With
  `@bf` they are looked up where the macro is used; with a string they are
  looked up in `Main` (pass `mod` to `bf` to change this).
- A non-linear formula given as a string, `bf("..."; nl = true)`, defines a
  Julia function at run time. Inside a function, call `brm` through
  `Base.invokelatest(brm, f, d; ...)`. The `@bf` form has no such
  restriction.
- Matrices for `car()`, `sar()`, `fcor()` and `gr(g, cov = A)` come from
  `data2` (or, failing that, the calling module). Julia matrices have no row
  names: their rows and columns must be in the sorted order of the grouping
  levels.

## Data

- Data is any Tables.jl table: a `DataFrame`, a NamedTuple of vectors, a
  CSV.jl file, and so on.
- Strings, `Bool`s and categorical vectors are factors; numbers are numeric.
  Factor levels are sorted by code point, whereas R sorts them by locale, so
  `"b" < "B"` in R but not in Julia. The level order sets the reference level
  and the order of the coefficients. Labels that sort the same in both (one
  case, zero-padded numbers such as `"g01"`, `"g02"`, ...) give identical
  models. A `CategoricalArray`'s own level order is respected.
- An ordinal response is an integer 1, 2, ... or a categorical vector, as an
  ordered factor in R.

## Results that differ

- `gammafamily()` uses the log link by default; brms's `Gamma()` uses the
  inverse link.
- Thin-plate smooths `s(x)` fit the same curves as brms, but their `sds`
  parameter is on a different scale, because mgcv reparameterises the basis
  internally. The bases `bs = "cr"`, `"ps"`, `"cc"` and `"re"` match brms
  exactly. `bs = "cs"` shrinks its null space by a different amount than mgcv.
- Missing covariates modelled with `mi()` are by default integrated out where
  that is exact or accurate, rather than sampled; `brm(...; mi = :sample)`
  samples all of them, as brms does. The posterior is the same.
- `lp__` follows Stan's convention (constants dropped, Jacobians included).
  [`log_lik`](@ref) includes all constants, so LOO and WAIC values match
  brms's.
