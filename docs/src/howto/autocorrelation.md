# Autocorrelation and spatial terms

Observations close in time or space are often more alike than the predictors
explain. These terms model that dependence, with brms's syntax and
parameterisations.

## Time series

`ar`, `ma` and `arma` add autoregressive and moving-average residual
structures within each series. `time` orders the observations and `gr`
separates the series:

```julia
brm(@bf(y ~ x + ar(time = day, gr = site, p = 1)), df)
brm(@bf(y ~ x + arma(day, site, 1, 1)), df)              # positional: time, gr, p, q
brm(@bf(y ~ x + ar(day, site) + ma(day, site)), df)      # the same as arma
```

The parameters are `ar[1]`, ..., `ma[1]`, .... By default these are brms's
residual formulation, for `gaussian` and `student` responses. With
`cov = true` (order 1 only) the correlation enters the residual covariance
matrix instead, which also works for other families through latent residuals:

```julia
brm(@bf(y ~ x + ar(gr = site, cov = true)), df)
brm(@bf(count ~ x + ar(gr = site, cov = true)), df; family = poisson())
```

Other residual structures:

```julia
brm(@bf(y ~ x + cosy(gr = site)), df)            # compound symmetry
brm(@bf(y ~ x + unstr(time = wave, gr = person)), df)   # unstructured correlations
brm(@bf(y ~ x + fcor(M)), df; data2 = (; M))     # a fixed covariance matrix M
```

`unstr()` is available for `gaussian` and `student` responses.

## Spatial terms

Both spatial terms take a matrix `W` of neighbourhood weights from `data2`.

`sar(W)` is a simultaneous autoregressive model, with `type = "lag"` (the
default) or `"error"`; `W` has one row and column per observation:

```julia
brm(@bf(y ~ x + sar(W, type = "error")), df; data2 = (; W))
```

`car(W, gr = region)` is a conditional autoregressive model over areas; `W`
has one row and column per area and `gr` gives each observation's area. The
types are brms's `"escar"` (the default), `"esicar"`, `"icar"` and `"bym2"`:

```julia
brm(@bf(cases ~ x + car(W, gr = region, type = "bym2")), df;
    family = poisson(), data2 = (; W))
```

Julia matrices have no row names, so the rows and columns of `W` must follow
the sorted order of the area labels (as `sort(unique(df.region))`), or `gr`
must hold the row numbers.

## Known correlation between groups

`gr(g, cov = A)` correlates the group-level effects of `g` through a known
matrix `A`, such as a phylogenetic or kinship matrix:

```julia
brm(@bf(trait ~ x + (1 | gr(species, cov = A))), df; data2 = (; A))
```

The rows and columns of `A` follow the sorted order of the levels of `g`.

## Predictions

Predictions for new data are available for the time-series structures, which
are applied within the new data's own series. For `car()`, `sar()`,
`fcor()` and the latent (non-gaussian) structures, predictions with
`newdata` raise an error.
