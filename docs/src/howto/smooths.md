# Smooths and Gaussian processes

For a predictor whose effect is a curve of unknown shape, use a smooth or a
Gaussian process. Both are penalised: the data decide how wiggly the curve is.

## Smooths

`s(x)` is a thin-plate regression spline, as in mgcv and brms:

```julia
fit = brm(@bf(y ~ s(x)), df)
fit = brm(@bf(y ~ s(x, k = 20)), df)            # more basis functions
fit = brm(@bf(y ~ s(x, bs = "cr")), df)         # cubic regression spline
fit = brm(@bf(y ~ s(x, by = group)), df)        # one curve per level of group
fit = brm(@bf(y ~ s(x, z)), df)                 # a two-dimensional smooth
fit = brm(@bf(y ~ t2(x, z)), df)                # a tensor-product smooth
```

The supported bases are `bs = "tp"` (the default), `"ts"`, `"cr"`, `"cs"`,
`"ps"`, `"cc"` and `"re"`. `by` may be a factor (a curve per level) or a
numeric variable (the curve is multiplied by it).

Each smooth has a standard deviation `sds_sx_1` that sets how wiggly the curve
can be, and, for bases with an unpenalised part, a linear coefficient
`b_sx_1`.

!!! note
    Thin-plate smooths fit the same curves as brms, but their `sds` values
    are on a different scale, because mgcv reparameterises the basis
    internally. Compare fitted curves, not `sds` values. The `cr`, `ps`,
    `cc` and `re` bases give the same `sds` as brms.

### Look at the curve

[`conditional_smooths`](@ref) evaluates a smooth along a grid of its
covariate; [`posterior_smooths`](@ref) gives its draws at the data or at new
data:

```julia
cs = conditional_smooths(fit; smooth = "s(x)")
cs.estimate, cs.lower, cs.upper
plot_effects(fit, :x)              # the whole predictor along x
```

## Gaussian processes

`gp(x)` is a Gaussian process with brms's squared-exponential kernel and
priors:

```julia
fit = brm(@bf(y ~ gp(x)), df)
fit = brm(@bf(y ~ gp(x, z)), df)                          # two inputs
fit = brm(@bf(y ~ gp(x, cov = "matern32")), df)           # other kernels: matern52, exponential
fit = brm(@bf(y ~ gp(x, k = 20, c = 5/4)), df)            # approximate GP
```

The parameters are `sdgp_gpx` (the marginal standard deviation) and
`lscale_gpx` (the length scale). An exact GP costs a Cholesky factorisation
of an n x n matrix per gradient, where n is the number of distinct input
values, and becomes slow as n grows. Give `k` (the number of basis functions)
and `c` (the boundary factor) for brms's Hilbert-space approximation, whose
cost grows linearly with n; it is usually much faster beyond a few dozen
distinct values.

## Splines as fixed effects

`ns(x, df)` and `bs(x, df)` (natural and B-spline bases, as in R's splines
package) and `poly(x, degree)` (orthogonal polynomials) enter as ordinary,
unpenalised terms:

```julia
brm(@bf(y ~ ns(x, 4) + poly(z, 2)), df)
```
