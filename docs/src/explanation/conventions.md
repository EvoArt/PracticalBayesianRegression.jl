# How brms's conventions are followed

A model specified here and the same model specified in brms should give the
same posterior, up to Monte Carlo error. That requires more than the same
formula: brms makes many choices on the user's behalf, and this package makes
the same ones.

## Design matrices

Population-level design matrices follow R's `model.matrix`: treatment
contrasts for factors, R's rules for when a factor is coded with all its
levels (for example in `0 + f` or in interactions whose margins are absent),
and R's term order. Coefficient names follow brms, which removes characters
that Stan names cannot contain: `log(x)` becomes `logx` and `I(x^2)` becomes
`IxE2`.

`poly()`, `ns()` and `bs()` reproduce R's orthogonal polynomials and splines
package bases, so new data is coded exactly as R codes it.

The one difference is the order of factor levels. R sorts text by the
locale's collation rules; Julia sorts by code point. Labels that sort the same
both ways give identical models.

## Centring

As brms does by default, the population-level predictors of each linear
predictor are centred before sampling. The sampler works with the intercept
at the predictors' means (`Intercept`), which is much less correlated with
the slopes, and the reported `b_Intercept` is converted back to the
intercept at zero. Priors on class `:Intercept` apply to the centred
intercept, as in brms. `center = false` in the formula turns this off.

## Default priors

The default priors are brms's, including their dependence on the data:

- `b`: flat;
- `Intercept`: Student-t with 3 degrees of freedom, located at the median of
  the response on the link scale and scaled by its median absolute
  deviation (at least 2.5);
- `sd`, `sigma`: half Student-t with 3 degrees of freedom and a scale taken
  from the median absolute deviation of the response (at least 2.5);
- `cor`: `lkj(1)`;
- family parameters (`shape`, `nu`, `phi`, ...): brms's choice for each
  family.

[`get_prior`](@ref) lists them for a given model and data.

## Parameterisations

Group-level effects are non-centred, `r = diag(sd) * L * z` with standard
normal `z` and the Cholesky factor `L` of their correlation matrix, as in
brms's Stan code. Correlation matrices use Stan's transform to the
unconstrained scale, ordinal thresholds Stan's ordered transform, simplexes
Stan's stick-breaking transform. Families use brms's parameterisations (for
example the mean and standard deviation, rather than the location and scale,
of the skew-normal).

## Names

Parameters are named as brms names them: `b_x`, `b_sigma_x`,
`sd_g__Intercept`, `cor_g__Intercept__x`, `r_g[level,x]`, `sigma`,
`shape`, `rescor__y1__y2`, `bsp_mox`, `simo_mox1`, `sds_sx_1`, `ar[1]`, and
so on. Code written for brms output, and comparisons between the two, can
match parameters by name.

## Log densities

`lp__` in the output is Stan's: the log posterior on the unconstrained scale
with constant terms dropped and the Jacobians of the transforms included.
[`log_lik`](@ref) returns fully normalised pointwise log-likelihoods, so
`loo`, `waic` and model comparisons give the same numbers as brms.
