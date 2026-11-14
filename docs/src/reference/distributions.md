# Distribution functions

```@meta
CurrentModule = PracticalBayesianRegression
```

brms's density (`d`), distribution (`p`), quantile (`q`) and random-draw
(`r`) functions for its families, with brms's argument order and
parameterisations. They take scalars; broadcast them with a dot,
`dskew_normal.(x, 0, 1, 2)`. Densities take `log = false`, distribution
functions `lower_tail = true, log_p = false` (R's `lower.tail` and `log.p`),
and random draws an `rng` keyword.

## Continuous

```@docs
dstudent_t
pstudent_t
qstudent_t
rstudent_t
dskew_normal
pskew_normal
qskew_normal
rskew_normal
dexgaussian
pexgaussian
rexgaussian
dasym_laplace
pasym_laplace
qasym_laplace
rasym_laplace
dgen_extreme_value
pgen_extreme_value
qgen_extreme_value
rgen_extreme_value
```

## Positive

```@docs
dfrechet
pfrechet
qfrechet
rfrechet
dinv_gaussian
pinv_gaussian
rinv_gaussian
dshifted_lnorm
pshifted_lnorm
qshifted_lnorm
rshifted_lnorm
```

## Circular and response times

```@docs
dvon_mises
pvon_mises
rvon_mises
dwiener
rwiener
```

## Counts and proportions

```@docs
dbeta_binomial
pbeta_binomial
rbeta_binomial
dzero_inflated_poisson
pzero_inflated_poisson
dzero_inflated_negbinomial
pzero_inflated_negbinomial
dzero_inflated_binomial
pzero_inflated_binomial
dzero_inflated_beta_binomial
pzero_inflated_beta_binomial
dzero_inflated_beta
pzero_inflated_beta
dhurdle_poisson
phurdle_poisson
dhurdle_negbinomial
phurdle_negbinomial
dhurdle_gamma
phurdle_gamma
dhurdle_lognormal
phurdle_lognormal
```

## Multivariate

```@docs
ddirichlet
rdirichlet
dmulti_normal
rmulti_normal
dmulti_student_t
rmulti_student_t
dlogistic_normal
rlogistic_normal
```

## Scaled logit and other helpers

```@docs
logit_scaled
inv_logit_scaled
logm1
expp1
```
