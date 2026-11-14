# Fitting

```@meta
CurrentModule = PracticalBayesianRegression
```

```@docs
PracticalBayesianRegression
```

## Fitting a model

```@docs
brm
BrmsFit
BrmsMVFit
update
brm_multiple
combine_fits
```

## Draws and sampler diagnostics

```@docs
variables(::BrmsFit)
nvariables
draws
as_draws_matrix
ndraws
nchains
niterations
rhat(::Union{BrmsFit, BrmsMVFit})
neff_ratio
nuts_params
log_posterior
prior_draws
```

## Building and sampling in two steps

```@docs
build_model
build_mv_model
sample_model
```
