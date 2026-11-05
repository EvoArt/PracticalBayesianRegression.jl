# Posterior summaries and predictions

```@meta
CurrentModule = PracticalBayesianRegression
```

## Parameters

```@docs
posterior_summary
fixef
ranef
group_coef
VarCorr
posterior_interval
hypothesis
HypothesisResult
```

## Predictions

These take `newdata` (a table with the predictors) to predict for new
observations; see [Predict for new data](@ref).

```@docs
posterior_epred
posterior_predict
posterior_linpred
log_lik
fitted_summary
predict_summary
predictive_interval
predictive_error
residual_draws
```

## Effects and smooths

```@docs
conditional_effects
posterior_smooths
conditional_smooths
pp_mixture
```

## Explained variance

```@docs
bayes_R2
loo_R2
```
