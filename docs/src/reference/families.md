# Families

```@meta
CurrentModule = PracticalBayesianRegression
```

A family is the distribution of the response. Each constructor takes an
optional `link` for the mean parameter `mu`; the other parameters use fixed
links (`log` for standard deviations, shapes and precisions, `logit` for
probabilities). Model any of them with its own formula, e.g.
`@bf(y ~ x, sigma ~ x)`. [Choose a family and link](@ref) gives an overview.

## Continuous responses

```@docs
gaussian
student
skew_normal
exgaussian
gen_extreme_value
asym_laplace
```

## Positive responses

```@docs
lognormal
shifted_lognormal
gammafamily
weibull
exponential
frechet
inverse_gaussian
```

## Binary and binomial responses

```@docs
bernoulli
Base.binomial()
beta_binomial
```

## Counts

```@docs
poisson
negbinomial
negbinomial2
geometric
discrete_weibull
com_poisson
```

## Zero-inflated and hurdle families

```@docs
zero_inflated_poisson
zero_inflated_negbinomial
zero_inflated_binomial
zero_inflated_beta_binomial
hurdle_poisson
hurdle_negbinomial
hurdle_gamma
hurdle_lognormal
```

## Proportions

```@docs
betafamily
zero_inflated_beta
zero_one_inflated_beta
xbeta
```

## Ordinal and categorical responses

```@docs
cumulative
sratio
cratio
acat
hurdle_cumulative
categorical
```

## Multi-category responses

```@docs
multinomial
dirichlet()
logistic_normal
```

## Other families

```@docs
von_mises
wiener
cox
mixture
custom_family
brmsfamily
```
