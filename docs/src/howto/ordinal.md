# Ordinal and categorical responses

## Ordered categories

Ratings, grades and other ordered responses use an ordinal family. The
response is an integer 1, 2, ..., K, or a categorical vector whose levels are
in order (a `CategoricalArray` keeps its own level order; plain strings are
sorted).

```julia
fit = brm(@bf(rating ~ treatment + (1 | rater)), df; family = cumulative())
```

The model has a latent continuous variable cut at `K - 1` ordered thresholds,
reported as `b_Intercept[1]`, ..., `b_Intercept[K-1]`. The population-level
coefficients shift the latent variable, so a positive coefficient moves
responses towards the higher categories.

[`posterior_epred`](@ref) returns the probability of each category: an array
of draws x observations x categories.

```julia
p = posterior_epred(fit)
size(p)                 # (ndraws, nobs, K)
```

### Which ordinal family

- [`cumulative`](@ref): one latent variable and thresholds; the usual choice.
- [`sratio`](@ref), [`cratio`](@ref): sequential processes, where a response
  passes through the categories in turn.
- [`acat`](@ref): adjacent-category logits.

All take the links `logit` (default), `probit`, `probit_approx`, `cloglog`
and `cauchit`; `cumulative` and `acat` also `softit`.

### Category-specific effects

With `sratio`, `cratio` and `acat`, `cs(x)` gives `x` a separate effect at
each threshold (brms does not allow this for `cumulative`):

```julia
brm(@bf(rating ~ cs(treatment) + age), df; family = acat())
```

### Thresholds

`thres(gr = g)` estimates a separate set of thresholds for each level of `g`;
`thres(K - 1)` fixes the number of thresholds when not every category is
observed:

```julia
brm(@bf(rating | thres(gr = rater) ~ treatment), df; family = cumulative())
brm(@bf(rating | thres(6) ~ treatment), df; family = cumulative())
```

### Discrimination

A formula for `disc` models the scale of the latent variable:

```julia
brm(@bf(rating ~ treatment, disc ~ 0 + group), df; family = cumulative())
```

As in brms, `disc` is fixed at 1 unless it has a formula. Leave out the
intercept (`0 + ...`) or fix it, since the overall scale is not identified.

### Monotonic predictors

An ordinal *predictor* (education level, dose category) can enter with
`mo(x)`: its effect increases or decreases monotonically across the
categories, with free step sizes:

```julia
brm(@bf(y ~ mo(education) + age), df)
```

The coefficient `bsp_moeducation` is the average change per category; the
simplex `simo_moeducation1` holds the share of the total change at each
step.

## Unordered categories

[`categorical`](@ref) is the multinomial logit model. The response has K
categories; the first is the reference, and each other category gets its own
linear predictor, `mu2`, ..., `muK`:

```julia
fit = brm(@bf(choice ~ price), df; family = categorical())
```

Give a category its own formula with `mu3 ~ ...`. [`posterior_epred`](@ref)
returns the category probabilities, as for ordinal families.

## Counts or proportions in several categories

When each observation is a vector of counts (or proportions) over K
categories, give the columns with `cbind`:

```julia
brm(@bf(cbind(a, b, c) ~ x), df; family = multinomial())       # counts
brm(@bf(cbind(pa, pb, pc) ~ x), df; family = dirichlet())       # proportions summing to 1
brm(@bf(cbind(pa, pb, pc) ~ x), df; family = logistic_normal())
```

Every category but the first gets a predictor named after its column
(`mub`, `muc`).
