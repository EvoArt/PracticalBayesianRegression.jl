# Choose a family and link

The family is the distribution of the response given the predictors. Pass it
to `brm` with `family = ...`. Each family has a default link for its mean
`mu`; choose another with `link`:

```julia
brm(@bf(y ~ x), df; family = poisson())
brm(@bf(y ~ x), df; family = bernoulli(link = :probit))
brm(@bf(y ~ x), df; family = "negbinomial")      # a brms family name also works
```

## By type of response

| Response | Families |
|:---------|:---------|
| continuous | [`gaussian`](@ref), [`student`](@ref) (heavy tails), [`skew_normal`](@ref), [`exgaussian`](@ref), [`asym_laplace`](@ref) (quantile regression), [`gen_extreme_value`](@ref) |
| positive continuous | [`lognormal`](@ref), [`gammafamily`](@ref), [`weibull`](@ref), [`exponential`](@ref), [`inverse_gaussian`](@ref), [`frechet`](@ref), [`shifted_lognormal`](@ref) |
| 0 or 1 | [`bernoulli`](@ref) |
| successes out of `n` trials (`y \| trials(n)`) | [`binomial`](@ref Base.binomial()), [`beta_binomial`](@ref), [`zero_inflated_binomial`](@ref), [`zero_inflated_beta_binomial`](@ref) |
| counts | [`poisson`](@ref), [`negbinomial`](@ref), [`negbinomial2`](@ref), [`geometric`](@ref), [`com_poisson`](@ref), [`discrete_weibull`](@ref) |
| counts with extra zeros | [`zero_inflated_poisson`](@ref), [`zero_inflated_negbinomial`](@ref), [`hurdle_poisson`](@ref), [`hurdle_negbinomial`](@ref) |
| non-negative with exact zeros | [`hurdle_gamma`](@ref), [`hurdle_lognormal`](@ref) |
| proportions in (0, 1) | [`betafamily`](@ref) |
| proportions with 0 or 1 | [`zero_inflated_beta`](@ref), [`zero_one_inflated_beta`](@ref), [`xbeta`](@ref) |
| ordered categories | [`cumulative`](@ref), [`sratio`](@ref), [`cratio`](@ref), [`acat`](@ref), [`hurdle_cumulative`](@ref) |
| unordered categories | [`categorical`](@ref) |
| counts or proportions in several categories (`cbind(a, b, c) ~ x`) | [`multinomial`](@ref), [`dirichlet`](@ref dirichlet()), [`logistic_normal`](@ref) |
| angles | [`von_mises`](@ref) |
| response times with a decision | [`wiener`](@ref) |
| event times | [`cox`](@ref), and the positive families with `cens()` |

Each family's reference entry lists its parameters and links.

## Links

The link connects the linear predictor to `mu`: `mu = linkinv(eta)`. The
defaults are those of brms: `identity` for gaussian-type families, `log` for
counts and positive responses, `logit` for probabilities. Other links:

```julia
bernoulli(link = :probit)
poisson(link = :sqrt)
gammafamily(link = :identity)
cumulative(link = :cloglog)
```

The available names are `identity`, `log`, `inverse`, `sqrt`,
`inverse_squared`, `logit`, `probit`, `probit_approx`, `cloglog`, `cauchit`,
`softit`, `softplus`, `squareplus` and `tan_half`; each family accepts the
subset brms accepts for it. An unsupported link raises an error that lists
the family's links.

`gammafamily()` defaults to the log link, while brms's `Gamma()` defaults to
the inverse link. Write `gammafamily(link = :inverse)` to match a brms model
that relies on the default.

## Other parameters

The other parameters of a family (`sigma`, `shape`, `phi`, `zi`, ...) have
fixed links: `log` for positive parameters, `logit` for probabilities. Model
any of them with a formula, as in [A distributional model](@ref):

```julia
brm(@bf(y ~ x, shape ~ x), df; family = negbinomial())
brm(@bf(y ~ x, zi ~ z), df; family = zero_inflated_poisson())
```

## Mixtures

[`mixture`](@ref) combines families into a finite mixture. Each component has
its own predictors (`mu1`, `mu2`, ...) and the mixing proportions are
`theta1`, `theta2`, ...:

```julia
brm(@bf(y ~ x), df; family = mixture(gaussian(), gaussian()))
brm(@bf(y ~ 1, mu2 ~ x), df; family = mixture(gaussian(), student()))
```

When all components are of the same family, the component intercepts are
ordered so that the components are identified, as brms does by default.

## Your own family

[`custom_family`](@ref) defines a family from a log density written in
Julia, as brms's `custom_family` does from Stan code.
