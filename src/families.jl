# =============================================================================
# Response families.
#
# A family is defined by its distributional parameters ("dpars", brms's
# term), each living on a LINK scale: `mu` on the family's mean link, the
# auxiliary parameters on their own links (log for sigma/shape/phi, logm1
# for nu, logit for zi/hu). Every function takes the link-scale values as a
# tuple `eta = (eta_mu, eta_aux...)`.
#
# Per observation, a family implements:
#
#   _ll(fam, y, x, eta)            log-likelihood, data-only constants EXCLUDED,
#                                  generic in the number type (AD backends and
#                                  the tests differentiate this)
#   _pre(fam, eta)                 aux-only quantities (exp(-log sigma),
#                                  loggamma(shape), ...). When the aux
#                                  parameters are scalars this runs ONCE per
#                                  gradient instead of once per observation;
#                                  in a gaussian `y ~ x` model an exp per
#                                  observation dominates the gradient's cost
#   _ll_grad_pre(fam, y, x, m, p)  (value, d/d eta tuple), given eta_mu = m
#   _ll_const(fam, y, t)           the data-only constant (log y!, binomial
#                                  coefficient), summed once at build time
#
# `x` is per-observation family data (`_xdata`): binomial trials, log y for
# the positive families, (log(1-y), logit y) for beta. Data, so computed once
# at build time instead of a log per observation per gradient.
#
# Non-canonical links use a generic route: a family supplies `_llmu`, the
# value and derivative on the MEAN scale, and the fallbacks below chain it
# through `mu_eta`. Canonical links specialise `_ll`/`_ll_grad_pre` directly
# (the derivative simplifies and no log(exp(...)) round trip is taken).
#
# The tests check every `_ll_grad` against ForwardDiff of `_ll` and every
# normalised `_ll + _ll_const` against Distributions.jl, so a wrong hand
# derivative fails loudly instead of biasing a posterior.
# =============================================================================

abstract type Family end

# Every family is parameterised by its mean link.
struct Gaussian{L<:Link} <: Family end
struct Student{L<:Link} <: Family end
struct Lognormal{L<:Link} <: Family end
struct Bernoulli{L<:Link} <: Family end
struct Binomial{L<:Link} <: Family end
struct BetaBinomial{L<:Link} <: Family end
struct Poisson{L<:Link} <: Family end
struct NegBinomial{L<:Link} <: Family end
struct Geometric{L<:Link} <: Family end
struct GammaFamily{L<:Link} <: Family end
struct Weibull{L<:Link} <: Family end
struct Exponential{L<:Link} <: Family end
struct BetaFamily{L<:Link} <: Family end
struct ZeroInflatedPoisson{L<:Link} <: Family end
struct ZeroInflatedNegBinomial{L<:Link} <: Family end
struct ZeroInflatedBinomial{L<:Link} <: Family end
struct HurdlePoisson{L<:Link} <: Family end
struct HurdleNegBinomial{L<:Link} <: Family end
struct HurdleGamma{L<:Link} <: Family end
struct HurdleLognormal{L<:Link} <: Family end
struct Cumulative{L<:Link} <: Family end     # ordinal; see ordinal.jl
struct Sratio{L<:Link} <: Family end         # stopping ratio
struct Cratio{L<:Link} <: Family end         # continuation ratio
struct Acat{L<:Link} <: Family end           # adjacent category
const OrdinalFamily = Union{Cumulative,Sratio,Cratio,Acat}
struct Categorical{L<:Link,K} <: Family end  # K categories; see categorical.jl
# families differentiated by per-observation dual numbers (dualfamilies.jl)
abstract type DualFamily <: Family end
struct SkewNormal{L<:Link} <: DualFamily end
struct ExGaussian{L<:Link} <: DualFamily end
struct InverseGaussian{L<:Link} <: DualFamily end
struct Frechet{L<:Link} <: DualFamily end
struct ShiftedLognormal{L<:Link} <: DualFamily
    miny::Float64                           # upper bound of ndt, set from the data at build
end
ShiftedLognormal{L}() where {L} = ShiftedLognormal{L}(NaN)
struct AsymLaplace{L<:Link} <: DualFamily end          # quantile regression
struct ZeroInflatedAsymLaplace{L<:Link} <: DualFamily end
struct VonMises{L<:Link} <: DualFamily end             # circular responses
struct ZeroInflatedBeta{L<:Link} <: DualFamily end
struct ZeroOneInflatedBeta{L<:Link} <: DualFamily end
struct DiscreteWeibull{L<:Link} <: DualFamily end
struct NegBinomial2{L<:Link} <: DualFamily end         # negbinomial with sigma = 1 / shape
struct ComPoisson{L<:Link} <: DualFamily end           # Conway-Maxwell-Poisson, nu = shape
struct XBeta{L<:Link} <: DualFamily end                # beta with exact 0/1 (dualfamilies.jl)
struct ZeroInflatedBetaBinomial{L<:Link} <: DualFamily end   # extrafamilies.jl
struct DirichletFam{L<:Link,K} <: DualFamily           # see multicat.jl
    cats::Vector{String}
    Y::Matrix{Float64}
end
struct HurdleCumulative{L<:Link} <: Family end         # 0 = hurdle, then cumulative categories
struct GenExtremeValue{L<:Link} <: Family end          # see gev.jl
struct LogisticNormal{L<:Link,K} <: Family             # see multicat.jl
    cats::Vector{String}
    Y::Matrix{Float64}
end
struct Cox{L<:Link} <: Family                          # see cox.jl
    codes::Vector{Float64}       # 0 event, 1 right-censored, -1 left-censored
    lb::Vector{Float64}          # trunc(): per observation (empty: no truncation)
    ub::Vector{Float64}
end
struct Multinomial{L<:Link,K} <: Family                # see multicat.jl
    cats::Vector{String}
    Y::Matrix{Float64}
end
struct DirichletMultinomial{L<:Link,K} <: DualFamily   # see multicat.jl
    cats::Vector{String}
    Y::Matrix{Float64}
end
struct Dirichlet2{L<:Link,K} <: DualFamily             # see multicat.jl
    cats::Vector{String}
    Y::Matrix{Float64}
end


# name, default link, allowed links
const _FAMILY_TABLE = [
    (Gaussian, "gaussian", :identity, (:identity, :log, :inverse, :softplus, :squareplus, :logit, :probit, :probit_approx, :cloglog, :cauchit, :softit)),
    (Student, "student", :identity, (:identity, :log, :inverse, :softplus, :squareplus, :logit, :probit, :probit_approx, :cloglog, :cauchit, :softit)),
    (Lognormal, "lognormal", :identity, (:identity, :inverse)),
    (Bernoulli, "bernoulli", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (Binomial, "binomial", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (BetaBinomial, "beta_binomial", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity)),
    (Poisson, "poisson", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (NegBinomial, "negbinomial", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (Geometric, "geometric", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (GammaFamily, "gamma", :log, (:log, :identity, :inverse, :softplus, :squareplus)),
    (Weibull, "weibull", :log, (:log, :identity, :inverse, :softplus, :squareplus)),
    (Exponential, "exponential", :log, (:log, :identity, :inverse, :softplus, :squareplus)),
    (BetaFamily, "beta", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (ZeroInflatedPoisson, "zero_inflated_poisson", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (ZeroInflatedNegBinomial, "zero_inflated_negbinomial", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (ZeroInflatedBinomial, "zero_inflated_binomial", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (HurdlePoisson, "hurdle_poisson", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (HurdleNegBinomial, "hurdle_negbinomial", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (HurdleGamma, "hurdle_gamma", :log, (:log, :identity, :inverse, :softplus, :squareplus)),
    (HurdleLognormal, "hurdle_lognormal", :identity, (:identity, :inverse)),
    (Cumulative, "cumulative", :logit, (:logit, :probit, :probit_approx, :cloglog, :cauchit, :softit)),
    (Sratio, "sratio", :logit, (:logit, :probit, :probit_approx, :cloglog, :cauchit)),
    (Cratio, "cratio", :logit, (:logit, :probit, :probit_approx, :cloglog, :cauchit)),
    (Acat, "acat", :logit, (:logit, :probit, :probit_approx, :cloglog, :cauchit, :softit)),
    (Categorical, "categorical", :logit, (:logit,)),
    (SkewNormal, "skew_normal", :identity, (:identity, :log, :inverse, :softplus, :squareplus, :logit, :probit, :probit_approx, :cloglog, :cauchit, :softit)),
    (ExGaussian, "exgaussian", :identity, (:identity, :log, :inverse, :softplus, :squareplus)),
    (InverseGaussian, "inverse.gaussian", :inverse_squared, (:inverse_squared, :inverse, :identity, :log, :softplus, :squareplus)),
    (Frechet, "frechet", :log, (:log, :identity, :inverse, :softplus, :squareplus)),
    (ShiftedLognormal, "shifted_lognormal", :identity, (:identity, :inverse)),
    (AsymLaplace, "asym_laplace", :identity, (:identity, :log, :inverse, :softplus, :squareplus)),
    (ZeroInflatedAsymLaplace, "zero_inflated_asym_laplace", :identity, (:identity, :log, :inverse, :softplus, :squareplus)),
    (VonMises, "von_mises", :tan_half, (:tan_half, :identity)),
    (ZeroInflatedBeta, "zero_inflated_beta", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (ZeroOneInflatedBeta, "zero_one_inflated_beta", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (DiscreteWeibull, "discrete_weibull", :logit, (:logit, :probit, :probit_approx, :cloglog, :cauchit, :softit, :identity)),
    (NegBinomial2, "negbinomial2", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (ComPoisson, "com_poisson", :log, (:log, :identity, :sqrt, :softplus, :squareplus)),
    (HurdleCumulative, "hurdle_cumulative", :logit, (:logit, :probit, :probit_approx, :cloglog, :cauchit, :softit)),
    (Multinomial, "multinomial", :logit, (:logit,)),
    (Cox, "cox", :log, (:log, :identity, :softplus, :squareplus)),
    (LogisticNormal, "logistic_normal", :identity, (:identity,)),
    (GenExtremeValue, "gen_extreme_value", :identity, (:identity, :log, :inverse, :softplus, :squareplus)),
    (XBeta, "xbeta", :logit, (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log)),
    (DirichletFam, "dirichlet", :logit, (:logit,)),
    (DirichletMultinomial, "dirichlet_multinomial", :logit, (:logit,)),
    (Dirichlet2, "dirichlet2", :log, (:log,)),
]

# `link(fam)`: the link instance, by dispatch (no reflection on the hot path)
for row in _FAMILY_TABLE
    @eval link(::$(row[1]){L}) where {L} = L()
end

function _make_family(F, link)
    row = _FAMILY_TABLE[findfirst(r -> r[1] === F, _FAMILY_TABLE)]
    l = link === nothing ? row[3] : Symbol(link)
    l in row[4] || throw(ArgumentError("family $(row[2]) supports links $(join(row[4], ", ")), not `$l`"))
    # categorical's number of categories comes from the data (0 = not yet known)
    F === Categorical && return Categorical{typeof(_link(l)),0}()
    return F{typeof(_link(l))}()
end

family_name(f::Family) = _FAMILY_TABLE[findfirst(r -> f isa r[1], _FAMILY_TABLE)][2]

# "`log` (default), `identity`, ..." for a family constructor's docstring
function _links_doc(F)
    row = _FAMILY_TABLE[findfirst(r -> r[1] === F, _FAMILY_TABLE)]
    return join((l === row[3] ? "`$l` (default)" : "`$l`" for l in row[4]), ", ")
end

# brms-named constructors (keyword `link`, as brms's positional one)

"""
    gaussian(; link = :identity)

Normal distribution for continuous responses. Parameters: `mu` (mean) and
`sigma` (standard deviation).

Links for `mu`: $(_links_doc(Gaussian)).

# Examples
```julia
brm(@bf(y ~ x), data; family = gaussian())
brm(@bf(y ~ x, sigma ~ x), data; family = gaussian())   # sigma varies with x
```
"""
gaussian(; link=nothing) = _make_family(Gaussian, link)

"""
    student(; link = :identity)

Student-t distribution for continuous responses with heavy tails.
Parameters: `mu`, `sigma` (scale) and `nu` (degrees of freedom, above 1).

Links for `mu`: $(_links_doc(Student)).
"""
student(; link=nothing) = _make_family(Student, link)

"""
    lognormal(; link = :identity)

Lognormal distribution for positive responses: `log(y)` is normal with mean
`mu` and standard deviation `sigma`.

Links for `mu`: $(_links_doc(Lognormal)).
"""
lognormal(; link=nothing) = _make_family(Lognormal, link)

"""
    bernoulli(; link = :logit)

Bernoulli distribution for binary responses coded 0 and 1. Parameter: `mu`
(the probability of a 1).

Links for `mu`: $(_links_doc(Bernoulli)).
"""
bernoulli(; link=nothing) = _make_family(Bernoulli, link)

# `binomial` is Base's binomial coefficient. A zero-argument method cannot
# clash with any Base use, and keeps brms's spelling `binomial()` working.
"""
    binomial(; link = :logit)

Binomial distribution for the number of successes out of `n` trials; give the
trials with `y | trials(n)`. Parameter: `mu` (the success probability).
`binomial()` is a method of `Base.binomial`, so it needs no import.

Links for `mu`: $(_links_doc(Binomial)).

# Examples
```julia
brm(@bf(successes | trials(n) ~ x), data; family = binomial())
```
"""
Base.binomial(; link=nothing) = _make_family(Binomial, link)

"""
    beta_binomial(; link = :logit)

Beta-binomial distribution: a binomial with extra variation between
observations. Give the trials with `y | trials(n)`. Parameters: `mu` (mean
success probability) and `phi` (precision).

Links for `mu`: $(_links_doc(BetaBinomial)).
"""
beta_binomial(; link=nothing) = _make_family(BetaBinomial, link)

"""
    poisson(; link = :log)

Poisson distribution for counts. Parameter: `mu` (the mean). Use
`y | rate(t)` for an exposure `t`.

Links for `mu`: $(_links_doc(Poisson)).

# Examples
```julia
brm(@bf(count ~ x + (1 | site)), data; family = poisson())
```
"""
poisson(; link=nothing) = _make_family(Poisson, link)

"""
    negbinomial(; link = :log)

Negative binomial distribution for overdispersed counts. Parameters: `mu`
(the mean) and `shape`; the variance is `mu + mu^2 / shape`.

Links for `mu`: $(_links_doc(NegBinomial)).

See also [`negbinomial2`](@ref).
"""
negbinomial(; link=nothing) = _make_family(NegBinomial, link)

"""
    geometric(; link = :log)

Geometric distribution for counts: a negative binomial with `shape = 1`.
Parameter: `mu` (the mean).

Links for `mu`: $(_links_doc(Geometric)).
"""
geometric(; link=nothing) = _make_family(Geometric, link)

"""
    gammafamily(; link = :log)

Gamma distribution for positive responses, brms's `Gamma()` (the name
`Gamma` belongs to Distributions.jl). Parameters: `mu` (the mean) and
`shape`.

The default link is `log`; brms's `Gamma()` defaults to `inverse`, so pass
`link = :inverse` to reproduce a brms model that relies on that default.

Links for `mu`: $(_links_doc(GammaFamily)).
"""
gammafamily(; link=nothing) = _make_family(GammaFamily, link)

"""
    weibull(; link = :log)

Weibull distribution for positive responses such as survival times.
Parameters: `mu` (the mean) and `shape`.

Links for `mu`: $(_links_doc(Weibull)).
"""
weibull(; link=nothing) = _make_family(Weibull, link)

"""
    exponential(; link = :log)

Exponential distribution for positive responses. Parameter: `mu` (the mean).

Links for `mu`: $(_links_doc(Exponential)).
"""
exponential(; link=nothing) = _make_family(Exponential, link)

"""
    betafamily(; link = :logit)

Beta distribution for proportions strictly between 0 and 1, brms's `Beta()`
(the name `Beta` belongs to Distributions.jl). Parameters: `mu` (the mean)
and `phi` (precision).

Links for `mu`: $(_links_doc(BetaFamily)).

See also [`zero_inflated_beta`](@ref), [`zero_one_inflated_beta`](@ref) and
[`xbeta`](@ref) for responses that include 0 or 1.
"""
betafamily(; link=nothing) = _make_family(BetaFamily, link)

"""
    zero_inflated_poisson(; link = :log)

Poisson distribution with extra zeros. Parameters: `mu` (the Poisson mean)
and `zi` (the probability of a structural zero).

Links for `mu`: $(_links_doc(ZeroInflatedPoisson)).
"""
zero_inflated_poisson(; link=nothing) = _make_family(ZeroInflatedPoisson, link)

"""
    zero_inflated_negbinomial(; link = :log)

Negative binomial distribution with extra zeros. Parameters: `mu`, `shape`
and `zi` (the probability of a structural zero).

Links for `mu`: $(_links_doc(ZeroInflatedNegBinomial)).
"""
zero_inflated_negbinomial(; link=nothing) = _make_family(ZeroInflatedNegBinomial, link)

"""
    zero_inflated_binomial(; link = :logit)

Binomial distribution with extra zeros; give the trials with
`y | trials(n)`. Parameters: `mu` (the success probability) and `zi` (the
probability of a structural zero).

Links for `mu`: $(_links_doc(ZeroInflatedBinomial)).
"""
zero_inflated_binomial(; link=nothing) = _make_family(ZeroInflatedBinomial, link)

"""
    hurdle_poisson(; link = :log)

Hurdle model for counts: zero with probability `hu`, otherwise a
zero-truncated Poisson with parameter `mu`.

Links for `mu`: $(_links_doc(HurdlePoisson)).
"""
hurdle_poisson(; link=nothing) = _make_family(HurdlePoisson, link)

"""
    hurdle_negbinomial(; link = :log)

Hurdle model for counts: zero with probability `hu`, otherwise a
zero-truncated negative binomial with parameters `mu` and `shape`.

Links for `mu`: $(_links_doc(HurdleNegBinomial)).
"""
hurdle_negbinomial(; link=nothing) = _make_family(HurdleNegBinomial, link)

"""
    hurdle_gamma(; link = :log)

Hurdle model for non-negative continuous responses: zero with probability
`hu`, otherwise gamma with parameters `mu` and `shape`.

Links for `mu`: $(_links_doc(HurdleGamma)).
"""
hurdle_gamma(; link=nothing) = _make_family(HurdleGamma, link)

"""
    hurdle_lognormal(; link = :identity)

Hurdle model for non-negative continuous responses: zero with probability
`hu`, otherwise lognormal with parameters `mu` and `sigma`.

Links for `mu`: $(_links_doc(HurdleLognormal)).
"""
hurdle_lognormal(; link=nothing) = _make_family(HurdleLognormal, link)

"""
    cumulative(; link = :logit)

Cumulative ordinal model (ordered logit with the default link). The response
is an ordered category: integers 1, 2, ..., or a categorical vector, whose
levels are used in their own order (sorted, for plain strings). Parameters:
`mu` (the latent predictor) and the ordered thresholds, reported as
`b_Intercept[1]`, `b_Intercept[2]`, ... A formula `disc ~ ...` models the
discrimination.

Links: $(_links_doc(Cumulative)).
"""
cumulative(; link=nothing) = _make_family(Cumulative, link)

"""
    sratio(; link = :logit)

Stopping-ratio (sequential) ordinal model. Response and parameters as for
[`cumulative`](@ref); `cs(x)` gives category-specific effects.

Links: $(_links_doc(Sratio)).
"""
sratio(; link=nothing) = _make_family(Sratio, link)

"""
    cratio(; link = :logit)

Continuation-ratio ordinal model. Response and parameters as for
[`cumulative`](@ref); `cs(x)` gives category-specific effects.

Links: $(_links_doc(Cratio)).
"""
cratio(; link=nothing) = _make_family(Cratio, link)

"""
    acat(; link = :logit)

Adjacent-category ordinal model. Response and parameters as for
[`cumulative`](@ref); `cs(x)` gives category-specific effects.

Links: $(_links_doc(Acat)).
"""
acat(; link=nothing) = _make_family(Acat, link)

"""
    categorical(; link = :logit)

Categorical (multinomial logit) model for an unordered response with `K`
categories, coded 1, ..., K or as a categorical vector. The first category is
the reference; categories 2 to K get predictors `mu2`, ..., `muK`, which
share the main formula unless given their own (`mu3 ~ z`).
"""
categorical(; link=nothing) = _make_family(Categorical, link)

"""
    skew_normal(; link = :identity)

Skew-normal distribution for continuous responses, in brms's
parameterisation: `mu` (the mean), `sigma` (the standard deviation) and
`alpha` (skewness).
"""
skew_normal(; link=nothing) = _make_family(SkewNormal, link)

"""
    exgaussian(; link = :identity)

Exponentially modified Gaussian distribution, often used for reaction times.
Parameters: `mu` (the mean), `sigma` (the normal component's standard
deviation) and `beta` (the exponential component's scale).
"""
exgaussian(; link=nothing) = _make_family(ExGaussian, link)

"""
    inverse_gaussian(; link = :inverse_squared)

Inverse Gaussian distribution for positive responses, brms's
`inverse.gaussian()`. Parameters: `mu` (the mean) and `shape`.

Links for `mu`: $(_links_doc(InverseGaussian)).
"""
inverse_gaussian(; link=nothing) = _make_family(InverseGaussian, link)

"""
    frechet(; link = :log)

Fréchet distribution for positive responses. Parameters: `mu` (the mean) and
`nu` (shape).
"""
frechet(; link=nothing) = _make_family(Frechet, link)

"""
    shifted_lognormal(; link = :identity)

Lognormal distribution shifted to the right by `ndt`, often used for reaction
times. Parameters: `mu`, `sigma` (as in [`lognormal`](@ref)) and `ndt` (the
shift, between 0 and the smallest response).
"""
shifted_lognormal(; link=nothing) = _make_family(ShiftedLognormal, link)

"""
    asym_laplace(; link = :identity)

Asymmetric Laplace distribution, for quantile regression. Parameters: `mu`
(the `quantile`-th quantile of the response), `sigma` (scale) and `quantile`
(in (0, 1)). Fix the quantile with a constant prior, e.g.
`prior("constant(0.9)"; class = :quantile)`.

Links for `mu`: $(_links_doc(AsymLaplace)).
"""
asym_laplace(; link=nothing) = _make_family(AsymLaplace, link)

"""
    zero_inflated_asym_laplace(; link = :identity)

Asymmetric Laplace (quantile regression) with extra exact zeros, brms's
`zero_inflated_asym_laplace()`. Parameters: `mu`, `sigma`, `quantile` (as in
[`asym_laplace`](@ref)) and `zi`, the probability of an extra zero.

Links for `mu`: $(_links_doc(ZeroInflatedAsymLaplace)).
"""
zero_inflated_asym_laplace(; link=nothing) = _make_family(ZeroInflatedAsymLaplace, link)

"""
    von_mises(; link = :tan_half)

Von Mises distribution for angles in [-π, π]. Parameters: `mu` (the mean
direction) and `kappa` (concentration).

Links for `mu`: $(_links_doc(VonMises)).
"""
von_mises(; link=nothing) = _make_family(VonMises, link)

"""
    zero_inflated_beta(; link = :logit)

Beta distribution with extra zeros, for proportions in [0, 1). Parameters:
`mu`, `phi` (as in [`betafamily`](@ref)) and `zi` (the probability of a zero).

Links for `mu`: $(_links_doc(ZeroInflatedBeta)).
"""
zero_inflated_beta(; link=nothing) = _make_family(ZeroInflatedBeta, link)

"""
    zero_one_inflated_beta(; link = :logit)

Beta distribution with extra zeros and ones, for proportions in [0, 1].
Parameters: `mu`, `phi` (as in [`betafamily`](@ref)), `zoi` (the probability
of a 0 or 1) and `coi` (the probability that such a value is 1).

Links for `mu`: $(_links_doc(ZeroOneInflatedBeta)).
"""
zero_one_inflated_beta(; link=nothing) = _make_family(ZeroOneInflatedBeta, link)

"""
    discrete_weibull(; link = :logit)

Discrete Weibull distribution for counts. Parameters: `mu` (in (0, 1)) and
`shape`.
"""
discrete_weibull(; link=nothing) = _make_family(DiscreteWeibull, link)

"""
    negbinomial2(; link = :log)

Negative binomial distribution parameterised by `mu` (the mean) and
`sigma = 1 / shape`, so that `sigma` near 0 approaches the Poisson.

Links for `mu`: $(_links_doc(NegBinomial2)).
"""
negbinomial2(; link=nothing) = _make_family(NegBinomial2, link)

"""
    com_poisson(; link = :log)

Conway-Maxwell-Poisson distribution for counts that are under- or
overdispersed. Parameters: `mu` and `shape`.
"""
com_poisson(; link=nothing) = _make_family(ComPoisson, link)

"""
    hurdle_cumulative(; link = :logit)

Ordinal responses with a hurdle: 0 with probability `hu`, otherwise a
cumulative-model category 1, 2, ... (see [`cumulative`](@ref)).

Links: $(_links_doc(HurdleCumulative)).
"""
hurdle_cumulative(; link=nothing) = _make_family(HurdleCumulative, link)

"""
    brmsfamily(name; link = nothing)

A family by its brms name, given as a string or symbol: `"gaussian"`,
`"gamma"`, `"beta"`, `"inverse.gaussian"`, `"zero_inflated_poisson"`, and so
on for every family in this package, or a mixture such as
`"mixture(gaussian, student)"`. `link` is a link name; `nothing` takes the
family's default.

```julia
brmsfamily("gamma"; link = "inverse")     # brms's Gamma()
```
"""
function brmsfamily(name; link=nothing)
    n = lowercase(string(name))
    # "mixture(gaussian, student)": a mixture of named families
    if startswith(n, "mixture(") && endswith(n, ")")
        link === nothing || throw(ArgumentError("give links per component: mixture(gaussian(link = ...), ...)"))
        return mixture((brmsfamily(strip(c)) for c in split(n[9:(end - 1)], ","))...)
    end
    # families with their own constructors (defined in later files)
    n == "wiener" && return wiener(; link=link)
    n == "zero_inflated_beta_binomial" && return zero_inflated_beta_binomial(; link=link)
    k = findfirst(r -> r[2] == n, _FAMILY_TABLE)
    k === nothing && throw(ArgumentError("family `$name` is not supported yet"))
    return _make_family(_FAMILY_TABLE[k][1], link)
end

# --- distributional parameters ----------------------------------------------

dpars(::Union{Gaussian,Lognormal}) = (:mu, :sigma)
dpars(::Student) = (:mu, :sigma, :nu)
dpars(::Union{Bernoulli,Binomial,Poisson,Geometric,Exponential}) = (:mu,)
dpars(::Union{NegBinomial,GammaFamily,Weibull}) = (:mu, :shape)
dpars(::Union{BetaBinomial,BetaFamily}) = (:mu, :phi)
dpars(::Union{ZeroInflatedPoisson,ZeroInflatedBinomial}) = (:mu, :zi)
dpars(::ZeroInflatedNegBinomial) = (:mu, :shape, :zi)
dpars(::HurdlePoisson) = (:mu, :hu)
dpars(::Union{HurdleNegBinomial,HurdleGamma}) = (:mu, :shape, :hu)
dpars(::HurdleLognormal) = (:mu, :sigma, :hu)
# `thres` is internal: the thresholds, which come from the formula's
# intercept rather than a formula of their own
dpars(::OrdinalFamily) = (:mu, :thres)

"""
    aux_bounds(fam, dpar) -> (link, lb)

How an auxiliary parameter maps to its link scale: `(:log, lb)` means
`x = lb + exp(eta)`; `(:logit, 0)` means `x = logistic(eta)` on (0, 1).
"""
aux_bounds(::Family, d::Symbol) = d === :nu ? (:log, 1.0) : d in (:zi, :hu, :zoi, :coi, :quantile) ? (:logit, 0.0) : (:log, 0.0)
dpar_link_name(fam::Family, d::Symbol) = d === :mu ? link_name(link(fam)) : d === :nu ? "logm1" : string(aux_bounds(fam, d)[1])

mu_link(fam::Family) = Symbol(link_name(link(fam)))

# How brms transforms the response for its default intercept prior: the
# family's mean link, except that "log-scale" families use log.
prior_link(fam::Family) = mu_link(fam)
prior_link(::Union{Lognormal,HurdleLognormal}) = :log

uses_trials(::Family) = false
uses_trials(::Union{Binomial,BetaBinomial,ZeroInflatedBinomial}) = true

const _LOG_2PI_HALF = 0.5 * log(2π)

# --- generic machinery ----------------------------------------------------------

@inline _ll_grad(fam::Family, y, x, eta) = _ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta))
@inline _pre(::Family, eta) = nothing

# non-canonical links: mean-scale value + derivative, chained through mu_eta
@inline _ll(fam::Family, y, x, eta) = first(_llmu(fam, y, x, linkinv(link(fam), eta[1]), _pre(fam, eta)))
@inline function _ll_grad_pre(fam::Family, y, x, e, p)
    L = link(fam)
    v, g = _llmu(fam, y, x, linkinv(L, e), p)
    return v, (g[1] * mu_eta(L, e), Base.tail(g)...)
end

_xdata(::Family, y, trials) = Float64[]
_xdata(::Union{Binomial,BetaBinomial,ZeroInflatedBinomial}, y, trials) = Float64.(trials)
_xdata(::Union{Lognormal,GammaFamily,Weibull}, y, trials) = log.(y)
_xdata(::Union{HurdleGamma,HurdleLognormal}, y, trials) = [v > 0 ? log(v) : 0.0 for v in y]
_xdata(::BetaFamily, y, trials) = [(log1p(-v), log(v) - log1p(-v)) for v in y]

# zi / hu on the logit scale: log(pi), log(1 - pi), pi
@inline _logit_parts(z) = (-log1pexp(-z), -log1pexp(z), logistic(z))

# =============================================================================
# Continuous families
# =============================================================================

# --- gaussian: sigma log ------------------------------------------------------
@inline _pre(::Gaussian, eta) = (ls=eta[2], is=exp(-eta[2]))
@inline function _llmu(::Gaussian, y, x, mu, p)
    r = (y - mu) * p.is
    r2 = r * r
    return -r2 / 2 - p.ls, (r * p.is, r2 - 1)
end
_ll_const(::Gaussian, y, t) = -_LOG_2PI_HALF

# --- student: sigma log, nu logm1 (nu = 1 + exp(eta)) ---------------------------
@inline function _pre(::Student, eta)
    ls, ln = eta[2], eta[3]
    en = exp(ln)
    nu = 1 + en
    c0 = loggamma((nu + 1) / 2) - loggamma(nu / 2) - log(nu) / 2 - ls
    d0 = (digamma((nu + 1) / 2) - digamma(nu / 2)) / 2 - 1 / (2nu)
    return (is=exp(-ls), nu=nu, en=en, c0=c0, d0=d0)
end
@inline function _llmu(::Student, y, x, mu, p)
    nu = p.nu
    e = y - mu
    r2 = (e * p.is)^2
    q = nu + r2
    l1 = log1p(r2 / nu)
    v = p.c0 - (nu + 1) / 2 * l1
    g_mu = (nu + 1) * e * p.is * p.is / q
    g_ls = -1 + (nu + 1) * r2 / q
    dnu = p.d0 - l1 / 2 + (nu + 1) * r2 / (2nu * q)
    return v, (g_mu, g_ls, dnu * p.en)
end
_ll_const(::Student, y, t) = -log(π) / 2

# --- lognormal: mu, sigma of log(y); x = log y ----------------------------------
@inline _pre(::Lognormal, eta) = (ls=eta[2], is=exp(-eta[2]))
@inline function _llmu(::Lognormal, y, ly, mu, p)
    r = (ly - mu) * p.is
    r2 = r * r
    return -r2 / 2 - p.ls, (r * p.is, r2 - 1)
end
_ll_const(::Lognormal, y, t) = -log(y) - _LOG_2PI_HALF

# --- gamma: mean mu, shape alpha (log); x = log y -------------------------------
@inline function _pre(::GammaFamily, eta)
    la = eta[2]
    a = exp(la)
    return (la=la, a=a, lg=loggamma(a), dg=digamma(a))
end
@inline _ll(fam::GammaFamily{LogLink}, y, ly, eta) = first(_ll_grad_pre(fam, y, ly, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::GammaFamily{LogLink}, y, ly, e, p)
    a, la = p.a, p.la
    q = y * exp(-e)                 # y / mu
    v = a * (la - e) - p.lg + (a - 1) * ly - a * q
    return v, (a * (q - 1), a * (la + 1 - e - p.dg + ly - q))
end
@inline function _llmu(::GammaFamily, y, ly, mu, p)
    a, la = p.a, p.la
    lmu = _slog(mu)
    q = y / mu
    v = a * (la - lmu) - p.lg + (a - 1) * ly - a * q
    return v, (a * (q - 1) / mu, a * (la + 1 - lmu - p.dg + ly - q))
end
_ll_const(::GammaFamily, y, t) = 0.0

# --- weibull: mean mu (log), shape k (log); x = log y ---------------------------
# brms: y ~ weibull(k, lambda), lambda = mu / Gamma(1 + 1/k), so mu is the mean.
@inline function _pre(::Weibull, eta)
    lk = eta[2]
    k = exp(lk)
    return (lk=lk, k=k, lg1=loggamma(1 + 1 / k), dg1=digamma(1 + 1 / k))
end
@inline _ll(fam::Weibull{LogLink}, y, ly, eta) = first(_ll_grad_pre(fam, y, ly, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::Weibull{LogLink}, y, ly, e, p)
    k = p.k
    llam = e - p.lg1
    d = ly - llam
    z = exp(k * d)                  # (y / lambda)^k
    v = p.lk - k * llam + (k - 1) * ly - z
    dllam = k * (z - 1)             # d/d log(lambda)
    # log(lambda) depends on k too: d log(lambda)/dk = digamma(1 + 1/k) / k^2
    dk = 1 / k + d - z * d + dllam * p.dg1 / (k * k)
    return v, (dllam, dk * k)
end
_ll_const(::Weibull, y, t) = 0.0

# --- exponential: mean mu (log) -------------------------------------------------
@inline _ll(::Exponential{LogLink}, y, x, eta) = -eta[1] - y * exp(-eta[1])
@inline function _ll_grad_pre(::Exponential{LogLink}, y, x, e, p)
    q = y * exp(-e)
    return -e - q, (q - 1,)
end
_ll_const(::Exponential, y, t) = 0.0

# --- beta: mean mu, precision phi (log); x = (log(1-y), logit y) -----------------
@inline function _pre(::BetaFamily, eta)
    phi = exp(eta[2])
    return (phi=phi, lg=loggamma(phi), dg=digamma(phi))
end
@inline function _llmu(::BetaFamily, y, x, mu, p)
    phi = p.phi
    a = mu * phi
    b = phi - a
    l1y, lgt = x
    v = p.lg - loggamma(a) - loggamma(b) + phi * (l1y + mu * lgt)
    da, db = digamma(a), digamma(b)
    dmu = phi * (db - da + lgt)
    dphi = p.dg - mu * da - (1 - mu) * db + l1y + mu * lgt
    return v, (dmu, dphi * phi)
end
_ll_const(::BetaFamily, y, t) = -(log(y) + log1p(-y))

# =============================================================================
# Binary / binomial families
# =============================================================================

# --- bernoulli ------------------------------------------------------------------
# log1pexp and logistic share one exp(-|eta|)
@inline function _log1pexp_logistic(e)
    x = exp(-abs(e))
    l = max(e, zero(e)) + log1p(x)
    s = ifelse(e >= 0, 1 / (1 + x), x / (1 + x))
    return l, s
end
@inline _ll(::Bernoulli{LogitLink}, y, x, eta) = y * eta[1] - log1pexp(eta[1])
@inline function _ll_grad_pre(::Bernoulli{LogitLink}, y, x, e, p)
    l, s = _log1pexp_logistic(e)
    return y * e - l, (y - s,)
end
@inline _ll(::Bernoulli{ProbitLink}, y, x, eta) = y * normlogcdf(eta[1]) + (1 - y) * normlogcdf(-eta[1])
# one log-cdf and one exp per observation: y is 0 or 1, so only one tail
# is ever needed (computing both, plus each Mills ratio's own log-cdf, was
# four log-cdf calls, several times the cost)
@inline function _ll_grad_pre(fam::Bernoulli{ProbitLink}, y, x, e, p)
    s = y > 0 ? e : -e                 # log Phi(s) is the log-likelihood
    lc = normlogcdf(s)
    r = exp(normlogpdf(s) - lc)        # Mills ratio phi(s) / Phi(s)
    return lc, (y > 0 ? r : -r,)
end
# cloglog: mu = 1 - exp(-exp(eta)); log(1 - mu) = -exp(eta) exactly
@inline _ll(::Bernoulli{CloglogLink}, y, x, eta) = y * log(-expm1(-exp(eta[1]))) - (1 - y) * exp(eta[1])
@inline function _ll_grad_pre(fam::Bernoulli{CloglogLink}, y, x, e, p)
    ee = exp(e)
    m = -expm1(-ee)
    return y * log(m) - (1 - y) * ee, (y * exp(e - ee) / m - (1 - y) * ee,)
end
_ll_const(::Bernoulli, y, t) = 0.0
# any other link (brms: probit_approx, cauchit, softit, identity, log): the mean-scale route
@inline _llmu(::Bernoulli, y, x, mu, p) = (y * _slog(mu) + (1 - y) * _slog(1 - mu), (y / mu - (1 - y) / (1 - mu),))
@inline _llmu(::Binomial, y, t, mu, p) = (y * _slog(mu) + (t - y) * _slog(1 - mu), (y / mu - (t - y) / (1 - mu),))

# --- binomial: x = trials --------------------------------------------------------
@inline _ll(::Binomial{LogitLink}, y, t, eta) = y * eta[1] - t * log1pexp(eta[1])
@inline function _ll_grad_pre(::Binomial{LogitLink}, y, t, e, p)
    l, s = _log1pexp_logistic(e)
    return y * e - t * l, (y - t * s,)
end
@inline _ll(::Binomial{ProbitLink}, y, t, eta) = y * normlogcdf(eta[1]) + (t - y) * normlogcdf(-eta[1])
@inline function _ll_grad_pre(fam::Binomial{ProbitLink}, y, t, e, p)
    lp, lq = normlogcdf(e), normlogcdf(-e)
    lphi = normlogpdf(e)
    return y * lp + (t - y) * lq, (y * exp(lphi - lp) - (t - y) * exp(lphi - lq),)
end
@inline _ll(::Binomial{CloglogLink}, y, t, eta) = y * log(-expm1(-exp(eta[1]))) - (t - y) * exp(eta[1])
@inline function _ll_grad_pre(fam::Binomial{CloglogLink}, y, t, e, p)
    ee = exp(e)
    m = -expm1(-ee)
    return y * log(m) - (t - y) * ee, (y * exp(e - ee) / m - (t - y) * ee,)
end
_ll_const(::Union{Binomial,BetaBinomial}, y, t) = logabsbinomial(Int(t), Int(y))[1]

# --- beta-binomial: mean mu, precision phi (log); x = trials ---------------------
@inline function _pre(::BetaBinomial, eta)
    phi = exp(eta[2])
    return (phi=phi, lg=loggamma(phi), dg=digamma(phi))
end
@inline function _llmu(::BetaBinomial, y, t, mu, p)
    phi = p.phi
    a = mu * phi
    b = phi - a
    v = loggamma(y + a) + loggamma(t - y + b) - loggamma(t + phi) - loggamma(a) - loggamma(b) + p.lg
    da = digamma(y + a) - digamma(a)
    db = digamma(t - y + b) - digamma(b)
    dphi = mu * da + (1 - mu) * db + p.dg - digamma(t + phi)
    return v, (phi * (da - db), dphi * phi)
end

# =============================================================================
# Count families
# =============================================================================

# --- poisson ----------------------------------------------------------------------
@inline _ll(::Poisson{LogLink}, y, x, eta) = y * eta[1] - exp(eta[1])
@inline function _ll_grad_pre(::Poisson{LogLink}, y, x, e, p)
    m = exp(e)
    return y * e - m, (y - m,)
end
@inline _llmu(::Poisson, y, x, mu, p) = (y * _slog(mu) - mu, (y / mu - 1,))
_ll_const(::Poisson, y, t) = -loggamma(y + 1)

# --- negative binomial: mean mu, shape phi (log), variance mu + mu^2/phi ---------
@inline function _pre(::Union{NegBinomial,ZeroInflatedNegBinomial,HurdleNegBinomial}, eta)
    lp = eta[2]
    phi = exp(lp)
    return (lp=lp, phi=phi, lg=loggamma(phi), dg=digamma(phi))
end

# loggamma(y + phi) - loggamma(phi) and digamma(y + phi) - digamma(phi).
# For integer y these are the finite sums of log(phi + k) and 1/(phi + k),
# k = 0..y-1, which beat two special-function calls for small counts — the
# usual case for count data. Large y falls back to the special functions.
@inline function _nb_gamma_terms(y, phi, lg, dg)
    if y < 8
        a = zero(phi); b = zero(phi)
        for k in 0:(Int(y) - 1)
            a += log(phi + k)
            b += 1 / (phi + k)
        end
        return a, b
    end
    return loggamma(y + phi) - lg, digamma(y + phi) - dg
end
# value and (d/d eta, d/d log phi) of the NB log-pmf, log link
@inline function _nb_log(y, e, p)
    phi, lp = p.phi, p.lp
    lmp = logaddexp(e, lp)          # log(mu + phi)
    mu = exp(e)
    lgd, dgd = _nb_gamma_terms(y, phi, p.lg, p.dg)
    v = lgd + phi * (lp - lmp) + y * (e - lmp)
    g_e = phi * (y - mu) / (mu + phi)
    dphi = dgd + lp - lmp + 1 - (y + phi) / (mu + phi)
    return v, g_e, dphi * phi
end
# log P(y = 0) and its derivatives: phi * (log phi - log(mu + phi))
@inline function _nb_log0(e, p)
    phi, lp = p.phi, p.lp
    lmp = logaddexp(e, lp)
    mu = exp(e)
    l0 = phi * (lp - lmp)
    return l0, -phi * mu / (mu + phi), phi * (lp - lmp + 1 - phi / (mu + phi))
end
@inline _ll(fam::NegBinomial{LogLink}, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::NegBinomial{LogLink}, y, x, e, p)
    v, ge, gp = _nb_log(y, e, p)
    return v, (ge, gp)
end
@inline function _llmu(::NegBinomial, y, x, mu, p)
    phi, lp = p.phi, p.lp
    lmp = _slog(mu + phi)
    lgd, dgd = _nb_gamma_terms(y, phi, p.lg, p.dg)
    v = lgd + phi * (lp - lmp) + y * (_slog(mu) - lmp)
    dmu = y / mu - (y + phi) / (mu + phi)
    dphi = dgd + lp - lmp + 1 - (y + phi) / (mu + phi)
    return v, (dmu, dphi * phi)
end
_ll_const(::NegBinomial, y, t) = -loggamma(y + 1)
# NB with shape 1: the y! terms cancel, the pmf is mu^y / (1 + mu)^(y + 1)
_ll_const(::Geometric, y, t) = 0.0

# --- geometric: negative binomial with shape 1 -------------------------------------
@inline _ll(::Geometric{LogLink}, y, x, eta) = y * eta[1] - (y + 1) * log1pexp(eta[1])
@inline function _ll_grad_pre(::Geometric{LogLink}, y, x, e, p)
    l, s = _log1pexp_logistic(e)
    return y * e - (y + 1) * l, (y - (y + 1) * s,)
end

# =============================================================================
# Zero-inflated and hurdle families
#
# With pi = zi (or hu) on the logit scale z, and a base distribution with
# log P(0) = l0:
#   zero-inflated, y = 0:  log(pi + (1 - pi) P(0)) = logsumexp(log pi, log(1-pi) + l0)
#                          with weights wA = pi / (...), wB = 1 - wA:
#                          d/d base = wB * d l0,   d/dz = wA (1 - pi) - wB pi
#   zero-inflated, y > 0:  log(1 - pi) + base(y);  d/dz = -pi
#   hurdle, y = 0:         log pi;                  d/dz = 1 - pi
#   hurdle, y > 0:         log(1 - pi) + base(y) - log(1 - P(0))
#                          d/d base adds  dl0 / expm1(-l0)
# =============================================================================

@inline _pre(::Union{ZeroInflatedPoisson,ZeroInflatedBinomial,HurdlePoisson}, eta) = _logit_parts(eta[2])

@inline _ll(fam::ZeroInflatedPoisson{LogLink}, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::ZeroInflatedPoisson{LogLink}, y, x, e, p)
    lpi, l1pi, pi = p
    lam = exp(e)
    if y == 0
        lb = l1pi - lam
        v = logaddexp(lpi, lb)
        wA = exp(lpi - v); wB = 1 - wA
        return v, (-lam * wB, wA * (1 - pi) - wB * pi)
    end
    return l1pi + y * e - lam, (y - lam, -pi)
end
_ll_const(::Union{ZeroInflatedPoisson,HurdlePoisson,ZeroInflatedNegBinomial,HurdleNegBinomial}, y, t) = -loggamma(y + 1)

@inline _ll(fam::ZeroInflatedBinomial{LogitLink}, y, t, eta) = first(_ll_grad_pre(fam, y, t, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::ZeroInflatedBinomial{LogitLink}, y, t, e, p)
    lpi, l1pi, pi = p
    l, s = _log1pexp_logistic(e)
    if y == 0
        lb = l1pi - t * l                 # log(1-pi) + t log(1 - p)
        v = logaddexp(lpi, lb)
        wA = exp(lpi - v); wB = 1 - wA
        return v, (-t * s * wB, wA * (1 - pi) - wB * pi)
    end
    return l1pi + y * e - t * l, (y - t * s, -pi)
end
_ll_const(::ZeroInflatedBinomial, y, t) = logabsbinomial(Int(t), Int(y))[1]

@inline function _pre(::ZeroInflatedNegBinomial, eta)
    lp = eta[2]; phi = exp(lp)
    return (lp=lp, phi=phi, lg=loggamma(phi), dg=digamma(phi), z=_logit_parts(eta[3]))
end
@inline _ll(fam::ZeroInflatedNegBinomial{LogLink}, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::ZeroInflatedNegBinomial{LogLink}, y, x, e, p)
    lpi, l1pi, pi = p.z
    if y == 0
        l0, d0e, d0p = _nb_log0(e, p)
        v = logaddexp(lpi, l1pi + l0)
        wA = exp(lpi - v); wB = 1 - wA
        return v, (wB * d0e, wB * d0p, wA * (1 - pi) - wB * pi)
    end
    v, ge, gp = _nb_log(y, e, p)
    return l1pi + v, (ge, gp, -pi)
end

@inline _ll(fam::HurdlePoisson{LogLink}, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::HurdlePoisson{LogLink}, y, x, e, p)
    lpi, l1pi, pi = p
    y == 0 && return lpi, (zero(e), 1 - pi)
    lam = exp(e)
    # log(1 - P(0)) = log(1 - exp(-lam)); its derivative w.r.t. eta is lam / expm1(lam)
    v = l1pi + y * e - lam - log1mexp(-lam)
    return v, (y - lam - lam / expm1(lam), -pi)
end

@inline function _pre(::HurdleNegBinomial, eta)
    lp = eta[2]; phi = exp(lp)
    return (lp=lp, phi=phi, lg=loggamma(phi), dg=digamma(phi), z=_logit_parts(eta[3]))
end
@inline _ll(fam::HurdleNegBinomial{LogLink}, y, x, eta) = first(_ll_grad_pre(fam, y, x, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::HurdleNegBinomial{LogLink}, y, x, e, p)
    lpi, l1pi, pi = p.z
    y == 0 && return lpi, (zero(e), zero(e), 1 - pi)
    v, ge, gp = _nb_log(y, e, p)
    l0, d0e, d0p = _nb_log0(e, p)
    r = 1 / expm1(-l0)              # P0 / (1 - P0)
    return l1pi + v - log1mexp(l0), (ge + r * d0e, gp + r * d0p, -pi)
end

@inline function _pre(::HurdleGamma, eta)
    la = eta[2]; a = exp(la)
    return (la=la, a=a, lg=loggamma(a), dg=digamma(a), z=_logit_parts(eta[3]))
end
@inline _ll(fam::HurdleGamma{LogLink}, y, ly, eta) = first(_ll_grad_pre(fam, y, ly, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::HurdleGamma{LogLink}, y, ly, e, p)
    lpi, l1pi, pi = p.z
    y == 0 && return lpi, (zero(e), zero(e), 1 - pi)
    a, la = p.a, p.la
    q = y * exp(-e)
    v = a * (la - e) - p.lg + (a - 1) * ly - a * q
    return l1pi + v, (a * (q - 1), a * (la + 1 - e - p.dg + ly - q), -pi)
end
_ll_const(::HurdleGamma, y, t) = 0.0

@inline _pre(::HurdleLognormal, eta) = (ls=eta[2], is=exp(-eta[2]), z=_logit_parts(eta[3]))
@inline _ll(fam::HurdleLognormal{IdentityLink}, y, ly, eta) = first(_ll_grad_pre(fam, y, ly, eta[1], _pre(fam, eta)))
@inline function _ll_grad_pre(::HurdleLognormal{IdentityLink}, y, ly, mu, p)
    lpi, l1pi, pi = p.z
    y == 0 && return lpi, (zero(mu), zero(mu), 1 - pi)
    r = (ly - mu) * p.is
    r2 = r * r
    return l1pi - r2 / 2 - p.ls, (r * p.is, r2 - 1, -pi)
end
_ll_const(::HurdleLognormal, y, t) = y > 0 ? -log(y) - _LOG_2PI_HALF : 0.0

# =============================================================================
# Non-default links (brms 2.23's link lists): the mean-scale route. Each
# `_llmu` is the family's log density and its derivatives at the MEAN mu; the
# generic `_ll` / `_ll_grad_pre` above chain d/d mu through mu_eta. The
# hand-coded default-link methods are more specific, so they keep the fast
# path. A mean outside the support (identity link) is -Inf, a rejected
# proposal, as Stan's argument checks.
# =============================================================================

@inline function _llmu(::Geometric, y, x, mu, p)
    mu > 0 || return oftype(mu, -Inf), (zero(mu),)
    return y * log(mu) - (y + 1) * log1p(mu), (y / mu - (y + 1) / (1 + mu),)
end

# weibull / exponential in terms of log mu; d/d mu = d/d log mu / mu
@inline function _llmu(fam::Weibull, y, ly, mu, p)
    mu > 0 || return oftype(mu, -Inf), (zero(mu), zero(mu))
    v, (gl, gk) = _ll_grad_pre(Weibull{LogLink}(), y, ly, log(mu), p)
    return v, (gl / mu, gk)
end
@inline function _llmu(::Exponential, y, x, mu, p)
    mu > 0 || return oftype(mu, -Inf), (zero(mu),)
    q = y / mu
    return -log(mu) - q, ((q - 1) / mu,)
end

# zero-inflated: see the identities above the zero-inflated section
@inline function _llmu(::ZeroInflatedPoisson, y, x, mu, p)
    lpi, l1pi, pi = p
    mu > 0 || return oftype(mu, -Inf), (zero(mu), zero(mu))
    if y == 0
        v = logaddexp(lpi, l1pi - mu)
        wA = exp(lpi - v); wB = 1 - wA
        return v, (-wB, wA * (1 - pi) - wB * pi)
    end
    return l1pi + y * log(mu) - mu, (y / mu - 1, -pi)
end
@inline function _llmu(::ZeroInflatedBinomial, y, t, mu, p)
    lpi, l1pi, pi = p
    0 < mu < 1 || return oftype(mu, -Inf), (zero(mu), zero(mu))
    l1m = log1p(-mu)
    if y == 0
        v = logaddexp(lpi, l1pi + t * l1m)
        wA = exp(lpi - v); wB = 1 - wA
        return v, (-t / (1 - mu) * wB, wA * (1 - pi) - wB * pi)
    end
    return l1pi + y * log(mu) + (t - y) * l1m, (y / mu - (t - y) / (1 - mu), -pi)
end
# NB at mean mu: log P(0) = phi (log phi - log(mu + phi)) and its derivatives
# (d/d mu, d/d log phi)
@inline function _nb_log0_mu(mu, p)
    phi, lp = p.phi, p.lp
    lmp = log(mu + phi)
    return phi * (lp - lmp), -phi / (mu + phi), phi * (lp - lmp + 1 - phi / (mu + phi))
end
@inline function _llmu(::ZeroInflatedNegBinomial, y, x, mu, p)
    lpi, l1pi, pi = p.z
    mu > 0 || return oftype(mu, -Inf), (zero(mu), zero(mu), zero(mu))
    if y == 0
        l0, d0m, d0p = _nb_log0_mu(mu, p)
        v = logaddexp(lpi, l1pi + l0)
        wA = exp(lpi - v); wB = 1 - wA
        return v, (wB * d0m, wB * d0p, wA * (1 - pi) - wB * pi)
    end
    v, (gm, gp) = _llmu(NegBinomial{LogLink}(), y, x, mu, p)
    return l1pi + v, (gm, gp, -pi)
end

# hurdle: y = 0 is log hu whatever mu (as brms's lpmf, which does not
# evaluate the count part there)
@inline function _llmu(::HurdlePoisson, y, x, mu, p)
    lpi, l1pi, pi = p
    y == 0 && return lpi, (zero(mu), 1 - pi)
    mu > 0 || return oftype(mu, -Inf), (zero(mu), zero(mu))
    # log(1 - P(0)) = log(1 - exp(-mu)); d/d mu = 1 / expm1(mu)
    return l1pi + y * log(mu) - mu - log1mexp(-mu), (y / mu - 1 - 1 / expm1(mu), -pi)
end
@inline function _llmu(::HurdleNegBinomial, y, x, mu, p)
    lpi, l1pi, pi = p.z
    y == 0 && return lpi, (zero(mu), zero(mu), 1 - pi)
    mu > 0 || return oftype(mu, -Inf), (zero(mu), zero(mu), zero(mu))
    v, (gm, gp) = _llmu(NegBinomial{LogLink}(), y, x, mu, p)
    l0, d0m, d0p = _nb_log0_mu(mu, p)
    r = 1 / expm1(-l0)                  # P0 / (1 - P0)
    return l1pi + v - log1mexp(l0), (gm + r * d0m, gp + r * d0p, -pi)
end
@inline function _llmu(::HurdleGamma, y, ly, mu, p)
    lpi, l1pi, pi = p.z
    y == 0 && return lpi, (zero(mu), zero(mu), 1 - pi)
    mu > 0 || return oftype(mu, -Inf), (zero(mu), zero(mu), zero(mu))
    v, (gm, ga) = _llmu(GammaFamily{LogLink}(), y, ly, mu, p)
    return l1pi + v, (gm, ga, -pi)
end
# hurdle_lognormal: mu is the location of log y (identity or inverse link)
@inline function _llmu(::HurdleLognormal, y, ly, mu, p)
    lpi, l1pi, pi = p.z
    y == 0 && return lpi, (zero(mu), zero(mu), 1 - pi)
    r = (ly - mu) * p.is
    r2 = r * r
    return l1pi - r2 / 2 - p.ls, (r * p.is, r2 - 1, -pi)
end

# =============================================================================
# Response checks
# =============================================================================
_check_response(::Family, y, t) = nothing
_chk(ok, msg) = ok || throw(ArgumentError(msg))
_check_response(::Bernoulli, y, t) = _chk(all(v -> v == 0 || v == 1, y), "bernoulli responses must be 0 or 1")
_check_response(::Union{Binomial,BetaBinomial,ZeroInflatedBinomial}, y, t) =
    _chk(all(i -> isinteger(y[i]) && 0 <= y[i] <= t[i], eachindex(y)), "binomial responses must be integers in 0:trials")
_check_response(::Union{Poisson,NegBinomial,Geometric,ZeroInflatedPoisson,ZeroInflatedNegBinomial,HurdlePoisson,HurdleNegBinomial}, y, t) =
    _chk(all(v -> isinteger(v) && v >= 0, y), "count responses must be non-negative integers")
_check_response(::Union{GammaFamily,Weibull,Exponential,Lognormal}, y, t) = _chk(all(>(0), y), "responses must be positive")
_check_response(::Union{HurdleGamma,HurdleLognormal}, y, t) = _chk(all(>=(0), y), "hurdle responses must be non-negative")
_check_response(::BetaFamily, y, t) = _chk(all(v -> 0 < v < 1, y), "beta responses must be in (0, 1)")

# =============================================================================
# Predictive quantities: mean of the response (posterior_epred) and one draw
# (posterior_predict). `eta` is the link-scale tuple, `t` the trials.
# =============================================================================
const D_ = Distributions

_mean(fam::Family, eta, t) = linkinv(link(fam), eta[1])
_mean(fam::Union{Binomial,BetaBinomial}, eta, t) = t * linkinv(link(fam), eta[1])
_mu(fam, eta) = linkinv(link(fam), eta[1])
_mean(f::Lognormal, eta, t) = exp(_mu(f, eta) + exp(2eta[2]) / 2)
_mean(f::ZeroInflatedPoisson, eta, t) = (1 - logistic(eta[2])) * _mu(f, eta)
_mean(f::ZeroInflatedNegBinomial, eta, t) = (1 - logistic(eta[3])) * _mu(f, eta)
_mean(f::ZeroInflatedBinomial, eta, t) = (1 - logistic(eta[2])) * t * _mu(f, eta)
_mean(f::HurdlePoisson, eta, t) = (m = _mu(f, eta); (1 - logistic(eta[2])) * m / (-expm1(-m)))
function _mean(f::HurdleNegBinomial, eta, t)
    mu, phi = _mu(f, eta), exp(eta[2])
    p0 = (phi / (mu + phi))^phi
    return (1 - logistic(eta[3])) * mu / (1 - p0)
end
_mean(f::HurdleGamma, eta, t) = (1 - logistic(eta[3])) * _mu(f, eta)
_mean(f::HurdleLognormal, eta, t) = (1 - logistic(eta[3])) * exp(_mu(f, eta) + exp(2eta[2]) / 2)

_rand(rng, fam::Gaussian, eta, t) = _mu(fam, eta) + exp(eta[2]) * randn(rng)
_rand(rng, fam::Student, eta, t) = _mu(fam, eta) + exp(eta[2]) * rand(rng, D_.TDist(1 + exp(eta[3])))
_rand(rng, f::Lognormal, eta, t) = exp(_mu(f, eta) + exp(eta[2]) * randn(rng))
_rand(rng, fam::Bernoulli, eta, t) = Float64(rand(rng) < _mu(fam, eta))
_rand(rng, fam::Binomial, eta, t) = Float64(rand(rng, D_.Binomial(t, _mu(fam, eta))))
function _rand(rng, fam::BetaBinomial, eta, t)
    mu, phi = _mu(fam, eta), exp(eta[2])
    return Float64(rand(rng, D_.BetaBinomial(t, mu * phi, (1 - mu) * phi)))
end
_rand(rng, fam::Poisson, eta, t) = Float64(rand(rng, D_.Poisson(_mu(fam, eta))))
function _rand(rng, fam::NegBinomial, eta, t)
    mu, phi = _mu(fam, eta), exp(eta[2])
    return Float64(rand(rng, D_.NegativeBinomial(phi, phi / (mu + phi))))
end
_rand(rng, fam::Geometric, eta, t) = Float64(rand(rng, D_.Geometric(1 / (1 + _mu(fam, eta)))))
function _rand(rng, fam::GammaFamily, eta, t)
    mu, a = _mu(fam, eta), exp(eta[2])
    return rand(rng, D_.Gamma(a, mu / a))
end
function _rand(rng, f::Weibull, eta, t)
    k = exp(eta[2])
    return rand(rng, D_.Weibull(k, _mu(f, eta) / exp(loggamma(1 + 1 / k))))
end
_rand(rng, f::Exponential, eta, t) = rand(rng, D_.Exponential(_mu(f, eta)))
function _rand(rng, fam::BetaFamily, eta, t)
    mu, phi = _mu(fam, eta), exp(eta[2])
    return rand(rng, D_.Beta(mu * phi, (1 - mu) * phi))
end
_rand(rng, f::ZeroInflatedPoisson, eta, t) =
    rand(rng) < logistic(eta[2]) ? 0.0 : Float64(rand(rng, D_.Poisson(_mu(f, eta))))
_rand(rng, f::ZeroInflatedBinomial, eta, t) =
    rand(rng) < logistic(eta[2]) ? 0.0 : Float64(rand(rng, D_.Binomial(t, _mu(f, eta))))
function _rand(rng, f::ZeroInflatedNegBinomial, eta, t)
    rand(rng) < logistic(eta[3]) && return 0.0
    mu, phi = _mu(f, eta), exp(eta[2])
    return Float64(rand(rng, D_.NegativeBinomial(phi, phi / (mu + phi))))
end
# hurdle: zero with probability hu, else a draw from the base truncated at 0
function _rand(rng, f::HurdlePoisson, eta, t)
    rand(rng) < logistic(eta[2]) && return 0.0
    d = D_.Poisson(_mu(f, eta))
    while true
        v = rand(rng, d)
        v > 0 && return Float64(v)
    end
end
function _rand(rng, f::HurdleNegBinomial, eta, t)
    rand(rng) < logistic(eta[3]) && return 0.0
    mu, phi = _mu(f, eta), exp(eta[2])
    d = D_.NegativeBinomial(phi, phi / (mu + phi))
    while true
        v = rand(rng, d)
        v > 0 && return Float64(v)
    end
end
function _rand(rng, f::HurdleGamma, eta, t)
    rand(rng) < logistic(eta[3]) && return 0.0
    mu, a = _mu(f, eta), exp(eta[2])
    return rand(rng, D_.Gamma(a, mu / a))
end
_rand(rng, f::HurdleLognormal, eta, t) =
    rand(rng) < logistic(eta[3]) ? 0.0 : exp(_mu(f, eta) + exp(eta[2]) * randn(rng))
