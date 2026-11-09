# =============================================================================
# Priors.
#
# Every prior is lowered to a `ScalarPrior`: one concrete struct over a closed
# set of families, each with a hand-written log-density and derivative. Two
# reasons for not just holding `Distributions.UnivariateDistribution`s:
#
#   * the gradient path needs d(logpdf)/dx, and calling a different
#     distribution type per coefficient is dynamic dispatch in the hot loop;
#   * trimmability — a closed set of concrete types is what `juliac --trim`
#     can compile, an open `Vector{Any}` of distributions is not.
#
# Users may still write `Normal(0, 1)` or brms's own strings
# ("student_t(3, 0, 2.5)"); both convert here. The string form lets the same
# prior text be used in R and Julia.
#
# All densities are fully normalised (brms's `target += *_lpdf` keeps the
# constants), and a lower-bounded prior carries its truncation constant, so
# `lprior` agrees with brms up to the LKJ normaliser (see `lkj_logpdf`).
# =============================================================================

const _HALF_LOG_2PI = 0.5 * log(2π)

@enum PriorKind::UInt8 begin
    PK_FLAT
    PK_NORMAL       # a = mu, b = sigma
    PK_STUDENT      # a = nu, b = mu, c = sigma
    PK_CAUCHY       # a = mu, b = sigma
    PK_EXPONENTIAL  # a = rate
    PK_GAMMA        # a = shape, b = rate
    PK_INVGAMMA     # a = shape, b = scale
    PK_LOGNORMAL    # a = mu, b = sigma
    PK_BETA         # a, b (on (0, 1))
    PK_UNIFORM      # a = lower, b = upper
    PK_LOGISTIC     # a = mu, b = scale
    PK_LAPLACE      # a = mu, b = scale (brms double_exponential)
    PK_SKEWNORMAL   # a = xi, b = omega, c = alpha (Stan's parameterisation)
    PK_CONSTANT     # a = the fixed value (brms constant(): the parameter becomes data)
end

"""
    ScalarPrior

A prior on one scalar parameter. `lb` is the lower bound of the parameter it
is attached to (`-Inf`, or e.g. 0 for a standard deviation).

The log density is split as `kernel(x) + lnorm + lconst`: `lnorm` is the
density's own normalising constant (e.g. the loggamma terms of a
Student-t) and `lconst = -log(1 - F(lb))` the truncation constant. Both are
computed ONCE here. The sampling path evaluates only the kernel — constants
cannot change where the sampler goes, and the Student-t's two `loggamma`
calls were otherwise paid on every gradient — and reported quantities
(`lprior`) add them back.
"""
struct ScalarPrior
    kind::PriorKind
    a::Float64
    b::Float64
    c::Float64
    lb::Float64
    lnorm::Float64
    lconst::Float64
end
ScalarPrior(kind, a=0.0, b=0.0, c=0.0) = ScalarPrior(kind, a, b, c, -Inf, _lnorm(kind, a, b, c), 0.0)

function _lnorm(k, a, b, c)
    k == PK_NORMAL && return -log(b) - _HALF_LOG_2PI
    k == PK_STUDENT && return loggamma((a + 1) / 2) - loggamma(a / 2) - log(a * π) / 2 - log(c)
    k == PK_CAUCHY && return -log(π) - log(b)
    k == PK_EXPONENTIAL && return log(a)
    k == PK_GAMMA && return a * log(b) - loggamma(a)
    k == PK_INVGAMMA && return a * log(b) - loggamma(a)
    k == PK_LOGNORMAL && return -log(b) - _HALF_LOG_2PI
    k == PK_BETA && return -logbeta(a, b)
    k == PK_UNIFORM && return -log(b - a)
    k == PK_LOGISTIC && return -log(b)
    k == PK_LAPLACE && return -log(2b)
    k == PK_SKEWNORMAL && return log(2.0) - log(b) - _HALF_LOG_2PI
    return 0.0
end

isflat(p::ScalarPrior) = p.kind == PK_FLAT

# brms-named constructors. Not exported (`gamma`, `normal` would collide with
# common names); write `PracticalBayesianRegression.normal(0, 1)`, a string, or a
# Distributions.jl object instead.
flat() = ScalarPrior(PK_FLAT)
normal(mu, sigma) = ScalarPrior(PK_NORMAL, mu, sigma)
student_t(nu, mu, sigma) = ScalarPrior(PK_STUDENT, nu, mu, sigma)
cauchy(mu, sigma) = ScalarPrior(PK_CAUCHY, mu, sigma)
exponential(rate) = ScalarPrior(PK_EXPONENTIAL, rate)
gamma(shape, rate) = ScalarPrior(PK_GAMMA, shape, rate)
inv_gamma(shape, scale) = ScalarPrior(PK_INVGAMMA, shape, scale)
lognormal(mu, sigma) = ScalarPrior(PK_LOGNORMAL, mu, sigma)
beta_prior(a, b) = ScalarPrior(PK_BETA, a, b)
function uniform(lo, hi)
    lo < hi || throw(ArgumentError("uniform($lo, $hi): the lower bound must be below the upper"))
    return ScalarPrior(PK_UNIFORM, lo, hi)
end
logistic_prior(mu, s) = ScalarPrior(PK_LOGISTIC, mu, s)
double_exponential(mu, s) = ScalarPrior(PK_LAPLACE, mu, s)
skew_normal_prior(xi, omega, alpha) = ScalarPrior(PK_SKEWNORMAL, xi, omega, alpha)
constant(v) = ScalarPrior(PK_CONSTANT, v)
isconstant(p) = p isa ScalarPrior && p.kind == PK_CONSTANT

"""
    LKJ(eta)

LKJ prior on a correlation matrix (brms `lkj(eta)`), applied through its
Cholesky factor.
"""
struct LKJ
    eta::Float64
end

"""
    lkj(eta)

The LKJ prior on a correlation matrix, as brms's `lkj(eta)`, for the
correlation classes (`:cor`, `:rescor`, ...). `eta = 1` is uniform over
correlation matrices; larger values favour weaker correlations.

```julia
prior(lkj(2); class = :cor)
```
"""
lkj(eta) = LKJ(eta)

"""
    DirichletPrior(alpha)

Symmetric Dirichlet prior on a simplex (brms `dirichlet(alpha)`, used for
the monotonic-effect simplexes `simo`).
"""
struct DirichletPrior
    alpha::Float64
end
"""
    dirichlet(alpha)

The symmetric Dirichlet prior on a simplex, as brms's `dirichlet(alpha)`: the
prior of the simplexes of monotonic effects (class `:simo`). Called without
a positional argument, `dirichlet()` is the response family.

```julia
prior(dirichlet(2); class = :simo)
```
"""
dirichlet(alpha) = DirichletPrior(alpha)
Base.show(io::IO, p::DirichletPrior) = print(io, "dirichlet($(_g(p.alpha)))")

const _PRIOR_CONSTRUCTORS = Dict{Symbol,Function}(
    :normal => normal, :student_t => student_t, :cauchy => cauchy,
    :exponential => exponential, :gamma => gamma, :inv_gamma => inv_gamma,
    :lognormal => lognormal, :beta => beta_prior, :lkj => lkj, :dirichlet => dirichlet, :lkj_corr_cholesky => lkj,
    :uniform => uniform, :logistic => logistic_prior, :double_exponential => double_exponential,
    :skew_normal => skew_normal_prior, :constant => constant,
)

"""
    to_prior(x) -> Union{ScalarPrior,LKJ}

Convert a brms prior string (`"normal(0, 1)"`, `""` for flat), a
Distributions.jl distribution, `nothing` (flat), or a `ScalarPrior`.
"""
to_prior(p::ScalarPrior) = p
to_prior(p::LKJ) = p
to_prior(p::DirichletPrior) = p
to_prior(::Nothing) = flat()
function to_prior(s::AbstractString)
    s = strip(s)
    isempty(s) && return flat()
    ex = Meta.parse(s)
    (ex isa Expr && ex.head == :call && ex.args[1] isa Symbol) ||
        throw(ArgumentError("cannot parse prior `$s`"))
    f = ex.args[1] === :horseshoe ? horseshoe : ex.args[1] === :R2D2 ? R2D2 : get(_PRIOR_CONSTRUCTORS, ex.args[1], nothing)
    f === nothing && throw(ArgumentError("prior family `$(ex.args[1])` is not supported yet"))
    num(a) = a isa Number ? Float64(a) :
             (a isa Expr && a.head == :call && a.args[1] === :- && a.args[2] isa Number) ? -Float64(a.args[2]) :
             a in (:TRUE, :true) ? true : a in (:FALSE, :false) ? false :
             throw(ArgumentError("prior arguments must be numbers, got `$a` in `$s`"))
    pos = [num(a) for a in ex.args[2:end] if !(a isa Expr && a.head in (:kw, :parameters))]
    kws = [Symbol(a.args[1]) => num(a.args[2]) for a in ex.args[2:end] if a isa Expr && a.head == :kw]
    return f(pos...; kws...)
end

function to_prior(d::Distributions.Distribution)
    D = Distributions
    d isa D.Normal && return normal(d.μ, d.σ)
    d isa D.TDist && return student_t(d.ν, 0.0, 1.0)
    if d isa D.AffineDistribution && d.ρ isa D.TDist
        return student_t(d.ρ.ν, d.μ, d.σ)
    end
    d isa D.Cauchy && return cauchy(d.μ, d.σ)
    d isa D.Exponential && return exponential(1 / d.θ)
    d isa D.Gamma && return gamma(d.α, 1 / d.θ)
    d isa D.InverseGamma && return inv_gamma(D.shape(d), D.scale(d))
    d isa D.LogNormal && return lognormal(d.μ, d.σ)
    d isa D.Beta && return beta_prior(d.α, d.β)
    d isa D.Uniform && return uniform(d.a, d.b)
    d isa D.Logistic && return logistic_prior(d.μ, d.θ)
    d isa D.Laplace && return double_exponential(d.μ, d.θ)
    d isa D.SkewNormal && return skew_normal_prior(d.ξ, d.ω, d.α)
    d isa D.LKJ && return lkj(d.η)
    d isa D.LKJCholesky && return lkj(d.η)
    if d isa D.Truncated
        # the bound comes from the parameter class; a user-written
        # truncation is accepted only when it matches that bound
        return to_prior(d.untruncated)
    end
    nameof(typeof(d)) in (:Flat, :FlatPos) && return flat()
    throw(ArgumentError("prior $(d) is not supported yet"))
end

# Attach the parameter's lower bound and its truncation constant.
function with_bound(p::ScalarPrior, lb::Float64, ub::Float64=Inf)
    (lb == -Inf && ub == Inf || p.kind in (PK_FLAT, PK_CONSTANT)) && return ScalarPrior(p.kind, p.a, p.b, p.c, lb, p.lnorm, 0.0)
    Fl = lb == -Inf ? 0.0 : _cdf(p, lb)
    Fu = ub == Inf ? 1.0 : _cdf(p, ub)
    return ScalarPrior(p.kind, p.a, p.b, p.c, lb, p.lnorm, -log(Fu - Fl))
end

function _cdf(p::ScalarPrior, x)
    D = Distributions
    k = p.kind
    x == Inf && return 1.0
    k == PK_NORMAL && return D.cdf(D.Normal(p.a, p.b), x)
    k == PK_STUDENT && return D.cdf(D.TDist(p.a), (x - p.b) / p.c)
    k == PK_CAUCHY && return D.cdf(D.Cauchy(p.a, p.b), x)
    k == PK_EXPONENTIAL && return x <= 0 ? 0.0 : D.cdf(D.Exponential(1 / p.a), x)
    k == PK_GAMMA && return x <= 0 ? 0.0 : D.cdf(D.Gamma(p.a, 1 / p.b), x)
    k == PK_INVGAMMA && return x <= 0 ? 0.0 : D.cdf(D.InverseGamma(p.a, p.b), x)
    k == PK_LOGNORMAL && return x <= 0 ? 0.0 : D.cdf(D.LogNormal(p.a, p.b), x)
    k == PK_BETA && return x <= 0 ? 0.0 : D.cdf(D.Beta(p.a, p.b), x)
    k == PK_UNIFORM && return clamp((x - p.a) / (p.b - p.a), 0.0, 1.0)
    k == PK_LOGISTIC && return D.cdf(D.Logistic(p.a, p.b), x)
    k == PK_LAPLACE && return D.cdf(D.Laplace(p.a, p.b), x)
    k == PK_SKEWNORMAL && return D.cdf(D.SkewNormal(p.a, p.b, p.c), x)
    return 0.0
end

# A stored (Float64) prior constant in the precision of `x`: a Float32 sampling
# path then stays in Float32 instead of promoting every prior term to Float64
# (and returning a Float64 log density). Other number types (ForwardDiff duals,
# BigFloat) see the constant as stored, which their own promotion handles.
@inline _pv(x::T, v::Real) where {T<:Union{Float32,Float16}} = T(v)
@inline _pv(x, v) = v
# a dual number over Float32 (the dual-number families' per-observation
# gradient): the constant in its value type, so Dual{Float32} stays Dual{Float32}
@inline _pv(x::ForwardDiff.Dual{G,V}, v::Real) where {G,V<:Union{Float32,Float16}} = V(v)
const _LOG2 = log(2.0)
const _LOG_2PI = log(2π)

"""
    prior_kernel(p, x)

Log density of prior `p` at `x` WITHOUT its constants, generic in the number
type (AD backends differentiate this). See [`ScalarPrior`](@ref).
"""
function prior_kernel(p::ScalarPrior, x)
    k = p.kind
    T = typeof(x)
    (k == PK_FLAT || k == PK_CONSTANT) && return zero(T)
    pa, pb, pc = _pv(x, p.a), _pv(x, p.b), _pv(x, p.c)
    if k == PK_NORMAL
        r = (x - pa) / pb
        return -r * r / 2
    elseif k == PK_STUDENT
        r = (x - pb) / pc
        return -(pa + 1) / 2 * log1p(r * r / pa)
    elseif k == PK_CAUCHY
        r = (x - pa) / pb
        return -log1p(r * r)
    elseif k == PK_EXPONENTIAL
        return -pa * x
    elseif k == PK_GAMMA
        return (pa - 1) * log(x) - pb * x
    elseif k == PK_INVGAMMA
        return -(pa + 1) * log(x) - pb / x
    elseif k == PK_LOGNORMAL
        lx = log(x)
        r = (lx - pa) / pb
        return -r * r / 2 - lx
    elseif k == PK_BETA
        return (pa - 1) * log(x) + (pb - 1) * log1p(-x)
    elseif k == PK_UNIFORM
        # Stan: -Inf outside the support (pair with bounds on the parameter)
        return pa <= x <= pb ? zero(T) : T(-Inf)
    elseif k == PK_LOGISTIC
        z = (x - pa) / pb
        return -z - 2 * log1pexp(-z)
    elseif k == PK_LAPLACE
        return -abs(x - pa) / pb
    elseif k == PK_SKEWNORMAL
        z = (x - pa) / pb
        return -z * z / 2 + normlogcdf(pc * z)
    end
    error("unreachable")
end

"""
    prior_logpdf(p, x)

The fully normalised log density (kernel + normaliser + truncation constant).
"""
prior_logpdf(p::ScalarPrior, x) = prior_kernel(p, x) + p.lnorm + p.lconst

"""
    prior_kernel_grad(p, x) -> (kernel, d kernel / dx)

The closed-form derivative used by the analytic gradient. Checked against
ForwardDiff of [`prior_kernel`](@ref) in the tests.
"""
function prior_kernel_grad(p::ScalarPrior, x::T) where {T<:Real}
    k = p.kind
    (k == PK_FLAT || k == PK_CONSTANT) && return zero(T), zero(T)
    v = prior_kernel(p, x)
    pa, pb, pc = _pv(x, p.a), _pv(x, p.b), _pv(x, p.c)
    d = if k == PK_NORMAL
        -(x - pa) / (pb * pb)
    elseif k == PK_STUDENT
        nu, mu, s = pa, pb, pc
        e = x - mu
        -(nu + 1) * e / (nu * s * s + e * e)
    elseif k == PK_CAUCHY
        e = x - pa
        -2e / (pb * pb + e * e)
    elseif k == PK_EXPONENTIAL
        -pa
    elseif k == PK_GAMMA
        (pa - 1) / x - pb
    elseif k == PK_INVGAMMA
        -(pa + 1) / x + pb / (x * x)
    elseif k == PK_LOGNORMAL
        -(1 + (log(x) - pa) / (pb * pb)) / x
    elseif k == PK_BETA
        (pa - 1) / x - (pb - 1) / (1 - x)
    elseif k == PK_UNIFORM
        zero(T)
    elseif k == PK_LOGISTIC
        z = (x - pa) / pb
        (2 * logistic(-z) - 1) / pb
    elseif k == PK_LAPLACE
        -sign(x - pa) / pb
    elseif k == PK_SKEWNORMAL
        z = (x - pa) / pb
        (-z + pc * mills(pc * z)) / pb
    else
        error("unreachable")
    end
    return T(v), T(d)
end

# brms-syntax printing, used by prior_summary and in error messages
function Base.show(io::IO, p::ScalarPrior)
    k = p.kind
    s = k == PK_FLAT ? "(flat)" :
        k == PK_NORMAL ? "normal($(_g(p.a)), $(_g(p.b)))" :
        k == PK_STUDENT ? "student_t($(_g(p.a)), $(_g(p.b)), $(_g(p.c)))" :
        k == PK_CAUCHY ? "cauchy($(_g(p.a)), $(_g(p.b)))" :
        k == PK_EXPONENTIAL ? "exponential($(_g(p.a)))" :
        k == PK_GAMMA ? "gamma($(_g(p.a)), $(_g(p.b)))" :
        k == PK_INVGAMMA ? "inv_gamma($(_g(p.a)), $(_g(p.b)))" :
        k == PK_BETA ? "beta($(_g(p.a)), $(_g(p.b)))" :
        k == PK_UNIFORM ? "uniform($(_g(p.a)), $(_g(p.b)))" :
        k == PK_LOGISTIC ? "logistic($(_g(p.a)), $(_g(p.b)))" :
        k == PK_LAPLACE ? "double_exponential($(_g(p.a)), $(_g(p.b)))" :
        k == PK_SKEWNORMAL ? "skew_normal($(_g(p.a)), $(_g(p.b)), $(_g(p.c)))" :
        k == PK_CONSTANT ? "constant($(_g(p.a)))" :
        "lognormal($(_g(p.a)), $(_g(p.b)))"
    print(io, s)
end
Base.show(io::IO, p::LKJ) = print(io, "lkj($(_g(p.eta)))")
_g(x) = isinteger(x) ? string(Int(x)) : string(x)

# --- user prior specification -----------------------------------------------

"""
    PriorSpec

One row of a prior table: a prior and the parameters it applies to (class,
coefficient, group, distributional parameter, response, bounds). Made by
[`prior`](@ref); [`get_prior`](@ref) and [`prior_summary`](@ref) return them
too.
"""
struct PriorSpec
    prior::Any                 # ScalarPrior, LKJ, DirichletPrior or HorseshoePrior
    class::Symbol
    coef::String
    group::String
    dpar::String
    source::Symbol    # :user or :default
    lb::Float64       # bounds on the parameter (class b only, as brms's lb/ub)
    ub::Float64
    resp::String      # multivariate models: the response it applies to ("" = all)
end
PriorSpec(prior, class, coef, group, dpar, source) = PriorSpec(prior, class, coef, group, dpar, source, -Inf, Inf, "")
PriorSpec(prior, class, coef, group, dpar, source, lb, ub) = PriorSpec(prior, class, coef, group, dpar, source, lb, ub, "")

"""
    prior(p; class = :b, coef = "", group = "", dpar = "", nlpar = "", resp = "", lb = -Inf, ub = Inf) -> PriorSpec

One prior, as brms's `set_prior` / `prior()`. Pass a vector of them to
[`brm`](@ref) with `prior = [...]`; every parameter without one keeps its
default (see [`get_prior`](@ref)).

`p` is one of:
- a brms prior string: `"normal(0, 1)"`, `"student_t(3, 0, 2.5)"`,
  `"cauchy(0, 1)"`, `"exponential(1)"`, `"gamma(2, 0.1)"`,
  `"inv_gamma(0.4, 0.3)"`, `"lognormal(0, 1)"`, `"beta(1, 1)"`,
  `"uniform(0, 10)"`, `"logistic(0, 1)"`, `"double_exponential(0, 1)"`,
  `"skew_normal(0, 1, 2)"`, `"lkj(2)"`, `"dirichlet(1)"`, `"constant(0.5)"`
  (fixes the parameter), `"horseshoe(1)"` or `"R2D2(0.5, 2)"` (shrinkage
  priors on all coefficients of class `:b`);
- a Distributions.jl distribution: `Normal(0, 1)`, `Exponential(1)`,
  `Gamma(2, 10)`, `Cauchy(0, 1)`, `TDist(3)`, `LKJ(2, 2)`, ...;
- `nothing` or `""`: a flat (improper) prior.

The keywords pick the parameters the prior applies to, as in brms:
- `class`: `:b` (population-level effects), `:Intercept`, `:sd` (group-level
  standard deviations), `:cor` (group-level correlations), `:sigma`,
  `:shape`, `:nu`, `:phi`, `:zi`, `:hu`, ... (see [`get_prior`](@ref) for a
  model's classes).
- `coef`: one coefficient (`"x"`), else the whole class.
- `group`: one grouping factor (`"g"`), for `:sd` and `:cor`.
- `dpar`: a distributional parameter's formula (`"sigma"`).
- `nlpar`: a non-linear parameter.
- `resp`: one response of a multivariate model.
- `lb`, `ub`: bounds on `:b` coefficients, e.g. `lb = 0` for a positive
  effect.

The most specific matching prior wins.

# Examples
```julia
priors = [
    prior("normal(0, 1)"; class = :b),
    prior("normal(0, 5)"; class = :b, coef = "x"),
    prior(Exponential(1); class = :sigma),
    prior("student_t(3, 0, 1)"; class = :sd, group = "g"),
    prior(lkj(2); class = :cor),
]
fit = brm(@bf(y ~ x + z + (1 + x | g)), df; prior = priors)
```
"""
prior(p; class=:b, coef="", group="", dpar="", nlpar="", lb=-Inf, ub=Inf, resp="") =
    _with_resp(_prior_row(p, class, coef, group, dpar, nlpar, lb, ub), resp)
_with_resp(r::PriorSpec, resp) = PriorSpec(r.prior, r.class, r.coef, r.group, r.dpar, r.source, r.lb, r.ub, string(resp))
_prior_row(p, class, coef, group, dpar, nlpar, lb, ub) =
    PriorSpec(to_prior(p), Symbol(class), string(coef), string(group), string(nlpar == "" ? dpar : nlpar), :user,
              Float64(lb), Float64(ub))

# Most specific matching user row wins: coef+group > group > coef > class.
function _lookup_prior(rows::Vector{PriorSpec}, class::Symbol, coef::String, group::String, dpar::String)
    best = nothing; best_score = -1
    for r in rows
        r.class === class && r.dpar == dpar || continue
        (r.coef == "" || r.coef == coef) || continue
        (r.group == "" || r.group == group) || continue
        score = (r.coef != "" ? 2 : 0) + (r.group != "" ? 1 : 0)
        score >= best_score && (best = r; best_score = score)
    end
    return best
end

# --- prior draws (sample_prior = "yes") ---------------------------------------------

# a ScalarPrior as a Distributions.jl distribution (for drawing from it)
function _prior_dist(p::ScalarPrior)
    D = Distributions
    k = p.kind
    k == PK_NORMAL && return D.Normal(p.a, p.b)
    k == PK_STUDENT && return p.b + p.c * D.TDist(p.a)
    k == PK_CAUCHY && return D.Cauchy(p.a, p.b)
    k == PK_EXPONENTIAL && return D.Exponential(1 / p.a)
    k == PK_GAMMA && return D.Gamma(p.a, 1 / p.b)
    k == PK_INVGAMMA && return D.InverseGamma(p.a, p.b)
    k == PK_LOGNORMAL && return D.LogNormal(p.a, p.b)
    k == PK_BETA && return D.Beta(p.a, p.b)
    k == PK_UNIFORM && return D.Uniform(p.a, p.b)
    k == PK_LOGISTIC && return D.Logistic(p.a, p.b)
    k == PK_LAPLACE && return D.Laplace(p.a, p.b)
    k == PK_SKEWNORMAL && return D.SkewNormal(p.a, p.b, p.c)
    return nothing                       # flat / constant: nothing to draw
end

# n draws, restricted to (lb, ub) by rejection as brms's generated quantities
function _prior_draws(rng, p::ScalarPrior, n, lb=-Inf, ub=Inf)
    d = _prior_dist(p)
    d === nothing && return nothing
    out = Vector{Float64}(undef, n)
    for i in 1:n
        v = rand(rng, d)
        tries = 0
        while !(lb <= v <= ub) && tries < 10_000
            v = rand(rng, d); tries += 1
        end
        out[i] = v
    end
    return out
end
