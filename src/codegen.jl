# =============================================================================
# Model code generation: a built model as Turing / PracticalBayes `@model`
# source, plus the data it takes (the analogue of brms's stancode/standata).
#
# WHY: people who know Turing (or PracticalBayes) may prefer to fit, extend or
# inspect the model in that ecosystem. The generated code is plain, readable
# `@model` code written from the BUILT model (GLMDensity + ModelInfo), so it
# uses exactly the design matrices, coding, centring and resolved default
# priors that `brm` samples; nothing is re-derived from the formula.
#
# The two dialects are nearly the same language (PracticalBayes's `@model` is
# Turing-style); they differ only in the imports. Both write the likelihood as
# `y ~ arraydist(...)`, which each backend evaluates as one vectorised term.
#
# The parameterisation follows brms's Stan code, so the posteriors are the same
# model: centred population-level design `X` with the prior on the centred
# `Intercept` (`b_Intercept` is recorded with `:=`), non-centred group effects
# r = diag(sd) * L * z. The density differs from `brm`'s only by constants
# (Distributions' logpdfs are normalised; brm's sampling density is not) and by
# the unconstrained parameterisation the backend chooses for L (Bijectors'
# LKJCholesky transform rather than Stan's CPC/tanh one).
#
# This is the readable path, not the fast one: gradients come from the
# backend's AD, not the closed form. Features the generator does not know yet
# throw an ArgumentError naming the feature, rather than emitting a model that
# silently differs from the one `brm` fits.
# =============================================================================

"""
    ModelCode

Generated `@model` source (see [`turing_code`](@ref), [`pb_code`](@ref)).
Displays as the code; `string(c)` gives it as a `String`, and
`write("model.jl", c)` saves it.
"""
struct ModelCode
    backend::Symbol
    header::String     # the `using ...` line
    body::String       # helpers + the @model definition
    name::Symbol       # the model function's name
    fallback::String   # "" for explicit code; else the feature that forced the density fallback
end
ModelCode(backend, header, body, name) = ModelCode(backend, header, body, name, "")

"""
    model_logdensity(density, θ) -> Real

The log density that [`brm`](@ref) samples (constants dropped, Jacobians
included) of a built model at the unconstrained point `θ`, written in generic
Julia code that any automatic differentiation backend can differentiate.
Generated model code calls it for models it cannot write out explicitly (see
[`turing_code`](@ref)).
"""
model_logdensity(density, θ) = LogDensityProblems.logdensity(density, θ)

"""
    model_dimension(density) -> Int

The length of the unconstrained parameter vector `θ` of a built model.
"""
model_dimension(density) = LogDensityProblems.dimension(density)
Base.string(c::ModelCode) = c.header * "\n\n" * c.body
Base.print(io::IO, c::ModelCode) = print(io, string(c))
Base.show(io::IO, ::MIME"text/plain", c::ModelCode) = print(io, string(c))
Base.write(io::IO, c::ModelCode) = write(io, string(c))
Base.write(path::AbstractString, c::ModelCode) = open(io -> write(io, c), path, "w")

"""
    turing_code(formula, data; family, prior, ...) -> ModelCode
    turing_code(fit::BrmsFit)
    turing_code(ld::GLMDensity, info::ModelInfo)

Turing.jl `@model` source code for the model that `brm` fits, to read, edit
or extend by hand; the counterpart of brms's `stancode`. The arguments are
those of [`brm`](@ref), or a fit. The model takes the data from
[`model_data`](@ref) as positional arguments:

```julia
code = turing_code(@bf(y ~ x + (1 | g)), d; family = poisson())
write("model.jl", code)                  # edit it, then include("model.jl")
m = regression_model(model_data(@bf(y ~ x + (1 | g)), d; family = poisson())...)
```

[`turing_model`](@ref) does the evaluation step for you. The keyword `name`
(default `:regression_model`) names the generated function.

Every model that `brm` fits gets code. Where the generator can write a
model's features out, the code is explicit: named parameters,
Distributions.jl priors, the linear predictors and the likelihood. For other
models it samples the unconstrained parameter vector `θ` and adds the
package's own log density ([`model_logdensity`](@ref)); `code.fallback` then
names the feature responsible, and `fallback = false` raises an error
instead.

The generated model is sampled with Turing's samplers and automatic
differentiation, so it is not guaranteed to be as fast as `brm`.
"""
turing_code(args...; kw...) = _codegen(:turing, args...; kw...).code

"""
    pb_code(formula, data; family, prior, ...) -> ModelCode
    pb_code(fit::BrmsFit)
    pb_code(ld::GLMDensity, info::ModelInfo)

PracticalBayes.jl `@model` source code for the model that `brm` fits. It
works as [`turing_code`](@ref) does; only the package the code loads
differs. To run it, install PracticalBayes.jl, `include` the code and call
the model function with the arguments from [`model_data`](@ref).
"""
pb_code(args...; kw...) = _codegen(:pb, args...; kw...).code

"""
    model_data(formula, data; family, prior, ...) -> NamedTuple
    model_data(fit::BrmsFit)
    model_data(ld::GLMDensity, info::ModelInfo)

The data that the generated model code takes, as a NamedTuple in argument
order; the counterpart of brms's `standata`. Splat it into the generated
function: `regression_model(model_data(...)...)`.
"""
model_data(args...; kw...) = _codegen(:turing, args...; kw...).data

"""
    turing_model(formula, data; family, prior, ...) -> DynamicPPL.Model
    turing_model(fit::BrmsFit)

The generated Turing model ([`turing_code`](@ref)), evaluated and applied to
its data, ready for `sample(m, NUTS(), MCMCThreads(), 1000, 4)`,
`maximum_a_posteriori(m)` and so on. Needs `using Turing`.

The model function is defined when this is called. Inside a function or
loop that calls `turing_model` and then samples, call
`Base.invokelatest(sample, m, ...)`; at the REPL or the top level of a script
no extra step is needed.
"""
turing_model(args...; kw...) = _generated_model(:turing, "Turing", args...; kw...)

# Defined by the Turing extension:
# (Val(backend), code body) -> the module the body was evaluated in
function _eval_model_code end

function _generated_model(backend::Symbol, pkg::String, args...; kw...)
    hasmethod(_eval_model_code, Tuple{Val{backend},String}) ||
        throw(ArgumentError("$(backend)_model needs $pkg: run `using $pkg` first"))
    g = _codegen(backend, args...; kw...)
    m = Base.invokelatest(_eval_model_code, Val(backend), g.code.body)
    # the binding is newer than this world too (Julia 1.12 world-age rules)
    f = Base.invokelatest(getglobal, m, g.code.name)
    return Base.invokelatest(f, g.data...)
end

# Evaluate generated code in a fresh module that sees exactly `mods` (bound
# and `using`-ed by object, so they need not be in the active environment).
function _eval_in_module(body::AbstractString, mods::Module...)
    m = Module(:GeneratedModel)
    for mod in mods
        nm = nameof(mod)
        Core.eval(m, :(const $nm = $mod))
        Core.eval(m, Expr(:using, Expr(:., :., nm)))
    end
    for ex in Meta.parseall(body).args
        ex isa LineNumberNode && continue
        Core.eval(m, ex)
    end
    return m
end

# --- inputs ------------------------------------------------------------------

_codegen(backend::Symbol, fit::BrmsFit; kw...) = _codegen(backend, fit.density, fit.info; kw...)
_codegen(backend::Symbol, fit::BrmsMVFit; kw...) = _codegen(backend, fit.density, nothing; kw...)
function _codegen(backend::Symbol, formula, data; name::Symbol=:regression_model, fallback::Bool=true, kw...)
    fb = formula isa MVFormula ? formula : bf(formula)
    # invokelatest: a string non-linear formula compiles its mean function by eval
    dens, info = _is_mv(fb) ? Base.invokelatest(build_mv_model, fb, data; kw...) :
                              Base.invokelatest(build_model, fb, data; kw...)
    return _codegen(backend, dens, info; name=name, fallback=fallback)
end

# Explicit code where the generator knows every feature of the model; the
# density fallback otherwise (or an error with fallback = false).
function _codegen(backend::Symbol, dens::Union{GLMDensity,MVDensity}, info; name::Symbol=:regression_model,
                  fallback::Bool=true)
    haskey(_CG_HEADERS, backend) || throw(ArgumentError("unknown backend `$backend` (:turing or :pb)"))
    try
        return _codegen_explicit(backend, dens, info; name=name)
    catch e
        e isa _CGUnsupported || rethrow()
        fallback || throw(ArgumentError("model code generation does not write out $(e.what) yet " *
                                        "(pass fallback = true for code using the package's log density)"))
        return _codegen_density(backend, dens, info, e.what; name=name)
    end
end

# --- what the generator supports ---------------------------------------------

const _CG_FAMILIES = Union{Gaussian,Student,Lognormal,Bernoulli,Binomial,Poisson,NegBinomial,
                           GammaFamily,Exponential,BetaFamily}

# a feature the explicit generator cannot write out (caught: density fallback)
struct _CGUnsupported <: Exception
    what::String
end
_cg_unsupported(what) = throw(_CGUnsupported(string(what)))
_codegen_explicit(backend, dens, info; kw...) = _cg_unsupported("$(nameof(typeof(dens))) models (multivariate)")

# --- the density fallback ----------------------------------------------------
#
# Works for EVERY model: θ is the unconstrained vector the package's own
# samplers use (flat on the real line, so Turing's link is the identity and
# its log density equals ours exactly), and the posterior is added by the
# generic density path, which is plain Julia over any number type (it is the
# reference the closed-form gradient is tested against). Readable it is not;
# correct it is, by construction.

const _CG_DENSITY_HEADERS = Dict(:turing => "using Turing, PracticalBayesianRegression",
                                 :pb => "using PracticalBayes, Distributions, PracticalBayesianRegression")

function _codegen_density(backend::Symbol, dens, info, reason::String; name::Symbol=:regression_model)
    gen = backend === :turing ? "turing_code" : "pb_code"
    io = IOBuffer()
    println(io, "# Generated by PracticalBayesianRegression.$gen.")
    _cg_print_formula(io, info)
    println(io, "#")
    println(io, "# This model uses $reason, which $gen cannot write out as explicit")
    println(io, "# @model code yet. The model therefore samples the unconstrained parameter")
    println(io, "# vector θ (the layout brm samples: `variables(density)` names the parameters,")
    println(io, "# `constrain(density, θ)` gives their values) and adds PracticalBayesianRegression's")
    println(io, "# own log density: the posterior brm samples (Jacobians included, constants")
    println(io, "# dropped), in plain Julia, so any AD backend can differentiate it.")
    println(io, "# Data: `model_data(...)` returns `(; density)`.")
    println(io)
    println(io, "@model function $name(density)")
    println(io, "    θ ~ filldist(Flat(), model_dimension(density))")
    println(io, "    @addlogprob! model_logdensity(density, θ)")
    println(io, "end")
    code = ModelCode(backend, _CG_DENSITY_HEADERS[backend], String(take!(io)), name, reason)
    return (code=code, data=(; density=dens), params=θ -> (; θ=θ))
end

function _cg_print_formula(io, info)
    info isa ModelInfo || return
    fstr = try
        sprint(show, info.formula)
    catch
        return
    end
    println(io, "# formula: ", replace(fstr, "\n" => "\n#          "))
end

# Every GLMDensity's likelihood is, by definition (the generic density path),
# sum_i w_i * _ll(family, y_i, x_i, eta_i) over the per-dpar linear predictors;
# so ANY family can be written out (as FamilyDist, below) once its predictors
# can. What is checked here is the parameter side. A term this list does not
# know is a silent error, which is why the tests compare every explicit
# model's likelihood with the package's at random points.
function _cg_check(ld::GLMDensity)
    # a family with its own logdensity method is not the per-observation sum:
    # TimeCov's joint residual covariance is written out below (_cg_timecov!);
    # any other (gev's data-scaled xi, ...) is not yet
    if ld.family isa TimeCov
        # (unstr's correlation is an LnCorBlock named L_cortime; see the preds loop)
    elseif ld.family isa GenExtremeValue
        # its xi is rescaled from the data (written out in _codegen_explicit)
    elseif which(LogDensityProblems.logdensity, Tuple{typeof(ld),Vector{Float64}}) !==
           which(LogDensityProblems.logdensity, Tuple{GLMDensity,AbstractVector})
        _cg_unsupported("the $(_cg_try(() -> family_name(ld.family), string(nameof(typeof(ld.family))))) family's joint likelihood")
    end
    if !isempty(ld.ordered)
        all(k -> any(p -> p isa LinPred && p.intercept == k, ld.preds), ld.ordered) ||
            _cg_unsupported("ordered parameters other than intercepts")
    end
    ld.family isa GenExtremeValue && !(ld.preds[3] isa ScalarDpar) && _cg_unsupported("a gev xi with a formula")
    isempty(ld.Jmi) || _cg_unsupported("mi() responses")
    # a non-linear mean's parameters are linear predictors like any other
    allp = reduce(vcat, [p isa NLPred ? collect(Any, p.pars) : Any[p] for p in ld.preds]; init=Any[])
    for p in allp
        if p isa LinPred
            p.r2d2.on && p.r2d2.sigma[] != 0 &&
                !any(q -> q isa ScalarDpar && q.idx == p.r2d2.sigma[], ld.preds) &&
                _cg_unsupported("R2D2 autoscale with a predicted sigma")
            p.hs.on && p.hs.sigma[] > 0 &&
                !any(q -> q isa ScalarDpar && q.idx == p.hs.sigma[], ld.preds) &&
                _cg_unsupported("horseshoe autoscale with a predicted sigma")
            isempty(p.mi) || _cg_unsupported("mi() predictors")
            for (j, lb) in enumerate(p.b_lb)
                isflat(p.b_priors[j]) && lb == -Inf && isfinite(p.b_ub[j]) &&
                    _cg_unsupported("flat priors with only an upper bound")
            end
        elseif !(p isa Union{ScalarDpar,Thresholds,ThresholdsCS,ThetaSimplex,LnCorBlock,BhazSimplex,NLPred})
            _cg_unsupported("$(nameof(typeof(p))) predictors")
        end
    end
    return nothing
end

"""
    FamilyDist(family, x, trials, eta...)

The distribution of one observation under a package family, given its
distributional parameters on the LINK scale (`eta`, in the family's dpar
order: mu, sigma, ... as `log(sigma)`, ...) and the family's per-observation
data `x` (e.g. `log(y)`, censoring codes, truncation bounds) and `trials`.
`logpdf` is the package's own (normalised) log-likelihood and `rand` its
posterior-predictive draw, so it is exact for every family `brm` fits.

Generated model code ([`turing_code`](@ref)) uses it for families with no
Distributions.jl equivalent, the way brms's Stan code defines such families
in its `functions` block.
"""
struct FamilyDist{F<:Family,X,E<:Tuple} <: Distributions.ContinuousUnivariateDistribution
    family::F
    x::X
    trials::Int
    eta::E
end
FamilyDist(family::Family, x, trials, eta...) = FamilyDist(family, x, Int(trials), eta)
Distributions.logpdf(d::FamilyDist, y::Real) =
    _ll(d.family, y, d.x, d.eta) + _ll_const_obs(d.family, y, d.trials, d.x)
Distributions.pdf(d::FamilyDist, y::Real) = exp(Distributions.logpdf(d, y))
Base.rand(rng::Random.AbstractRNG, d::FamilyDist) = _rand(rng, d.family, d.eta, d.trials)
Base.minimum(::FamilyDist) = -Inf
Base.maximum(::FamilyDist) = Inf
Distributions.insupport(::FamilyDist, ::Real) = true

# The package's normalised log-likelihood at θ, with the definition FamilyDist
# uses: the reference the explicit generated code is checked against.
function _cg_loglik(ld::GLMDensity, θraw)
    θ = Vector{promote_type(eltype(θraw), Float64)}(first(_order_generic(ld.ordered, θraw)))   # mixtures
    for (k, c) in zip(ld.pinned, ld.pinvals)          # constant() priors
        θ[k] = c
    end
    etas = map(p -> _eta_generic(p, θ, ld.N), ld.preds)
    s = zero(eltype(θ))
    joint = ld.family isa Union{TimeCov,GenExtremeValue}
    if ld.family isa TimeCov
        # the joint (per group) likelihood, normalised
        s = _tc_ll_generic(ld, etas) + sum(_ll_const(ld.family.fam, ld.y[i], 0) for i in 1:ld.N)
    elseif ld.family isa GenExtremeValue
        xi = _gev_xi_generic(ld, etas)
        for i in 1:ld.N
            v = _gev_ll(ld.y[i], _at(etas[1], i), _at(etas[2], i), xi)
            s += isempty(ld.weights) ? v : ld.weights[i] * v
        end
    end
    for i in 1:(joint ? 0 : ld.N)
        x = _xd(ld, i)
        t = isempty(ld.trials) ? 0 : ld.trials[i]
        v = _ll(ld.family, ld.y[i], x, map(e -> _at(e, i), etas)) + _ll_const_obs(ld.family, ld.y[i], t, x)
        s += isempty(ld.weights) ? v : ld.weights[i] * v
    end
    # me(): the observed covariate's measurement model is data likelihood too
    # (the generated code observes it), counted once per latent variable
    seen = Set{Int}()
    for p in ld.preds
        p isa LinPred || continue
        for mc in p.mecor                       # correlated latents first
            K = length(mc.terms)
            W = first(_cholcorr(θ[mc.cor], K)) * reshape(θ[mc.z], K, :)
            for (j, m) in enumerate(mc.terms)
                push!(seen, m.meanme)
                xme = θ[m.meanme] .+ exp(θ[m.sdme]) .* W[j, :]
                s += sum(Distributions.logpdf.(Distributions.Normal.(xme, m.sdx), m.x))
            end
        end
        for m in p.me
            m.meanme in seen && continue
            push!(seen, m.meanme)
            xme = θ[m.meanme] .+ exp(θ[m.sdme]) .* θ[m.zme]
            s += sum(Distributions.logpdf.(Distributions.Normal.(xme, m.sdx), m.x))
        end
    end
    return s
end

# --- priors as Distributions.jl source ------------------------------------------

_cg_num(x) = _g(Float64(x))
_cg_inv(rate) = isone(rate) ? "1" : "1 / " * _cg_num(rate)

function _cg_base(p::ScalarPrior)
    k, a, b, c = p.kind, p.a, p.b, p.c
    k == PK_NORMAL && return "Normal($(_cg_num(a)), $(_cg_num(b)))"
    if k == PK_STUDENT
        t = "TDist($(_cg_num(a)))"
        isone(c) || (t = "$(_cg_num(c)) * " * t)
        iszero(b) || (t = "$(_cg_num(b)) + " * t)
        return t
    end
    k == PK_CAUCHY && return "Cauchy($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_EXPONENTIAL && return "Exponential($(_cg_inv(a)))"
    k == PK_GAMMA && return "Gamma($(_cg_num(a)), $(_cg_inv(b)))"
    k == PK_INVGAMMA && return "InverseGamma($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_LOGNORMAL && return "LogNormal($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_BETA && return "Beta($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_UNIFORM && return "Uniform($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_LOGISTIC && return "Logistic($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_LAPLACE && return "Laplace($(_cg_num(a)), $(_cg_num(b)))"
    k == PK_SKEWNORMAL && return "SkewNormal($(_cg_num(a)), $(_cg_num(b)), $(_cg_num(c)))"
    _cg_unsupported("the prior $(p)")
end

# the prior on a parameter with support (lb, ub): truncated only where the
# bound actually cuts the distribution's own support
function _cg_dist(p::ScalarPrior, lb::Float64=p.lb, ub::Float64=Inf)
    if isflat(p)
        lb == -Inf && ub == Inf && return "Flat()"
        ub == Inf && return "FlatPos($(_cg_num(lb)))"
        # flat on a bounded interval is uniform (the same density up to a constant)
        lb > -Inf && return "Uniform($(_cg_num(lb)), $(_cg_num(ub)))"
        _cg_unsupported("flat priors with only an upper bound")
    end
    d = _prior_dist(p)
    lo = lb > minimum(d) ? lb : -Inf
    hi = ub < maximum(d) ? ub : Inf
    s = _cg_base(p)
    lo == -Inf && hi == Inf && return s
    lo == -Inf && return "truncated($s; upper=$(_cg_num(hi)))"
    hi == Inf && return "truncated($s; lower=$(_cg_num(lo)))"
    return "truncated($s; lower=$(_cg_num(lo)), upper=$(_cg_num(hi)))"
end

# a vector of n priors: filldist when they are all the same
function _cg_vecdist(ps::AbstractVector{ScalarPrior}, n::Int, lb=nothing)
    ds = [lb === nothing ? _cg_dist(p) : _cg_dist(p, Float64(lb)) for p in ps]
    allequal(ds) && return "filldist($(first(ds)), $n)"
    return "arraydist([" * join(ds, ", ") * "])"
end

# --- the generator -----------------------------------------------------------

_cg_ident(s) = (t = replace(string(s), r"[^A-Za-z0-9_]" => "_"); isempty(t) || isdigit(t[1]) ? "v" * t : t)

# vectorised inverse links, as source
function _cg_linkinv(L::Link, e::String)
    L isa IdentityLink && return e
    L isa LogLink && return "exp.($e)"
    L isa LogitLink && return "invlogit.($e)"
    L isa ProbitLink && return "cdf.(Normal(), $e)"
    L isa CloglogLink && return "1 .- exp.(.-exp.($e))"
    L isa InverseLink && return "inv.($e)"
    L isa SqrtLink && return "$e .^ 2"
    _cg_unsupported("the $(link_name(L)) link")
end

mutable struct _CG
    lines::Vector{String}
    args::Vector{Pair{Symbol,Any}}
    used::Set{String}
    needs_invlogit::Bool
    # each declared parameter: its name and how to compute its value from the
    # package's θ (tests evaluate the generated model at the package's points)
    params::Vector{Pair{Symbol,Any}}
    shared::Dict{Int,Any}            # shared group IDs: sid -> the joint block's names
    latents::Dict{Int,String}        # me(): θ index of meanme -> the latent's name (declared once)
    scalars::Dict{Int,String}        # θ index of a scalar dpar -> its name (horseshoe autoscale)
    ordered::String                  # the dialect's ordered-vector constructor
    backend::Symbol
    ordint::Dict{Int,String}         # mixtures: θ index of an ordered intercept -> its expression
    pins::Dict{Int,Float64}          # constant() priors: θ index -> its fixed unconstrained value
    used_pins::Set{Int}              # the pins the generator wrote as constants
end
_CG(backend::Symbol=:turing) =
    _CG(String[], Pair{Symbol,Any}[], Set(["y", "trials", "i", "family", "xdata", "weights"]), false,
        Pair{Symbol,Any}[], Dict{Int,Any}(), Dict{Int,String}(), Dict{Int,String}(),
        backend === :pb ? "PracticalBayes.Bijectors.ordered" : "ordered", backend, Dict{Int,String}(),
        Dict{Int,Float64}(), Set{Int}())
# a constant() prior's value, marked as written
_cg_pin!(g::_CG, k::Int) = (push!(g.used_pins, k); g.pins[k])

# a prior term that is not a distribution (an improper CAR prior): Turing
# counts it as prior; PracticalBayes's @addlogprob! has only the likelihood form
_cg_addprior(g::_CG, ex::String) =
    g.backend === :turing ? "@addlogprob! (; logprior = $ex)" : "@addlogprob! $ex"

"""
    log_mixing_weights(theta)

Mixing proportions `theta` (a simplex) as the `theta` argument of
[`FamilyDist`](@ref) for a mixture family.
"""
log_mixing_weights(theta::AbstractVector) = LogTheta(log.(theta))

"""
    residual_correlation(L)

The Cholesky factor `L` of logistic_normal's residual correlation, as its
argument of [`FamilyDist`](@ref).
"""
residual_correlation(L::AbstractMatrix) = LnCorL(L)

"""
    thresholds(tau)

Ordinal cut-points `tau` as the threshold argument of [`FamilyDist`](@ref)
(ordinal families take them in the slot of their `thres` parameter).
"""
thresholds(tau::AbstractVector) = Thr(tau)

# ordinal thresholds (ordinal.jl): brms's Intercept[k], ordered for cumulative
# (Stan's ordered transform), one block per level with thres(gr = g); returns
# the FamilyDist argument (per observation)
function _cg_thresholds!(g::_CG, p::Thresholds)
    _cg_line!(g, "# ordinal thresholds (brms Intercept)" * (p.ordered ? ", ordered" : ""))
    blocks = collect(_blocks(p))
    names = String[]
    for (b, (s, n)) in enumerate(blocks)
        nm = _cg_fresh!(g, length(blocks) == 1 ? "Intercept" : "Intercept_" * _cg_ident(p.labels[b]))
        ds = [_cg_dist(p.priors[k]) for k in s:(s + n - 1)]
        d = allequal(ds) ? "filldist($(first(ds)), $n)" : "arraydist([" * join(ds, ", ") * "])"
        _cg_line!(g, "$nm ~ " * (p.ordered ? "$(g.ordered)($d)" : d))
        r = s:(s + n - 1)
        _cg_par!(g, nm, θ -> _eta_generic(p, θ, 0).tau[r])
        push!(names, nm)
    end
    length(blocks) == 1 && return "Ref(thresholds($(names[1])))"
    # grouped: each observation sees its own block
    grp = _cg_arg!(g, _cg_fresh!(g, "thres_group"), copy(p.grp))
    _cg_line!(g, "thres_blocks = [$(join(names, ", "))]")
    push!(g.used, "thres_blocks")
    return "thresholds.(thres_blocks[$grp])"
end

# category-specific effects cs() (ordinal.jl): TAU[k, n] = tau_k - sum_c bcs[c, k] Xcs[n, c]
function _cg_thresholds!(g::_CG, p::ThresholdsCS)
    K1, Kc = _K1(p), _Kcs(p)
    _cg_line!(g, "# ordinal thresholds (brms Intercept) and category-specific effects cs()")
    tau = _cg_fresh!(g, "Intercept")
    ds = [_cg_dist(p.base.priors[k]) for k in 1:K1]
    _cg_line!(g, "$tau ~ " * (allequal(ds) ? "filldist($(first(ds)), $K1)" : "arraydist([" * join(ds, ", ") * "])"))
    ib = p.base.idx
    _cg_par!(g, tau, θ -> θ[ib])
    bcs = _cg_fresh!(g, "bcs")
    cd = [_cg_dist(p.priors[c]) for c in 1:Kc]
    # rows: the cs() columns (each with its prior), columns: the thresholds
    _cg_line!(g, "$bcs ~ " * (allequal(cd) ? "filldist($(first(cd)), $Kc, $K1)" :
                               "arraydist(repeat([" * join(cd, ", ") * "], 1, $K1))"))
    rb = p.bcs
    _cg_par!(g, bcs, θ -> reshape(θ[rb], Kc, K1))
    Xcs = _cg_arg!(g, _cg_fresh!(g, "Xcs"), Matrix{Float64}(p.Xcs))
    TAU = _cg_fresh!(g, "TAU")
    _cg_line!(g, "$TAU = $tau .- transpose($bcs) * transpose($Xcs)")
    return "thresholds.(eachcol($TAU))"
end
_cg_par!(g::_CG, name::String, f) = (push!(g.params, Symbol(name) => f); name)
_cg_line!(g::_CG, s::String="") = (push!(g.lines, isempty(s) ? "" : "    " * s); nothing)
function _cg_arg!(g::_CG, name::String, value)
    push!(g.args, Symbol(name) => value)
    push!(g.used, name)
    return name
end
function _cg_fresh!(g::_CG, base::String)
    nm = base; k = 1
    while nm in g.used
        k += 1; nm = "$(base)_$k"
    end
    push!(g.used, nm)
    return nm
end

# one linear predictor; returns the name of the vector holding it (link scale)
function _cg_linpred!(g::_CG, p::LinPred, sfx::String)
    pre = sfx == "" ? "" : "_" * sfx
    K = length(p.b)
    lhs = p.dpar === :mu ? "eta" : "eta" * pre
    terms = String[]
    hasint = p.intercept > 0
    if hasint || K > 0
        _cg_line!(g, "# population-level effects" * (sfx == "" ? "" : " for $sfx") *
                     (hasint && K > 0 ? " (X is centred; Intercept is the centred intercept, as in brms)" : ""))
    end
    if hasint
        nm = "Intercept" * pre
        if haskey(g.ordint, p.intercept)
            # a mixture component's intercept: an element of the ordered vector
            _cg_line!(g, "$nm = $(g.ordint[p.intercept])")
        elseif haskey(g.pins, p.intercept)
            _cg_line!(g, "$nm = $(_cg_num(_cg_pin!(g, p.intercept)))      # constant() prior")
        else
            _cg_line!(g, "$nm ~ " * _cg_dist(p.int_prior))
            k0 = p.intercept
            _cg_par!(g, nm, θ -> θ[k0])
        end
        push!(terms, nm)
    end
    if K > 0
        X = _cg_arg!(g, "X" * pre, Matrix{Float64}(p.Xc))
        b = "b" * pre
        if p.hs.on
            _cg_horseshoe!(g, p, b, pre)
        elseif p.r2d2.on
            _cg_r2d2!(g, p, b, pre)
        elseif any(k -> haskey(g.pins, k), p.b)
            # constant() priors on some coefficients: those are numbers, the rest b_free
            free = [j for (j, k) in enumerate(p.b) if !haskey(g.pins, k)]
            bf = b * "_free"; push!(g.used, bf)
            if !isempty(free)
                ds = [_cg_dist(p.b_priors[j], p.b_lb[j], p.b_ub[j]) for j in free]
                _cg_line!(g, "$bf ~ " * (allequal(ds) ? "filldist($(first(ds)), $(length(free)))" :
                                         "arraydist([" * join(ds, ", ") * "])"))
                _cg_par!(g, bf, θ -> _bvalues(p, θ)[free])
            end
            parts = String[]
            for (j, k) in enumerate(p.b)
                if haskey(g.pins, k)
                    push!(parts, _cg_num(first(_btrans(p.b_lb[j], p.b_ub[j], _cg_pin!(g, k)))))
                else
                    push!(parts, "$bf[$(findfirst(==(j), free))]")
                end
            end
            _cg_line!(g, "$b = [" * join(parts, ", ") * "]      # constant() priors fix some coefficients")
        elseif p.bounded
            # bounded coefficients (prior(...; lb, ub)): Turing's own transform
            ds = [_cg_dist(p.b_priors[j], p.b_lb[j], p.b_ub[j]) for j in 1:K]
            _cg_line!(g, "$b ~ " * (allequal(ds) ? "filldist($(first(ds)), $K)" : "arraydist([" * join(ds, ", ") * "])"))
            _cg_par!(g, b, θ -> _bvalues(p, θ))
        else
            _cg_line!(g, "$b ~ " * _cg_vecdist(p.b_priors, K))
            _cg_par!(g, b, θ -> _bvalues(p, θ))
        end
        push!(terms, "$X * $b")
        if hasint
            mX = _cg_arg!(g, "means_X" * pre, Vector{Float64}(p.means))
            _cg_line!(g, "b$(pre)_Intercept := Intercept$pre - dot($mX, $b)")
        end
    elseif hasint
        _cg_line!(g, "b$(pre)_Intercept := Intercept$pre")
    end
    if p.has_offset
        push!(terms, _cg_arg!(g, "offset" * pre, Vector{Float64}(p.offset)))
    end
    for gb in p.groups
        push!(terms, _cg_group!(g, gb, sfx))
    end
    for m in p.mo
        push!(terms, _cg_mo!(g, m, pre))
    end
    for mc in p.mecor
        _cg_mecor!(g, mc, pre)          # declares its terms' latents; their bsp terms come below
    end
    for m in p.me
        push!(terms, _cg_me!(g, m, pre))
    end
    for sb in p.smooths
        push!(terms, _cg_smooth!(g, sb, pre))
    end
    for t in p.gp
        push!(terms, _cg_gp!(g, t, pre))
    end
    for t in p.car
        push!(terms, _cg_car!(g, t, pre))
    end
    for t in p.lac
        push!(terms, _cg_lac!(g, t, pre))
    end
    first_vec = findfirst(t -> !startswith(t, "Intercept"), terms)
    if isempty(terms)
        # nothing on this predictor (e.g. an ordinal model with only cs() effects)
        _cg_line!(g, "$lhs = zeros(length(y))")
    elseif first_vec === nothing
        # intercept only: a vector of the right length
        _cg_line!(g, "$lhs = fill($(terms[1]), length(y))")
    else
        _cg_line!(g, "$lhs = " * join(terms, " .+ "))
    end
    _has_ar(p.ar) && _cg_arma!(g, p.ar, lhs, pre)
    return lhs
end

# ar / ma / arma (cov = FALSE), as brms's Stan code (special.jl): within each
# group in time order, e = y - (base + MA part), then the AR part on e
function _cg_arma!(g::_CG, a::ARTerm, lhs::String, pre::String)
    arn, man = "ar" * pre, "ma" * pre
    push!(g.used, arn, man, "e_arma", "T_arma", "m_arma")
    _cg_line!(g, "# autocorrelation (brms ar/ma/arma, cov = FALSE): lags of the residuals within groups")
    if a.p > 0
        ds = [_cg_dist(pr) for pr in a.priors]
        _cg_line!(g, "$arn ~ " * (allequal(ds) ? "filldist($(first(ds)), $(a.p))" : "arraydist([" * join(ds, ", ") * "])"))
        ki = a.idx
        _cg_par!(g, arn, θ -> θ[ki])
    end
    if a.q > 0
        ds = [_cg_dist(pr) for pr in a.ma_priors]
        _cg_line!(g, "$man ~ " * (allequal(ds) ? "filldist($(first(ds)), $(a.q))" : "arraydist([" * join(ds, ", ") * "])"))
        km = a.ma_idx
        _cg_par!(g, man, θ -> θ[km])
    end
    prev = _cg_arg!(g, _cg_fresh!(g, "prev_arma"), copy(a.prev))
    ord = _cg_arg!(g, _cg_fresh!(g, "order_arma"), copy(a.order))
    ya = _cg_arg!(g, _cg_fresh!(g, "y_arma"), Vector{Float64}(a.y))
    pars = [a.p > 0 ? arn : nothing, a.q > 0 ? man : nothing]
    _cg_line!(g, "T_arma = promote_type(eltype($lhs), " * join(["eltype($v)" for v in pars if v !== nothing], ", ") * ")")
    _cg_line!(g, "$lhs = Vector{T_arma}($lhs)")
    _cg_line!(g, "e_arma = zeros(T_arma, length($lhs))")
    _cg_line!(g, "for n in $ord")
    _cg_line!(g, "    m_arma = $lhs[n]")
    a.q > 0 && _cg_line!(g, "    for i in 1:$(a.q); k = $prev[n, i]; k > 0 && (m_arma += $man[i] * e_arma[k]); end")
    _cg_line!(g, "    e_arma[n] = $ya[n] - m_arma")
    a.p > 0 && _cg_line!(g, "    for j in 1:$(a.p); k = $prev[n, j]; k > 0 && (m_arma += $arn[j] * e_arma[k]); end")
    _cg_line!(g, "    $lhs[n] = m_arma")
    _cg_line!(g, "end")
    return nothing
end

# --- non-linear models ----------------------------------------------------------
#
# brms bf(y ~ f(a, b, x), a ~ ..., b ~ ..., nl = TRUE): one linear predictor
# per non-linear parameter, then the user's expression broadcast over the
# observations. The expression is re-read from the formula text (the compiled
# closure cannot be printed); a mismatch would fail the likelihood check.

_cg_subst(ex, m::Dict{Symbol,String}) = ex
_cg_subst(s::Symbol, m::Dict{Symbol,String}) = haskey(m, s) ? Symbol(m[s]) : s
_cg_subst(ex::Expr, m::Dict{Symbol,String}) =
    ex.head === :call ? Expr(:call, ex.args[1], map(a -> _cg_subst(a, m), ex.args[2:end])...) :
                        Expr(ex.head, map(a -> _cg_subst(a, m), ex.args)...)

function _cg_nl!(g::_CG, p::NLPred, info)
    f = info isa ModelInfo ? info.formula : nothing
    (f isa BrmsFormula && f.nl) || _cg_unsupported("a non-linear model without its formula")
    ex = _cg_try(() -> Meta.parse("(" * f.text * ",)"), nothing)
    (ex isa Expr && ex.head === :tuple && _is_tilde(ex.args[1])) ||
        _cg_unsupported("a non-linear formula that cannot be re-read from its text")
    rhs = ex.args[1].args[3]
    m = Dict{Symbol,String}()
    for (k, q) in enumerate(p.pars)
        m[p.names[k]] = _cg_linpred!(g, q, string(p.names[k]))
        _cg_line!(g)
    end
    for (k, v) in enumerate(f.nlvars)
        m[v] = _cg_arg!(g, _cg_fresh!(g, _cg_ident(v)), Vector{Float64}(p.data[k]))
    end
    _cg_line!(g, "# non-linear mean (brms nl = TRUE), evaluated per observation")
    _cg_line!(g, "eta = @. " * string(_cg_subst(rhs, m)))
    push!(g.used, "eta")
    return "eta"
end

# --- special terms ------------------------------------------------------------
# Each follows the term's generic implementation (special.jl, splines.jl,
# gp.jl, horseshoe.jl) line for line, with the unconstrained parameters
# replaced by their natural-scale versions and priors as `~`.

# regularised horseshoe on b (horseshoe.jl):
#   b = zb .* tau .* lambda .* sqrt(c2 / (c2 + tau^2 lambda^2)),  c2 = scale_slab^2 * slab
function _cg_horseshoe!(g::_CG, p::LinPred, b::String, pre::String)
    h = p.hs; K = length(p.b)
    zb, loc, glob, slab = "zb" * pre, "hs_local" * pre, "hs_global" * pre, "hs_slab" * pre
    push!(g.used, zb, loc, glob, slab)
    sc = _cg_num(h.scale_global)
    if h.sigma[] > 0
        sc = "$sc * $(g.scalars[h.sigma[]])"     # autoscale: the global scale times sigma
    end
    _cg_line!(g, "# regularised horseshoe prior on $b (brms horseshoe())")
    _cg_line!(g, "$zb ~ filldist(Normal(), $K)")
    _cg_line!(g, "$loc ~ filldist(truncated(TDist($(_cg_num(h.df))); lower=0), $K)")
    _cg_line!(g, "$glob ~ truncated(($sc) * TDist($(_cg_num(h.df_global))); lower=0)")
    _cg_line!(g, "$slab ~ InverseGamma($(_cg_num(h.df_slab / 2)), $(_cg_num(h.df_slab / 2)))")
    c2 = "$(_cg_num(h.scale_slab^2)) * $slab"
    _cg_line!(g, "$b = $zb .* $glob .* $loc .* sqrt.(($c2) ./ (($c2) .+ $glob^2 .* $loc .^ 2))")
    br, lr, gi, si = p.b, h.loc, h.glob, h.slab
    _cg_par!(g, zb, θ -> θ[br])
    _cg_par!(g, loc, θ -> exp.(θ[lr]))
    _cg_par!(g, glob, θ -> exp(θ[gi]))
    _cg_par!(g, slab, θ -> exp(θ[si]))
    return nothing
end

# R2D2 on b (r2d2.jl): b = zb .* sqrt.(phi .* tau2), tau2 = sigma^2 R2 / (1 - R2)
function _cg_r2d2!(g::_CG, p::LinPred, b::String, pre::String)
    t = p.r2d2; K = length(p.b)
    zb, r2, phi, tau2 = "zb" * pre, "R2D2_R2" * pre, "R2D2_phi" * pre, "R2D2_tau2" * pre
    push!(g.used, zb, r2, phi, tau2)
    _cg_line!(g, "# R2D2 prior on $b (brms R2D2())")
    _cg_line!(g, "$zb ~ filldist(Normal(), $K)")
    _cg_line!(g, "$r2 ~ Beta($(_cg_num(t.a)), $(_cg_num(t.b)))")
    br, kr, kp = p.b, t.r2, t.phi
    _cg_par!(g, zb, θ -> θ[br])
    _cg_par!(g, r2, θ -> logistic(θ[kr]))
    if K == 1
        _cg_line!(g, "$phi = [1.0]")           # one coefficient: the simplex is trivial
    else
        _cg_line!(g, "$phi ~ Dirichlet($K, $(_cg_num(t.cons)))")
        _cg_par!(g, phi, θ -> first(_stick(θ[kp], K)))
    end
    s2 = t.sigma[] > 0 ? "$(g.scalars[t.sigma[]])^2 * " : ""
    _cg_line!(g, "$tau2 = $s2$r2 / (1 - $r2)")
    _cg_line!(g, "$b = $zb .* sqrt.($phi .* $tau2)")
    return nothing
end

# mo(x) (brms): bsp * prod_f D_f * cumsum([0; simo_f])[x_f + 1] * covariates
function _cg_mo!(g::_CG, m::MoTerm, pre::String)
    lab = _cg_ident(m.label) * pre
    bsp = _cg_fresh!(g, "bsp_" * lab)
    _cg_line!(g, "# monotonic effect $(m.label) (brms mo()): simplex increments, Dirichlet prior")
    _cg_line!(g, "$bsp ~ " * _cg_dist(m.bsp_prior))
    kb = m.bsp
    _cg_par!(g, bsp, θ -> θ[kb])
    factors = String[]
    for f in eachindex(m.Ds)
        D = m.Ds[f]
        simo = _cg_fresh!(g, "simo_$(lab)$f")
        _cg_line!(g, "$simo ~ Dirichlet($D, $(_cg_num(m.alphas[f])))")
        rs = m.simos[f]
        _cg_par!(g, simo, θ -> first(_stick(θ[rs], D)))
        xv = _cg_arg!(g, _cg_fresh!(g, "Xmo_$(lab)$f"), copy(m.xs[f]))
        push!(factors, "($D .* vcat(0, cumsum($simo))[$xv .+ 1])")
    end
    csp = isempty(m.csp) ? "" : " .* " * _cg_arg!(g, _cg_fresh!(g, "Csp_" * lab), copy(m.csp))
    return "$bsp .* " * join(factors, " .* ") * csp
end

# me(x, sdx) (brms): latent Xme = meanme + sdme * zme, observed x ~ N(Xme, sdx)
function _cg_me!(g::_CG, m::MeTerm, pre::String)
    lab = _cg_ident(m.label) * pre
    bsp = _cg_fresh!(g, "bsp_" * lab)
    _cg_line!(g, "$bsp ~ " * _cg_dist(m.bsp_prior))
    kb = m.bsp
    _cg_par!(g, bsp, θ -> θ[kb])
    if !haskey(g.latents, m.meanme)
        lat = _cg_ident(m.latent)
        Xme = _cg_fresh!(g, "X" * lat)
        mean, sd, z = "meanme_" * lat, "sdme_" * lat, "zme_" * lat
        push!(g.used, mean, sd, z)
        xo = _cg_arg!(g, _cg_fresh!(g, "xobs_" * lat), Vector{Float64}(m.x))
        sx = _cg_arg!(g, _cg_fresh!(g, "sdx_" * lat), Vector{Float64}(m.sdx))
        _cg_line!(g, "# measurement error (brms me()): latent $Xme, observed $xo with known sd $sx")
        _cg_line!(g, "$mean ~ " * _cg_dist(m.mean_prior))
        _cg_line!(g, "$sd ~ " * _cg_dist(m.sd_prior, 0.0))
        _cg_line!(g, "$z ~ filldist(Normal(), $(length(m.zme)))")
        _cg_line!(g, "$Xme = $mean .+ $sd .* $z")
        _cg_line!(g, "$xo ~ arraydist(Normal.($Xme, $sx))")
        km, ks, kz = m.meanme, m.sdme, m.zme
        _cg_par!(g, mean, θ -> θ[km])
        _cg_par!(g, sd, θ -> exp(θ[ks]))
        _cg_par!(g, z, θ -> θ[kz])
        g.latents[m.meanme] = Xme
    end
    return "$bsp .* $(g.latents[m.meanme])"
end

# car(W, gr, type) (special.jl, brms's Stan code): icar / bym2 / escar / esicar
function _cg_car!(g::_CG, t::CarTerm, pre::String)
    tag = _cg_fresh!(g, "car" * pre)
    sd, z, r, nsz, rho = "sd" * tag, "z" * tag, "r" * tag, "nsz" * tag, "rho" * tag
    push!(g.used, sd, z, r, nsz, rho, "tau_" * tag, "Wr_" * tag)
    Jc = _cg_arg!(g, "J_" * tag, copy(t.J))
    n = t.Nloc
    _cg_line!(g, "# spatial effects car(type = $(t.type)) as brms")
    _cg_line!(g, "$sd ~ " * _cg_dist(t.sd_prior, 0.0))
    ks, kz, kn, kr = t.sd, t.z, t.nsz, t.rho
    _cg_par!(g, sd, θ -> exp(θ[ks]))
    if t.type in (:icar, :bym2)
        e1 = _cg_arg!(g, "edge1_" * tag, copy(t.e1)); e2 = _cg_arg!(g, "edge2_" * tag, copy(t.e2))
        _cg_line!(g, "$z ~ filldist(Flat(), $n)")
        _cg_par!(g, z, θ -> θ[kz])
        _cg_line!(g, "# ICAR prior: pairwise differences over the edges, soft sum-to-zero")
        _cg_line!(g, _cg_addprior(g, "-sum(abs2, $z[$e1] .- $z[$e2]) / 2 - (sum($z) / $(_cg_num(0.001 * n)))^2 / 2"))
        if t.type === :icar
            _cg_line!(g, "$r = $sd .* $z")
        else
            _cg_line!(g, "$nsz ~ filldist(Normal(), $n)")
            _cg_line!(g, "$rho ~ " * _cg_dist(t.rho_prior, 0.0, 1.0))
            _cg_par!(g, nsz, θ -> θ[kn])
            _cg_par!(g, rho, θ -> logistic(θ[kr]))
            _cg_line!(g, "$r = $sd .* (sqrt(1 - $rho) .* $nsz .+ sqrt($rho / $(_cg_num(t.scale))) .* $z)")
        end
        return "$r[$Jc]"
    end
    W = zeros(n, n)
    for (i, j) in zip(t.e1, t.e2)
        W[i, j] = W[j, i] = 1
    end
    Wd = _cg_arg!(g, "W_" * tag, W)
    nn = _cg_arg!(g, "nneigh_" * tag, copy(t.nneigh))
    tau, Wr = "tau_" * tag, "Wr_" * tag
    if t.type === :escar
        eig = _cg_arg!(g, "eig_" * tag, copy(t.eig))
        _cg_line!(g, "$r ~ filldist(Flat(), $n)")
        _cg_line!(g, "$rho ~ " * _cg_dist(t.rho_prior, 0.0, 1.0))
        _cg_par!(g, r, θ -> θ[kz])
        _cg_par!(g, rho, θ -> logistic(θ[kr]))
        _cg_line!(g, "$tau = 1 / $sd^2")
        _cg_line!(g, "$Wr = $Wd * $r")
        _cg_line!(g, "# proper CAR prior (brms escar)")
        _cg_line!(g, _cg_addprior(g, "($n * log($tau) + sum(log1p.(-$rho .* $eig)) - " *
                                     "$tau * (sum($nn .* $r .^ 2) - $rho * sum($r .* $Wr))) / 2"))
    else
        _cg_line!(g, "$z ~ filldist(Flat(), $(n - 1))")
        _cg_par!(g, z, θ -> θ[kz])
        _cg_line!(g, "$r = vcat($z, -sum($z))")
        _cg_line!(g, "$tau = 1 / $sd^2")
        _cg_line!(g, "$Wr = $Wd * $r")
        _cg_line!(g, "# intrinsic CAR prior with a hard sum-to-zero constraint (brms esicar)")
        _cg_line!(g, _cg_addprior(g, "($(n - 1) * log($tau) - $tau * (sum($nn .* $r .^ 2) - sum($r .* $Wr))) / 2"))
    end
    return "$r[$Jc]"
end

# correlated me() latents (brms mecor, special.jl):
#   Xme_k = meanme_k + sdme_k * (Lme * zme)[k, :],  zme ~ N(0, I) (K x N), Lme ~ LKJ
function _cg_mecor!(g::_CG, mc::MeCorTerm, pre::String)
    K = length(mc.terms); N = length(mc.terms[1].x)
    tag = _cg_fresh!(g, "mecor" * pre)
    Lm, zm, Wm = "Lme_" * tag, "zme_" * tag, "Wme_" * tag
    push!(g.used, Lm, zm, Wm)
    _cg_line!(g, "# correlated measurement-error latents (brms me() with mecor = TRUE)")
    _cg_line!(g, "$Lm ~ LKJCholesky($K, $(_cg_num(mc.eta)))")
    _cg_line!(g, "$zm ~ filldist(Normal(), $K, $N)")
    _cg_line!(g, "$Wm = $Lm.L * $zm")
    kc, kz = mc.cor, mc.z
    _cg_par!(g, Lm, θ -> LinearAlgebra.Cholesky(first(_cholcorr(θ[kc], K)), 'L', 0))
    _cg_par!(g, zm, θ -> reshape(θ[kz], K, N))
    for (j, m) in enumerate(mc.terms)
        lat = _cg_ident(m.latent)
        Xme = _cg_fresh!(g, "X" * lat)
        mean, sd = "meanme_" * lat, "sdme_" * lat
        push!(g.used, mean, sd)
        xo = _cg_arg!(g, _cg_fresh!(g, "xobs_" * lat), Vector{Float64}(m.x))
        sx = _cg_arg!(g, _cg_fresh!(g, "sdx_" * lat), Vector{Float64}(m.sdx))
        _cg_line!(g, "$mean ~ " * _cg_dist(m.mean_prior))
        _cg_line!(g, "$sd ~ " * _cg_dist(m.sd_prior, 0.0))
        _cg_line!(g, "$Xme = $mean .+ $sd .* $Wm[$j, :]")
        _cg_line!(g, "$xo ~ arraydist(Normal.($Xme, $sx))")
        km, ks = m.meanme, m.sdme
        _cg_par!(g, mean, θ -> θ[km])
        _cg_par!(g, sd, θ -> exp(θ[ks]))
        g.latents[m.meanme] = Xme
    end
    return nothing
end

# the penalised part of s() / t2() (splines.jl): Zs * (sds .* zs), zs ~ N(0, 1)
function _cg_smooth!(g::_CG, sb::SmoothBlock, pre::String)
    lab = _cg_ident(sb.label) * pre
    sds = _cg_fresh!(g, "sds_" * lab)
    tag = replace(sds, r"^sds_" => "")
    zs = "zs_" * tag; push!(g.used, zs)
    Zs = _cg_arg!(g, "Zs_" * tag, Matrix{Float64}(sb.Z))
    _cg_line!(g, "# smooth $(sb.label): penalised basis, as brms (random-effect form)")
    _cg_line!(g, "$sds ~ " * _cg_dist(sb.sds_prior, 0.0))
    _cg_line!(g, "$zs ~ filldist(Normal(), $(length(sb.zs)))")
    ks, kz = sb.sds, sb.zs
    _cg_par!(g, sds, θ -> exp(θ[ks]))
    _cg_par!(g, zs, θ -> θ[kz])
    return "$Zs * ($sds .* $zs)"
end

# gp() (gp.jl): exact (Cholesky of the kernel matrix) or approximate (basis functions)
function _cg_gp!(g::_CG, t::GPTerm, pre::String)
    lab = _cg_ident(t.label) * pre
    sdgp = _cg_fresh!(g, "sdgp_" * lab)
    tag = replace(sdgp, r"^sdgp_" => "")
    ls, zgp, fg = "lscale_" * tag, "zgp_" * tag, "f_" * tag
    push!(g.used, ls, zgp, fg)
    Jg = _cg_arg!(g, "Jgp_" * tag, copy(t.J))
    _cg_line!(g, "# Gaussian process $(t.label) ($(t.cov) kernel$(t.approx ? ", approximate (Riutort-Mayol et al.)" : "")), as brms")
    _cg_line!(g, "$sdgp ~ " * _cg_dist(t.sd_prior, 0.0))
    _cg_line!(g, "$ls ~ " * _cg_dist(t.ls_prior, 0.0))
    _cg_line!(g, "$zgp ~ filldist(Normal(), $(_nz(t)))")
    ks_, kl, kz = t.sdgp, t.lscale, t.z
    _cg_par!(g, sdgp, θ -> exp(θ[ks_]))
    _cg_par!(g, ls, θ -> exp(θ[kl]))
    _cg_par!(g, zgp, θ -> θ[kz])
    if t.approx
        D = size(t.ks, 2)
        Phi = _cg_arg!(g, "Phi_" * tag, Matrix{Float64}(t.Phi))
        lam = _cg_arg!(g, "lambda_" * tag, Vector{Float64}(t.lam))
        # sqrt of the spectral density divided by sdgp (brms spd_gp_*)
        if t.cov === :exp_quad
            spd = "sqrt.((sqrt(2π) * $ls)^$D .* exp.(-$ls^2 .* $lam ./ 2))"
        else
            ν = t.cov === :matern52 ? 5.0 : t.cov === :matern32 ? 3.0 : 1.0
            logC = D * log(2) + D / 2 * log(π) + loggamma((D + ν) / 2) - log(sqrt(π)) +
                   (t.cov === :matern52 ? 2.5 * log(5) - log(0.75) : t.cov === :matern32 ? 1.5 * log(3) - log(0.5) : 0.0)
            spd = "sqrt.($(_cg_num(exp(logC))) .* $ls^$D .* ($(_cg_num(ν)) .+ $ls^2 .* $lam) .^ $(_cg_num(-(D + ν) / 2)))"
        end
        _cg_line!(g, "$fg = $Phi * ($sdgp .* $spd .* $zgp)")
    else
        D2 = _cg_arg!(g, "D2gp_" * tag, Matrix{Float64}(t.D2))
        K = "K_" * tag; C = "C_" * tag; push!(g.used, K, C)
        if t.cov === :exp_quad
            _cg_line!(g, "$K = $sdgp^2 .* exp.(-$D2 ./ (2 * $ls^2))")
        else
            a = "a_" * tag; push!(g.used, a)
            if t.cov === :matern52
                _cg_line!(g, "$a = sqrt(5) .* sqrt.($D2) ./ $ls")
                _cg_line!(g, "$K = $sdgp^2 .* (1 .+ $a .+ $a .^ 2 ./ 3) .* exp.(-$a)")
            elseif t.cov === :matern32
                _cg_line!(g, "$a = sqrt(3) .* sqrt.($D2) ./ $ls")
                _cg_line!(g, "$K = $sdgp^2 .* (1 .+ $a) .* exp.(-$a)")
            else
                _cg_line!(g, "$a = sqrt.($D2) ./ $ls")
                _cg_line!(g, "$K = $sdgp^2 .* exp.(-$a)")
            end
        end
        # a numerically singular kernel is a rejected proposal (as Stan), not an error
        _cg_line!(g, "$C = cholesky(Symmetric($K + $(_cg_num(_GP_JITTER)) * I); check=false)")
        _cg_line!(g, "if !issuccess($C)")
        _cg_line!(g, "    @addlogprob! -Inf")
        _cg_line!(g, "    return nothing")
        _cg_line!(g, "end")
        _cg_line!(g, "$fg = $C.L * $zgp")
    end
    return "$fg[$Jg]"
end

# The plain (1 | g) case keeps vector-valued effects; everything else uses a
# q x J matrix r (rows = the block's coefficients, columns = levels).
_cg_simple(gb::GroupBlock) = _q(gb) == 1 && gb.sid == 0 && !_hascov(gb) && !_student(gb)

# the parameters of one group-level block: sd, L, z (and student / cov parts),
# and r = the effects. Returns the names the members need.
function _cg_group_params!(g::_CG, gb::GroupBlock, sfx::String)
    lab = _cg_ident(gb.label) * (sfx == "" ? "" : "_" * sfx)
    q, J = _q(gb), _J(gb)
    sd = _cg_fresh!(g, "sd_" * lab)
    tag = replace(sd, r"^sd_" => "")
    z, r, Lc = "z_" * tag, "r_" * tag, "L_" * tag
    push!(g.used, z, r, Lc)
    Jv = _ismm(gb) ? "" : _cg_arg!(g, "J_" * tag, copy(gb.idx))
    NJ = _cg_arg!(g, "N_" * tag, J)
    desc = gb.sid > 0 ? "shared ID block" : "($(join(gb.coefs, " + ")) | $(gb.label)$(gb.sdsfx))"
    _cg_line!(g, "# group-level effects $desc" * (sfx == "" ? "" : " for $sfx") * ", non-centred as in brms")
    # standard deviations live on (0, Inf) whatever the prior row says
    sdr, zr, cr = gb.sd, gb.z, gb.corr
    if _cg_simple(gb)
        _cg_line!(g, "$sd ~ " * _cg_dist(gb.sd_priors[1], 0.0))
        _cg_par!(g, sd, θ -> exp(θ[first(sdr)]))
        _cg_line!(g, "$z ~ filldist(Normal(), $NJ)")
        _cg_par!(g, z, θ -> θ[zr])
        _cg_line!(g, "$r = $sd .* $z")
        return (r=r, J=Jv, tag=tag, simple=true, idx=gb.idx)
    end
    _cg_line!(g, "$sd ~ " * _cg_vecdist(gb.sd_priors, q, 0.0))
    _cg_par!(g, sd, θ -> exp.(θ[sdr]))
    _cg_line!(g, "$z ~ filldist(Normal(), $q, $NJ)")
    _cg_par!(g, z, θ -> reshape(θ[zr], q, J))
    M = z
    if _has_corr(gb)
        _cg_line!(g, "$Lc ~ LKJCholesky($q, $(_cg_num(gb.lkj)))")
        _cg_par!(g, Lc, θ -> LinearAlgebra.Cholesky(first(_cholcorr(θ[cr], q)), 'L', 0))
        M = "$Lc.L * $z"
    end
    if _hascov(gb)
        # gr(cov = A): known covariance among the levels, Lcov = cholesky(A).L
        Lcov = _cg_arg!(g, "Lcov_" * tag, Matrix{Float64}(gb.Lcov))
        M = "($M) * transpose($Lcov)"
    end
    if _student(gb)
        # gr(dist = "student"): level j's effects scaled by sqrt(df * udf_j),
        # udf_j ~ inv_chi_square(df) = InverseGamma(df / 2, 1 / 2)
        df, udf = "df_" * tag, "udf_" * tag
        push!(g.used, df, udf)
        kd, ku = gb.df, gb.udf
        _cg_line!(g, "$df ~ " * _cg_dist(gb.df_prior, 1.0))
        _cg_par!(g, df, θ -> 1 + exp(θ[kd]))
        _cg_line!(g, "$udf ~ filldist(InverseGamma($df / 2, 1 / 2), $NJ)")
        _cg_par!(g, udf, θ -> exp.(θ[ku]))
        _cg_line!(g, "$r = ($sd .* ($M)) .* transpose(sqrt.($df .* $udf))")
    else
        _cg_line!(g, "$r = $sd .* ($M)")
    end
    return (r=r, J=Jv, tag=tag, simple=false, idx=gb.idx)
end

# one group-level term; returns its contribution to the predictor (source)
function _cg_group!(g::_CG, gb::GroupBlock, sfx::String)
    # a shared ID block's parameters are declared once, by its first member
    nm = gb.sid > 0 && haskey(g.shared, gb.sid) ? g.shared[gb.sid] : _cg_group_params!(g, gb, sfx)
    gb.sid > 0 && (g.shared[gb.sid] = nm)
    r, tag = nm.r, nm.tag
    qm = _qm(gb)
    ones_col = [all(isone, view(gb.Z, :, k)) for k in 1:qm]
    Zm = all(ones_col) ? "" : _cg_arg!(g, _cg_fresh!(g, "Z_" * tag), Matrix{Float64}(gb.Z))
    zpart(k, idx) = (rk = nm.simple ? "$r[$idx]" : "$r[$(gb.rows[k]), $idx]";
                     ones_col[k] ? rk : "$rk .* $Zm[:, $k]")
    if _ismm(gb)
        # mm(g1, g2): every observation belongs to several levels, with weights
        JM = _cg_arg!(g, _cg_fresh!(g, "JM_" * tag), Matrix{Int}(gb.mmJ))
        W = _cg_arg!(g, _cg_fresh!(g, "W_" * tag), Matrix{Float64}(gb.mmW))
        parts = String[]
        for m in axes(gb.mmJ, 2)
            inner = join([zpart(k, "$JM[:, $m]") for k in 1:qm], " .+ ")
            push!(parts, "$W[:, $m] .* ($inner)")
        end
        return join(parts, " .+ ")
    end
    Jv = nm.J
    if gb.sid > 0 && (isempty(Jv) || gb.idx != nm.idx)
        Jv = _cg_arg!(g, _cg_fresh!(g, "J_" * tag), copy(gb.idx))
    end
    return join([zpart(k, Jv) for k in 1:qm], " .+ ")
end

# the likelihood line, given natural-scale names per dpar and the link-scale mu
function _cg_likelihood(fam, nat::Dict{Symbol,String}, vecs::Set{String}, eta::String, canon::Bool)
    mu = get(nat, :mu, "")
    sg = get(nat, :sigma, ""); sh = get(nat, :shape, ""); ph = get(nat, :phi, ""); nu = get(nat, :nu, "")
    # check_args=false: during warm-up a mean or precision can underflow to 0.
    # Distributions' constructors then THROW (DomainError), which kills the
    # chain; unchecked, the logpdf is -Inf/NaN and the sampler rejects the step
    # (beta and gamma models fail without it).
    # The canonical-link forms work on the link scale and cannot underflow.
    nc = "; check_args=false"
    fam isa Gaussian && return "y ~ arraydist(Normal.($mu, $sg$nc))"
    fam isa Student && return "y ~ arraydist([$(mu)[i] + $(_cg_at(sg, vecs)) * TDist($(_cg_at(nu, vecs))) for i in eachindex(y)])"
    fam isa Lognormal && return "y ~ arraydist(LogNormal.($mu, $sg$nc))"
    if fam isa Bernoulli
        return canon ? "y ~ arraydist(BernoulliLogit.($eta))" : "y ~ arraydist(Bernoulli.($mu$nc))"
    end
    if fam isa Binomial
        return canon ? "y ~ arraydist(BinomialLogit.(trials, $eta))" : "y ~ arraydist(Binomial.(trials, $mu$nc))"
    end
    if fam isa Poisson
        return canon ? "y ~ arraydist(LogPoisson.($eta))" : "y ~ arraydist(Poisson.($mu$nc))"
    end
    # brms's negbinomial(mu, shape): Distributions' NegativeBinomial(r, p) with r = shape
    fam isa NegBinomial && return "y ~ arraydist(NegativeBinomial.($sh, $sh ./ ($sh .+ $mu)$nc))"
    # brms's Gamma(mu, shape) is mean-parameterised: scale = mu / shape
    fam isa GammaFamily && return "y ~ arraydist(Gamma.($sh, $mu ./ $sh$nc))"
    fam isa Exponential && return "y ~ arraydist(Exponential.($mu$nc))"
    fam isa BetaFamily && return "y ~ arraydist(Beta.($mu .* $ph, (1 .- $mu) .* $ph$nc))"
    _cg_unsupported("the $(nameof(typeof(fam))) family")
end
# element i of a dpar inside a comprehension: indexed if it has a formula
_cg_at(nm::String, vecs) = nm in vecs ? "$nm[i]" : nm

const _CG_HEADERS = Dict(:turing => "using Turing, LinearAlgebra",
                         :pb => "using PracticalBayes, Distributions, LinearAlgebra")

# Residual covariance structures (timecov.jl), gaussian / student, identity
# link: per group (brms orders by (gr, time)) the residuals are multivariate
# normal / t with covariance diag(s) R diag(s), R the leading n x n block of
# one correlation matrix; sar has its own closed form. As brms's Stan code.
function _cg_timecov!(g::_CG, ld::GLMDensity, nat::Dict{Symbol,String}, linkof::Dict{Int,String})
    t = ld.family
    N = ld.N
    mu = linkof[1]                                       # identity link
    sig = haskey(nat, :sigma) ? nat[:sigma] : "exp.($(linkof[2]))"
    vec_sig = !haskey(nat, :sigma)
    nu = _is_student(t) ? nat[:nu] : ""
    _cg_line!(g, "# likelihood: residual correlation structure $(t.kind) (brms), joint per group")
    if _is_sar(t)
        rho = nat[t.kind]
        W = _cg_arg!(g, "W_sar", Matrix{Float64}(t.M))
        eig = _cg_arg!(g, "eig_sar", copy(t.eig))
        push!(g.used, "r_sar")
        _cg_line!(g, "if any($rho .* $eig .>= 1)")
        _cg_line!(g, "    @addlogprob! -Inf")
        _cg_line!(g, "    return nothing")
        _cg_line!(g, "end")
        if t.kind === :lagsar
            _cg_line!(g, "r_sar = y .- $rho .* ($W * y) .- $mu")
        else
            _cg_line!(g, "r_sar = (y .- $mu) .- $rho .* ($W * (y .- $mu))")
        end
        res = _is_student(t) ? "MvTDist($nu, zeros($N), Matrix($sig^2 * I, $N, $N))" : "MvNormal(zeros($N), $sig^2 * I)"
        _cg_line!(g, "# log|I - rho W| (the Jacobian of y -> r) plus the residuals' density")
        _cg_line!(g, "@addlogprob! sum(log1p.(-$rho .* $eig)) + logpdf($res, r_sar)")
        return nothing
    end
    groups = [t.perm[t.beg[k]:(t.beg[k + 1] - 1)] for k in 1:(length(t.beg) - 1)]
    G = _cg_arg!(g, "groups_ac", groups)
    push!(g.used, "R_ac", "C_ac", "S_ac", "obs", "k_ac", "J_ac")
    if t.kind === :fcor
        _cg_arg!(g, "R_ac", t.M * t.M')
    elseif t.kind === :unstr
        # unstr(): one correlation over the time LEVELS; each group uses the
        # rows / columns of the levels it has (brms)
        _cg_line!(g, "R_ac = L_cortime.L * transpose(L_cortime.L)")
        tl = [t.tlev[t.beg[k]:(t.beg[k + 1] - 1)] for k in 1:(length(t.beg) - 1)]
        _cg_arg!(g, "times_ac", tl)
    else
        _cg_Rlines!(g, t, nat, "R_ac", "C_ac")
    end
    sv = vec_sig ? "$sig[obs]" : sig
    blk = t.kind === :unstr ? "R_ac[J_ac, J_ac]" : "R_ac[1:k_ac, 1:k_ac]"
    _cg_line!(g, t.kind === :unstr ? "for (obs, J_ac) in zip($G, times_ac)" : "for obs in $G")
    _cg_line!(g, "    k_ac = length(obs)")
    _cg_line!(g, "    S_ac = Symmetric(" * (vec_sig ? "$sv .* $blk .* transpose($sv)" : "$sig^2 .* $blk") * ")")
    dist = _is_student(t) ? "MvTDist($nu, $mu[obs], Matrix(S_ac))" : "MvNormal($mu[obs], S_ac)"
    _cg_line!(g, "    @addlogprob! logpdf($dist, y[obs])")
    _cg_line!(g, "end")
    return nothing
end

# the residual correlation matrix of the time structures (timecov.jl _tc_R),
# nmax x nmax, and its Cholesky check (not positive definite: rejected)
function _cg_Rlines!(g::_CG, t::TimeCov, nat, R::String, C::String)
    n = t.nmax
    if t.kind === :cosy
        c = nat[:cosy]
        _cg_line!(g, "$R = [i == j ? one($c) : $c for i in 1:$n, j in 1:$n]")
    elseif t.q == 0
        a = nat[:ar]
        _cg_line!(g, "$R = [$a^abs(i - j) / (1 - $a^2) for i in 1:$n, j in 1:$n]")
    elseif t.p == 0
        m = nat[:ma]
        _cg_line!(g, "$R = [i == j ? 1 + $m^2 : abs(i - j) == 1 ? $m : zero($m) for i in 1:$n, j in 1:$n]")
    else
        a, m = nat[:ar], nat[:ma]
        _cg_line!(g, "$R = [i == j ? (1 + $m^2 + 2 * $a * $m) / (1 - $a^2) : " *
                     "(1 + $a * $m) * ($a + $m) * $a^(abs(i - j) - 1) / (1 - $a^2) for i in 1:$n, j in 1:$n]")
    end
    _cg_line!(g, "# a correlation matrix that is not positive definite is a rejected proposal")
    _cg_line!(g, "$C = cholesky(Symmetric($R); check=false)")
    _cg_line!(g, "if !issuccess($C)")
    _cg_line!(g, "    @addlogprob! -Inf")
    _cg_line!(g, "    return nothing")
    _cg_line!(g, "end")
    return nothing
end

# latent residual autocorrelation (timecov.jl, non-gaussian cov = TRUE):
# err[group] = sderr * L_n * zerr[group], L = chol(R(rho)), zerr in (gr, time) order
function _cg_lac!(g::_CG, t::LatentAC, pre::String)
    tc = t.tc
    tag = _cg_fresh!(g, "lac" * pre)
    sd, z, err = "sderr_" * tag, "zerr_" * tag, "err_" * tag
    push!(g.used, sd, z, err, "R_" * tag, "C_" * tag, "T_" * tag)
    _cg_line!(g, "# latent residual autocorrelation ($(tc.kind), brms cov = TRUE for non-gaussian families)")
    _cg_line!(g, "$sd ~ " * _cg_dist(t.sd_prior, 0.0))
    ks, kz = t.sd, t.z
    _cg_par!(g, sd, θ -> exp(θ[ks]))
    nat = Dict{Symbol,String}()
    for k in eachindex(t.rho)
        d = Symbol(replace(t.names[k], r"\[.*\]" => ""))           # "ar[1]" -> :ar
        nm = string(d, "_", tag); push!(g.used, nm)
        _cg_line!(g, "$nm ~ " * _cg_dist(t.rho_priors[k], t.lb[k], t.ub[k]))
        lb, ub, kr = t.lb[k], t.ub[k], t.rho[k]
        _cg_par!(g, nm, θ -> first(_btrans(lb, ub, θ[kr])))
        nat[d] = nm
    end
    _cg_line!(g, "$z ~ filldist(Normal(), $(length(t.z)))")
    _cg_par!(g, z, θ -> θ[kz])
    _cg_Rlines!(g, tc, nat, "R_" * tag, "C_" * tag)
    groups = [tc.perm[tc.beg[k]:(tc.beg[k + 1] - 1)] for k in 1:(length(tc.beg) - 1)]
    pos = [tc.beg[k]:(tc.beg[k + 1] - 1) for k in 1:(length(tc.beg) - 1)]
    G = _cg_arg!(g, "groups_" * tag, groups); P = _cg_arg!(g, "zpos_" * tag, pos)
    _cg_line!(g, "T_$tag = promote_type(eltype($z), typeof($sd), eltype(R_$tag))")
    _cg_line!(g, "$err = zeros(T_$tag, $(length(t.z)))")
    _cg_line!(g, "for (obs, r) in zip($G, $P)")
    _cg_line!(g, "    $err[obs] = $sd .* (LowerTriangular(C_$tag.L[1:length(r), 1:length(r)]) * $z[r])")
    _cg_line!(g, "end")
    return err
end

_cg_try(f, default) = try
    f()
catch
    default
end
_cg_readable(fam) = fam isa _CG_FAMILIES &&
    _cg_try(() -> link(fam), nothing) isa Union{IdentityLink,LogLink,LogitLink,ProbitLink,CloglogLink,InverseLink,SqrtLink}

# a scalar dpar's natural value back on the scale FamilyDist takes (the
# inverse of aux_bounds' transform: log(x - lb), logit, identity)
function _cg_tolink(p::ScalarDpar, nm::String)
    p.kind === :identity && return nm
    if p.kind === :logit
        p.lb == 0 && p.ub == 1 && return "(log($nm) - log1p(-$nm))"
        return "log(($nm - $(_cg_num(p.lb))) / ($(_cg_num(p.ub)) - $nm))"
    end
    return p.lb == 0 ? "log($nm)" : "log($nm - $(_cg_num(p.lb)))"
end

function _codegen_explicit(backend::Symbol, ld::GLMDensity, info; name::Symbol=:regression_model)
    g = _CG(backend)
    r = _cg_glm!(g, ld, info)
    return _cg_assemble(g, backend, info, name, r.readable, r.fname, ld.family)
end

# one GLMDensity's model body into g (its data arguments, parameters and
# lines); withlik = false leaves its likelihood out (a multivariate rescor model
# writes the joint one)
function _cg_glm!(g::_CG, ld::GLMDensity, info; withlik::Bool=true)
    _cg_check(ld)
    fam = ld.family
    for (k, c) in zip(ld.pinned, ld.pinvals)
        g.pins[k] = Float64(c)
    end
    readable = _cg_readable(fam)
    weighted = !isempty(ld.weights)
    discrete = readable && fam isa Union{Bernoulli,Binomial,Poisson,NegBinomial}
    push!(g.args, :y => discrete ? round.(Int, ld.y) : Vector{Float64}(ld.y))
    if readable
        uses_trials(fam) && push!(g.args, :trials => copy(ld.trials))
    elseif fam isa TimeCov
        # the joint likelihood is written out (_cg_timecov!); no family object needed
    else
        # the family object carries per-observation data of its own (censoring
        # codes, truncation bounds, ...); xdata is what its log-likelihood reads
        push!(g.args, :family => fam)
        isempty(ld.xdata) || push!(g.args, :xdata => collect(ld.xdata))
        isempty(ld.trials) || push!(g.args, :trials => copy(ld.trials))
    end
    weighted && push!(g.args, :weights => Vector{Float64}(ld.weights))

    nat = Dict{Symbol,String}()
    vecs = Set{String}()
    links = String[]               # link-scale value per dpar, for FamilyDist
    eta_mu = ""
    canon = readable && ((fam isa Union{Bernoulli,Binomial} && link(fam) isa LogitLink) ||
                         (fam isa Poisson && link(fam) isa LogLink))
    # scalar dpars first: a horseshoe's global scale can depend on sigma
    linkof = Dict{Int,String}()          # pred index -> its FamilyDist argument
    if !isempty(ld.ordered)
        # mixtures of one family: the components' mu intercepts are ordered (brms order = "mu")
        ps = [only(filter(p -> p isa LinPred && p.intercept == k, collect(ld.preds))) for k in ld.ordered]
        ds = [_cg_dist(p.int_prior) for p in ps]
        nm = _cg_fresh!(g, "Intercept_ordered")
        d = allequal(ds) ? "filldist($(first(ds)), $(length(ds)))" : "arraydist([" * join(ds, ", ") * "])"
        _cg_line!(g, "# mixture components identified by ordering their intercepts (brms order = \"mu\")")
        _cg_line!(g, "$nm ~ $(g.ordered)($d)")
        ko = copy(ld.ordered)
        _cg_par!(g, nm, θ -> first(_order_generic(ko, θ))[ko])
        for (k, idx) in enumerate(ld.ordered)
            g.ordint[idx] = "$nm[$k]"
        end
    end
    for (ip, p) in enumerate(ld.preds)
        p isa ScalarDpar || continue
        # gev's xi is brms's unbounded tmp_xi, rescaled from the data below
        nm = fam isa GenExtremeValue && p.dpar === :xi ? "tmp_xi" : string(p.dpar)
        if haskey(g.pins, p.idx)
            _cg_line!(g, "$nm = $(_cg_num(_natural(p, _cg_pin!(g, p.idx))))      # constant() prior")
        else
            _cg_line!(g, "$nm ~ " * _cg_dist(p.prior, p.lb, p.ub))
            _cg_par!(g, nm, θ -> _natural(p, θ[p.idx]))
        end
        nat[p.dpar] = nm
        g.scalars[p.idx] = nm
        linkof[ip] = _cg_tolink(p, nm)
    end
    any(p -> p isa ScalarDpar, ld.preds) && _cg_line!(g)
    for (ip, p) in enumerate(ld.preds)
        p isa ScalarDpar && continue
        if p isa Union{Thresholds,ThresholdsCS}
            linkof[ip] = _cg_thresholds!(g, p)
            _cg_line!(g)
            continue
        elseif p isa ThetaSimplex
            # mixing proportions without formulas: a simplex, Dirichlet prior (brms)
            th = _cg_fresh!(g, string(p.dpar))
            K = _K(p); kt = p.idx
            _cg_line!(g, "$th ~ Dirichlet($K, $(_cg_num(p.alpha)))")
            _cg_par!(g, th, θ -> first(_stick(θ[kt], K)))
            linkof[ip] = "Ref(log_mixing_weights($th))"
            _cg_line!(g)
            continue
        elseif p isa LnCorBlock
            # an LKJ correlation: logistic_normal's residual correlation, or
            # unstr()'s correlation over the time levels (dpar :cortime)
            Lr = _cg_fresh!(g, p.dpar === :cortime ? "L_cortime" : "L_resid")
            D = p.D; kc = p.idx
            _cg_line!(g, "$Lr ~ LKJCholesky($D, $(_cg_num(p.eta)))")
            _cg_par!(g, Lr, θ -> LinearAlgebra.Cholesky(first(_cholcorr(θ[kc], D)), 'L', 0))
            linkof[ip] = "Ref(residual_correlation($Lr.L))"
            _cg_line!(g)
            continue
        elseif p isa BhazSimplex
            # cox: baseline hazard weights on the M-/I-spline bases, a simplex (brms sbhaz)
            sb = _cg_fresh!(g, "sbhaz")
            K = _K(p); kb = p.idx
            Zb = _cg_arg!(g, "Zbhaz", Matrix{Float64}(p.Zb)); Zc = _cg_arg!(g, "Zcbhaz", Matrix{Float64}(p.Zc))
            _cg_line!(g, "# cox baseline hazard: h0 = Zbhaz * sbhaz, cumulative H0 = Zcbhaz * sbhaz (brms)")
            _cg_line!(g, "$sb ~ Dirichlet($K, $(_cg_num(p.alpha)))")
            _cg_par!(g, sb, θ -> first(_stick(θ[kb], K)))
            linkof[ip] = "tuple.($Zb * $sb, $Zc * $sb)"
            _cg_line!(g)
            continue
        end
        d = p.dpar
        begin
            sfx = d === :mu ? "" : string(d)
            e = p isa NLPred ? _cg_nl!(g, p, info) : _cg_linpred!(g, p, sfx)
            linkof[ip] = e
            if !readable
                # FamilyDist takes link-scale predictors as they are
            elseif d === :mu
                eta_mu = e
                L = link(fam)
                if !(canon || L isa IdentityLink)
                    s = _cg_linkinv(L, e)
                    L isa LogitLink && (g.needs_invlogit = true)
                    _cg_line!(g, "mu = $s")
                    nat[d] = "mu"
                else
                    nat[d] = e
                end
            else
                kind, lb = aux_bounds(fam, d)
                nm = string(d)
                if kind === :logit
                    g.needs_invlogit = true
                    _cg_line!(g, "$nm = invlogit.($e)")
                else
                    _cg_line!(g, "$nm = " * (iszero(lb) ? "exp.($e)" : "$(_cg_num(lb)) .+ exp.($e)"))
                end
                nat[d] = nm
                push!(vecs, nm)
            end
        end
        _cg_line!(g)
    end
    if fam isa GenExtremeValue
        # brms: xi = inv_logit(tmp_xi) * (ub - lb) + lb, with bounds that keep
        # the density positive at every standardised residual
        push!(g.used, "x_gev", "b1_gev", "b2_gev", "xi")
        _cg_line!(g, "# gev shape: brms rescales tmp_xi so every observation is inside the support")
        _cg_line!(g, "x_gev = (y .- $(linkof[1])) .* exp.(-$(linkof[2]))")
        _cg_line!(g, "b1_gev = -1 / minimum(x_gev); b2_gev = -1 / maximum(x_gev)")
        _cg_line!(g, "xi = inv(1 + exp(-tmp_xi)) * (max(b1_gev, b2_gev) - min(b1_gev, b2_gev)) + min(b1_gev, b2_gev)")
        _cg_line!(g)
        linkof[3] = "xi"
    end
    links = [linkof[ip] for ip in eachindex(ld.preds)]   # FamilyDist takes them in dpar order
    fname = _cg_try(() -> family_name(fam), lowercase(string(nameof(typeof(fam)))))
    if !withlik
        # (the joint likelihood is written by the caller)
    elseif ld.prior_only
        _cg_line!(g, "# sample_prior = :only: no likelihood (y is not observed)")
    elseif fam isa TimeCov
        _cg_timecov!(g, ld, nat, linkof)
    else
        if readable
            lik = _cg_likelihood(fam, nat, vecs, eta_mu, canon)
            dv = match(r"^y ~ arraydist\((.*)\)$", lik).captures[1]
            _cg_line!(g, "# likelihood" * (occursin("check_args=false", lik) ?
                         " (check_args=false: a parameter that underflows to 0 is a rejected step, not an error)" : ""))
        else
            xs = isempty(ld.xdata) ? "0.0" : "xdata"
            ts = isempty(ld.trials) ? "0" : "trials"
            dv = "FamilyDist.(Ref(family), $xs, $ts, $(join(links, ", ")))"
            _cg_line!(g, "# likelihood: the package's $fname log-likelihood (see ?FamilyDist), given")
            _cg_line!(g, "# the distributional parameters on the link scale, in the order " *
                         _cg_try(() -> join(string.(dpars(fam)), ", "), "of the family's parameters"))
            lik = "y ~ arraydist($dv)"
        end
        if weighted
            _cg_line!(g, "# weights(): each observation's log-likelihood is multiplied by its weight, as brms")
            _cg_line!(g, "@addlogprob! sum(weights .* logpdf.($dv, y))")
        else
            _cg_line!(g, lik)
        end
    end
    # a constant() prior somewhere the generator did not write as a constant
    # (a group sd, a special term's parameter, ...) would be sampled: refuse
    issubset(keys(g.pins), g.used_pins) || _cg_unsupported("constant() priors on this parameter class")
    return (readable=readable, fname=fname, nat=nat, linkof=linkof)
end

# --- multivariate models ------------------------------------------------------------
#
# Each response's model is generated on its own and its names suffixed with the
# response (y -> y_yg, eta -> eta_yg, b -> b_yg, ...); without residual
# correlations the joint density is their sum (as the package's MVDensity).
# With rescor (gaussian responses, brms) the responses' likelihoods are replaced
# by one multivariate normal per observation with an LKJ residual correlation.

# suffix the model-local names of one generated line (code only, not its comment)
function _cg_rename(line::String, names::Vector{String}, sfx::String)
    k = findfirst('#', line)
    code, cmt = k === nothing ? (line, "") : (line[1:prevind(line, k)], line[k:end])
    for nm in names
        code = replace(code, Regex("(?<![A-Za-z0-9_.])" * nm * "(?![A-Za-z0-9_])") => nm * sfx)
    end
    return code * cmt
end

function _codegen_explicit(backend::Symbol, mv::MVDensity, infos; name::Symbol=:regression_model)
    mv.cross && _cg_unsupported("group-level IDs shared across responses")
    isempty(mv.marg) || _cg_unsupported("mi() predictors integrated out")
    mv.nuidx > 0 && _cg_unsupported("student residual correlations")
    isempty(mv.pinned) || _cg_unsupported("constant() priors")
    K = length(mv.subs)
    g = _CG(backend)
    readable = true
    mus = String[]; sigs = String[]; ys = String[]
    for (k, sub) in enumerate(mv.subs)
        gk = _CG(backend)
        info_k = infos isa AbstractVector && length(infos) >= k ? infos[k] : nothing
        r = _cg_glm!(gk, sub, info_k; withlik=!mv.rescor)
        readable &= r.readable
        sfx = "_" * _cg_ident(mv.resp[k])
        names = sort!(collect(union(Set(string.(first.(gk.args))), Set(string.(first.(gk.params))), gk.used));
                      by=length, rev=true)
        _cg_line!(g, "# ---- response $(mv.resp[k]) ----")
        append!(g.lines, (_cg_rename(l, names, sfx) for l in gk.lines))
        append!(g.args, (Symbol(string(a) * sfx) => v for (a, v) in gk.args))
        append!(g.params, (Symbol(string(a) * sfx) => f for (a, f) in gk.params))
        g.needs_invlogit |= gk.needs_invlogit
        if mv.rescor
            (sub.family isa Gaussian && link(sub.family) isa IdentityLink) ||
                _cg_unsupported("residual correlations of non-gaussian responses")
            push!(mus, _cg_rename(r.nat[:mu], names, sfx))
            sv = sub.preds[mv.ksig[k]] isa ScalarDpar
            push!(sigs, _cg_rename(r.nat[:sigma], names, sfx) * (sv ? "" : "[n]"))
            push!(ys, "y" * sfx)
        end
    end
    if mv.rescor
        push!(g.used, "Lrescor", "sig_n", "S_n", "n")
        ri = mv.ridx
        _cg_line!(g, "# ---- residual correlations between the responses (brms rescor) ----")
        _cg_line!(g, "Lrescor ~ LKJCholesky($K, $(_cg_num(mv.lkj)))")
        _cg_par!(g, "Lrescor", θ -> LinearAlgebra.Cholesky(first(_cholcorr(θ[ri], K)), 'L', 0))
        _cg_line!(g, "for n in 1:$(mv.N)")
        _cg_line!(g, "    sig_n = [" * join(sigs, ", ") * "]")
        _cg_line!(g, "    S_n = Symmetric(Diagonal(sig_n) * (Lrescor.L * transpose(Lrescor.L)) * Diagonal(sig_n))")
        _cg_line!(g, "    @addlogprob! logpdf(MvNormal([" * join(("$m[n]" for m in mus), ", ") * "], S_n), [" *
                     join(("$y[n]" for y in ys), ", ") * "])")
        _cg_line!(g, "end")
    end
    return _cg_assemble(g, backend, nothing, name, readable, "multivariate (" * join(mv.resp, ", ") * ")", nothing)
end

# the reference for the multivariate likelihood: the sum of the responses',
# or with rescor the joint normal per observation (normalised)
function _cg_loglik(mv::MVDensity, θ)
    mv.rescor || return sum(_cg_loglik(sub, θ) for sub in mv.subs)
    K, N = length(mv.subs), mv.N
    mu = [_eta_generic(sub.preds[mv.kmu[k]], θ, N) for (k, sub) in enumerate(mv.subs)]
    ls = [_eta_generic(sub.preds[mv.ksig[k]], θ, N) for (k, sub) in enumerate(mv.subs)]
    L = first(_cholcorr(θ[mv.ridx], K))
    s = zero(eltype(θ))
    for n in 1:N
        sig = [exp(_at(ls[k], n)) for k in 1:K]
        z = [(mv.subs[k].y[n] - _at(mu[k], n)) / sig[k] for k in 1:K]
        v = LowerTriangular(L) \ z
        s += -sum(abs2, v) / 2 - sum(log, sig) - sum(log(L[i, i]) for i in 1:K) - K * log(2π) / 2
    end
    return s
end

# the model source and data from a filled _CG
function _cg_assemble(g::_CG, backend::Symbol, info, name::Symbol, readable::Bool, fname::String, fam)
    io = IOBuffer()
    println(io, "# Generated by PracticalBayesianRegression.$(backend === :turing ? "turing_code" : "pb_code").")
    _cg_print_formula(io, info)
    lname = _cg_try(() -> link_name(link(fam)), "")
    println(io, "# family:  ", fname, isempty(lname) ? "" : " ($lname link)")
    println(io, "# Data: `model_data(...)` returns the arguments in order.")
    g.needs_invlogit && println(io, "\ninvlogit(x) = inv(one(x) + exp(-x))")
    println(io)
    argnames = join(string.(first.(g.args)), ", ")
    println(io, "@model function $name($argnames)")
    for l in g.lines
        println(io, l)
    end
    println(io, "end")
    body = String(take!(io))
    # trailing blank lines inside the model body are noise
    body = replace(body, r"\n\n+(\s*end\n)$" => s"\n\1")
    data = NamedTuple{Tuple(first.(g.args))}(Tuple(last.(g.args)))
    hdr = readable ? _CG_HEADERS[backend] : _CG_HEADERS[backend] * ", PracticalBayesianRegression"
    pn = Tuple(first.(g.params)); pf = last.(g.params)
    params = θ -> NamedTuple{pn}(Tuple(f(θ) for f in pf))
    return (code=ModelCode(backend, hdr, body, name), data=data, params=params)
end
