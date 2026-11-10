# =============================================================================
# brm(): build, sample, and wrap the result.
#
# Defaults follow brms/Stan: 4 chains, iter = 2000 with the first half as
# warm-up, adapt_delta 0.8, max treedepth 10, inits uniform(-2, 2) on the
# unconstrained scale, diagonal metric with Stan's windowed adaptation
# (AdvancedHMC's StanHMCAdaptor). Matching these matters when comparing with
# brms: a different warm-up budget alone changes ESS/s several-fold.
# =============================================================================

"""
    BrmsFit

A fitted model, returned by [`brm`](@ref). Displaying it prints brms's
summary table. Get the parameter names with [`variables`](@ref) and their
draws with [`draws`](@ref); `fit.chain` holds the draws as a
`FlexiChains.FlexiChain` keyed by the same names.
"""
struct BrmsFit{D<:GLMDensity}
    info::ModelInfo
    density::D
    chain::FlexiChains.FlexiChain{String}
    draws::Array{Float64,3}
    sampler::String
    iter::Int
    warmup::Int
    thin::Int
    sampling_time::Vector{Float64}   # seconds per chain, warm-up included
    call::Dict{Symbol,Any}           # data, formula, family, prior, options (for update / kfold)
end

"""
    nchains(fit) -> Int

The number of chains in a fit.
"""
nchains(fit::BrmsFit) = size(fit.draws, 3)

"""
    ndraws(fit) -> Int

The total number of post-warm-up draws in a fit, over all chains.
"""
ndraws(fit::BrmsFit) = size(fit.draws, 2) * size(fit.draws, 3)

"""
    brm(formula, data; family = gaussian(), prior = PriorSpec[], kwargs...) -> BrmsFit

Fit a Bayesian regression model, as brms's `brm`. By default the posterior
is sampled with NUTS: 4 chains of 2000 iterations, the first half warm-up.

# Arguments
- `formula`: a formula from [`@bf`](@ref) or [`bf`](@ref), or a formula
  string. Adding formulas (`@bf(y1 ~ x) + @bf(y2 ~ x)`) gives a multivariate
  model, whose result is a [`BrmsMVFit`](@ref).
- `data`: any Tables.jl table, such as a `DataFrame` or a NamedTuple of
  vectors. Rows with missing values in the variables the model uses are
  dropped, with a warning, unless the model imputes them (`mi()`).

# Keywords
- `family`: the response distribution, e.g. `poisson()` or
  `bernoulli(link = :probit)`, or a brms family name (`"poisson"`); see
  [`brmsfamily`](@ref). A vector gives one family per response of a
  multivariate model.
- `prior`: a vector of [`prior`](@ref) rows. Parameters without one get
  brms's default prior (see [`get_prior`](@ref)).
- `data2`: a NamedTuple of extra data that is not one row per observation,
  such as the matrices of `car(W)` or `gr(g, cov = A)`.
- `chains = 4`, `iter = 2000`, `warmup = iter ÷ 2`, `thin = 1`: the number of
  chains, iterations per chain (warm-up included), warm-up iterations and the
  thinning interval.
- `seed`: an integer for reproducible results.
- `delta = 0.8`: the target acceptance rate (brms's `adapt_delta`). Raise it
  (to 0.95, say) when the fit reports divergent transitions.
- `max_depth = 10`: the maximum tree depth (brms's `max_treedepth`).
- `metric = :diag`: the mass matrix adapted during warm-up, `:diag` or
  `:dense` (see the sampler guide).
- `init = 2.0`: initial values are drawn uniformly from `(-init, init)` on the
  unconstrained scale.
- `threads = true`: run the chains in parallel when Julia has more than one
  thread (start Julia with `-t 4`).
- `sampler = :nuts`: the package's NUTS; `:advancedhmc` or an
  `AdvancedHMC.AbstractHMCSampler` uses AdvancedHMC.jl instead.
- `algorithm = :sampling`: or `:laplace`, `:meanfield`, `:fullrank`,
  `:pathfinder` for an approximate posterior of `ndraws = 1000` draws.
- `sample_prior = :no`: `:yes` also draws from the priors (for
  [`hypothesis`](@ref)); `:only` samples the prior alone, which needs a
  proper prior on every parameter.
- `mi = :auto`: how missing values modelled with `mi()` are handled.
  `:auto` integrates them out where that is exact or accurate by quadrature
  and samples the rest; `:sample` samples every missing value, as brms does.
- `T = Float64`: the floating-point type of the sampler (`Float32` for single
  precision).

# Examples
```julia
fit = brm(@bf(y ~ x + (1 | g)), df; family = poisson(), seed = 1)
fit = brm("y ~ x", df; prior = [prior("normal(0, 1)"; class = :b)])
fit = brm(@bf(y ~ x), df; chains = 4, iter = 4000, delta = 0.95)
```
"""
function brm(formula, data; family=gaussian(), prior=PriorSpec[], chains::Int=4, iter::Int=2000,
             warmup::Int=iter ÷ 2, thin::Int=1, seed=nothing, sampler=:nuts, delta=0.8,
             max_depth::Int=10, metric::Symbol=:diag, init=2.0, threads::Bool=true,
             T::Type{<:AbstractFloat}=Float64, sample_prior=:no, data2=nothing, mi::Symbol=:auto,
             algorithm::Symbol=:sampling, ndraws::Int=1000)
    fb = formula isa MVFormula ? formula : bf(formula)
    # inference barrier: a run-time branch to the multivariate path made every
    # univariate fit infer it (and, through it, the approximate algorithms)
    _is_mv(fb) && return Base.inferencebarrier(_brm_mv)(fb, data; family, prior, chains, iter, warmup, thin, seed,
                                                        sampler, delta, max_depth, metric, init, threads, T,
                                                        sample_prior, mi, algorithm, ndraws)
    iter > warmup || throw(ArgumentError("iter must exceed warmup"))
    mi in (:auto, :sample) || throw(ArgumentError("mi must be :auto (integrate missing values out where exact) or :sample (brms)"))
    ld, info = build_model(formula, data; family=family, prior=prior, T=T, sample_prior=sample_prior, data2=data2,
                           mi_mode=mi)
    fit = sample_model(ld, info; chains, iter, warmup, thin, seed, sampler, delta, max_depth, metric, init, threads,
                       algorithm, ndraws)
    Symbol(sample_prior) === :yes && (fit = _with_prior_draws(fit, seed))
    merge!(fit.call, Dict{Symbol,Any}(:formula => formula, :data => data, :family => family, :prior => prior,
        :chains => chains, :iter => iter, :warmup => warmup, :thin => thin, :sampler => sampler,
        :delta => delta, :max_depth => max_depth, :metric => metric, :init => init, :threads => threads, :T => T,
        :sample_prior => sample_prior, :algorithm => algorithm))
    return fit
end

"""
    sample_model(ld, info; kwargs...) -> BrmsFit

Sample an already-built model (see [`build_model`](@ref)); the keywords are
those of [`brm`](@ref). Useful for timing sampling separately from building.
"""
function sample_model(ld::GLMDensity{F,P,W,T}, info::ModelInfo; chains::Int=4, iter::Int=2000,
                      warmup::Int=iter ÷ 2, thin::Int=1, seed=nothing, sampler=:nuts, delta=0.8,
                      max_depth::Int=10, metric::Symbol=:diag, init=2.0, threads::Bool=true,
                      algorithm::Symbol=:sampling, ndraws::Int=1000) where {F,P,W,T}
    # brms algorithm = "laplace" / "meanfield" / "fullrank" / "pathfinder"
    # (approx.jl), behind an inference barrier: with `algorithm` a run-time
    # Symbol, a plain branch made every sampling fit infer and compile all four
    # approximations (16% of a first fit's compile time). The assertion keeps
    # this method's return type concrete.
    algorithm === :sampling ||
        return Base.inferencebarrier(_approx_fit)(ld, info, algorithm; seed, ndraws, init)::BrmsFit{typeof(ld)}
    draws, results, name = _sample_core(ld, T; chains, iter, warmup, thin, seed, sampler, delta, max_depth, metric,
                                        init, threads)
    return _finish_fit(ld, info, draws, results, name, iter, warmup, thin, seed)
end

function _approx_fit(ld, info, algorithm; seed, ndraws, init)
    draws, results, name = _approx_draws(ld, algorithm; seed, ndraws, init)
    return _finish_fit(ld, info, draws, results, name, ndraws, 0, 1, seed)
end

function _finish_fit(ld, info, draws, results, name, iter, warmup, thin, seed)
    # integrated-out missing values: their draws, as generated quantities past θ
    draws = _mi_generate(ld, draws, seed === nothing ? Random.default_rng() : Random.Xoshiro(seed + 104729))
    chain = _make_chain(ld, draws, results)
    fit = BrmsFit(info, ld, chain, draws, name, iter, warmup, thin, [r.time for r in results], Dict{Symbol,Any}())
    _warn_divergences(results)
    return fit
end

# run the chains of any density with the package's interface (GLMDensity,
# MVDensity); returns the unconstrained draws, per-chain results and the name
function _sample_core(ld, ::Type{T}; chains::Int=4, iter::Int=2000, warmup::Int=iter ÷ 2, thin::Int=1,
                      seed=nothing, sampler=:nuts, delta=0.8, max_depth::Int=10, metric::Symbol=:diag,
                      init=2.0, threads::Bool=true) where {T}
    seed0 = seed === nothing ? rand(Random.RandomDevice(), UInt32) : seed
    nkeep = (iter - warmup) ÷ thin
    run = c -> begin
        # (seed, chain) hashed together: `seed0 + c` made seed s chain c the same
        # stream as seed s + 1 chain c - 1, so "different seeds" shared chains
        rng = Random.Xoshiro(hash((seed0, c)))
        θ0 = _initial_values(ld, rng, init, T)
        _run_chain(fresh_workspace(ld), sampler, nkeep, warmup, thin, θ0, rng, T(delta), max_depth;
                   metric=metric)
    end
    results = if threads && Threads.nthreads() > 1 && chains > 1
        # One BLAS thread per chain while chains run in parallel: otherwise
        # every chain's gemv spawns OpenBLAS's own thread pool on the same
        # cores, which makes models with many columns markedly slower.
        nb = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            tasks = [Threads.@spawn run(c) for c in 1:chains]
            map(fetch, tasks)
        finally
            BLAS.set_num_threads(nb)
        end
    else
        map(run, 1:chains)
    end

    draws = Array{Float64,3}(undef, ld.dim, nkeep, chains)
    for (c, r) in enumerate(results)
        draws[:, :, c] .= r.draws
    end
    name = sampler isa Symbol ? string(sampler) : string(nameof(typeof(sampler)))
    sampler === :nuts && metric === :dense && (name *= " (dense metric)")
    _order_draws!(ld, draws)        # ordered intercepts: store tau, not u
    return draws, results, name
end

# Stan's initialisation: uniform(-init, init) on the unconstrained scale,
# redrawn (up to 100 attempts, as Stan) until the log density and its gradient
# are finite. Needed e.g. for ma(): |ma| > 1 makes the error recursion explode
# over a long series, so half of all uniform(-2, 2) draws are unusable.
function _initial_values(ld, rng, init, T)
    ws = fresh_workspace(ld)
    g = zeros(T, ld.dim)
    for _ in 1:100
        θ0 = T.(init .* (2 .* rand(rng, ld.dim) .- 1))
        # a domain error at a random start is a rejected start, as in Stan (redraw)
        lp = _safe_grad!((gg, x) -> logdensity_and_gradient!(gg, ws, x), g, θ0)
        isfinite(lp) && all(isfinite, g) && return θ0
    end
    throw(ArgumentError("no finite log density in 100 random initialisations (uniform(-$init, $init) " *
                        "on the unconstrained scale); try a smaller `init`"))
end

function _run_chain(ld, sampler, nkeep, nwarm, thin, θ0, rng, delta, max_depth; metric=:diag)
    t0 = time()
    if sampler === :nuts
        # the package's own allocation-free NUTS (nuts.jl): Stan's algorithm
        gradf! = (g, x) -> logdensity_and_gradient!(g, ld, x)
        dr, st = nuts_chain(gradf!, θ0, nkeep, nwarm; delta=delta, max_depth=max_depth, thin=thin,
                            metric=metric, rng=rng)
        return (draws=dr, stats=st, ndiv=round(Int, sum(st["divergent__"])), time=time() - t0)
    end
    # behind an inference barrier: with `sampler` a run-time value, a plain
    # branch made every NUTS fit infer and compile AdvancedHMC (17% of a first
    # fit's compile time). Both branches return a _ChainResult.
    return Base.inferencebarrier(_run_chain_ahmc)(ld, sampler, nkeep, nwarm, thin, θ0, rng, delta, max_depth,
                                                  t0)::_ChainResult
end

const _ChainResult = @NamedTuple{draws::Matrix{Float64}, stats::Dict{String,Vector{Float64}}, ndiv::Int,
                                 time::Float64}

function _run_chain_ahmc(ld, sampler, nkeep, nwarm, thin, θ0, rng, delta, max_depth, t0)
    sampler isa Symbol && sampler !== :advancedhmc &&
        throw(ArgumentError("unknown sampler `:$sampler` (use :nuts, :advancedhmc or an AdvancedHMC sampler)"))
    spl = sampler === :advancedhmc ? AdvancedHMC.NUTS(delta; max_depth=max_depth) : sampler
    model = AbstractMCMC.LogDensityModel(ld)
    ts = AbstractMCMC.sample(rng, model, spl, nkeep; n_adapts=nwarm, discard_initial=nwarm,
                             thinning=thin, initial_params=θ0, progress=false)
    draws = reduce(hcat, [Float64.(t.z.θ) for t in ts])
    stats = Dict{String,Vector{Float64}}()
    for (k, nm) in ((:log_density, "lp__"), (:acceptance_rate, "accept_stat__"),
                    (:step_size, "stepsize__"), (:tree_depth, "treedepth__"),
                    (:n_steps, "n_leapfrog__"), (:numerical_error, "divergent__"))
        haskey(ts[1].stat, k) && (stats[nm] = [Float64(getfield(t.stat, k)) for t in ts])
    end
    ndiv = haskey(stats, "divergent__") ? Int(sum(stats["divergent__"])) : 0
    return (draws=draws, stats=stats, ndiv=ndiv, time=time() - t0)
end

# brms sample_prior = "yes": independent draws from each prior, one variable per
# class (or per coefficient when the coefficients' priors differ), restricted to
# the parameter's support by rejection as brms's generated quantities do
function _with_prior_draws(fit::BrmsFit, seed)
    info = fit.info
    rng = Random.Xoshiro(seed === nothing ? rand(Random.RandomDevice(), UInt32) : seed + 7919)
    niter, nch = size(fit.draws, 2), size(fit.draws, 3)
    groups = Dict{Tuple{Symbol,String,String},Vector{PriorSpec}}()
    order = Tuple{Symbol,String,String}[]
    for r in info.priors
        r.prior isa ScalarPrior && !isflat(r.prior) && !isconstant(r.prior) || continue
        key = (r.class, r.dpar, r.group)
        haskey(groups, key) || push!(order, key)
        push!(get!(groups, key, PriorSpec[]), r)
    end
    d = Dict{Any,Matrix{Float64}}(k => Matrix{Float64}(fit.chain[k]) for k in keys(fit.chain))
    for key in order
        rows = groups[key]
        cls, dp, gr = key
        lb, ub = _prior_support(info.family, cls, rows[1])
        base = "prior_" * string(cls) * (dp == "" ? "" : "_" * dp) * (gr == "" ? "" : "_" * gr)
        same = all(r -> r.prior == rows[1].prior, rows)
        for r in (same ? rows[1:1] : rows)
            name = same || r.coef == "" ? base : base * "_" * r.coef
            v = _prior_draws(rng, r.prior, niter * nch, lb, ub)
            v === nothing || (d[FlexiChains.Parameter(name)] = reshape(v, niter, nch))
        end
    end
    chain = FlexiChains.FlexiChain{String}(niter, nch, d; sampling_time=fit.sampling_time)
    return BrmsFit(info, fit.density, chain, fit.draws, fit.sampler, fit.iter, fit.warmup, fit.thin,
                   fit.sampling_time, fit.call)
end
function _prior_support(fam, cls, row)
    cls in (:b, :Intercept) && return row.lb, row.ub
    cls in (:sd, :sds) && return 0.0, Inf
    cls === :df && return 1.0, Inf                  # student-t group effects
    cls in dpars(fam) || return -Inf, Inf
    lk, lb = aux_bounds(fam, cls)
    lk === :identity && return -Inf, Inf
    return lb, lk === :logit ? aux_ub(fam, cls) : Inf
end

function _make_chain(ld, draws, results)
    _, niter, nch = size(draws)
    names = variables(ld)
    vals = Array{Float64,3}(undef, length(names), niter, nch)
    for c in 1:nch, i in 1:niter
        vals[:, i, c] .= constrain(ld, view(draws, :, i, c))
    end
    d = Dict{Any,Matrix{Float64}}()
    for (k, n) in enumerate(names)
        d[FlexiChains.Parameter(n)] = vals[k, :, :]
    end
    for key in keys(results[1].stats)
        d[FlexiChains.Extra(key)] = reduce(hcat, [r.stats[key] for r in results])
    end
    return FlexiChains.FlexiChain{String}(niter, nch, d; sampling_time=[r.time for r in results])
end

function _warn_divergences(results)
    n = sum(r -> r.ndiv, results)
    n > 0 && @warn "There were $n divergent transitions after warmup. Increasing `delta` above $(0.8) may help."
    return nothing
end

# --- draws access ------------------------------------------------------------

"""
    variables(fit) -> Vector{String}

The names of a fit's parameters, as brms names them: `b_Intercept`, `b_x`,
`sd_g__Intercept`, `cor_g__Intercept__x`, `sigma`, `r_g[a,Intercept]`, ...
"""
variables(fit::BrmsFit) = variables(fit.density)

"""
    draws(fit, name) -> Matrix{Float64}
    draws(fit) -> (Matrix{Float64}, Vector{String})

Posterior draws on the parameters' natural scale. With a parameter `name`, an
iterations x chains matrix; without, a draws x parameters matrix (chains
stacked) and the parameter names.

```julia
b = vec(draws(fit, "b_x"))       # all draws of the slope
```
"""
draws(fit::BrmsFit, name::AbstractString) = Matrix{Float64}(fit.chain[FlexiChains.Parameter(String(name))])
function draws(fit::BrmsFit)
    names = variables(fit)
    M = reduce(hcat, [vec(draws(fit, n)) for n in names])
    return M, names
end
