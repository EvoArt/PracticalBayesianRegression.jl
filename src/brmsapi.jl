# =============================================================================
# Smaller brms functions, under brms's names: accessors (vcov, ngrps, logLik,
# nvariables, as_draws_matrix / as_draws_df), formula helpers (mvbf), the
# remaining loo-package entry points (reloo, loo_moment_match,
# loo_predictive_interval, loo_model_weights, kfold_predict), conditions for
# conditional_effects (make_conditions) and brms's scaled-logit helpers.
# =============================================================================

"""
    vcov(fit; correlation = false) -> (Matrix, names)

Posterior covariance (or correlation) matrix of the population-level effects,
as brms's `vcov`, with the coefficient names.
"""
function StatsAPI.vcov(fit::BrmsFit; correlation::Bool=false)
    nm = [r.variable for r in fixef(fit)]
    M = reduce(hcat, [vec(draws(fit, "b_" * n)) for n in nm])
    C = correlation ? cor(M) : cov(M)
    return C, nm
end

"""
    logLik(fit; kwargs...) -> Matrix

brms's `logLik`: an alias of [`log_lik`](@ref) (draws x observations).
"""
logLik(fit; kwargs...) = log_lik(fit; kwargs...)

"""
    ngrps(fit) -> Dict{String,Int}

The number of levels of each grouping factor, as brms's `ngrps`.
"""
function ngrps(fit::BrmsFit)
    out = Dict{String,Int}()
    for p in fit.density.preds
        for q in (p isa NLPred ? collect(p.pars) : p isa LinPred ? [p] : LinPred[])
            for gb in q.groups
                out[gb.label] = max(get(out, gb.label, 0), length(gb.levels))
            end
        end
    end
    return out
end

"""
    nvariables(fit) -> Int

The number of parameters stored per draw, as `posterior::nvariables`.
"""
nvariables(fit::Union{BrmsFit,BrmsMVFit}) = length(variables(fit))

"""
    as_draws_matrix(fit) -> (Matrix, names)
    as_draws_df(fit) -> NamedTuple of columns

The posterior draws as a draws x parameters matrix (chains stacked), or as a
table with brms's `.chain`, `.iteration` and `.draw` columns (any Tables.jl
sink takes it, e.g. `DataFrame(as_draws_df(fit))`), as the posterior package's
converters that brms re-exports.
"""
as_draws_matrix(fit::BrmsFit) = draws(fit)
function as_draws_df(fit::BrmsFit)
    M, nm = draws(fit)
    ni, nc = size(fit.draws, 2), size(fit.draws, 3)
    cols = Pair{Symbol,Vector}[Symbol(n) => M[:, k] for (k, n) in enumerate(nm)]
    push!(cols, Symbol(".chain") => repeat(1:nc; inner=ni), Symbol(".iteration") => repeat(1:ni, nc),
          Symbol(".draw") => collect(1:(ni * nc)))
    return (; cols...)
end

"""
    mvbf(formulas...; rescor = nothing) -> MVFormula

brms's `mvbf`: a multivariate model from several formulas, the same as
adding them (`f1 + f2`); `rescor` sets residual correlations as `set_rescor`.
"""
function mvbf(fs::BrmsFormula...; rescor::Union{Nothing,Bool}=nothing)
    length(fs) >= 1 || throw(ArgumentError("mvbf() needs formulas"))
    m = MVFormula(collect(BrmsFormula, fs))
    return rescor === nothing ? m : MVFormula(m.forms, rescor)
end

"""
    loo_moment_match(fit; k_threshold = 0.7, kwargs...) -> LOOResult

brms's `loo_moment_match`: PSIS-LOO with moment matching for the observations
whose Pareto k exceeds `k_threshold` (the same as `loo(fit; moment_match = true)`).
"""
loo_moment_match(fit; kwargs...) = loo(fit; moment_match=true, kwargs...)

"""
    reloo(fit; k_threshold = 0.7, loo_result = loo(fit), kwargs...) -> LOOResult

brms's `reloo`: for every observation whose Pareto k exceeds `k_threshold`,
refit the model without it and use its exact leave-one-out log predictive
density. Extra keywords go to the refits (e.g. `chains = 2`).
"""
function reloo(fit::BrmsFit; k_threshold=0.7, loo_result=loo(fit), kwargs...)
    isempty(fit.call) && throw(ArgumentError("reloo needs a fit made by `brm`"))
    bad = findall(>(k_threshold), loo_result.pareto_k)
    isempty(bad) && return loo_result
    cols = Tables.columntable(fit.call[:data])
    rows = fit.info.rows
    sub(idx) = NamedTuple{keys(cols)}(map(v -> v[idx], values(cols)))
    elpd = copy(loo_result.pointwise); ks = copy(loo_result.pareto_k)
    for i in bad
        fi = update(fit; newdata=sub(rows[setdiff(eachindex(rows), i)]), kwargs...)
        ll = vec(log_lik(fi; newdata=sub(rows[i:i]), allow_new_levels=true))
        elpd[i] = logsumexp(ll) - log(length(ll))
        ks[i] = 0.0
    end
    LL = log_lik(fit)
    lppd = [logsumexp(view(LL, :, i)) - log(size(LL, 1)) for i in axes(LL, 2)]
    return LOOResult(:loo, sum(elpd), _se(elpd), sum(lppd .- elpd), _se(lppd .- elpd), elpd, ks)
end

"""
    loo_predictive_interval(fit; prob = 0.9) -> Matrix

The central `prob` interval of each observation's leave-one-out predictive
distribution (PSIS-weighted quantiles), as brms's `loo_predictive_interval`.
"""
function loo_predictive_interval(fit::BrmsFit; prob=0.9, kwargs...)
    a = (1 - prob) / 2
    Q = loo_predict(fit; type=:quantile, probs=[a, 1 - a], kwargs...)
    return Q isa AbstractMatrix && size(Q, 2) == 2 ? Q : reduce(vcat, permutedims.(Q))
end

"""
    loo_model_weights(fits...; method = :stacking, kwargs...)

brms's `loo_model_weights`: [`model_weights`](@ref) with LOO (`method =
:stacking` or `:pseudobma`).
"""
loo_model_weights(fits...; method::Symbol=:stacking, kwargs...) =
    model_weights(fits...; weights=method, criterion=loo, kwargs...)

"""
    kfold_predict(fit; K = 10, seed = 1, method = :predict, kwargs...) -> (; yrep, y)

brms's `kfold_predict`: K-fold cross-validated predictions. Every
observation's draws come from the fit that left its fold out (`method =
:predict` for posterior-predictive draws, `:epred` for expected values).
Extra keywords go to the refits.
"""
function kfold_predict(fit::BrmsFit; K::Int=10, seed=1, method::Symbol=:predict, rng=Random.Xoshiro(seed), kwargs...)
    isempty(fit.call) && throw(ArgumentError("kfold_predict needs a fit made by `brm`"))
    cols = Tables.columntable(fit.call[:data])
    rows = fit.info.rows
    n = length(rows)
    fold = shuffle(Random.Xoshiro(seed), repeat(1:K, cld(n, K))[1:n])
    sub(idx) = NamedTuple{keys(cols)}(map(v -> v[idx], values(cols)))
    yrep = nothing
    for k in 1:K
        test = findall(==(k), fold)
        fk = update(fit; newdata=sub(rows[findall(!=(k), fold)]), kwargs...)
        P = method === :epred ? posterior_epred(fk; newdata=sub(rows[test]), allow_new_levels=true) :
            posterior_predict(fk; newdata=sub(rows[test]), allow_new_levels=true, rng)
        yrep === nothing && (yrep = fill(NaN, size(P, 1), n))
        size(P, 1) == size(yrep, 1) || throw(ArgumentError("the refits returned different numbers of draws"))
        yrep[:, test] .= P
    end
    return (; yrep, y=Float64.(fit.density.y))
end

"""
    make_conditions(data, vars) -> Vector{NamedTuple}

Conditions for [`conditional_effects`](@ref), as brms's `make_conditions`: for
a numeric variable its mean and the mean -/+ one sd, for a factor each level;
one condition per combination. Pass one at a time, e.g.
`[conditional_effects(fit, :x; conditions = c) for c in make_conditions(d, [:z])]`.
"""
function make_conditions(data, vars)
    cols = Tables.columntable(data)
    opts = map(vars) do v
        x = collect(skipmissing(getproperty(cols, Symbol(v))))
        eltype(x) <: Real && !(eltype(x) <: Bool) ? (m = mean(x); s = std(x); [m - s, m, m + s]) : sort(unique(x))
    end
    return [NamedTuple{Tuple(Symbol.(vars))}(c) for c in Iterators.product(opts...)][:]
end

"""
    logit_scaled(x, lb = 0, ub = 1)

brms's helper: the logit of `(x - lb) / (ub - lb)`.
"""
logit_scaled(x, lb=0, ub=1) = (p = (x - lb) / (ub - lb); log(p) - log1p(-p))
"""
    inv_logit_scaled(x, lb = 0, ub = 1)

brms's helper: the inverse of [`logit_scaled`](@ref), `lb + (ub - lb) * logistic(x)`.
"""
inv_logit_scaled(x, lb=0, ub=1) = lb + (ub - lb) * logistic(x)
"""
    logm1(x)

brms's helper: `log(x - 1)` (the link of student's `nu`).
"""
logm1(x) = log(x - 1)
"""
    expp1(x)

brms's helper: `exp(x) + 1`, the inverse of [`logm1`](@ref).
"""
expp1(x) = exp(x) + 1

# --- loo_subsample (Magnusson et al. 2019; the loo package's defaults) ------------------
"""
    LOOSubsampleResult

PSIS-LOO estimated from a subsample of observations, as `loo::loo_subsample`:
`elpd`, its `se` and the extra `subsampling_se`; `p` (p_loo) likewise; the
subsample `idx` with its pointwise `elpd_sub` and `pareto_k`, and the
approximation `approx` used for every observation.
"""
struct LOOSubsampleResult
    elpd::Float64
    se::Float64
    subsampling_se::Float64
    p::Float64
    p_se::Float64
    p_subsampling_se::Float64
    idx::Vector{Int}
    elpd_sub::Vector{Float64}
    pareto_k::Vector{Float64}
    approx::Vector{Float64}
end
function Base.show(io::IO, ::MIME"text/plain", r::LOOSubsampleResult)
    println(io, "PSIS-LOO by subsampling: ", length(r.idx), " of ", length(r.approx), " observations")
    print(io, @sprintf("  elpd_loo  %10.1f   SE %6.1f   subsampling SE %6.1f\n", r.elpd, r.se, r.subsampling_se))
    print(io, @sprintf("  p_loo     %10.1f   SE %6.1f   subsampling SE %6.1f\n", r.p, r.p_se, r.p_subsampling_se))
    nbad = count(>(0.7), r.pareto_k)
    nbad > 0 && println(io, "  ", nbad, " subsampled observations with Pareto k > 0.7")
end

# the loo package's estimators: difference estimator under simple random sampling
# (srs_diff_est) and the plain SRS estimator (srs_est)
function _srs_diff_est(yapprox, y, idx)
    N = length(yapprox); m = length(y)
    e = y .- yapprox[idx]
    tpi = sum(yapprox); tpi2 = sum(abs2, yapprox)
    te = N * mean(e)
    the = N * mean(y .^ 2 .- yapprox[idx] .^ 2)
    yhat = tpi + te
    vyhat = N^2 * (1 - m / N) * var(e) / m
    hatvy = (tpi2 + the) - (1 / N) * (te^2 - vyhat + 2 * tpi * yhat - tpi^2)
    return yhat, vyhat, hatvy
end
function _srs_est(y, N)
    m = length(y)
    return N * mean(y), N^2 * (1 - m / N) * var(y) / m, N * var(y)
end

"""
    loo_subsample(fit; observations = 400, loo_approximation = :plpd, seed = 1) -> LOOSubsampleResult

Approximate PSIS-LOO for large data, as brms's `loo_subsample` (loo package
defaults): every observation's elpd is approximated cheaply (`:plpd`, the log
predictive density at the posterior mean of the parameters; or `:lpd`, the
log posterior mean density), exact PSIS-LOO is computed for a simple random
subsample of `observations`, and the difference estimator corrects the
approximation. `se` is the usual SE of elpd_loo, `subsampling_se` the extra
uncertainty from subsampling. (The pointwise log likelihood is still computed
for every observation here; the subsample saves the PSIS fits.)
"""
function loo_subsample(fit::BrmsFit; observations::Int=400, loo_approximation::Symbol=:plpd, seed=1,
                       rng=Random.Xoshiro(seed))
    loo_approximation in (:plpd, :lpd) || throw(ArgumentError("loo_approximation must be :plpd or :lpd"))
    LL = log_lik(fit)
    S, N = size(LL)
    m = min(observations, N)
    approx = if loo_approximation === :lpd
        [logsumexp(view(LL, :, i)) - log(S) for i in 1:N]
    else
        # the log likelihood at the posterior mean of the (unconstrained) parameters
        θbar = vec(mean(reshape(fit.draws, size(fit.draws, 1), :); dims=2))
        tmp = BrmsFit(fit.info, fit.density, fit.chain, reshape(θbar, :, 1, 1), fit.sampler, fit.iter, fit.warmup,
                      fit.thin, fit.sampling_time, fit.call)
        vec(log_lik(tmp))
    end
    idx = sort(randperm(rng, N)[1:m])
    niter, nch = size(fit.draws, 2), size(fit.draws, 3)
    elpd = zeros(m); plo = zeros(m); ks = zeros(m)
    for (j, i) in enumerate(idx)
        ll = view(LL, :, i)
        reff = clamp(only(MCMCDiagnosticTools.ess(reshape(exp.(ll .- maximum(ll)), niter, nch, 1); kind=:basic)) / S, 1e-3, 1.0)
        res = PSIS.psis(-collect(ll), reff; warn=false)
        elpd[j] = logsumexp(res.log_weights .+ ll)
        ks[j] = res.pareto_shape
        plo[j] = logsumexp(ll) - log(S) - elpd[j]
    end
    yhat, vyhat, hatvy = _srs_diff_est(approx, elpd, idx)
    ph, pv, phv = _srs_est(plo, N)
    return LOOSubsampleResult(yhat, sqrt(max(hatvy, 0.0)), sqrt(max(vyhat, 0.0)), ph, sqrt(phv), sqrt(pv), idx, elpd, ks, approx)
end
