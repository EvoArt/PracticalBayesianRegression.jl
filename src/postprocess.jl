# =============================================================================
# Post-processing: group-level effects (ranef / group_coef), Bayesian R²,
# and approximate leave-one-out cross-validation (PSIS-LOO, WAIC).
#
# LOO follows the `loo` R package that brms calls: per observation, Pareto-
# smoothed importance sampling of the posterior with log ratios -log_lik,
# with the relative efficiency of exp(log_lik) as `r_eff`. Estimates and SEs
# are the loo package's: elpd = sum of pointwise elpd, se = sqrt(N var).
# =============================================================================

# draws x chains matrix of one quantity, chains as in the fit
_chainmat(fit::BrmsFit, v::AbstractVector) = reshape(v, size(fit.draws, 2), size(fit.draws, 3))

"""
    ranef(fit; prob=0.95) -> Dict{String, Dict{String, Vector{SummaryRow}}}

Group-level effects, as brms's `ranef`: `ranef(fit)["g"]["Intercept"]` is one
row per level of `g` (the row's `variable` is the level).
"""
function ranef(fit::BrmsFit; prob=0.95)
    out = Dict{String,Dict{String,Vector{SummaryRow}}}()
    for p in _linpreds(fit.density.preds)
        for gb in p.groups
            r = p.dpar === :mu ? "r_$(gb.label)" : "r_$(gb.label)__$(p.dpar)"
            byc = get!(out, gb.label, Dict{String,Vector{SummaryRow}}())
            for c in gb.coefs
                key = p.dpar === :mu ? c : "$(p.dpar)_$(c)"
                append!(get!(byc, key, SummaryRow[]),
                        [_relabel(_summarise("$(r)[$(l),$(c)]", draws(fit, "$(r)[$(l),$(c)]"), prob), l)
                         for l in gb.levels])
            end
        end
    end
    return out
end

_relabel(r::SummaryRow, name) = SummaryRow(name, r.estimate, r.est_error, r.lower, r.upper, r.rhat, r.bulk_ess, r.tail_ess)

"""
    group_coef(fit; prob=0.95)

Population-level plus group-level effects per level — brms's `coef()`
(named differently here so as not to clash with `StatsAPI.coef`, which
returns the population-level posterior means). A coefficient that varies by
`g` gets `b_coef + r_g[level, coef]`; one that does not is reported as the
population-level value for every level.
"""
function group_coef(fit::BrmsFit; prob=0.95)
    out = Dict{String,Dict{String,Vector{SummaryRow}}}()
    vars = Set(variables(fit))
    for p in fit.density.preds
        p isa LinPred && p.dpar === :mu || continue
        for gb in p.groups
            byc = get!(out, gb.label, Dict{String,Vector{SummaryRow}}())
            for c in vcat(p.intercept > 0 ? ["Intercept"] : String[], p.coefnames)
                bname = "b_$(c)"
                bname in vars || continue
                b = draws(fit, bname)
                byc[c] = map(gb.levels) do l
                    rn = "r_$(gb.label)[$(l),$(c)]"
                    x = rn in vars ? b .+ draws(fit, rn) : b
                    _relabel(_summarise(rn, x, prob), l)
                end
            end
        end
    end
    return out
end

"""
    bayes_R2(fit; prob=0.95) -> SummaryRow

Bayesian R² (Gelman, Goodrich, Gabry & Vehtari 2019), as brms's
`bayes_R2`: per draw, var(E[y]) / (var(E[y]) + var(y - E[y])).
"""
function bayes_R2(fit::BrmsFit; prob=0.95)
    fit.info.family isa Union{_Ord,Categorical} && throw(ArgumentError("bayes_R2 is not defined for ordinal or categorical models"))
    E = posterior_epred(fit)
    y = Float64.(fit.density.y)
    r2 = map(axes(E, 1)) do s
        e = view(E, s, :)
        ve = var(e; corrected=true)
        vr = var(y .- e; corrected=true)
        ve / (ve + vr)
    end
    return _summarise("R2", _chainmat(fit, r2), prob)
end

# --- PSIS-LOO and WAIC ----------------------------------------------------------

"""
    LOOResult

The result of [`loo`](@ref), [`waic`](@ref) or [`kfold`](@ref). Fields:
`elpd` (the expected log predictive density) and its standard error `se`;
`p` (the effective number of parameters) and `p_se`; `pointwise`, the elpd of
each observation; and for PSIS-LOO, `pareto_k`, each observation's Pareto k
diagnostic. Displaying it prints brms's table, including the information
criterion `-2 elpd`.
"""
struct LOOResult
    criterion::Symbol
    elpd::Float64
    se::Float64
    p::Float64
    p_se::Float64
    pointwise::Vector{Float64}
    pareto_k::Vector{Float64}
end

function Base.show(io::IO, ::MIME"text/plain", r::LOOResult)
    n = length(r.pointwise)
    name = string(r.criterion)
    println(io, "Computed from $(n) observations ($(uppercase(name)))")
    println(io, rpad("", 10), lpad("Estimate", 10), lpad("SE", 9))
    println(io, rpad("elpd_$(name)", 10), lpad(_fmt(r.elpd), 10), lpad(_fmt(r.se), 9))
    println(io, rpad("p_$(name)", 10), lpad(_fmt(r.p), 10), lpad(_fmt(r.p_se), 9))
    println(io, rpad("$(name)ic", 10), lpad(_fmt(-2r.elpd), 10), lpad(_fmt(2r.se), 9))
    if r.criterion === :loo
        bad = count(>(0.7), r.pareto_k)
        meh = count(k -> 0.5 < k <= 0.7, r.pareto_k)
        if bad + meh == 0
            print(io, "\nAll Pareto k estimates are good (k <= 0.5).")
        else
            print(io, "\nPareto k: $(meh) in (0.5, 0.7], $(bad) > 0.7 — consider moment matching or kfold for those.")
        end
    end
end

_se(v) = sqrt(length(v) * var(v; corrected=true))

"""
    loo(fit; moment_match = false, k_threshold = 0.7) -> LOOResult

Approximate leave-one-out cross-validation by Pareto-smoothed importance
sampling (Vehtari, Gelman & Gabry 2017), as brms's `loo`.

Observations with Pareto k above 0.7 make the estimate unreliable. With
`moment_match = true`, the draws are adapted for each observation whose k
exceeds `k_threshold` (Paananen et al. 2021), as brms's
`loo(fit, moment_match = TRUE)`; [`kfold`](@ref) refits the model instead.

```julia
loo(fit)
loo_compare("full" => fit1, "reduced" => fit2)
```
"""
function loo(fit::Union{BrmsFit,BrmsMVFit}; moment_match::Bool=false, k_threshold=0.7, max_iters::Int=30)
    LL = log_lik(fit)
    S, N = size(LL)
    niter, nch = size(fit.draws, 2), size(fit.draws, 3)
    elpd = Vector{Float64}(undef, N)
    plo = Vector{Float64}(undef, N)
    ks = Vector{Float64}(undef, N)
    for i in 1:N
        ll = view(LL, :, i)
        m = maximum(ll)
        # relative efficiency of exp(log_lik), the loo package's r_eff
        w = reshape(exp.(ll .- m), niter, nch, 1)
        reff = clamp(only(MCMCDiagnosticTools.ess(w; kind=:basic)) / S, 1e-3, 1.0)
        res = PSIS.psis(-collect(ll), reff; warn=false)
        elpd[i] = logsumexp(res.log_weights .+ ll)
        ks[i] = res.pareto_shape
        lppd = logsumexp(ll) - log(S)
        plo[i] = lppd - elpd[i]
        if moment_match && ks[i] > k_threshold
            e, k = _moment_match_obs(fit, i, reff; k_threshold, max_iters)
            if k < ks[i]
                elpd[i] = e; ks[i] = k; plo[i] = lppd - e
            end
        end
    end
    return LOOResult(:loo, sum(elpd), _se(elpd), sum(plo), _se(plo), elpd, ks)
end

# --- moment matching (Paananen, Piironen, Buerkner & Vehtari 2021) -------------------
#
# For an observation whose PSIS weights are unreliable (Pareto k high), move the
# posterior draws by affine maps that match the importance-weighted moments
# (shift the mean, then scale each dimension, then the full covariance), keeping
# a map only when it lowers k. With theta' = T(theta), the proposal density of
# the moved draws is q(theta) / |det T|, so the leave-one-out log ratio is
#     lr = lp(theta') - ll_i(theta') - (lp(theta) - log |det T|)
# Without the loo package's "split" refinement.

function _mm_lp(fit::BrmsFit, U)                  # log posterior at the columns of U
    ld = fit.density
    return [LogDensityProblems.logdensity(ld, view(U, :, s)) for s in axes(U, 2)]
end
function _mm_ll(fit::BrmsFit, U, i)              # log-likelihood of observation i
    D = reshape(U, size(U, 1), size(U, 2), 1)
    tmp = BrmsFit(fit.info, fit.density, fit.chain, D, fit.sampler, fit.iter, fit.warmup, fit.thin,
                  fit.sampling_time, fit.call)
    return log_lik(tmp)[:, i]
end
function _moment_match_obs(fit::BrmsFit, i, reff; k_threshold=0.7, max_iters=30)
    ld = fit.density
    (isempty(ld.ordered) && isempty(ld.pinned)) ||
        throw(ArgumentError("moment matching is not supported for models with ordered or constant() parameters yet"))
    U0 = reshape(copy(fit.draws), size(fit.draws, 1), :)
    lp0 = _mm_lp(fit, U0)
    U, logdet = U0, 0.0
    ll = _mm_ll(fit, U, i)
    lr = -ll
    res = PSIS.psis(lr, reff; warn=false)
    k = res.pareto_shape
    lw = res.log_weights .- logsumexp(res.log_weights)
    for _ in 1:max_iters
        k <= k_threshold && break
        improved = false
        for kind in (:shift, :scale, :cov)
            w = exp.(lw)
            μ = vec(mean(U; dims=2)); μw = U * w
            Un, ld_add = if kind === :shift
                U .+ (μw .- μ), 0.0
            elseif kind === :scale
                sd = vec(std(U; dims=2)); sdw = sqrt.(max.(((U .- μw) .^ 2) * w, 1e-300))
                r = sdw ./ sd
                ((U .- μ) .* r) .+ μw, sum(log, r)
            else
                Σ = cov(U; dims=2) + 1e-10I
                Σw = ((U .- μw) .* w') * (U .- μw)' + 1e-10I
                L, Lw = cholesky(Symmetric(Σ)).L, cholesky(Symmetric(Σw)).L
                A = Lw / L
                (A * (U .- μ)) .+ μw, logabsdet(A)[1]
            end
            lpn = _mm_lp(fit, Un); lln = _mm_ll(fit, Un, i)
            lrn = lpn .- lln .- (lp0 .- (logdet + ld_add))
            rn = PSIS.psis(lrn, reff; warn=false)
            if rn.pareto_shape < k
                U, logdet, ll, k = Un, logdet + ld_add, lln, rn.pareto_shape
                lw = rn.log_weights .- logsumexp(rn.log_weights)
                improved = true
                break
            end
        end
        improved || break
    end
    return logsumexp(lw .+ ll), k
end

"""
    waic(fit) -> LOOResult

The widely applicable information criterion, as brms's `waic`.
"""
function waic(fit::Union{BrmsFit,BrmsMVFit})
    LL = log_lik(fit)
    S, N = size(LL)
    lppd = [logsumexp(view(LL, :, i)) - log(S) for i in 1:N]
    pw = [var(view(LL, :, i); corrected=true) for i in 1:N]
    elpd = lppd .- pw
    return LOOResult(:waic, sum(elpd), _se(elpd), sum(pw), _se(pw), elpd, Float64[])
end

"""
    loo_compare(fits...; criterion=loo) -> Vector of (name, elpd_diff, se_diff)

Compare models by expected log predictive density, as `loo::loo_compare`:
differences are relative to the best model, with the SE of the paired
pointwise differences. Pass `name => fit` pairs to label them.
"""
function loo_compare(fits...; criterion=loo)
    named = [f isa Pair ? (string(first(f)), last(f)) : ("model$(k)", f) for (k, f) in enumerate(fits)]
    res = [(n, criterion(f)) for (n, f) in named]
    best = argmax([r.elpd for (_, r) in res])
    rb = res[best][2]
    out = map(res) do (n, r)
        d = r.pointwise .- rb.pointwise
        (name=n, elpd_diff=r.elpd - rb.elpd, se_diff=n == res[best][1] ? 0.0 : _se(d), elpd=r.elpd)
    end
    return sort(out; by=r -> -r.elpd_diff)
end

# --- refitting: update and kfold ------------------------------------------------------

"""
    update(fit; newdata=nothing, formula=nothing, kwargs...)

Refit with some arguments changed, as brms's `update` (any `brm` keyword:
`family`, `prior`, `chains`, `iter`, `seed`, ...). Only fits made by `brm`
remember their arguments.
"""
function update(fit::BrmsFit; newdata=nothing, formula=nothing, kwargs...)
    isempty(fit.call) && throw(ArgumentError("this fit was not made by `brm`; refit it directly"))
    c = copy(fit.call)
    f = formula === nothing ? pop!(c, :formula) : (pop!(c, :formula); formula)
    d = newdata === nothing ? pop!(c, :data) : (pop!(c, :data); newdata)
    for (k, v) in kwargs
        c[k] = v
    end
    return brm(f, d; (Symbol(k) => v for (k, v) in c)...)
end

"""
    kfold(fit; K=10, seed=1, kwargs...) -> LOOResult

K-fold cross-validation, as brms's `kfold`: refit K times leaving one fold
out, and score the held-out observations by their log predictive density.
Extra keywords go to the refits (e.g. `chains=2`).
"""
function kfold(fit::BrmsFit; K::Int=10, seed=1, kwargs...)
    isempty(fit.call) && throw(ArgumentError("kfold needs a fit made by `brm`"))
    cols = Tables.columntable(fit.call[:data])
    rows = fit.info.rows                       # the complete cases used in fitting
    n = length(rows)
    fold = shuffle(Random.Xoshiro(seed), repeat(1:K, cld(n, K))[1:n])
    sub(idx) = NamedTuple{keys(cols)}(map(v -> v[idx], values(cols)))
    elpd = zeros(n)
    for k in 1:K
        test = findall(==(k), fold)
        train = findall(!=(k), fold)
        fk = update(fit; newdata=sub(rows[train]), kwargs...)
        LL = log_lik(fk; newdata=sub(rows[test]), allow_new_levels=true)
        S = size(LL, 1)
        elpd[test] = [logsumexp(view(LL, :, i)) - log(S) for i in axes(LL, 2)]
    end
    # p_kfold as the loo package: lppd of the full fit minus elpd_kfold
    LLf = log_lik(fit)
    lppd = [logsumexp(view(LLf, :, i)) - log(size(LLf, 1)) for i in axes(LLf, 2)]
    return LOOResult(:kfold, sum(elpd), _se(elpd), sum(lppd .- elpd), _se(lppd .- elpd), elpd, Float64[])
end

"""
    loo_R2(fit; prob=0.95) -> SummaryRow

LOO-adjusted R² (brms's `loo_R2`): the R² of PSIS-LOO predictive means, with
a Bayesian-bootstrap posterior over the observation weights.
"""
function loo_R2(fit::BrmsFit; prob=0.95, ndraws::Int=4000, rng=Random.Xoshiro(1))
    E = posterior_epred(fit)
    LL = log_lik(fit)
    y = Float64.(fit.density.y)
    S, N = size(E)
    niter, nch = size(fit.draws, 2), size(fit.draws, 3)
    ypred = similar(y)
    for i in 1:N
        ll = view(LL, :, i)
        reff = clamp(only(MCMCDiagnosticTools.ess(reshape(exp.(ll .- maximum(ll)), niter, nch, 1); kind=:basic)) / S, 1e-3, 1.0)
        w = PSIS.psis(-collect(ll), reff; warn=false).weights
        ypred[i] = sum(w .* view(E, :, i))
    end
    err = y .- ypred
    r2 = map(1:ndraws) do _
        g = -log.(rand(rng, N)); b = g ./ sum(g)            # Dirichlet(1,...,1) weights
        mv(v) = (m = sum(b .* v); sum(b .* (v .- m) .^ 2))
        vy = mv(y); ve = mv(err)
        1 - ve / vy
    end
    a = (1 - prob) / 2
    return SummaryRow("R2", mean(r2), std(r2), quantile(r2, a), quantile(r2, 1 - a), NaN, NaN, NaN)
end

"""
    fitted_summary(fit; newdata = nothing, prob = 0.95, kwargs...)

Per observation, the posterior mean, standard deviation and central `prob`
interval of the expected response, as brms's `fitted()`. Returns a
NamedTuple of vectors `(Estimate, EstError, lower, upper)`. Other keywords
go to [`posterior_epred`](@ref).
"""
fitted_summary(fit::BrmsFit; prob=0.95, kwargs...) = _obs_summary(posterior_epred(fit; kwargs...), prob)

"""
    predict_summary(fit; newdata = nothing, prob = 0.95, kwargs...)

Per observation, the mean, standard deviation and central `prob` interval of
the posterior predictive distribution, as brms's `predict()`. Returns a
NamedTuple of vectors `(Estimate, EstError, lower, upper)`. Other keywords go
to [`posterior_predict`](@ref).
"""
predict_summary(fit::BrmsFit; prob=0.95, kwargs...) = _obs_summary(posterior_predict(fit; kwargs...), prob)
function _obs_summary(M::AbstractMatrix, prob)
    a = (1 - prob) / 2
    return (Estimate=vec(mean(M; dims=1)), EstError=vec(std(M; dims=1)),
            lower=[quantile(view(M, :, i), a) for i in axes(M, 2)],
            upper=[quantile(view(M, :, i), 1 - a) for i in axes(M, 2)])
end

# --- StatsAPI ----------------------------------------------------------------------

StatsAPI.coef(fit::BrmsFit) = [r.estimate for r in fixef(fit)]
StatsAPI.coefnames(fit::BrmsFit) = [r.variable for r in fixef(fit)]
StatsAPI.nobs(fit::BrmsFit) = fit.info.nobs
StatsAPI.fitted(fit::BrmsFit) = vec(mean(posterior_epred(fit); dims=1))
StatsAPI.residuals(fit::BrmsFit) = Float64.(fit.density.y) .- StatsAPI.fitted(fit)
StatsAPI.predict(fit::BrmsFit; newdata=nothing) = vec(mean(posterior_epred(fit; newdata=newdata); dims=1))
StatsAPI.loglikelihood(fit::BrmsFit) = vec(sum(log_lik(fit); dims=2))
