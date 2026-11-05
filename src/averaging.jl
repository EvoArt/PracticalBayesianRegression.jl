# =============================================================================
# Several fits at once (brms): brm_multiple / combine_models for multiply
# imputed data, model weights (stacking, pseudo-BMA+, LOO), and model-averaged
# draws (posterior_average, pp_average); plus brms's interval helpers.
# =============================================================================

"""
    combine_fits(fits) -> BrmsFit

Pool the chains of several fits of the same model, as brms's
`combine_models`: for example, fits to multiply imputed data sets. Every fit
must have the same parameters and number of draws per chain.
"""
function combine_fits(fits::AbstractVector{<:BrmsFit})
    isempty(fits) && throw(ArgumentError("no fits to combine"))
    f1 = fits[1]
    names = variables(f1.density)
    all(f -> variables(f.density) == names && size(f.draws, 2) == size(f1.draws, 2), fits) ||
        throw(ArgumentError("combine_fits: the fits must be the same model with the same number of draws per chain"))
    draws = cat((f.draws for f in fits)...; dims=3)
    d = Dict{Any,Matrix{Float64}}()
    for k in keys(f1.chain)
        d[k] = reduce(hcat, [Matrix{Float64}(collect(parent(f.chain[k]))) for f in fits])
    end
    times = reduce(vcat, [f.sampling_time for f in fits])
    chain = FlexiChains.FlexiChain{String}(size(draws, 2), size(draws, 3), d; sampling_time=times)
    call = merge(f1.call, Dict{Symbol,Any}(:combined => length(fits)))
    return BrmsFit(f1.info, f1.density, chain, draws, f1.sampler, f1.iter, f1.warmup, f1.thin, times, call)
end

"""
    brm_multiple(formula, datasets; seed = nothing, kwargs...) -> BrmsFit

Fit the same model to each data set (for example, multiple imputations) and
pool the chains, as brms's `brm_multiple`. Other keywords go to
[`brm`](@ref). Because the data sets differ, R-hat of the pooled chains is
not a convergence check; check the fits one by one.
"""
function brm_multiple(formula, datasets::AbstractVector; seed=nothing, kwargs...)
    fits = [brm(formula, d; seed=seed === nothing ? nothing : seed + 7919 * k, kwargs...)
            for (k, d) in enumerate(datasets)]
    return combine_fits(fits)
end

# --- model weights ----------------------------------------------------------------------

"""
    model_weights(fits...; weights = :stacking, criterion = loo) -> Vector{Float64}

Weights for averaging the predictions of several models, from their pointwise
leave-one-out log predictive densities, as brms's `model_weights`:
- `:stacking`: stacking of predictive distributions (Yao et al. 2018);
- `:pseudobma`: pseudo-BMA+ weights with the Bayesian bootstrap;
- `:loo`: weights proportional to `exp(elpd_loo)`.

`criterion` can be [`waic`](@ref) instead of [`loo`](@ref).

```julia
w = model_weights(fit1, fit2; weights = :stacking)
```
"""
function model_weights(fits...; weights::Symbol=:stacking, criterion=loo, rng=Random.Xoshiro(1))
    length(fits) >= 2 || throw(ArgumentError("model_weights needs at least two fits"))
    P = reduce(hcat, [criterion(f).pointwise for f in fits])              # N x K
    N, K = size(P)
    if weights === :loo
        e = vec(sum(P; dims=1)); w = exp.(e .- maximum(e)); return w ./ sum(w)
    elseif weights === :pseudobma
        W = zeros(K)
        for _ in 1:1000
            α = -log.(rand(rng, N)); α ./= sum(α)                          # Dirichlet(1, ..., 1)
            z = N .* vec(α' * P)
            w = exp.(z .- maximum(z)); W .+= w ./ sum(w)
        end
        return W ./ 1000
    elseif weights === :stacking
        # softmax(u_1..u_{K-1}, 0), maximised with L-BFGS
        Pm = P .- maximum(P; dims=2)
        E = exp.(Pm)
        function fg!(g, u)
            v = vcat(u, 0.0); w = exp.(v .- maximum(v)); w ./= sum(w)
            s = E * w
            f = sum(log, s)
            r = vec(sum(E ./ s; dims=1))                    # d f / d w
            gw = w .* (r .- dot(w, r))                      # through the softmax
            g .= gw[1:(K - 1)]
            return f
        end
        u, _, _ = _lbfgs(fg!, zeros(K - 1))
        v = vcat(u, 0.0); w = exp.(v .- maximum(v))
        return w ./ sum(w)
    end
    throw(ArgumentError("weights must be :stacking, :pseudobma or :loo"))
end

# draws per model for a total of n, proportional to the weights
_alloc(w, n) = (c = floor.(Int, w .* n); c[argmax(w)] += n - sum(c); c)

"""
    posterior_average(fits...; variables, weights = :stacking, ndraws = 4000) -> Dict

brms `posterior_average`: draws of `variables` (present in every model) mixed
across the models in proportion to the model weights.
"""
function posterior_average(fits...; variables::AbstractVector{<:AbstractString}, weights=:stacking, ndraws::Int=4000,
                           rng=Random.Xoshiro(1))
    w = weights isa Symbol ? model_weights(fits...; weights) : Float64.(collect(weights))
    cnt = _alloc(w, ndraws)
    out = Dict{String,Vector{Float64}}()
    for v in variables
        out[v] = reduce(vcat, [rand(rng, vec(draws(f, v)), c) for (f, c) in zip(fits, cnt) if c > 0])
    end
    return out
end

"""
    pp_average(fits...; weights = :stacking, ndraws = 4000, newdata = nothing, method = posterior_predict)

brms `pp_average`: posterior predictive (or `method = posterior_epred`) draws
mixed across models in proportion to the model weights (draws x observations).
"""
function pp_average(fits...; weights=:stacking, ndraws::Int=4000, newdata=nothing, method=posterior_predict,
                    rng=Random.Xoshiro(1), kw...)
    w = weights isa Symbol ? model_weights(fits...; weights) : Float64.(collect(weights))
    cnt = _alloc(w, ndraws)
    parts = Matrix{Float64}[]
    for (f, c) in zip(fits, cnt)
        c > 0 || continue
        Y = method(f; newdata, kw...)
        push!(parts, Y[rand(rng, axes(Y, 1), c), :])
    end
    return reduce(vcat, parts)
end

# --- interval helpers (brms) ------------------------------------------------------------

"""
    posterior_interval(fit; prob = 0.95) -> Dict{String, Tuple{Float64, Float64}}

The central `prob` posterior interval of every parameter, as brms's
`posterior_interval`: `variable => (lower, upper)`.
"""
function posterior_interval(fit; prob=0.95)
    a = (1 - prob) / 2
    return Dict(v => (quantile(vec(draws(fit, v)), a), quantile(vec(draws(fit, v)), 1 - a)) for v in variables(fit))
end

"""
    predictive_interval(fit; prob = 0.9, kwargs...) -> Matrix{Float64}

The central `prob` interval of each observation's posterior predictive
distribution, as brms's `predictive_interval`: an observations x 2 matrix of
lower and upper bounds. Other keywords (`newdata`, ...) go to
[`posterior_predict`](@ref).
"""
function predictive_interval(fit; prob=0.9, kw...)
    Y = posterior_predict(fit; kw...)
    a = (1 - prob) / 2
    return reduce(vcat, [[quantile(view(Y, :, i), a) quantile(view(Y, :, i), 1 - a)] for i in axes(Y, 2)])
end

"""
    predictive_error(fit; kwargs...) -> Matrix{Float64}

Draws of `y - y_rep` for the training data, where `y_rep` is a posterior
predictive draw, as brms's `predictive_error`: a draws x observations
matrix.
"""
predictive_error(fit::BrmsFit; kw...) = Float64.(fit.density.y)' .- posterior_predict(fit; kw...)
