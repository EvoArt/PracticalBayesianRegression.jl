# =============================================================================
# brms `posterior_smooths` / `conditional_smooths`: draws of one smooth term's
# contribution to a linear predictor -- its unpenalised columns (uncentred)
# times their coefficients plus its penalised blocks Z (sds .* zs) -- at the
# training rows, new data, or (conditional_smooths) along a grid of its first
# covariate with everything else at its reference value.
# =============================================================================

function _smooth_lookup(fit::BrmsFit, smooth::AbstractString, dpar::Symbol)
    k = findfirst(d -> d.dpar === dpar, fit.info.dinfo)
    k === nothing && throw(ArgumentError("no predictor for `$dpar`"))
    di = fit.info.dinfo[k]
    texts = [sm[1][5] for sm in di.smooths]
    j = findfirst(==(smooth), texts)
    j === nothing && throw(ArgumentError("no smooth `$smooth` for $dpar (smooths: $(join(texts, ", ")))"))
    spec = di.smooths[j][1]
    return spec, string(spec[1]) * join(brms_rename.(getfield.(spec[2], :label)))
end

"""
    posterior_smooths(fit; smooth = "s(x)", newdata = nothing, dpar = :mu) -> draws x observations

brms `posterior_smooths`: the term's contribution to the (link-scale)
predictor of `dpar`, per draw, without the intercept or other terms.
"""
function posterior_smooths(fit::BrmsFit; smooth::AbstractString, newdata=nothing, dpar::Symbol=:mu)
    _, lab = _smooth_lookup(fit, smooth, dpar)
    preds, N, _ = _prediction_preds(fit, newdata)
    k = findfirst(p -> p isa PredLin && p.p.dpar === dpar, preds)
    pl = preds[k]; p = pl.p
    mine(n) = startswith(n, lab * "_") || startswith(n, lab * ":")
    cols = findall(mine, p.coefnames)
    blocks = findall(sb -> mine(sb.label), p.smooths)
    Xs = pl.Xc[:, cols] .+ Float64.(p.means[cols])'          # uncentred unpenalised columns
    D = reshape(fit.draws, size(fit.draws, 1), :)
    out = Matrix{Float64}(undef, size(D, 2), N)
    for s in axes(D, 2)
        θ = view(D, :, s)
        b = _bvalues(p, θ)
        v = Xs * Float64.(b[cols])
        for bl in blocks
            sb = p.smooths[bl]
            v .+= pl.smZ[bl] * (exp(θ[sb.sds]) .* θ[sb.zs])
        end
        out[s, :] .= v
    end
    return out
end

"""
    conditional_smooths(fit; smooth = "s(x)", resolution = 100, prob = 0.95, dpar = :mu)

brms `conditional_smooths`: the smooth along a grid of its (first) covariate
-- a NamedTuple of the grid, `estimate`, `lower`, `upper`.
"""
function conditional_smooths(fit::BrmsFit; smooth::AbstractString, resolution::Int=100, prob=0.95, dpar::Symbol=:mu)
    spec, _ = _smooth_lookup(fit, smooth, dpar)
    x = spec[2][1].ex
    x isa Symbol || throw(ArgumentError("conditional_smooths: the smooth's covariate must be a plain column"))
    base = _reference_row(fit)
    xs = _effect_grid(fit, x, resolution)
    nd = Dict{Symbol,Any}(k => fill(v, length(xs)) for (k, v) in pairs(base))
    nd[x] = xs
    P = posterior_smooths(fit; smooth, newdata=NamedTuple(nd), dpar)
    a = (1 - prob) / 2
    return (; x => xs, :estimate => vec(mean(P; dims=1)),
            :lower => [quantile(view(P, :, i), a) for i in axes(P, 2)],
            :upper => [quantile(view(P, :, i), 1 - a) for i in axes(P, 2)])
end
