# =============================================================================
# Posterior predictions: brms's posterior_linpred / posterior_epred /
# posterior_predict / log_lik, each returning a (draws x observations)
# matrix, draws ordered chain by chain.
#
# For new data the fit's stored design information re-codes the data
# exactly as the training data was coded (same factor levels, same columns,
# centred with the TRAINING column means, so the centred intercept keeps its
# meaning).
#
# Keywords shared by every prediction function, as in brms:
#   re_formula        nothing (all group terms), :none / false (none), or the
#                     group labels to keep (e.g. ["g"])
#   allow_new_levels  allow levels of a grouping factor not seen in fitting
#   sample_new_levels how their effects are drawn: :uncertainty (default; one
#                     existing level's effects per draw and new level),
#                     :old_levels (a fresh existing level per observation),
#                     :gaussian (from the fitted sd/correlation)
# =============================================================================

# A group block as seen at prediction time: the fitted block plus where the
# new-level observations are (`idx[i] > J` marks new level `idx[i] - J`).
struct PredGroup{G}
    gb::G
    idx::Vector{Int}
    Z::Matrix{Float64}
    nnew::Int
    mmJ::Matrix{Int}          # mm(): memberships at these observations (0 x 0 otherwise)
    mmW::Matrix{Float64}
end
PredGroup(gb, idx, Z, nnew) = PredGroup(gb, idx, Z, nnew, Matrix{Int}(gb.mmJ), Matrix{Float64}(gb.mmW))

struct PredLin{P}
    p::P                      # the fitted LinPred (θ layout, means, ...)
    Xc::Matrix{Float64}
    offset::Vector{Float64}
    groups::Vector{PredGroup}
    mox::Vector{Any}          # per mo term: (levels per factor, numeric covariate product) of these rows
    use_ar::Bool              # AR lags need the observed responses: training data only
    smZ::Vector{Matrix{Float64}}   # penalised smooth designs at these observations
    mex::Vector{Vector{Float64}}   # me(): observed x for new data (empty: training -> latent values)
    gpX::Vector{Matrix{Float64}}   # gp(): covariates of new data (empty: training)
end
PredLin(p, Xc, offset, groups) =
    PredLin(p, Xc, offset, groups, Any[(m.xs, m.csp) for m in p.mo], _has_ar(p.ar), [Float64.(sb.Z) for sb in p.smooths],
            Vector{Float64}[], Matrix{Float64}[])

struct PredNL{F,P,D}
    f::F
    pars::P
    data::D
end

struct _PredOpts
    rng::AbstractRNG
    mode::Symbol
end

_keep_group(label, re_formula::Nothing) = true
_keep_group(label, re_formula::Bool) = re_formula
_keep_group(label, re_formula::Symbol) = re_formula === :none ? false : string(re_formula) == label
_keep_group(label, re_formula::AbstractString) = re_formula == label
_keep_group(label, re_formula::AbstractVector) = label in string.(re_formula)

function _group_index_new(gi::GroupInfo, gt::GroupTerm, vals, allow_new)
    labels = _group_values(gt, vals)
    newlv = String[]
    idx = map(labels) do l
        haskey(gi.index, l) && return gi.index[l]
        allow_new || throw(ArgumentError("group level `$l` of `$(group_label(gt))` was not in the data the model was built from; pass `allow_new_levels=true` to predict for it"))
        k = findfirst(==(l), newlv)
        k === nothing && (push!(newlv, l); k = length(newlv))
        length(gi.levels) + k
    end
    return idx, length(newlv)
end

function _prediction_preds(fit::BrmsFit, newdata; re_formula=nothing, allow_new_levels=false)
    info, f = fit.info, fit.info.formula
    if newdata === nothing
        trials = fit.density.trials
        N = fit.density.N
        train(p) = PredLin(p, Float64.(p.Xc), Float64.(p.offset),
                           PredGroup[PredGroup(gb, gb.idx, Float64.(gb.Z), 0) for gb in p.groups if _keep_group(gb.label, re_formula)])
        preds = map(fit.density.preds) do p
            p isa NLPred && return PredNL(p.f, map(train, p.pars), map(d -> Float64.(d), p.data))
            p isa LinPred || return p
            train(p)
        end
        return preds, N, trials
    end
    cols = Tables.columntable(newdata)
    atoms = Atom[]
    for d in f.dpars
        _rhs_atoms!(atoms, d.rhs)
    end
    uses_trials(info.family) && f.trials isa Atom && push!(atoms, f.trials)
    append!(atoms, Atom.(f.nlvars))
    vals = Dict{Atom,AbstractVector}()
    for a in unique(atoms)
        v = _eval_atom(a, cols, f.mod)
        any(ismissing, v) && throw(ArgumentError("newdata has missing values in `$(a.label)`"))
        vals[a] = v
    end
    N = _nrows(cols)
    newlin = p -> begin
        di = info.dinfo[findfirst(d -> d.dpar === p.dpar, info.dinfo)]
        X = model_matrix(di.spec, vals, info.infos, N)
        smZ = Matrix{Float64}[]
        for (spec, t, lvs) in di.smooths
            Xcov = _smooth_cols(t, [vals[a] for a in spec[2]])
            Xs, Zs = smooth_basis(t, Xcov)
            if isempty(lvs) && spec[4] !== nothing
                bz = Float64.(vals[spec[4]])                    # numeric by
                X = hcat(X, Xs .* bz); append!(smZ, [Zb .* bz for Zb in Zs])
            elseif isempty(lvs)
                X = hcat(X, Xs); append!(smZ, Zs)
            else
                bv = [_level_name(v) for v in vals[spec[4]]]
                for l in lvs
                    ind = bv .== l
                    X = hcat(X, Xs .* ind); append!(smZ, [Zb .* ind for Zb in Zs])
                end
            end
        end
        offset = zeros(N)
        for a in di.rhs.offsets
            offset .+= Float64.(vals[a])
        end
        gs = PredGroup[]
        for (gb, (gt, gi, gspec)) in zip(p.groups, di.groups)
            _keep_group(gb.label, re_formula) || continue
            Zx = model_matrix(gspec, vals, info.infos, N)
            Z = gt.intercept ? hcat(ones(N), Zx) : Zx
            if !isempty(gt.mm)
                mmJ = reduce(hcat, [map(_group_values(GroupTerm(true, Vector{Atom}[], m, false), vals)) do l
                                        haskey(gi.index, l) || throw(ArgumentError("mm(): new levels are not supported for predictions"))
                                        gi.index[l]
                                    end for m in gt.mm])
                M = length(gt.mm)
                mmW = isempty(gt.mmw) ? fill(1 / M, N, M) : reduce(hcat, [Float64.(vals[a]) for a in gt.mmw])
                gt.mmscale && (mmW = mmW ./ sum(mmW; dims=2))
                push!(gs, PredGroup(gb, ones(Int, N), Z, 0, mmJ, mmW))
                continue
            end
            if gb.bylevel != ""
                # gr(by =): only this level's rows; the others get Z = 0
                inb = [_level_name(v) == gb.bylevel for v in vals[gt.by]]
                sub = Dict{Atom,AbstractVector}(a => v[inb] for (a, v) in vals)
                idxb, nnew = _group_index_new(gi, gt, sub, allow_new_levels)
                idx = ones(Int, N); idx[inb] .= idxb
                push!(gs, PredGroup(gb, idx, Z .* inb, nnew))
                continue
            end
            idx, nnew = _group_index_new(gi, gt, vals, allow_new_levels)
            push!(gs, PredGroup(gb, idx, Z, nnew))
        end
        # a non-linear parameter's intercept is a design column of ones
        if !isempty(p.coefnames) && p.coefnames[1] == "Intercept" && size(X, 2) == length(p.coefnames) - 1
            X = hcat(ones(N), X)
        end
        # per mo term (plain mo(x) first, then interaction terms, as built)
        mox = Any[]
        for (k, a) in enumerate(di.rhs.mo)
            t = p.mo[k]
            push!(mox, ([_mo_levels(vals[a], info.infos[a], t.mins[1], t.Ds[1])], Float64[]))
        end
        for t in p.mo[(length(di.rhs.mo) + 1):end]
            xs = [_mo_levels(vals[a], info.infos[a], t.mins[f], t.Ds[f]) for (f, a) in enumerate(t.moatoms)]
            csp = _mo_csp(vals, t.cspatoms, t.cspfac)
            push!(mox, (xs, csp))
        end
        _has_ar(p.ar) && @warn "predictions for new data ignore the ar() term (it needs the observed responses)" maxlog = 1
        mex = Vector{Float64}[Float64.(vals[a]) for (a, _, _) in di.rhs.me]
        gpX = Matrix{Float64}[reduce(hcat, [Float64.(vals[a]) for a in spec[1]]) for spec in di.rhs.gp]
        PredLin(p, X .- Float64.(p.means'), offset, gs, mox, false, smZ, mex, gpX)
    end
    preds = map(fit.density.preds) do p
        p isa NLPred && return PredNL(p.f, map(newlin, p.pars), Tuple(Float64.(vals[Atom(v)]) for v in f.nlvars))
        p isa ThresholdsCS && return typeof(p)(p.base, model_matrix(p.spec, vals, info.infos, N), p.bcs,
                                                 p.priors, p.labels, p.spec)
        if p isa Thresholds{true}
            # grouped thresholds: each new row takes its level's block
            gv = [_level_name(v) for v in _eval_atom(p.gatom, cols, f.mod)]
            return _with_grp(p, map(gv) do l
                k = findfirst(==(l), p.labels)
                k === nothing && throw(ArgumentError("thres(gr = ): level `$l` has no thresholds (not in the data)"))
                k
            end)
        end
        p isa LinPred || return p
        newlin(p)
    end
    trials = uses_trials(info.family) ? (f.trials isa Int ? fill(f.trials, N) : Int.(vals[f.trials])) : Int[]
    return preds, N, trials
end

# group-level effects for the new levels of one block, for one draw
function _new_level_effects(pg::PredGroup, R, θ, opts::_PredOpts)
    gb = pg.gb
    q, J = _q(gb), _J(gb)
    if opts.mode === :gaussian
        sd = exp.(θ[gb.sd])
        L = _has_corr(gb) ? first(_cholcorr(θ[gb.corr], q)) : Matrix(1.0I, q, q)
        Rn = sd .* (L * randn(opts.rng, q, pg.nnew))     # joint rows; the caller takes its own
        if _student(gb)
            # student-t effects: each new level's scale sqrt(df * udf), udf ~ inv_chi_square(df)
            dfv = 1 + exp(θ[gb.df])
            Rn .*= [sqrt(dfv / rand(opts.rng, Distributions.Chisq(dfv))) for _ in 1:pg.nnew]'
        end
        return Rn
    elseif opts.mode === :uncertainty
        return R[:, rand(opts.rng, 1:J, pg.nnew)]
    end
    return nothing       # :old_levels — drawn per observation below
end

function _eta_pred(pl::PredLin, θ, N, opts)
    p = pl.p
    η = copy(pl.offset)
    p.intercept > 0 && (η .+= θ[p.intercept])
    isempty(p.b) || (η .+= pl.Xc * _bvalues(p, θ))
    for (sb, Z) in zip(p.smooths, pl.smZ)
        η .+= Z * (exp(θ[sb.sds]) .* θ[sb.zs])
    end
    for (m, mx) in zip(p.mo, pl.mox)
        xs, csp = mx
        η .+= first(_mo_generic(m, θ, xs, csp))
    end
    for (k, m) in enumerate(p.me)
        if !isempty(pl.mex)
            η .+= θ[m.bsp] .* pl.mex[k]
        elseif isempty(m.zme)
            # a correlated latent (mecor): its row of meanme + diag(sdme) Lme z
            mc = only(p.mecor)
            K = length(mc.terms); j = findfirst(t -> t.meanme == m.meanme, mc.terms)
            L, _ = _cholcorr(θ[mc.cor], K)
            w = (L * reshape(θ[mc.z], K, :))[j, :]
            η .+= θ[m.bsp] .* (θ[m.meanme] .+ exp(θ[m.sdme]) .* w)
        else
            η .+= θ[m.bsp] .* (θ[m.meanme] .+ exp(θ[m.sdme]) .* θ[m.zme])
        end
    end
    for t in p.car
        N == length(t.J) || throw(ArgumentError("predictions for new data with car() terms are not supported yet"))
        η .+= first(_car_generic(t, θ))
    end
    for t in p.lac
        # latent autocorrelated residuals: training rows only (brms draws new
        # residuals for new time points; not yet)
        N == length(t.z) || throw(ArgumentError("predictions for new data with latent autocorrelation are not supported yet"))
        η .+= first(_lac_generic(t, θ))
    end
    for m in p.mi
        # (training rows only: new data would need its own imputation model)
        N == length(m.x) || throw(ArgumentError("predictions for new data with mi() terms are not supported yet"))
        η .+= first(_mi_generic(m, θ))
    end
    for (k, t) in enumerate(p.gp)
        η .+= isempty(pl.gpX) ? first(_gp_generic(t, θ)) : _gp_new(t, θ, pl.gpX[k], opts.rng)
    end
    pl.use_ar && (η = _ar_generic(η, p.ar, θ))
    for pg in pl.groups
        gb = pg.gb
        R = _group_R(gb, θ)
        J = _J(gb)
        Rn = pg.nnew > 0 ? _new_level_effects(pg, R, θ, opts) : nothing
        for i in 1:N
            if !isempty(pg.mmJ)
                for m in axes(pg.mmJ, 2)
                    η[i] += pg.mmW[i, m] * dot(view(pg.Z, i, :), view(R, gb.rows, pg.mmJ[i, m]))
                end
                continue
            end
            j = pg.idx[i]
            rows = gb.rows
            col = j <= J ? view(R, rows, j) :
                  Rn === nothing ? view(R, rows, rand(opts.rng, 1:J)) : view(Rn, rows, j - J)
            η[i] += dot(view(pg.Z, i, :), col)
        end
    end
    return η
end
_eta_pred(p, θ, N, opts) = _eta_generic(p, θ, N)
function _eta_pred(pn::PredNL, θ, N, opts)
    inner = map(q -> _eta_pred(q, θ, N, opts), pn.pars)
    return [pn.f(map(e -> e[i], inner)..., map(d -> d[i], pn.data)...) for i in 1:N]
end

# apply f(eta_tuple_for_obs_i, trials_i, i) to every draw and observation
function _per_draw(f, fit::BrmsFit, newdata; re_formula=nothing, allow_new_levels=false,
                   sample_new_levels=:uncertainty, rng=Random.default_rng())
    preds, N, trials = _prediction_preds(fit, newdata; re_formula, allow_new_levels)
    opts = _PredOpts(rng, Symbol(sample_new_levels))
    D = reshape(fit.draws, size(fit.draws, 1), :)
    out = Matrix{Float64}(undef, size(D, 2), N)
    gev = fit.info.family isa GenExtremeValue
    for s in axes(D, 2)
        θ = view(D, :, s)
        etas = map(p -> _eta_pred(p, θ, N, opts), preds)
        if gev
            # xi is scaled by the TRAINING residuals of this draw (brms)
            ld = fit.density
            xi = _gev_xi_generic(ld, map(p -> _eta_generic(p, θ, ld.N), ld.preds))
            etas = (etas[1], etas[2], xi)
        end
        for i in 1:N
            out[s, i] = f(map(e -> _at(e, i), etas), isempty(trials) ? 0 : trials[i], i)
        end
    end
    return out
end

"""
    posterior_linpred(fit; newdata=nothing, dpar=:mu, transform=false, re_formula=nothing,
                      allow_new_levels=false, sample_new_levels=:uncertainty)

Draws of the linear predictor of `dpar` (on the link scale; `transform=true`
applies the inverse link for `mu`).
"""
function posterior_linpred(fit::BrmsFit; newdata=nothing, dpar::Symbol=:mu, transform::Bool=false, kw...)
    k = findfirst(==(dpar), dpars(fit.info.family))
    k === nothing && throw(ArgumentError("no parameter `$dpar` in this family"))
    fam = fit.info.family
    return _per_draw(fit, newdata; kw...) do eta, t, i
        transform && dpar === :mu ? linkinv(link(fam), eta[1]) : eta[k]
    end
end

"""
    posterior_epred(fit; newdata=nothing, re_formula=nothing, allow_new_levels=false,
                    sample_new_levels=:uncertainty)

Draws of the expected value of the response (for binomial: trials x p).
For an ordinal or categorical model, as in brms, the category probabilities instead:
a (draws x observations x categories) array.
"""
function posterior_epred(fit::BrmsFit; newdata=nothing, kw...)
    fam = fit.info.family
    fam isa Union{_Ord,Categorical} && return _ordinal_probs(fit, newdata; kw...)
    fam isa _MultiCat && return _multicat_draws(fit, newdata, :epred; kw...)
    fam isa TimeCov && return _tc_draws(fit, newdata, :epred; kw...)
    return _per_draw((eta, t, i) -> _mean(fam, eta, t), fit, newdata; kw...)
end

function _ordinal_probs(fit::BrmsFit, newdata; re_formula=nothing, allow_new_levels=false,
                        sample_new_levels=:uncertainty, rng=Random.default_rng())
    fam = fit.info.family
    preds, N, _ = _prediction_preds(fit, newdata; re_formula, allow_new_levels)
    opts = _PredOpts(rng, Symbol(sample_new_levels))
    D = reshape(fit.draws, size(fit.draws, 1), :)
    # (grouped thresholds: a level with fewer categories gets probability 0 for the rest)
    K = fam isa _Ord ? maximum(_thresholds(fit.density).len) + 1 : length(dpars(fam)) + 1
    out = zeros(size(D, 2), N, K)
    for s in axes(D, 2)
        θ = view(D, :, s)
        etas = map(p -> _eta_pred(p, θ, N, opts), preds)
        for i in 1:N
            pr = fam isa Disc ? _cum_probs(fam, etas[1][i], _at(etas[2], i).tau, exp(etas[3][i])) :
                 fam isa OrdinalFamily ? _cum_probs(fam, etas[1][i], _at(etas[2], i).tau) :
                 _cat_probs(fam, map(e -> e[i], etas))
            out[s, i, 1:length(pr)] .= pr
        end
    end
    return out
end

"""
    posterior_predict(fit; newdata=nothing, rng=Random.default_rng(), re_formula=nothing,
                      allow_new_levels=false, sample_new_levels=:uncertainty)

Draws from the posterior predictive distribution.
"""
function posterior_predict(fit::BrmsFit; newdata=nothing, rng=Random.default_rng(), kw...)
    fam = fit.info.family
    fam isa _MultiCat && return _multicat_draws(fit, newdata, :predict; rng=rng, kw...)
    fam isa TimeCov && return _tc_draws(fit, newdata, :predict; rng=rng, kw...)
    return _per_draw((eta, t, i) -> _rand(rng, fam, eta, t), fit, newdata; rng=rng, kw...)
end

"""
    log_lik(fit; newdata=nothing, re_formula=nothing, allow_new_levels=false)

Pointwise log-likelihood draws (draws x observations), the input to PSIS-LOO.
Includes all normalising constants and, for weighted models, the weights.
"""
function log_lik(fit::BrmsFit; newdata=nothing, kw...)
    fam = fit.info.family
    fam isa TimeCov && return _tc_draws(fit, newdata, :log_lik; kw...)
    newdata !== nothing && fam isa _MultiCat &&
        throw(ArgumentError("log_lik with new data is not supported for $(family_name(fam)) yet"))
    if newdata === nothing
        y = Float64.(fit.density.y)
        w = Float64.(fit.density.weights)
        trials = fit.density.trials
    else
        f = fit.info.formula
        cols = Tables.columntable(newdata)
        yv = _eval_atom(f.response, cols, f.mod)
        if fam isa _Ord && !(nonmissingtype(eltype(yv)) <: Real)
            lv = fit.info.infos[f.response].levels
            yv = [findfirst(isequal(v), lv) for v in yv]
        end
        y = Float64.(yv)
        w = f.weights === nothing ? Float64[] : Float64.(_eval_atom(f.weights, cols, f.mod))
        trials = uses_trials(fam) ? (f.trials isa Int ? fill(f.trials, length(y)) : Int.(_eval_atom(f.trials, cols, f.mod))) : Int[]
    end
    # the family's per-observation data (log y, trials, ...) — what `_ll`
    # reads as its `x` argument, exactly as in the density
    if newdata !== nothing && fam isa Censored
        f = fit.info.formula
        cols = Tables.columntable(newdata)
        codes = cens_code.(_eval_atom(f.cens, cols, f.mod))
        y2 = f.cens_y2 === nothing ? zeros(length(y)) :
             [ismissing(v) ? 0.0 : Float64(v) for v in _eval_atom(f.cens_y2, cols, f.mod)]
        fam = Censored(fam.fam, codes, y2)
    end
    xd = _xdata(fam, y, trials)
    return _per_draw(fit, newdata; kw...) do eta, t, i
        x = isempty(xd) ? 0.0 : xd[i]
        v = _ll(fam, y[i], x, eta) + _ll_const_obs(fam, y[i], t, x)
        isempty(w) ? v : w[i] * v
    end
end

"""
    residual_draws(fit) -> Matrix   (draws x observations)

Draws of the response residuals `y - E[y]` (brms `residuals(..., summary=FALSE)`).
"""
function residual_draws(fit::BrmsFit)
    E = posterior_epred(fit)
    return Float64.(fit.density.y)' .- E
end

# multinomial / dirichlet: expected values or draws per category
function _multicat_draws(fit::BrmsFit, newdata, what; re_formula=nothing, allow_new_levels=false,
                         sample_new_levels=:uncertainty, rng=Random.default_rng(), trials=nothing)
    fam = fit.info.family
    preds, N, _ = _prediction_preds(fit, newdata; re_formula, allow_new_levels)
    opts = _PredOpts(rng, Symbol(sample_new_levels))
    n = newdata === nothing ? vec(sum(fam.Y; dims=2)) :
        trials === nothing ? fill(fam isa Multinomial ? NaN : 1.0, N) : Float64.(trials)
    fam isa Multinomial && what === :predict && any(isnan, n) &&
        throw(ArgumentError("posterior_predict for new data needs `trials =` (the total counts)"))
    D = reshape(fit.draws, size(fit.draws, 1), :)
    K = _ncat(fam)
    out = Array{Float64,3}(undef, size(D, 2), N, K)
    for s in axes(D, 2)
        θ = view(D, :, s)
        etas = map(p -> _eta_pred(p, θ, N, opts), preds)
        for i in 1:N
            eta = map(e -> _at(e, i), etas)
            if fam isa LogisticNormal
                out[s, i, :] .= what === :epred ? _ln_mean(fam, eta) : _ln_draw(rng, fam, eta)
                continue
            end
            p = _mc_probs(fam, eta)
            if what === :epred
                out[s, i, :] .= fam isa Multinomial ? n[i] .* p : p
            elseif fam isa Multinomial
                out[s, i, :] .= rand(rng, Distributions.Multinomial(round(Int, n[i]), p))
            else
                out[s, i, :] .= rand(rng, Distributions.Dirichlet(exp(eta[end]) .* p))
            end
        end
    end
    return out
end
