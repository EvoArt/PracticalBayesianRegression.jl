# =============================================================================
# Formula + data + family + priors -> GLMDensity.
#
# This is the "front end" half of the package: dynamic, allocation-happy,
# run once per model. Nothing here is on the sampling path, so none of it
# needs to be trimmable — the GLMDensity it produces is the part that does.
# =============================================================================

"""
    DparInfo

What was learned from the data for one distributional parameter's formula:
the population-level design spec, the atom schemas, and the levels of each
grouping factor. Kept on the fit so new data can be coded identically.
"""
struct DparInfo
    dpar::Symbol
    rhs::RHS
    spec::DesignSpec
    groups::Vector{Tuple{GroupTerm,GroupInfo,DesignSpec}}
    smooths::Vector{Any}      # (spec, smooth object, by levels) per smooth term
end

"""
    ModelInfo

Everything about a built model except the numbers: formula, family, the
per-dpar design information and the resolved prior table.
"""
struct ModelInfo
    formula::BrmsFormula
    family::Family
    dinfo::Vector{DparInfo}
    infos::Dict{Atom,AtomInfo}
    priors::Vector{PriorSpec}     # every parameter class, defaults included
    nobs::Int
    rows::Vector{Int}             # rows of the input data used (complete cases)
    atom_values::Dict{Atom,AbstractVector}   # training values (conditional_effects grids)
end

# --- default priors (brms's def_scale_prior) ---------------------------------
#
# Verified against brms 2.23.0 `get_prior()` output for gaussian, student,
# poisson, bernoulli, binomial, negbinomial, Gamma and a distributional
# gaussian model. The rule: for an identity/log mean
# link, take the response on the link scale (zeros nudged to 0.1 for log),
# location = round(median, 1) and scale = max(2.5, round(mad, 1)); for every
# other link, location 0 and scale 2.5. Intercepts use the location; sd and
# sigma use location 0 with a lower bound of 0.

# Statistics.median written out: its median! has an error path that prints
# the array (repr), which juliac --trim cannot compile, and model binaries
# recompute these defaults for each data set. Same result: sort, then the
# middle value (Statistics.middle, a/2 + b/2, for even n).
function _tmedian(x::AbstractVector{<:Real})
    s = sort(Float64.(x)); n = length(s); k = (n + 1) ÷ 2
    return isodd(n) ? s[k] : Statistics.middle(s[k], s[k + 1])
end
_rmad(x) = 1.4826 * _tmedian(abs.(x .- _tmedian(x)))   # R's mad(): constant 1.4826

_default_loc_scale(fam::Family, y) = _loc_scale_link(prior_link(fam), y)

# brms's def_scale_prior.brmsterms on the response transformed by `link`.
# A transform undefined for some y (log of a negative) makes brms's median / mad
# NA, and it keeps the fallback 0 / 2.5: NaN here does the same.
function _loc_scale_link(link::Symbol, y)
    # brms's transformable links ("tlinks"); logit/probit/cloglog/softplus... are not
    link in (:identity, :log, :inverse, :sqrt, Symbol("1/mu^2")) || return 0.0, 2.5
    nz(v) = v == 0 ? v + 0.1 : Float64(v)
    yy = if link === :log
        [(u = nz(v); u > 0 ? log(u) : NaN) for v in y]
    elseif link === :inverse
        [1 / nz(v) for v in y]
    elseif link === Symbol("1/mu^2")
        [1 / nz(v)^2 for v in y]
    elseif link === :sqrt
        [v >= 0 ? sqrt(Float64(v)) : NaN for v in y]
    else
        Float64.(y)
    end
    any(isnan, yy) && return 0.0, 2.5
    scale = 2.5
    s = round(_rmad(yy); digits=1)
    isfinite(s) && (scale = max(scale, s))
    loc = round(_tmedian(yy); digits=1)
    return isfinite(loc) ? loc : 0.0, scale
end

# brms's def_dpar_prior: a scalar sigma's default scale is computed with the
# identity link (log for the log-scale families), whatever mu's link is; sigma
# with its own formula gets its Intercept scale on its own (log) link.
_has_logscale(::Family) = false
_has_logscale(::Union{Lognormal,HurdleLognormal,ShiftedLognormal}) = true
_has_logscale(f::Union{Censored,Truncated,TimeCov}) = _has_logscale(f.fam)
_sigma_scale(fam, y) = last(_loc_scale_link(_has_logscale(fam) ? :log : :identity, y))

# The Intercept of a distributional parameter with its own formula: brms 2.23's
# def_dpar_prior for a non-identity dpar link (every default dpar link here).
# Checked against brms's get_prior.
function _dpar_intercept_prior(dpar::Symbol, y)
    dpar === :sigma && return student_t(3, 0, last(_loc_scale_link(:log, y)))
    dpar === :nu && return normal(2.7, 0.8)
    dpar in (:zi, :hu, :zoi, :coi, :quantile, :bias) && return logistic_prior(0, 1)
    dpar === :kappa && return normal(5.0, 0.8)
    dpar === :beta && return normal(1.7, 1.3)
    dpar in (:alpha, :xi) && return normal(0, 4)
    dpar === :disc && return normal(0, 1)
    dpar === :bs && return normal(-0.6, 1.3)
    dpar === :ndt && return flat()
    return student_t(3, 0, 2.5)                # shape, phi, and the rest
end

# Defaults as brms 2.23 `get_prior` reports them (checked for every family
# here).
# mixtures: a component's parameters get that component family's defaults;
# theta intercepts logistic(0, 1) (brms)
function _default_prior(fam::Mixture, class::Symbol, dpar::Symbol, y)
    if _is_mix_theta(dpar)
        return class === :Intercept ? logistic_prior(0, 1) : _default_prior(fam.comps[1], class, :theta, y)
    elseif occursin(r"\d+$", string(class)) && class !== :Intercept
        k, base = _mix_split(class)                  # :sigma2 -> component 2's sigma
        return _default_prior(fam.comps[k], base, :mu, y)
    end
    k, base = _mix_split(dpar)                       # :mu2, :sigma2 (with a formula)
    return _default_prior(fam.comps[k], class, base, y)
end

# cens() / trunc() wrappers: the base family's defaults (its `isa` rules, e.g.
# negbinomial's inv_gamma shape, must see the base family, not the wrapper)
_default_prior(c::Union{Censored,Truncated}, class::Symbol, dpar::Symbol, y) = _default_prior(c.fam, class, dpar, y)

function _default_prior(fam::Family, class::Symbol, dpar::Symbol, y)
    loc, scale = _default_loc_scale(fam, y)
    if class === :Intercept
        # brms: von_mises' mean intercept gets student_t(1, 0, 1) (tan_half link only)
        fam isa VonMises{TanHalfLink} && dpar === :mu && return student_t(1, 0, 1)
        dpar === :mu && return student_t(3, loc, scale)
        return _dpar_intercept_prior(dpar, y)
    elseif class === :b
        return flat()
    elseif class === :df
        return gamma(2, 0.1)                   # student-t group effects (brms)
    elseif class in (:sd, :sds, :sdgp, :sdcar, :sderr)
        return dpar === :mu ? student_t(3, 0, scale) : student_t(3, 0, 2.5)
    elseif class in (:cor, :lncor, :cortime, :corme)
        return lkj(1)
    elseif fam isa LogisticNormal && startswith(string(class), "sigma")
        return student_t(3, 0, 2.5)
    elseif class === :sigma
        fam isa NegBinomial2 && return gamma(0.4, 0.3)
        return student_t(3, 0, _sigma_scale(fam, y))
    elseif class === :shape
        return fam isa Union{NegBinomial,ZeroInflatedNegBinomial,HurdleNegBinomial} ?
               inv_gamma(0.4, 0.3) : gamma(0.01, 0.01)
    elseif class === :phi
        return gamma(0.01, 0.01)
    elseif class === :nu
        return gamma(2, 0.1)
    elseif class in (:zi, :hu, :zoi, :coi, :quantile, :rhocar)
        return beta_prior(1, 1)
    elseif class === :kappa
        return fam isa VonMises ? gamma(2, 0.01) : gamma(0.01, 0.01)      # von_mises / xbeta (brms)
    elseif class === :xi
        return normal(0, 2.5)                  # brms: on the unscaled tmp_xi
    elseif class in (:ar, :ma, :meanme, :sdme, :car, :cosy)
        return flat()
    elseif class === :alpha
        return normal(0, 4)
    elseif class === :beta
        return gamma(1, 0.1)
    elseif class === :ndt
        return flat()          # brms: uniform(0, min_Y), i.e. flat on the bounded scale
    end
    throw(ArgumentError("no default prior for class $class"))
end

# Look up a user prior or fall back to the default, and record which was used.
function _resolve!(table, user, fam, y, class, dpar::Symbol; coef="", group="", lb=-Inf)
    dstr = dpar === :mu ? "" : string(dpar)
    row = _lookup_prior(user, class, coef, group, dstr)
    p = row === nothing ? _default_prior(fam, class, dpar, y) : row.prior
    src = row === nothing ? :default : :user
    push!(table, PriorSpec(p, class, coef, group, dstr, src))
    p isa LKJ && return p
    return with_bound(p, lb)
end

# class b: the prior plus the coefficient's bounds (brms's lb/ub on b)
function _resolve_b!(table, user, fam, y, dpar::Symbol, coef)
    dstr = dpar === :mu ? "" : string(dpar)
    row = _lookup_prior(user, :b, coef, "", dstr)
    p = row === nothing ? _default_prior(fam, :b, dpar, y) : row.prior
    lb, ub = row === nothing ? (-Inf, Inf) : (row.lb, row.ub)
    lb < ub || throw(ArgumentError("prior bounds for b $(coef) need lb < ub"))
    push!(table, PriorSpec(p, :b, coef, "", dstr, row === nothing ? :default : :user, lb, ub))
    return with_bound(p, lb, ub), lb, ub
end

# --- data handling -----------------------------------------------------------

function _all_atoms(f::BrmsFormula)
    atoms = Atom[f.response]
    f.trials isa Atom && push!(atoms, f.trials)
    f.weights isa Atom && push!(atoms, f.weights)
    f.cens isa Atom && push!(atoms, f.cens)
    for v in f.nlvars
        push!(atoms, Atom(v))
    end
    haskey(f.extra, :se) && push!(atoms, f.extra[:se][1])
    haskey(f.extra, :subset) && push!(atoms, f.extra[:subset])
    haskey(f.extra, :vars) && append!(atoms, f.extra[:vars])
    haskey(f.extra, :dec) && push!(atoms, f.extra[:dec])
    haskey(f.extra, :thres) && f.extra[:thres][2] !== nothing && push!(atoms, f.extra[:thres][2])
    if haskey(f.extra, :trunc)
        foreach(b -> b isa Atom && push!(atoms, b), f.extra[:trunc])
    end
    for d in f.dpars
        _rhs_atoms!(atoms, d.rhs)
    end
    return unique(atoms)
end

function _rhs_atoms!(atoms, rhs::RHS)
    foreach(t -> append!(atoms, t), rhs.terms)
    append!(atoms, rhs.offsets)
    append!(atoms, rhs.mo)
    for t in rhs.moint, a in t
        push!(atoms, _is_mo(a) ? _atom(a.ex.args[2]) : a)
    end
    append!(atoms, rhs.mi)
    foreach(c -> push!(atoms, c[2]), rhs.car)
    for gt in rhs.groups
        gt.by === nothing || push!(atoms, gt.by)
        append!(atoms, gt.mmw)
    end
    for (a, s, _) in rhs.me
        push!(atoms, a, s)
    end
    for gpspec in rhs.gp
        append!(atoms, gpspec[1])
    end
    for spec in rhs.sm
        append!(atoms, spec[2])
        spec[4] === nothing || push!(atoms, spec[4])
    end
    foreach(t -> append!(atoms, t), rhs.cs)
    if rhs.ar !== nothing
        rhs.ar[1] === nothing || push!(atoms, rhs.ar[1])
        rhs.ar[2] === nothing || push!(atoms, rhs.ar[2])
    end
    for gt in rhs.groups
        foreach(t -> append!(atoms, t), gt.terms)
        append!(atoms, gt.group)
    end
    return atoms
end

_complete(v) = map(!ismissing, v)

# --- building ----------------------------------------------------------------

"""
    build_model(formula, data; family=gaussian(), prior=PriorSpec[], T=Float64)
        -> (GLMDensity, ModelInfo)

Construct the log density of a brms-style model without sampling it. `data`
is any Tables.jl table (a DataFrame, a NamedTuple of vectors, ...). `T` is
the element type of the parameter vector and of the stored design matrices
(use `Float32` for a single-precision sampling path). `sample_prior=:only`
drops the likelihood (brms's `sample_prior = "only"`): the fit is then a
draw from the prior, and its posterior_predict is the prior predictive.
"""
function build_model(formula, data; family=gaussian(), prior=PriorSpec[], T::Type{<:AbstractFloat}=Float64,
                     sample_prior=:no, next::Base.RefValue{Int}=Ref(1), shared_ids=nothing, resp::String="",
                     data2=nothing, mi_mode::Symbol=:auto, mi_marg=Set{String}())
    f = bf(formula)
    # (a multivariate build filters each response's rows itself and passes resp)
    haskey(f.extra, :subset) && resp == "" &&
        throw(ArgumentError("subset() is for the responses of multivariate models (as in brms)"))
    _DATA2[] = (data2, f.mod)
    fam = family isa Family ? family : brmsfamily(family)
    user = prior isa PriorSpec ? [prior] : collect(PriorSpec, prior)
    cols = Tables.columntable(data)
    N0 = _nrows(cols)

    # evaluate every atom once over the full data, then keep complete rows
    atoms = _all_atoms(f)
    raw = Dict{Atom,AbstractVector}(a => _eval_atom(a, cols, f.mod) for a in atoms)
    # y | mi() and mi(x): those variables may be missing (they are imputed)
    mivars = Set{Atom}(vcat(get(f.extra, :mi, false) ? [f.response] : Atom[], reduce(vcat, [d.rhs.mi for d in f.dpars]; init=Atom[])))
    keep = trues(N0)
    for (a, v) in raw
        a in mivars && continue
        keep .&= _complete(v)
    end
    rows = findall(keep)
    if length(rows) < N0
        @warn "Rows containing missing values were excluded from the model." n_excluded = N0 - length(rows)
    end
    vals = Dict{Atom,AbstractVector}(a => a in mivars ? [ismissing(x) ? NaN : Float64(x) for x in v[rows]] :
                                           collect(skipmissing(v[rows])) for (a, v) in raw)
    # y2 is only meaningful on interval-censored rows, and brms allows it to be
    # missing elsewhere, so it is kept out of the complete-case filter
    y2raw = f.cens_y2 === nothing ? nothing : _eval_atom(f.cens_y2, cols, f.mod)[rows]
    N = length(rows)
    N > 0 || throw(ArgumentError("no complete observations"))
    infos = Dict{Atom,AtomInfo}(a => _atom_info(a, v) for (a, v) in vals)

    # response
    yv = vals[f.response]
    if fam isa Union{OrdinalFamily,Categorical} && !(nonmissingtype(eltype(yv)) <: Real)
        # ordered categories: code by level order (a CategoricalArray's own
        # order, else sorted), as brms does for an ordered factor
        lv = collect(DataAPI.levels(yv))
        yv = [findfirst(isequal(v), lv) for v in yv]
    end
    if fam isa _MultiCat
        # a matrix response: cbind(a, b, c) (categories named after the columns)
        ex = f.response.ex
        (ex isa Expr && ex.head == :call && ex.args[1] === :cbind) ||
            throw(ArgumentError("$(family_name(fam)) needs a matrix response: cbind(a, b, c) ~ ..."))
        # yv holds one K-vector per row (see `_interp` for cbind): stack to N x K
        Ym = Float64.(permutedims(reduce(hcat, yv)))
        cats = [brms_rename(string(a)) for a in ex.args[2:end]]
        fam = _with_matrix(fam, cats, Ym)
        yv = vec(sum(Ym; dims=2))          # the trials (multinomial) / 1 (dirichlet)
    end
    eltype(yv) <: Real || throw(ArgumentError("the response must be numeric"))
    y = Float64.(yv)
    fam isa ShiftedLognormal && (fam = typeof(fam)(minimum(y)))   # ndt < min(y)
    if fam isa Categorical
        K = Int(maximum(y))
        K >= 2 || throw(ArgumentError("a categorical response needs at least two categories"))
        fam = _with_ncat(fam, K)
    end
    trials = if uses_trials(fam)
        f.trials === nothing && throw(ArgumentError("the binomial family needs `y | trials(n) ~ ...`"))
        f.trials isa Int ? fill(f.trials, N) : Int.(vals[f.trials])
    else
        f.trials === nothing || fam isa Multinomial ||
            throw(ArgumentError("trials() is only meaningful for the binomial family"))
        Int[]
    end
    if f.cens !== nothing && fam isa Cox
        # cox: the censoring codes are the family's per-observation data (brms)
        codes = Float64.(cens_code.(vals[f.cens]))
        all(c -> c in (-1, 0, 1), codes) || throw(ArgumentError("cox(): only left / right censoring is supported"))
        fam = typeof(fam)(codes)
    elseif f.cens !== nothing
        fam isa _CENSORABLE || throw(ArgumentError(
            "censoring is implemented for gaussian (identity), lognormal, weibull, exponential and the count families " *
            "(poisson, negbinomial, negbinomial2, geometric, binomial, beta_binomial, discrete_weibull, com_poisson) so far, " *
            "not $(family_name(fam))"))
        codes = cens_code.(vals[f.cens])
        y2 = y2raw === nothing ? zeros(N) : [ismissing(v) ? 0.0 : Float64(v) for v in y2raw]
        for i in eachindex(codes)
            codes[i] == 2 && !(y2[i] > y[i]) &&
                throw(ArgumentError("interval-censored observation $i needs y2 > y (got y = $(y[i]), y2 = $(y2[i]))"))
        end
        fam = Censored(fam, codes, y2)
    end
    if haskey(f.extra, :se)
        fam isa Gaussian{IdentityLink} || throw(ArgumentError("se() is supported for the gaussian family (identity link)"))
        a, sig = f.extra[:se]
        se = Float64.(vals[a])
        all(>(0), se) || throw(ArgumentError("standard errors in se() must be positive"))
        fam = GaussianSE{IdentityLink,sig}(se)
    end
    if haskey(f.extra, :trunc)
        f.cens === nothing || throw(ArgumentError("trunc() together with cens() is not supported yet"))
        fam isa _CENSORABLE || throw(ArgumentError(
            "trunc() is implemented for gaussian (identity), lognormal, weibull, exponential and the count families so far"))
        bv(b) = b isa Atom ? Float64.(vals[b]) : fill(Float64(b), N)
        lbv, ubv = bv(f.extra[:trunc][1]), bv(f.extra[:trunc][2])
        all(lbv .< ubv) || throw(ArgumentError("trunc() needs lb < ub"))
        fam = Truncated(fam, lbv, ubv)
    end
    Jmi = Int[]
    if get(f.extra, :mi, false)
        fam isa Union{Gaussian{IdentityLink},Student{IdentityLink}} ||
            throw(ArgumentError("y | mi(): imputation is supported for gaussian/student (identity) responses"))
        Jmi = findall(isnan, y)
        y[Jmi] .= 0.0                        # placeholders: the values are θ[ymi]
        # :auto (a univariate model, or a response no mi() term uses): the missing
        # responses integrate out exactly -- their rows leave the likelihood
        # (weight 0) and their draws are generated after sampling (mimarg.jl)
        mi_mode in (:auto, :sample, :drop) || throw(ArgumentError("mi must be :auto or :sample"))
        mi_mode === :auto && (mi_mode = :drop)
    end
    # dec(): the wiener family's boundary per response (and ndt < min(rt))
    if haskey(f.extra, :dec)
        fam isa Wiener || throw(ArgumentError("dec() is for the wiener family"))
        fam = _with_dec(fam, vals[f.extra[:dec]], y)
    end
    # vreal() / vint(): a custom family's per-observation data
    if haskey(f.extra, :vars)
        fam isa CustomFamily || throw(ArgumentError("vreal() / vint() are for custom families"))
        fam = _with_vars(fam, reduce(hcat, [Float64.(vals[a]) for a in f.extra[:vars]]))
    end
    # residual covariance structures (timecov.jl): the family carries the structure
    ac = f.dpars[1].rhs.ar
    # (other families' ar/ma/arma(cov = TRUE) and cosy(): brms's latent residuals,
    # a term of mu's predictor -- see _build_linpred)
    if ac !== nothing && ac[5] !== :arma &&
       (fam isa Union{Gaussian{IdentityLink},Student{IdentityLink}} || !(ac[5] in (:arma_cov, :cosy)))
        (f.cens === nothing && !haskey(f.extra, :se) && !haskey(f.extra, :trunc) && isempty(Jmi) && f.weights === nothing) ||
            throw(ArgumentError("$(ac[5]) terms cannot be combined with cens(), se(), trunc(), mi() or weights() yet"))
        fam = _timecov_family(fam, ac, vals, y, rows, N0)
        for d in f.dpars
            d.dpar in _ac_dpars(fam) && throw(ArgumentError("`$(d.dpar)` cannot have a formula"))
            d.dpar === :nu && throw(ArgumentError("nu cannot have a formula in a model with $(ac[5]) (brms: one multi_student_t)"))
            d.dpar === :sigma && _is_sar(fam) && throw(ArgumentError("sar(): sigma cannot have a formula (brms)"))
        end
    end
    any(d -> d.rhs.ar !== nothing && d.rhs.ar[5] !== :arma, f.dpars[2:end]) &&
        throw(ArgumentError("autocorrelation structures belong in the formula of mu"))
    _check_response(fam, y, trials)
    weights = f.weights === nothing ? Float64[] : Float64.(vals[f.weights])
    if !isempty(Jmi) && mi_mode === :drop
        isempty(weights) && (weights = ones(N))
        weights[Jmi] .= 0.0
    end

    # a mixture's theta: one simplex, or a formula per component (theta2 ~ ...)
    fam isa Mixture && (fam = _with_theta_formulas(fam, any(d -> _is_mix_theta(d.dpar), f.dpars)))

    # ordinal discrimination: a parameter only when it has a formula (brms fixes disc = 1)
    fam isa Union{OrdinalFamily,HurdleCumulative} && any(d -> d.dpar === :disc, f.dpars) && (fam = Disc{typeof(fam)}())

    # check the dpar formulas against the family
    fdp = dpars(fam)
    for d in f.dpars
        fam isa Union{Categorical,Mixture,_MultiCat} && d.dpar === :mu && continue    # the shared formula of every mu<k>
        d.dpar in f.nlpars && continue                       # non-linear parameters
        d.dpar in fdp || throw(ArgumentError("family $(family_name(fam)) has no parameter `$(d.dpar)` (it has $(join(fdp, ", ")))"))
    end

    table = PriorSpec[]
    # (ID, group) -> members; a multivariate model passes one registry for all
    # responses and finishes it itself, so IDs can span responses
    shared = shared_ids === nothing ? Dict{Tuple{String,String},Vector{Any}}() : shared_ids
    nbefore = Dict(k => length(v) for (k, v) in shared)
    # `next` may be shared with other responses (multivariate models)
    take(n) = (r = next[]:(next[] + n - 1); next[] += n; r)
    preds = Any[]
    dinfo = DparInfo[]
    for dp in fdp
        k = findfirst(d -> d.dpar === dp, f.dpars)
        if dp === :thres
            # ordinal thresholds: one per category boundary, prior class
            # Intercept with coef = threshold number (brms's convention)
            k === nothing || throw(ArgumentError("thresholds cannot have a formula"))
            f.dpars[1].rhs.intercept || throw(ArgumentError("ordinal models need an intercept (it becomes the thresholds)"))
            # (hurdle_cumulative: 0 is the hurdle, categories are 1..K as for cumulative)
            nthr, tgr = get(f.extra, :thres, (nothing, nothing))
            ordered = _ord_core(fam) isa Union{Cumulative,HurdleCumulative}
            if tgr === nothing
                K1 = nthr === nothing ? Int(maximum(y)) - 1 : nthr
                K1 >= 1 || throw(ArgumentError("an ordinal response needs at least two categories"))
                K1 >= Int(maximum(y)) - 1 ||
                    throw(ArgumentError("thres($K1): the response has $(Int(maximum(y))) categories"))
                pri = [_resolve!(table, user, fam, y, :Intercept, :mu; coef=string(j)) for j in 1:K1]
                # only cumulative's thresholds are ordered (brms: `ordered` vs `vector`)
                thr = Thresholds(take(K1), pri, ordered)
            else
                # thres(gr = g): one threshold vector per level of g, each with
                # max(y in the level) - 1 entries unless thres(n, ...) fixes n (brms)
                isempty(f.dpars[1].rhs.cs) || throw(ArgumentError("thres(gr = ) together with cs() is not supported"))
                gv = [_level_name(v) for v in vals[tgr]]
                labs = sort(unique(gv))
                gidx = Dict(l => k for (k, l) in enumerate(labs))
                lens = map(labs) do l
                    my = Int(maximum(y[gv .== l]))
                    n = nthr === nothing ? my - 1 : nthr
                    n >= max(my - 1, 1) || throw(ArgumentError("thres(): level $l of $(tgr.label) has $my categories"))
                    n
                end
                starts = cumsum([1; lens[1:(end - 1)]])
                pri = [_resolve!(table, user, fam, y, :Intercept, :mu; coef=string(j), group=l)
                       for (l, n) in zip(labs, lens) for j in 1:n]
                thr = Thresholds(take(sum(lens)), pri, ordered, [gidx[v] for v in gv], starts, lens, labs, tgr)
            end
            csterms = f.dpars[1].rhs.cs
            if isempty(csterms)
                push!(preds, thr)
            else
                _ord_core(fam) isa Union{Sratio,Cratio,Acat} ||
                    throw(ArgumentError("cs() is supported for sratio, cratio and acat (brms: not cumulative)"))
                cspec = design_spec(false, csterms, infos)
                Xcs = model_matrix(cspec, vals, infos, N)
                labs = brms_rename.(cspec.colnames)
                cpri = [first(_resolve_b!(table, user, fam, y, :mu, l)) for l in labs]
                push!(preds, ThresholdsCS{T}(thr, Xcs, take(size(Xcs, 2) * K1), cpri, labs, cspec))
            end
            continue
        end
        if dp === :mu && f.nl
            push!(preds, _build_nlpred(f, vals, infos, N, T, fam, y, user, table, take, dinfo))
            continue
        end
        if dp === :cortime
            # unstr(): LKJ correlation over the time levels (class cortime, brms lkj(1))
            D = fam.nmax
            lk = _resolve!(table, user, fam, y, :cortime, :mu)
            push!(preds, LnCorBlock(:cortime, take(D * (D - 1) ÷ 2), D, lk.eta))
            continue
        end
        if dp === :lncor
            # logistic_normal: LKJ correlation of the log-ratios (class lncor)
            D = _ncat(fam) - 1
            lk = _resolve!(table, user, fam, y, :lncor, :mu)
            push!(preds, LnCorBlock(:lncor, take(D * (D - 1) ÷ 2), D, lk.eta))
            continue
        end
        if dp === :sbhaz
            # cox baseline hazard: M/I-spline basis in the response, simplex weights
            row = _lookup_prior(user, :sbhaz, "", "", "")
            alpha = row === nothing ? 1.0 : Float64(row.prior.alpha)
            push!(table, PriorSpec(DirichletPrior(alpha), :sbhaz, "", "", "", row === nothing ? :default : :user))
            bb = BhazBasis(y)
            Zb, Zc = _bhaz_matrices(bb, y)
            push!(preds, BhazSimplex{T}(:sbhaz, take(size(Zb, 2) - 1), alpha, T.(Zb), T.(Zc), bb))
            continue
        end
        if dp === :theta
            # mixture proportions: a simplex with a symmetric dirichlet prior
            row = _lookup_prior(user, :theta, "", "", "")
            alpha = row === nothing ? 1.0 : Float64(row.prior.alpha)
            push!(table, PriorSpec(DirichletPrior(alpha), :theta, "", "", "", row === nothing ? :default : :user))
            push!(preds, ThetaSimplex(:theta, take(_ncomp(fam) - 1), alpha))
            continue
        end
        if k === nothing && fam isa Mixture && _is_mix_theta(dp)
            # a component without its own theta formula: intercept only
            lp, di = _build_linpred(dp, RHS(true, Vector{Atom}[], GroupTerm[], Atom[]), vals, infos, N, T, fam, y,
                                    user, table, take; shared)
            push!(preds, lp); push!(dinfo, di)
            continue
        end
        if k === nothing && (fam isa Categorical || (fam isa Mixture && _is_mix_mu(dp)) ||
                             (fam isa _MultiCat && _is_cat_mu(fam, dp)))
            # mu<k> without its own formula takes the main one
            lp, di = _build_linpred(dp, f.dpars[1].rhs, vals, infos, N, T, fam, y, user, table, take; shared)
            push!(preds, lp); push!(dinfo, di)
            continue
        end
        if k === nothing
            lk, lb = aux_bounds(fam, dp)
            ub = lk === :logit ? aux_ub(fam, dp) : Inf
            p = _resolve!(table, user, fam, y, dp, :mu; lb=lk === :identity ? -Inf : lb)
            push!(preds, ScalarDpar(dp, first(take(1)), lk === :identity ? -Inf : lb, ub, lk, p))
            continue
        end
        fam isa GenExtremeValue && dp === :xi &&
            throw(ArgumentError("gen_extreme_value(): xi cannot have a formula (brms scales it by the data)"))
        rhs = f.dpars[k].rhs
        # grouped thresholds: brms does not centre the design (mu += X * b)
        center = !(dp === :mu && haskey(f.extra, :thres) && f.extra[:thres][2] !== nothing) &&
                 get(f.extra, :center, true)       # bf(..., center = false): brms's uncentred intercept
        lp, di = _build_linpred(dp, rhs, vals, infos, N, T, fam, y, user, table, take; shared, center,
                                mi_marg=dp === :mu ? mi_marg : Set{String}(), mecor=get(f.extra, :mecor, true))
        push!(preds, lp); push!(dinfo, di)
    end
    if shared_ids === nothing
        _finish_shared!(preds, shared, table, user, fam, y, T, take)
    else
        # tag this response's members with what finishing them needs
        for (k, mem) in shared, j in (get(nbefore, k, 0) + 1):length(mem)
            mem[j] = merge(mem[j], (preds=preds, table=table, user=user, fam=fam, y=y, resp=resp))
        end
    end
    dim = next[] - 1
    # horseshoe autoscale: point at the scalar sigma (none if sigma has a formula)
    sig = findfirst(p -> p isa ScalarDpar && p.dpar === :sigma, preds)
    for p in preds
        p isa LinPred && p.hs.on && p.hs.sigma[] == -1 && (p.hs.sigma[] = sig === nothing ? 0 : preds[sig].idx)
        p isa LinPred && p.r2d2.on && p.r2d2.sigma[] == -1 && (p.r2d2.sigma[] = sig === nothing ? 0 : preds[sig].idx)
    end

    # data-only likelihood constants
    lconst = 0.0
    xd = _xdata(fam, y, trials)
    for i in 1:N
        c = _ll_const_obs(fam, y[i], isempty(trials) ? 0 : trials[i], isempty(xd) ? 0.0 : xd[i])
        lconst += isempty(weights) ? c : weights[i] * c
    end
    fam isa TimeCov && (lconst -= fam.ldM / 2)          # fcor: the normaliser's log det M

    prior_only = Symbol(sample_prior) === :only
    if prior_only
        flats = [r for r in table if r.prior isa ScalarPrior && isflat(r.prior)]
        isempty(flats) || throw(ArgumentError(
            "sampling from the prior needs proper priors, but $(join(unique("$(r.class)" * (r.coef == "" ? "" : " ($(r.coef))") for r in flats), ", ")) " *
            "have flat priors; set them with `prior(...)`"))
    end
    ordered = Int[]
    if fam isa Mixture && fam.order
        for k in 1:_ncomp(fam)
            j = findfirst(p -> p isa LinPred && p.dpar === Symbol(:mu, k), preds)
            (j === nothing || preds[j].intercept == 0) &&
                throw(ArgumentError("ordering the mixture components needs an intercept in every mu formula; " *
                                    "use mixture(...; order = false)"))
            push!(ordered, preds[j].intercept)
        end
    end
    pinned, pinvals = _constant_pins(preds)
    isempty(intersect(pinned, ordered)) ||
        throw(ArgumentError("a constant() prior on an ordered mixture intercept is not supported"))
    # sampled: θ entries; dropped: generated entries beyond θ (a multivariate
    # build renumbers them after the joint dimension is known)
    ymi = isempty(Jmi) || mi_mode === :drop ? (1:0) : take(length(Jmi))
    dim = next[] - 1
    # dropped: generated entries just past θ (a multivariate build renumbers
    # them once the joint dimension is known)
    isempty(Jmi) || mi_mode !== :drop || (ymi = (dim + 1):(dim + length(Jmi)))
    ld = GLMDensity(fam, Tuple(preds), T.(y), trials, T.(weights), dim, lconst; prior_only=prior_only,
                    ordered=ordered, pinned=pinned, pinvals=pinvals, Jmi=Jmi, ymi=ymi)
    info = ModelInfo(f, fam, dinfo, infos, table, N, rows, vals)
    return ld, info
end

# non-tabular data for terms such as car(W): brms's `data2`, else the formula's module
const _DATA2 = Ref{Any}((nothing, Main))
function _lookup_data2(ex)
    d2, mod = _DATA2[]
    if ex isa Symbol
        d2 !== nothing && haskey(d2, ex) && return getindex(d2, ex)
        isdefined(mod, ex) && return getfield(mod, ex)
    end
    throw(ArgumentError("`$ex` not found: pass it in `data2 = (; $ex = ...)`"))
end

# constant() priors -> (θ indices, unconstrained values)
function _constant_pins(preds)
    idx = Int[]; val = Float64[]
    pin(k, v) = (push!(idx, k); push!(val, v))
    for p in preds
        if p isa LinPred
            p.intercept > 0 && isconstant(p.int_prior) && pin(p.intercept, p.int_prior.a)
            for (j, pr) in enumerate(p.b_priors)
                isconstant(pr) || continue
                (p.b_lb[j] == -Inf && p.b_ub[j] == Inf) ||
                    throw(ArgumentError("a constant() prior on a bounded coefficient is not supported"))
                (p.hs.on || p.r2d2.on) && throw(ArgumentError("constant() cannot be combined with horseshoe/R2D2"))
                pin(p.b[j], pr.a)
            end
            for gb in p.groups
                _owns(gb) || continue
                for (j, pr) in enumerate(gb.sd_priors)
                    isconstant(pr) || continue
                    pr.a > 0 || throw(ArgumentError("a constant sd must be positive"))
                    pin(gb.sd[j], log(pr.a))
                end
            end
            for (j, pr) in enumerate(p.ar.priors)
                isconstant(pr) && pin(p.ar.idx[j], pr.a)
            end
            for (j, pr) in enumerate(p.ar.ma_priors)
                isconstant(pr) && pin(p.ar.ma_idx[j], pr.a)
            end
        elseif p isa ScalarDpar && isconstant(p.prior)
            c = p.prior.a
            u = p.kind === :identity ? c : p.kind === :log ? log(c - p.lb) : logit((c - p.lb) / (p.ub - p.lb))
            isfinite(u) || throw(ArgumentError("constant($c) is outside the support of $(p.dpar)"))
            pin(p.idx, u)
        end
    end
    return idx, val
end

# allocate every shared-ID group block: one sd vector, one correlation and one
# z matrix over all members' columns; each member predictor gets its rows
function _finish_shared!(preds, shared, table, user, fam, y, T, take)
    sid = 0
    for ((id, label), mem) in sort!(collect(shared); by=first)
        J = length(mem[1].levels)
        qs = [size(m.Z, 2) for m in mem]
        qt = sum(qs)
        cor = all(m -> m.cor, mem) && qt > 1
        m1 = mem[1]
        lk = cor ? _resolve!(get(m1, :table, table), get(m1, :user, user), get(m1, :fam, fam), get(m1, :y, y),
                             :cor, m1.dp; group=label) : LKJ(1.0)
        all(m -> m.levels == m1.levels, mem) ||
            throw(ArgumentError("group-term ID `$id`: the grouping factor $label has different levels across responses"))
        sd_r = take(qt)
        corr_r = cor ? take(qt * (qt - 1) ÷ 2) : (1:0)
        z_r = take(qt * J)
        single = length(mem) == 1          # an ID used once is an ordinary block
        single || (sid += 1)
        sdp = reduce(vcat, [m.sd_p for m in mem])
        # joint labels "<dpar>__<coef>", prefixed "<resp>::" across responses
        jl = reduce(vcat, [[(haskey(m, :resp) && m.resp != "" ? m.resp * "::" : "") * "$(m.dp)__$(c)" for c in m.coefs]
                           for m in mem])
        r0 = 0
        for (n, m) in enumerate(mem)
            role = single ? 0 : n == 1 ? 1 : n == length(mem) ? 3 : 2
            ps = get(m, :preds, preds)
            k = findfirst(p -> p isa LinPred && p.dpar === m.dp, ps)
            ps[k].groups[m.pos] = GroupBlock{T}(label, m.idx, m.levels, m.Z, m.coefs, sd_r, corr_r, z_r, sdp, lk.eta;
                                                    rows=(r0 + 1):(r0 + qs[n]), role=role, sid=single ? 0 : sid,
                                                    jlabels=jl)
            r0 += qs[n]
        end
    end
    return preds
end

function _build_nlpred(f, vals, infos, N, T, fam, y, user, table, take, dinfo)
    pars = map(f.nlpars) do a
        k = findfirst(d -> d.dpar === a, f.dpars)
        lp, di = _build_linpred(a, f.dpars[k].rhs, vals, infos, N, T, fam, y, user, table, take; nlpar=true)
        push!(dinfo, di)
        any(isflat, lp.b_priors) && throw(ArgumentError(
            "non-linear parameter `$a` needs proper priors on its coefficients " *
            "(e.g. `prior(\"normal(0, 5)\"; class=:b, nlpar=:$a)`), as in brms"))
        lp
    end
    data = Tuple(T.(Float64.(vals[Atom(v)])) for v in f.nlvars)
    return NLPred(:mu, f.nlfun, Tuple(pars), copy(f.nlpars), data)
end

function _build_linpred(dp, rhs, vals, infos, N, T, fam, y, user, table, take; nlpar::Bool=false, shared=nothing,
                        center::Bool=true, mi_marg=Set{String}(), mecor::Bool=true)
    spec = design_spec(rhs.intercept, rhs.terms, infos)
    X = model_matrix(spec, vals, infos, N)
    coefs = brms_rename.(spec.colnames)
    # smooths: unpenalised columns join the population design (brms bs_sx_1,
    # here b_sx_1), penalised columns become a random-effect block below
    sminfo = Any[]
    smZ = Any[]              # (label, Z, prior coef) per penalised block
    for spec in rhs.sm
        kind, covs, k, by, text = spec
        bs = length(spec) >= 6 ? spec[6] : :tp
        Xcov = bs === :re ? [vals[a] for a in covs] : reduce(hcat, [Float64.(vals[a]) for a in covs])
        lab = string(kind) * join(brms_rename.(getfield.(covs, :label)))
        # a numeric `by` multiplies an UNcentred smooth (mgcv applies no constraint then)
        numby = by !== nothing && !iscategorical(infos[by])
        t = kind === :t2 ? T2Smooth(Xcov; k, label=lab) :
            bs === :re ? res(Xcov, Union{Nothing,Vector{String}}[iscategorical(infos[a]) ?
                                 [_level_name(l) for l in infos[a].levels] : nothing for a in covs]; label=lab) :
            bs === :cr ? crs(Xcov; k, label=lab, center=!numby) :
            bs === :cs ? crs(Xcov; k, label=lab, center=!numby, shrink=true) :
            bs === :ts ? tprs(Xcov; k, label=lab, center=!numby, shrink=true) :
            bs === :ps ? pss(Xcov; k, label=lab, center=!numby) :
            bs === :cc ? ccs(Xcov; k, label=lab, center=!numby) : tprs(Xcov; k, label=lab, center=!numby)
        Xs, Zs = smooth_basis(t, Xcov)
        if numby
            # s(x, by = z), z numeric: brms names the smooth s<x>:<z>
            bz = Float64.(vals[by])
            ll = lab * ":" * brms_rename(by.label)
            X = hcat(X, Xs .* bz)
            append!(coefs, [ll * "_$j" for j in 1:size(Xs, 2)])
            for (b, Zb) in enumerate(Zs)
                push!(smZ, (ll * "_$b", Zb .* bz, text))
            end
            push!(sminfo, (spec, t, String[]))
        elseif by === nothing
            X = hcat(X, Xs)
            append!(coefs, [lab * "_$j" for j in 1:size(Xs, 2)])
            for (b, Zb) in enumerate(Zs)
                push!(smZ, (lab * "_$b", Zb, text))
            end
            push!(sminfo, (spec, t, String[]))
        else
            # s(x, by = g): a smooth per level of g (factor)
            lvs = [_level_name(l) for l in infos[by].levels]
            bv = [_level_name(v) for v in vals[by]]
            for l in lvs
                ind = bv .== l
                ll = lab * ":" * brms_rename(by.label) * l
                X = hcat(X, Xs .* ind)
                append!(coefs, [ll * "_$j" for j in 1:size(Xs, 2)])
                for (b, Zb) in enumerate(Zs)
                    push!(smZ, (ll * "_$b", Zb .* ind, text))
                end
            end
            push!(sminfo, (spec, t, lvs))
        end
    end
    if nlpar && rhs.intercept
        # a non-linear parameter's intercept is an ordinary, uncentred
        # coefficient (brms: class b, coef Intercept)
        X = hcat(ones(N), X)
        coefs = vcat("Intercept", coefs)
        rhs = _with_intercept(rhs, false)
    end
    # brms centres the population-level design whenever there is an intercept
    # (but not in a model with category-specific cs() terms)
    means = center && rhs.intercept && isempty(rhs.cs) ? vec(mean(X; dims=1)) : zeros(size(X, 2))
    # in an ordinal model the formula's intercept is the thresholds block,
    # so the linear predictor itself has none (but is still centred)
    ordinal = _ord_core(fam) isa Union{OrdinalFamily,HurdleCumulative} && dp === :mu
    if ordinal
        rhs = _with_intercept(rhs, false)
    end
    Xc = X .- means'
    offset = zeros(N)
    for a in rhs.offsets
        offset .+= Float64.(vals[a])
    end

    int_idx = 0
    int_prior = flat()
    if rhs.intercept
        int_prior = _resolve!(table, user, fam, y, :Intercept, dp)
        int_idx = first(take(1))
    end
    b_idx = take(size(X, 2))
    # a horseshoe prior on class b replaces every coefficient prior of this dpar
    hsrow = _lookup_prior(user, :b, "", "", dp === :mu ? "" : string(dp))
    hsp = hsrow !== nothing && hsrow.prior isa HorseshoePrior ? hsrow.prior : nothing
    r2p = hsrow !== nothing && hsrow.prior isa R2D2Prior ? hsrow.prior : nothing
    if r2p !== nothing
        isempty(coefs) && throw(ArgumentError("an R2D2 prior needs population-level coefficients"))
        push!(table, PriorSpec(r2p, :b, "", "", dp === :mu ? "" : string(dp), :user))
        b_priors = fill(flat(), length(coefs))
        b_lb = fill(-Inf, length(coefs)); b_ub = fill(Inf, length(coefs))
    elseif hsp !== nothing
        isempty(coefs) && throw(ArgumentError("a horseshoe prior needs population-level coefficients"))
        push!(table, PriorSpec(hsp, :b, "", "", dp === :mu ? "" : string(dp), :user))
        b_priors = fill(flat(), length(coefs))
        b_lb = fill(-Inf, length(coefs)); b_ub = fill(Inf, length(coefs))
    else
        bres = [_resolve_b!(table, user, fam, y, dp, c) for c in coefs]
        b_priors = [r[1] for r in bres]
        b_lb = [r[2] for r in bres]; b_ub = [r[3] for r in bres]
    end

    groups = GroupBlock{T}[]
    ginfo = Tuple{GroupTerm,GroupInfo,DesignSpec}[]
    for gt in rhs.groups
        gi = _group_info(gt, vals, infos)
        idx = isempty(gt.mm) ? _group_index(gi, gt, vals) : ones(Int, N)
        gspec = design_spec(gt.intercept, gt.terms, infos)
        Zx = model_matrix(gspec, vals, infos, N)
        Z = gt.intercept ? hcat(ones(N), Zx) : Zx
        gcoefs = vcat(gt.intercept ? ["Intercept"] : String[], brms_rename.(gspec.colnames))
        q = size(Z, 2)
        q > 0 || throw(ArgumentError("group term for $(group_label(gt)) has no effects"))
        label = group_label(gt)
        sd_p = [_resolve!(table, user, fam, y, :sd, dp; coef=c, group=label, lb=0.0) for c in gcoefs]
        if gt.by !== nothing
            # gr(g, by = h): a block per level of h (g must be nested in h)
            byv = [_level_name(v) for v in vals[gt.by]]
            bylv = iscategorical(infos[gt.by]) ? [_level_name(l) for l in infos[gt.by].levels] : sort(unique(byv))
            for j in 1:length(gi.levels)
                length(unique(byv[idx .== j])) == 1 ||
                    throw(ArgumentError("gr(): each level of $(label) must belong to one level of $(gt.by.label)"))
            end
            cor = gt.cor && q > 1
            lk = cor ? _resolve!(table, user, fam, y, :cor, dp; group=label) : LKJ(1.0)
            for b in bylv
                inb = byv .== b
                any(inb) || continue
                lvj = sort(unique(idx[inb]))
                loc = Dict(j => n for (n, j) in enumerate(lvj))
                idxb = [inb[i] ? loc[idx[i]] : 1 for i in 1:N]
                Zb = Z .* inb
                sd_r = take(q)
                corr_r = cor ? take(q * (q - 1) ÷ 2) : (1:0)
                z_r = take(q * length(lvj))
                levb = gi.levels[lvj]
                push!(groups, GroupBlock{T}(label, idxb, levb, T.(Zb), gcoefs, sd_r, corr_r, z_r, sd_p, lk.eta;
                                            bylevel=b, sdsfx=":" * brms_rename(gt.by.label) * b))
                push!(ginfo, (gt, GroupInfo(levb, Dict(l => n for (n, l) in enumerate(levb))), gspec))
            end
            continue
        end
        mmJ = zeros(Int, 0, 0); mmW = zeros(0, 0)
        if !isempty(gt.mm)
            # mm(g1, g2): level of each membership, and the weights
            M = length(gt.mm)
            mmJ = reduce(hcat, [[gi.index[l] for l in _group_values(GroupTerm(true, Vector{Atom}[], m, false), vals)]
                                for m in gt.mm])
            mmW = isempty(gt.mmw) ? fill(1 / M, N, M) : reduce(hcat, [Float64.(vals[a]) for a in gt.mmw])
            gt.mmscale && (mmW = mmW ./ sum(mmW; dims=2))
        end
        if gt.id != ""
            shared === nothing && throw(ArgumentError("group-term IDs are not supported in non-linear formulas yet"))
            # (terms | id | g): a placeholder now; the joint block is allocated
            # once every predictor is built (`_finish_shared!`)
            push!(get!(shared, (gt.id, label), Any[]),
                  (dp=dp, Z=T.(Z), coefs=gcoefs, idx=idx, levels=gi.levels, sd_p=sd_p, cor=gt.cor, pos=length(groups) + 1))
            push!(groups, GroupBlock{T}(label, idx, gi.levels, T.(Z), gcoefs, 1:0, 1:0, 1:0, sd_p, 1.0; role=-1))
            push!(ginfo, (gt, gi, gspec))
            continue
        end
        sd_r = take(q)
        cor = gt.cor && q > 1
        lk = cor ? _resolve!(table, user, fam, y, :cor, dp; group=label) : LKJ(1.0)
        corr_r = cor ? take(q * (q - 1) ÷ 2) : (1:0)
        z_r = take(q * length(gi.levels))
        # gr(g, dist = "student"): df (class df, group g; brms gamma(2, 0.1) on (1, Inf)) and one udf per level
        dfi, udf_r, dfp = 0, 1:0, flat()
        if gt.student
            dfp = _resolve!(table, user, fam, y, :df, dp; group=label, lb=1.0)
            dfi = first(take(1)); udf_r = take(length(gi.levels))
        end
        # gr(g, cov = A): A's rows / columns are the levels in sorted order (Julia
        # matrices have no row names; brms matches A's names to the levels)
        Lcov = zeros(T, 0, 0)
        if gt.cov !== nothing
            A = Float64.(_lookup_data2(gt.cov))
            nl = length(gi.levels)
            size(A) == (nl, nl) ||
                throw(ArgumentError("gr(cov = $(gt.cov)): A must be $nl x $nl (the levels of $label, sorted)"))
            CA = cholesky(Symmetric(A); check=false)
            issuccess(CA) || throw(ArgumentError("gr(cov = $(gt.cov)): A must be positive definite"))
            Lcov = T.(Matrix(CA.L))
        end
        push!(groups, GroupBlock{T}(label, idx, gi.levels, T.(Z), gcoefs, sd_r, corr_r, z_r, sd_p, lk.eta;
                                    mmJ=mmJ, mmW=mmW, df=dfi, udf=udf_r, df_prior=dfp, Lcov=Lcov))
        push!(ginfo, (gt, gi, gspec))
    end
    # monotonic effects: coefficient bsp (class b, coef "mo<x>"), simplex simo
    mos = MoTerm[]
    for a in rhs.mo
        x = _mo_levels(vals[a], infos[a])
        D = maximum(x)
        D >= 1 || throw(ArgumentError("mo($(a.label)) needs at least two levels"))
        lab = "mo" * brms_rename(a.label)
        bp, _, _ = _resolve_b!(table, user, fam, y, dp, lab)
        bsp = first(take(1))
        row = _lookup_prior(user, :simo, lab * "1", "", dp === :mu ? "" : string(dp))
        alpha = row === nothing ? 1.0 : row.prior.alpha
        push!(table, PriorSpec(row === nothing ? DirichletPrior(1.0) : row.prior, :simo, lab * "1", "",
                               dp === :mu ? "" : string(dp), row === nothing ? :default : :user))
        push!(mos, MoTerm(lab, x, D, bsp, take(D - 1), bp, alpha, _mo_min(vals[a], infos[a])))
    end
    # monotonic interaction terms (brms): mo(x):z, mo(x1):mo(x2); one coefficient
    # (coef "mox:z") and one simplex per mo factor (simo coef "mox:z1", "mox1:mox22")
    for t in rhs.moint
        moat = [_atom(a.ex.args[2]) for a in t if _is_mo(a)]
        csat = [a for a in t if !_is_mo(a)]
        # mo(x):f with a factor f: brms codes f fully (one column per level, no
        # contrasts), each its own term -- coef "mox:fa", simplex "mox:fa1"
        facs = [a for a in csat if iscategorical(infos[a])]
        nums = [a for a in csat if !iscategorical(infos[a])]
        xs = [_mo_levels(vals[a], infos[a]) for a in moat]
        Ds = maximum.(xs)
        all(>=(1), Ds) || throw(ArgumentError("mo() variables need at least two levels"))
        levs = [[_level_name(l) for l in infos[a].levels] for a in facs]
        for combo in (isempty(facs) ? [()] : vec(collect(Iterators.product(levs...))))
            fl = Dict(zip(facs, combo))
            lab = join([_is_mo(a) ? "mo" * brms_rename(_atom(a.ex.args[2]).label) :
                        haskey(fl, a) ? brms_rename(a.label * fl[a]) : brms_rename(a.label) for a in t], ":")
            bp, _, _ = _resolve_b!(table, user, fam, y, dp, lab)
            bsp = first(take(1))
            simos = UnitRange{Int}[]; alphas = Float64[]
            for (f, D) in enumerate(Ds)
                row = _lookup_prior(user, :simo, lab * string(f), "", dp === :mu ? "" : string(dp))
                push!(alphas, row === nothing ? 1.0 : row.prior.alpha)
                push!(table, PriorSpec(row === nothing ? DirichletPrior(1.0) : row.prior, :simo, lab * string(f), "",
                                       dp === :mu ? "" : string(dp), row === nothing ? :default : :user))
                push!(simos, take(D - 1))
            end
            cf = Tuple{Atom,String}[(a, fl[a]) for a in facs]
            csp = _mo_csp(vals, nums, cf)
            push!(mos, MoTerm(lab, xs, Ds, bsp, simos, bp, alphas, csp, moat, nums, [_mo_min(vals[a], infos[a]) for a in moat], cf))
        end
    end
    # spatial CAR terms car(W, gr = loc, type = )
    cars = CarTerm{T}[]
    for (Wex, ga, type) in rhs.car
        W = _lookup_data2(Wex)
        (W isa AbstractMatrix && size(W, 1) == size(W, 2) && issymmetric(W)) ||
            throw(ArgumentError("car(): W must be a symmetric adjacency matrix"))
        W = Float64.(W .!= 0)
        nl = size(W, 1)
        lv = vals[ga]
        J = all(v -> v isa Integer || (v isa Real && isinteger(v)), lv) && all(v -> 1 <= v <= nl, lv) ? Int.(lv) :
            (u = sort(unique(string.(lv))); length(u) == nl ||
                throw(ArgumentError("car(): $(nl) locations in W but $(length(u)) distinct values of gr"));
             [findfirst(==(string(v)), u) for v in lv])
        e1, e2 = _car_edges(W)
        nn = vec(sum(W; dims=2))
        any(iszero, nn) && throw(ArgumentError("car(): every location needs at least one neighbour"))
        Dh = Diagonal(1 ./ sqrt.(nn))
        eig = type in (:escar, :esicar) ? eigvals(Symmetric(Dh * W * Dh)) : Float64[]
        scale = type === :bym2 ? _car_scale(W) : 1.0
        sdp = _resolve!(table, user, fam, y, :sdcar, dp; lb=0.0)
        sd_i = first(take(1))
        z_r = take(type === :esicar ? nl - 1 : nl)
        nsz_r = type === :bym2 ? take(nl) : (1:0)
        rp = type === :bym2 ? _resolve!(table, user, fam, y, :rhocar, dp) : type === :escar ? _resolve!(table, user, fam, y, :car, dp) : flat()
        rho_i = type in (:bym2, :escar) ? first(take(1)) : 0
        push!(cars, CarTerm{T}(type, J, nl, e1, e2, nn, eig, scale, sd_i, z_r, nsz_r, rho_i, sdp, rp))
    end
    # imputed predictors mi(x) (linked to x's model by the multivariate build)
    mis = MiTerm{T}[]
    for a in rhs.mi
        xv = Float64.(vals[a])
        lab = "mi" * brms_rename(a.label)
        bp, _, _ = _resolve_b!(table, user, fam, y, dp, lab)
        var = replace(brms_rename(a.label), "_" => "", "." => "")
        push!(mis, MiTerm{T}(lab, var, T.(xv), findall(isnan, xv), Ref(1:0), first(take(1)), bp, var in mi_marg))
    end
    # measurement-error terms me(x, sdx)
    mes = MeTerm{T}[]
    # brms mecor = TRUE (default): two or more me() terms share one correlated
    # latent block (MeCorTerm, special.jl); their own zme stay empty
    cor_me = mecor && length(rhs.me) >= 2
    for (a, sa, lab) in rhs.me
        sdx = Float64.(vals[sa])
        all(>(0), sdx) || throw(ArgumentError("me(): standard errors must be positive"))
        latent = "me" * brms_rename(a.label)
        bp, _, _ = _resolve_b!(table, user, fam, y, dp, lab)
        mp = _resolve!(table, user, fam, y, :meanme, dp; coef=latent)
        sp = _resolve!(table, user, fam, y, :sdme, dp; coef=latent, lb=0.0)
        push!(mes, MeTerm{T}(lab, latent, T.(vals[a]), T.(sdx), first(take(1)), first(take(1)), first(take(1)),
                             cor_me ? (1:0) : take(N), bp, mp, sp))
    end
    mecors = MeCorTerm{T}[]
    if cor_me
        K = length(mes)
        lk = _resolve!(table, user, fam, y, :corme, dp)
        z = take(K * N)
        cr = take(K * (K - 1) ÷ 2)
        lat = [m.latent for m in mes]
        push!(mecors, MeCorTerm{T}(copy(mes), cr, z, lk.eta,
                                   ["corme__$(lat[j])__$(lat[k])" for k in 2:K for j in 1:(k - 1)]))
    end
    # Gaussian processes gp(x, ...)
    gps = GPTerm{T}[]
    for spec in rhs.gp
        covs, k, c, scale = spec
        cov = length(spec) >= 5 ? spec[5] : :exp_quad
        X = reduce(hcat, [Float64.(vals[a]) for a in covs])
        J, Xs, dmax, cmeans, Lb, ks = _gp_data(X, k, c, scale)
        lab = "gp" * join(brms_rename.(getfield.(covs, :label)))
        sdp = _resolve!(table, user, fam, y, :sdgp, dp; coef=lab, lb=0.0)
        lrow = _lookup_prior(user, :lscale, lab, "", dp === :mu ? "" : string(dp))
        lsp0 = lrow === nothing ? _lscale_prior(Xs) : lrow.prior
        push!(table, PriorSpec(lsp0, :lscale, lab, "", dp === :mu ? "" : string(dp), lrow === nothing ? :default : :user))
        lsp = with_bound(lsp0, 0.0)
        approx = k !== nothing
        Phi, lam = approx ? _gp_basis(Xs .- cmeans', ks, Lb) : (zeros(0, 0), Float64[])
        nz = approx ? size(Phi, 2) : size(Xs, 1)
        push!(gps, GPTerm{T}(lab, J, Xs, approx ? zeros(T, 0, 0) : T.(_sqdist(Xs)), approx, T.(Phi), T.(lam),
                             dmax, cmeans, Lb, ks, first(take(1)), first(take(1)), take(nz), sdp, lsp, cov))
    end
    # ARMA residual terms (mu of gaussian/student only, brms's cov = FALSE form)
    ar = ARTerm{T}()
    if rhs.ar !== nothing && rhs.ar[5] === :arma
        dp === :mu && fam isa Union{Gaussian{IdentityLink},Student{IdentityLink}} ||
            throw(ArgumentError("ar()/ma()/arma() are supported on the mean of gaussian/student (identity) models"))
        ta, ga, p, q = rhs.ar
        time = ta === nothing ? collect(1:N) : Float64.(vals[ta])
        gr = ga === nothing ? nothing : vals[ga]
        pri = [_resolve!(table, user, fam, y, :ar, :mu; coef=string(k)) for k in 1:p]
        mpri = [_resolve!(table, user, fam, y, :ma, :mu; coef=string(k)) for k in 1:q]
        prev, order = _ar_prev(time, gr, max(p, q))
        ar = ARTerm{T}(p, take(p), q, take(q), prev, order, T.(y), pri, mpri)
    end
    # latent residual autocorrelation (non-gaussian families; brms err = sderr * L zerr)
    lacs = LatentAC{T}[]
    # (gaussian / student were already wrapped in TimeCov by build_model: the
    # structure is in their likelihood, so no latent term -- never both)
    if rhs.ar !== nothing && rhs.ar[5] in (:arma_cov, :cosy) &&
       !(fam isa Union{TimeCov,Gaussian{IdentityLink},Student{IdentityLink}})
        dp === :mu || throw(ArgumentError("autocorrelation structures belong in the formula of mu"))
        tc = _timecov_family(Gaussian{IdentityLink}(), rhs.ar, vals, y, 1:N, N)
        cls = collect(Symbol, _ac_dpars(tc))
        sdp = _resolve!(table, user, fam, y, :sderr, dp; lb=0.0)
        rpri = [_resolve!(table, user, fam, y, c, dp) for c in cls]
        sd_i = first(take(1)); rho_r = take(length(cls)); z_r = take(N)
        lbs = [c === :cosy ? 0.0 : -1.0 for c in cls]; ubs = ones(length(cls))
        push!(lacs, LatentAC{T}(tc, sd_i, rho_r, lbs, ubs, z_r, sdp, rpri, [_scalar_label(c) for c in cls]))
    end
    hs = HSTerm()
    if hsp !== nothing
        K = length(coefs)
        sg = isnan(hsp.par_ratio) ? hsp.scale_global : hsp.par_ratio / sqrt(N)
        glob = first(take(1)); slab = first(take(1)); loc = take(K)
        # autoscale uses a scalar sigma; its index is filled in once it exists
        hs = HSTerm(true, hsp.df, sg, hsp.df_global, hsp.scale_slab, hsp.df_slab, glob, slab, loc,
                    Ref(hsp.autoscale && :sigma in dpars(fam) ? -1 : 0))
    end
    r2d2 = R2D2Term()
    if r2p !== nothing
        K = length(coefs)
        r2d2 = R2D2Term(true, r2p.mean_R2 * r2p.prec_R2, (1 - r2p.mean_R2) * r2p.prec_R2, r2p.cons_D2,
                        first(take(1)), take(K - 1), Ref(r2p.autoscale && :sigma in dpars(fam) ? -1 : 0))
    end
    # smooth random parts: sds (class sds, coef "s(x)") and standard-normal zs
    sblocks = SmoothBlock{T}[]
    for (blab, Zs, text) in smZ
        pr = _resolve!(table, user, fam, y, :sds, dp; coef=text, lb=0.0)
        push!(sblocks, SmoothBlock{T}(blab, T.(Zs), first(take(1)), take(size(Zs, 2)), pr))
    end
    lp = LinPred{T}(dp, int_idx, T.(Xc), T.(means), coefs, b_idx, groups, T.(offset),
                    int_prior, b_priors, any(!isflat, b_priors); b_lb=b_lb, b_ub=b_ub, mo=mos, me=mes, gp=gps, mi=mis, car=cars, ar=ar, hs=hs,
                    r2d2=r2d2, smooths=sblocks, lac=lacs, mecor=mecors)
    return lp, DparInfo(dp, rhs, spec, ginfo, sminfo)
end
