# =============================================================================
# Model recipes: build a GLMDensity for NEW data without the front end.
#
# A model binary (binary.jl) must refit a fixed formula on any data with the
# right columns, as a CmdStan executable does. The front end that turns a
# formula + table into a GLMDensity (formula.jl, design.jl, build.jl) is
# dynamic by design -- atoms are interpreted Exprs, values live in
# Dict{Atom,AbstractVector}, levels in Vector{Any} -- and cannot be compiled
# by `juliac --trim`. But everything structural about a model (family, which
# terms, how each factor is coded, which parameters have formulas) is fixed by
# the formula, and the density's TYPE depends only on that structure, not on
# N, the number of columns or levels.
#
# So: `model_recipe` (build time) builds the model once with the real front
# end, checks it uses only supported features, and extracts a concretely typed
# ModelRecipe. `assemble` (run time, trimmable) replays build_model's rules on
# new atom values: complete cases, factor levels, R's dummy coding and column
# order, centring, group indices, brms default priors (the package's own
# `_default_prior` / `with_bound`), and the same parameter-layout order
# (`take`). Before returning, `model_recipe` re-assembles the training data and
# requires EXACT agreement with build_model (type, dimension, variable names,
# density and gradient), so a rule this file does not replicate fails at build
# time instead of producing a wrong binary.
#
# Supported (v1): every family whose object does not depend on the data
# (gaussian, student, poisson, bernoulli, binomial, negbinomial, gamma,
# lognormal, beta, weibull, exponential, zero-inflated / hurdle, skew_normal,
# ... -- `_recipe_family_ok`) and shifted_lognormal; formulas for any distributional parameter; numeric atoms including
# elementwise transforms (log(x), I(x^2)); factor columns with R's treatment /
# full dummy coding; interactions; offset(); trials(); weights(); group terms
# (1 + x | g), (x || g), (1 | g1:g2), correlated or not; scalar priors and
# lkj() on cor, by class / coef / group / dpar. Not yet: censoring /
# truncation / se, ordinal and categorical families, mixtures, smooths, GPs,
# mo(), autocorrelation, me(), mi(), car(), poly/bs/ns, mm(), gr(by, cov,
# dist), group IDs, horseshoe / R2D2 / constant() priors, non-linear formulas.
# =============================================================================

struct RTerm
    atoms::Vector{Int}               # atom ids; first varies fastest (R)
end

struct RGroup
    label::String                    # group_label: "g", or "g1:g2"
    group::Vector{Int}               # the grouping atoms
    intercept::Bool
    terms::Vector{RTerm}
    cor::Bool
end

struct RLin
    dpar::Symbol
    intercept::Bool
    terms::Vector{RTerm}
    offsets::Vector{Int}
    groups::Vector{RGroup}
end

struct RScalar
    dpar::Symbol
    kind::Symbol                     # aux_bounds' link: :log, :logit, :identity
    lb::Float64                      # (the upper bound of a :logit one is aux_ub at run time:
end                                  #  shifted_lognormal's ndt < min(y) depends on the data)

# Families a recipe supports: a family type with no fields cannot carry data,
# so it is the same object for any data set -- except the ordinal and
# categorical ones (thresholds / category count come from the data). Plus
# shifted_lognormal, whose one datum (min y) `_refamily` recomputes. Anything
# holding data (Censored, Truncated, Cox, mixtures, TimeCov, wiener, custom
# families) is out until it gets a refamily rule and a new-data test.
_recipe_family_ok(fam::Family) =
    (Base.issingletontype(typeof(fam)) && !(fam isa Union{Categorical,_Ord,HurdleCumulative,Disc})) ||
    fam isa ShiftedLognormal
_refamily(fam::Family, y) = fam
_refamily(fam::ShiftedLognormal, y) = typeof(fam)(minimum(y))   # build_model: ndt < min(y)

# a user prior row, typed (PriorSpec.prior is Any)
struct RPrior
    class::Symbol
    coef::String
    group::String
    dpar::String
    prior::ScalarPrior
    lb::Float64
    ub::Float64
end

struct RLKJ
    coef::String
    group::String
    dpar::String
    eta::Float64
end

"""
    ModelRecipe

The structure of a model, without its data: what `assemble` needs to build
the GLMDensity for any data with the right columns. Made by
[`model_recipe`](@ref).
"""
struct ModelRecipe{F<:Family,S<:Tuple}
    family::F
    slots::S                         # one RLin / RScalar per distributional parameter, dpars(family) order
    labels::Vector{String}           # atom labels
    cat::Vector{Bool}                # atom is a factor
    resp::Int
    trials::Int                      # atom id of trials(n), 0 if none
    ntrials::Int                     # trials(10): a constant, 0 if none
    weights::Int
    priors::Vector{RPrior}
    lkj::Vector{RLKJ}
end

"""
    AtomVals

Every atom of a recipe evaluated on a data set (before the complete-case
filter): numeric values (NaN for factors), level names (R's labels; for a
numeric atom the value as written), and missingness.
"""
struct AtomVals
    num::Vector{Vector{Float64}}
    str::Vector{Vector{String}}
    miss::Vector{BitVector}
end

# ------------------------------------------------------------------ run time

# what the per-dpar assembly needs
struct _RCtx{F}
    family::F
    labels::Vector{String}
    cat::Vector{Bool}
    num::Vector{Vector{Float64}}
    str::Vector{Vector{String}}
    lev::Vector{Vector{String}}
    y::Vector{Float64}
    N::Int
    priors::Vector{RPrior}
    lkj::Vector{RLKJ}
    next::Base.RefValue{Int}
end

function _rtake(c::_RCtx, n::Int)
    r = c.next[]:(c.next[] + n - 1)
    c.next[] += n
    return r
end

# _lookup_prior's rule: the most specific matching row (coef 2, group 1), the
# last one on ties
function _rlookup(rows, class::Symbol, coef::String, group::String, dpar::String)
    best = 0; best_score = -1
    for (k, r) in enumerate(rows)
        r.class === class && r.dpar == dpar || continue
        (r.coef == "" || r.coef == coef) || continue
        (r.group == "" || r.group == group) || continue
        score = (r.coef != "" ? 2 : 0) + (r.group != "" ? 1 : 0)
        score >= best_score && (best = k; best_score = score)
    end
    return best
end
function _rlookup_lkj(rows::Vector{RLKJ}, group::String, dpar::String)
    best = 0; best_score = -1
    for (k, r) in enumerate(rows)
        r.dpar == dpar || continue
        (r.group == "" || r.group == group) || continue
        score = r.group != "" ? 1 : 0
        score >= best_score && (best = k; best_score = score)
    end
    return best
end

# _resolve!: a user prior or the brms default, bounded below at lb
function _rresolve(c::_RCtx, class::Symbol, dpar::Symbol; coef::String="", group::String="", lb::Float64=-Inf)
    dstr = dpar === :mu ? "" : string(dpar)
    k = _rlookup(c.priors, class, coef, group, dstr)
    p = k == 0 ? _default_prior(c.family, class, dpar, c.y) : c.priors[k].prior
    p isa ScalarPrior || throw(ArgumentError("no scalar prior for class $class"))
    return with_bound(p, lb)
end
# _resolve_b!: the prior and the coefficient's bounds
function _rresolve_b(c::_RCtx, dpar::Symbol, coef::String)
    dstr = dpar === :mu ? "" : string(dpar)
    k = _rlookup(c.priors, :b, coef, "", dstr)
    p = k == 0 ? _default_prior(c.family, :b, dpar, c.y) : c.priors[k].prior
    p isa ScalarPrior || throw(ArgumentError("no scalar prior for class b"))
    lb = k == 0 ? -Inf : c.priors[k].lb
    ub = k == 0 ? Inf : c.priors[k].ub
    return with_bound(p, lb, ub), lb, ub
end
function _rlkj_eta(c::_RCtx, dpar::Symbol, group::String)
    k = _rlookup_lkj(c.lkj, group, dpar === :mu ? "" : string(dpar))
    return k == 0 ? 1.0 : c.lkj[k].eta            # brms default lkj(1)
end

# design_spec's coding rules: which factor in which term gets FULL dummy
# coding (rule 1: a factor whose margin is absent; rule 2: no intercept, the
# first factor with more than one level)
function _rfull(c::_RCtx, intercept::Bool, terms::Vector{RTerm})
    full = [falses(length(t.atoms)) for t in terms]
    for ti in eachindex(terms)
        t = terms[ti].atoms
        for ai in eachindex(t)
            a = t[ai]
            c.cat[a] || continue
            rest = Set{Int}(x for x in t if x != a)
            isempty(rest) && continue
            any(pj -> issubset(rest, Set{Int}(terms[pj].atoms)), 1:(ti - 1)) && continue
            full[ti][ai] = true
        end
    end
    if !intercept
        found = false
        for ti in eachindex(terms)
            t = terms[ti].atoms
            for ai in eachindex(t)
                a = t[ai]
                if c.cat[a] && length(c.lev[a]) > 1
                    full[ti][ai] = true
                    found = true
                    break
                end
            end
            found && break
        end
    end
    return full
end

# one atom's columns and their names (_atom_block / _atom_colnames)
function _rblock(c::_RCtx, a::Int, full::Bool)
    if !c.cat[a]
        return reshape(copy(c.num[a]), :, 1), [c.labels[a]]
    end
    lv = full ? c.lev[a] : c.lev[a][2:end]
    B = zeros(Float64, c.N, length(lv))
    v = c.str[a]
    for k in eachindex(lv), i in 1:c.N
        v[i] == lv[k] && (B[i, k] = 1.0)
    end
    return B, [c.labels[a] * l for l in lv]
end

# model_matrix + the column names of design_spec: interactions expand with the
# first atom varying fastest
function _rdesign(c::_RCtx, intercept::Bool, terms::Vector{RTerm})
    full = _rfull(c, intercept, terms)
    cols = Vector{Float64}[]
    names = String[]
    for ti in eachindex(terms)
        t = terms[ti].atoms
        cur, cn = _rblock(c, t[1], full[ti][1])
        for j in 2:length(t)
            nxt, nn = _rblock(c, t[j], full[ti][j])
            M = Matrix{Float64}(undef, c.N, size(cur, 2) * size(nxt, 2))
            mn = String[]
            k = 0
            for n in axes(nxt, 2), cc in axes(cur, 2)
                k += 1
                for i in 1:c.N
                    M[i, k] = cur[i, cc] * nxt[i, n]
                end
                push!(mn, cn[cc] * ":" * nn[n])
            end
            cur, cn = M, mn
        end
        for k in axes(cur, 2)
            push!(cols, cur[:, k])
        end
        append!(names, cn)
    end
    X = zeros(Float64, c.N, length(cols))
    for k in eachindex(cols)
        X[:, k] .= cols[k]
    end
    return X, names
end

# levels and per-observation index of a grouping factor (_group_info /
# _group_index): a factor keeps its levels, a numeric group sorts numerically,
# a combination g1:g2 sorts its joined labels "a_b"
function _rgroup(c::_RCtx, g::RGroup)
    if length(g.group) == 1
        a = g.group[1]
        lab = c.str[a]
        if c.cat[a]
            lv = c.lev[a]
        else
            v = c.num[a]
            uv = sort(unique(v))
            # each distinct value named as it was first written
            lv = String[]
            for u in uv
                i = findfirst(==(u), v)
                push!(lv, lab[i::Int])
            end
        end
    else
        lab = [join((c.str[a][i] for a in g.group), "_") for i in 1:c.N]
        lv = sort(unique(lab))
    end
    index = Dict{String,Int}(l => k for (k, l) in enumerate(lv))
    idx = Int[index[l] for l in lab]
    return lv, idx
end

function _rslot(c::_RCtx, s::RScalar)
    lbp = s.kind === :identity ? -Inf : s.lb
    ub = s.kind === :logit ? Float64(aux_ub(c.family, s.dpar)) : Inf
    p = _rresolve(c, s.dpar, :mu; lb=lbp)
    return ScalarDpar(s.dpar, first(_rtake(c, 1)), lbp, ub, s.kind, p)
end

function _rslot(c::_RCtx, s::RLin)
    dp = s.dpar
    X, cn = _rdesign(c, s.intercept, s.terms)
    coefs = String[brms_rename(n) for n in cn]
    means = s.intercept ? vec(mean(X; dims=1)) : zeros(size(X, 2))
    Xc = X .- means'
    offset = zeros(c.N)
    for a in s.offsets
        offset .+= c.num[a]
    end
    int_idx = 0
    int_prior = flat()
    if s.intercept
        int_prior = _rresolve(c, :Intercept, dp)
        int_idx = first(_rtake(c, 1))
    end
    b_idx = _rtake(c, size(X, 2))
    b_priors = ScalarPrior[]; b_lb = Float64[]; b_ub = Float64[]
    for cf in coefs
        p, lb, ub = _rresolve_b(c, dp, cf)
        push!(b_priors, p); push!(b_lb, lb); push!(b_ub, ub)
    end
    groups = GroupBlock{Float64}[]
    for g in s.groups
        lv, idx = _rgroup(c, g)
        Zx, zn = _rdesign(c, g.intercept, g.terms)
        Z = g.intercept ? hcat(ones(c.N), Zx) : Zx
        gcoefs = vcat(g.intercept ? ["Intercept"] : String[], String[brms_rename(n) for n in zn])
        q = size(Z, 2)
        q > 0 || throw(ArgumentError("group term for $(g.label) has no effects"))
        sd_p = ScalarPrior[_rresolve(c, :sd, dp; coef=cf, group=g.label, lb=0.0) for cf in gcoefs]
        sd_r = _rtake(c, q)
        cor = g.cor && q > 1
        eta = cor ? _rlkj_eta(c, dp, g.label) : 1.0
        corr_r = cor ? _rtake(c, q * (q - 1) ÷ 2) : (1:0)
        z_r = _rtake(c, q * length(lv))
        push!(groups, GroupBlock{Float64}(g.label, idx, lv, Z, gcoefs, sd_r, corr_r, z_r, sd_p, eta))
    end
    return LinPred{Float64}(dp, int_idx, Xc, means, coefs, b_idx, groups, offset, int_prior, b_priors,
                            any(!isflat, b_priors); b_lb=b_lb, b_ub=b_ub)
end

"""
    assemble(recipe, atomvals) -> (GLMDensity, rows)

The model's density for new data, given every atom of the recipe evaluated on
it. `rows` are the complete cases used. Trimmable: this is what a model
binary runs.
"""
function assemble(rec::ModelRecipe, av::AtomVals)
    N0 = length(av.miss[rec.resp])
    keep = trues(N0)
    for m in av.miss
        length(m) == N0 || throw(ArgumentError("atom values of different lengths"))
        keep .&= .!m
    end
    rows = findall(keep)
    N = length(rows)
    N > 0 || throw(ArgumentError("no complete observations"))
    num = [v[rows] for v in av.num]
    str = [v[rows] for v in av.str]
    # factor levels: sorted unique (DataAPI.levels, R's factor() default)
    lev = [rec.cat[k] ? sort(unique(str[k])) : String[] for k in eachindex(rec.cat)]
    y = copy(num[rec.resp])
    trials = rec.trials > 0 ? Int[Int(v) for v in num[rec.trials]] :
             rec.ntrials > 0 ? fill(rec.ntrials, N) : Int[]
    fam = _refamily(rec.family, y)
    _check_response(fam, y, trials)
    weights = rec.weights > 0 ? copy(num[rec.weights]) : Float64[]
    c = _RCtx(fam, rec.labels, rec.cat, num, str, lev, y, N, rec.priors, rec.lkj, Ref(1))
    preds = map(s -> _rslot(c, s), rec.slots)
    dim = c.next[] - 1
    lconst = 0.0
    xd = _xdata(fam, y, trials)
    for i in 1:N
        cc = _ll_const_obs(fam, y[i], isempty(trials) ? 0 : trials[i], isempty(xd) ? 0.0 : xd[i])
        lconst += isempty(weights) ? cc : weights[i] * cc
    end
    return GLMDensity(fam, preds, y, trials, weights, dim, lconst), rows
end

# ---------------------------------------------------------------- build time

function _recipe_unsupported(f::BrmsFormula, fam, info, user)
    bad = String[]
    (f.nl || haskey(f.extra, :nlf)) && push!(bad, "non-linear formulas")
    f.cens === nothing || push!(bad, "cens()")
    for k in keys(f.extra)
        push!(bad, "addition term $(k)()")
    end
    (typeof(info.family) == typeof(fam) && _recipe_family_ok(info.family)) ||
        push!(bad, "family $(family_name(fam))")
    for d in f.dpars
        r = d.rhs
        isempty(r.mo) || push!(bad, "mo()")
        hasproperty(r, :moint) && !isempty(getproperty(r, :moint)) && push!(bad, "mo() interactions")
        r.ar === nothing || push!(bad, "autocorrelation terms")
        isempty(r.sm) || push!(bad, "smooth terms s() / t2()")
        isempty(r.cs) || push!(bad, "cs()")
        isempty(r.me) || push!(bad, "me()")
        isempty(r.gp) || push!(bad, "gp()")
        isempty(r.mi) || push!(bad, "mi()")
        isempty(r.car) || push!(bad, "car()")
        for gt in r.groups
            isempty(gt.mm) || push!(bad, "mm()")
            gt.by === nothing || push!(bad, "gr(by = )")
            gt.id == "" || push!(bad, "group-term IDs (1 | p | g)")
            gt.student && push!(bad, "gr(dist = \"student\")")
            gt.cov === nothing || push!(bad, "gr(cov = )")
        end
    end
    for (a, ai) in info.infos
        ai.basis === nothing || push!(bad, "$(a.label) (poly / bs / ns bases)")
        ai.levels !== nothing && !(a.ex isa Symbol) && push!(bad, "$(a.label) (a factor must be a data column)")
    end
    f.response.ex isa Expr && f.response.ex.head == :call && f.response.ex.args[1] === :cbind &&
        push!(bad, "matrix responses cbind()")
    for r in user
        r.resp == "" || push!(bad, "per-response priors")
        if r.prior isa ScalarPrior
            r.prior.kind == PK_CONSTANT && push!(bad, "constant() priors")
        elseif !(r.prior isa LKJ)
            push!(bad, "$(nameof(typeof(r.prior))) priors")
        end
    end
    return unique(bad)
end

function _rterms(terms, id)
    return RTerm[RTerm(Int[id[a] for a in t]) for t in terms]
end

"""
    model_recipe(formula, data; family=gaussian(), prior=PriorSpec[]) -> (recipe, atoms)

Build the model once with the front end and extract its [`ModelRecipe`](@ref).
Throws an ArgumentError naming every feature a recipe does not support yet,
and verifies that re-assembling `data` from the recipe reproduces
`build_model` exactly. `atoms` (the front end's atoms, in recipe order) is what
code generation for a model binary needs.
"""
function model_recipe(formula, data; family=gaussian(), prior=PriorSpec[])
    f = bf(formula)
    fam = family isa Family ? family : brmsfamily(family)
    user = prior isa PriorSpec ? PriorSpec[prior] : collect(PriorSpec, prior)
    ld, info = build_model(f, data; family=fam, prior=user)
    bad = _recipe_unsupported(f, fam, info, user)
    isempty(bad) || throw(ArgumentError("a model binary does not support yet: " * join(bad, ", ")))

    atoms = _all_atoms(f)
    id = Dict{Atom,Int}(a => k for (k, a) in enumerate(atoms))
    cat = Bool[iscategorical(info.infos[a]) for a in atoms]
    fam = info.family
    slots = Any[]
    for dp in dpars(fam)
        k = findfirst(d -> d.dpar === dp, f.dpars)
        if k === nothing
            lk, lb = aux_bounds(fam, dp)
            push!(slots, RScalar(dp, lk, Float64(lb)))
        else
            r = f.dpars[k].rhs
            groups = RGroup[RGroup(group_label(gt), Int[id[a] for a in gt.group], gt.intercept,
                                   _rterms(gt.terms, id), gt.cor) for gt in r.groups]
            push!(slots, RLin(dp, r.intercept, _rterms(r.terms, id), Int[id[a] for a in r.offsets], groups))
        end
    end
    priors = RPrior[RPrior(r.class, r.coef, r.group, r.dpar, r.prior, r.lb, r.ub) for r in user if r.prior isa ScalarPrior]
    lkjs = RLKJ[RLKJ(r.coef, r.group, r.dpar, r.prior.eta) for r in user if r.prior isa LKJ]
    rec = ModelRecipe(fam, Tuple(slots), String[a.label for a in atoms], cat, id[f.response],
                      f.trials isa Atom ? id[f.trials] : 0, f.trials isa Int ? f.trials : 0,
                      f.weights isa Atom ? id[f.weights] : 0, priors, lkjs)
    # the self-check: the recipe must rebuild THIS model exactly
    ld2, _ = assemble(rec, frontend_atomvals(atoms, f, data))
    msg = _density_mismatch(ld, ld2)
    msg === nothing || throw(ArgumentError("internal: the recipe does not reproduce build_model ($msg); " *
                                           "please report this formula"))
    # the binary's own names / constrained values (binary.jl) must equal
    # variables() / constrain() on this model
    _binary_names(ld2) == variables(ld) ||
        throw(ArgumentError("internal: a binary would name this model's parameters differently; please report"))
    rng = Random.Xoshiro(11)
    for _ in 1:3
        θ = 0.5 .* randn(rng, ld.dim)
        isapprox(_binary_values!(Float64[], ld2, θ), constrain(ld, θ); rtol=1e-12, atol=1e-12) ||
            throw(ArgumentError("internal: a binary would report this model's draws differently; please report"))
    end
    return rec, atoms
end

"""
    frontend_atomvals(atoms, formula, data) -> AtomVals

Atom values computed by the front end's own interpreter (the reference that
a model binary's generated atom code is checked against).
"""
function frontend_atomvals(atoms::Vector{Atom}, f::BrmsFormula, data)
    cols = Tables.columntable(data)
    num = Vector{Float64}[]; str = Vector{String}[]; miss = BitVector[]
    for a in atoms
        v = _eval_atom(a, cols, f.mod)
        m = BitVector(map(ismissing, v))
        push!(miss, m)
        push!(str, String[ismissing(x) ? "" : _level_name(x) for x in v])
        # numeric whenever the value is a number, factors included: a Bool
        # response is a factor to _atom_info but 0/1 to the likelihood
        push!(num, Float64[x isa Real ? Float64(x) : NaN for x in v])
    end
    return AtomVals(num, str, miss)
end

# nothing if the two densities are the same model, else what differs
function _density_mismatch(a::GLMDensity, b::GLMDensity; ntry::Int=3)
    typeof(a) == typeof(b) || return "type $(typeof(b)) vs $(typeof(a))"
    a.dim == b.dim || return "dimension $(b.dim) vs $(a.dim)"
    a.N == b.N || return "observations $(b.N) vs $(a.N)"
    variables(a) == variables(b) || return "parameter names"
    a.y == b.y || return "response"
    isapprox(a.lconst, b.lconst; rtol=1e-12, atol=1e-12) || return "likelihood constant"
    rng = Random.Xoshiro(20261104)
    wa, wb = fresh_workspace(a), fresh_workspace(b)
    ga, gb = zeros(a.dim), zeros(a.dim)
    for _ in 1:ntry
        θ = 0.5 .* randn(rng, a.dim)
        va = logdensity_and_gradient!(ga, wa, θ)
        vb = logdensity_and_gradient!(gb, wb, θ)
        isapprox(va, vb; rtol=1e-12) || return "log density $vb vs $va"
        isapprox(ga, gb; rtol=1e-10, atol=1e-10) || return "gradient"
    end
    return nothing
end
