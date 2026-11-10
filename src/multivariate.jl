# =============================================================================
# Multivariate models: several responses, each with its own formula and
# family, sharing one parameter vector — brms's `bf(y1 ~ x) + bf(y2 ~ z)` and
# `bf(mvbind(y1, y2) ~ x)`, with optional residual correlations
# (`set_rescor(TRUE)`, gaussian responses).
#
# Every response is an ordinary GLMDensity built with ONE shared parameter
# counter, so each sub-density indexes the joint θ directly and needs no
# offsets. Without residual correlations the posterior is the product of the
# responses' likelihoods, so value and gradient are plain sums. With them
# (brms's multi_normal_cholesky on each row's residuals), the sub-densities
# compute their predictors as usual, one joint loop over rows does the
# multivariate normal, and its d/d mu and d/d log sigma go back through each
# sub-density's own pullbacks:
#     s = (y - mu) ./ sigma,  v = L \ s,  a = L' \ v
#     ll_n = -|v|^2 / 2 - sum log sigma - sum log diag(L)      (constants dropped)
#     d/d mu_k = a_k / sigma_k,   d/d log sigma_k = a_k s_k - 1,   d/d L = a v'
# and d/d L feeds the same closed-form Cholesky-correlation reverse pass
# (`_corr_value_grad!`) the correlated group effects use, LKJ prior included.
# =============================================================================

"""
    MVFormula

Several [`BrmsFormula`](@ref)s fitted jointly. Build it with `+`
(`@bf(y1 ~ x) + @bf(y2 ~ z)`), from `@bf(mvbind(y1, y2) ~ x)`, and add
`set_rescor(true/false)` to choose residual correlations (default: on when
every response is gaussian, as brms).
"""
struct MVFormula
    forms::Vector{BrmsFormula}
    rescor::Union{Nothing,Bool}
end

struct _Rescor
    flag::Bool
end
"""
    set_rescor(flag=true)

Residual correlations between the responses of a multivariate model (gaussian
responses), as brms's `set_rescor`; add it to the formula with `+`.
"""
set_rescor(flag::Bool=true) = _Rescor(flag)

# mvbind(y1, y2) ~ rhs  ->  one formula per response
function _mv_split(f::BrmsFormula)
    ex = f.response.ex
    (ex isa Expr && ex.head == :call && ex.args[1] === :mvbind) || return [f]
    return [BrmsFormula((n === :response ? _atom(a) : getfield(f, n) for n in fieldnames(BrmsFormula))...)
            for a in ex.args[2:end]]
end
MVFormula(fs::Vector{BrmsFormula}) = MVFormula(reduce(vcat, _mv_split.(fs)), nothing)

Base.:+(a::BrmsFormula, b::BrmsFormula) = MVFormula([a, b])
Base.:+(a::MVFormula, b::BrmsFormula) = MVFormula(vcat(a.forms, _mv_split(b)), a.rescor)
Base.:+(a::MVFormula, r::_Rescor) = MVFormula(a.forms, r.flag)
Base.:+(a::BrmsFormula, r::_Rescor) = MVFormula(_mv_split(a), r.flag)

_is_mv(f) = f isa MVFormula || (f isa BrmsFormula && length(_mv_split(f)) > 1)
_mvformula(f::MVFormula) = f
_mvformula(f::BrmsFormula) = MVFormula(_mv_split(f), nothing)

# brms response names: the variable with non-alphanumerics removed
_resp_name(f::BrmsFormula) = replace(brms_rename(f.response.label), "_" => "", "." => "")

# --- the density -----------------------------------------------------------------

struct MVDensity{S<:Tuple,T}
    subs::S                        # one GLMDensity per response (shared θ indexing)
    resp::Vector{String}
    rescor::Bool
    ridx::UnitRange{Int}           # residual-correlation parameters (Stan's CPC)
    lkj::Float64
    dim::Int
    N::Int
    kmu::Vector{Int}               # position of each response's mu / sigma in its preds
    ksig::Vector{Int}
    # per-chain buffers
    gbuf::Vector{Vector{T}}        # one gradient per response (no-rescor sum)
    Y::Vector{Vector{T}}
    mus::Vector{Vector{T}}         # views into the sub-densities' workspaces
    wmus::Vector{Vector{T}}
    lsig::Vector{Vector{T}}        # log sigma: per observation, or a 1-vector
    wsig::Vector{Vector{T}}
    sigvec::Vector{Bool}
    sigidx::Vector{Int}            # θ index of a scalar sigma (0 if it has a formula)
    L::Matrix{T}
    A::Matrix{T}
    gc::Vector{T}
    s::Vector{T}
    v::Vector{T}
    a::Vector{T}
    # student rescor (brms multi_student_t): one shared nu; each response's own
    # nu slot is pinned while sampling and given the joint draws afterwards,
    # so the per-response views predict with the right marginal t
    nuidx::Int                     # θ index of log(nu - 1), 0 for gaussian
    nu_prior::ScalarPrior
    nu_alias::Vector{Int}
    # constant() priors of the responses, applied here on the rescor path
    pinned::Vector{Int}
    pinvals::Vector{T}
    obuf::Vector{T}
    cross::Bool                    # a group-term ID spans responses: phased gradient
    marg::Vector{MargMi}           # missing predictors integrated out (mimarg.jl): phased gradient
end

# share the GroupWS of every member of a group-term ID, recording which
# responses (by position) hold members. A recursion over the tuple of
# sub-densities and `map` over each one's (preds, ws) tuples: both unroll, so
# every call is static (a `for` over the heterogeneous tuples is not, --trim)
# θ index of the k-th predictor if it is a ScalarDpar, else 0 (static over the tuple)
_scalar_idx(ps::Tuple, k::Int) = k == 1 ? (ps[1] isa ScalarDpar ? ps[1].idx : 0) : _scalar_idx(Base.tail(ps), k - 1)
_scalar_idx(::Tuple{}, k::Int) = 0
_share_groups!(shared, members, ::Tuple{}, s) = nothing
function _share_groups!(shared, members, subs::Tuple, s)
    sub = subs[1]
    map((p, pw) -> _share_pred!(shared, members, p, pw, s), sub.preds, sub.ws)
    return _share_groups!(shared, members, Base.tail(subs), s + 1)
end
_share_pred!(shared, members, p, pw, s) = nothing
function _share_pred!(shared, members, p::LinPred, pw, s)
    for (j, gb) in enumerate(p.groups)
        gb.sid > 0 || continue
        haskey(shared, gb.sid) || (shared[gb.sid] = pw.groups[j])
        push!(get!(members, gb.sid, Set{Int}()), s)
        pw.groups[j] = shared[gb.sid]
    end
    return nothing
end

function MVDensity(subs::Tuple, resp, rescor, ridx, lkj, dim; nuidx=0, nu_prior=flat(), nu_alias=Int[],
                   marg=MargMi[])
    T = eltype(first(subs).y)
    # members of a group-term ID spanning responses work in ONE GroupWS
    # (a concretely typed Dict: Any here was an unresolvable convert under --trim)
    shared = Dict{Int,GroupWS{T}}(); members = Dict{Int,Set{Int}}()
    _share_groups!(shared, members, subs, 1)
    cross = any(m -> length(m) > 1, values(members))
    K = length(subs)
    N = first(subs).N
    kmu = Int[]; ksig = Int[]; mus = Vector{T}[]; wmus = Vector{T}[]
    lsig = Vector{T}[]; wsig = Vector{T}[]; sigvec = Bool[]; sigidx = Int[]
    if rescor
        # (a map over the tuple of sub-densities: unrolled, so typed, as for _share_groups!)
        map(subs) do sub
            km = findfirst(p -> p isa LinPred && p.dpar === :mu, sub.preds)::Int
            ks = findfirst(p -> p.dpar === :sigma, sub.preds)::Int
            push!(kmu, km); push!(ksig, ks)
            push!(mus, sub.ws[km].eta); push!(wmus, sub.ws[km].w)
            sidx = _scalar_idx(sub.preds, ks)
            if sidx == 0
                push!(lsig, sub.ws[ks].eta); push!(wsig, sub.ws[ks].w); push!(sigvec, true); push!(sigidx, 0)
            else
                push!(lsig, zeros(T, 1)); push!(wsig, sub.ws[ks].w); push!(sigvec, false); push!(sigidx, sidx)
            end
            nothing
        end
    end
    pinned = reduce(vcat, map(sub -> sub.pinned, subs); init=Int[])
    pinvals = reduce(vcat, map(sub -> sub.pinvals, subs); init=T[])
    return MVDensity(subs, collect(String, resp), rescor, ridx, Float64(lkj), dim, N, kmu, ksig,
                     [zeros(T, dim) for _ in subs], [copy(sub.y) for sub in subs], mus, wmus, lsig, wsig,
                     sigvec, sigidx, zeros(T, K, K), zeros(T, K, K), zeros(T, length(ridx)),
                     zeros(T, K), zeros(T, K), zeros(T, K), nuidx, nu_prior, collect(Int, nu_alias),
                     pinned, pinvals, zeros(T, dim), cross, collect(MargMi, marg))
end

function _order_draws!(mv::MVDensity, D::AbstractArray)
    foreach(s -> _order_draws!(s, D), mv.subs)
    if mv.nuidx > 0                       # the responses' nu = the shared nu
        for k in mv.nu_alias
            D[k, :, :] .= D[mv.nuidx, :, :]
        end
    end
    return D
end
fresh_workspace(mv::MVDensity) =
    MVDensity(map(fresh_workspace, mv.subs), mv.resp, mv.rescor, mv.ridx, mv.lkj, mv.dim;
              nuidx=mv.nuidx, nu_prior=mv.nu_prior, nu_alias=mv.nu_alias, marg=mv.marg)

LogDensityProblems.capabilities(::Type{<:MVDensity}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(mv::MVDensity) = mv.dim

# type-stable walks over the (heterogeneous) tuple of sub-densities
_mv_sum!(g, ::Tuple{}, bufs, θ, k) = 0
function _mv_sum!(g, subs::Tuple, bufs, θ, k)
    v = logdensity_and_gradient!(bufs[k], subs[1], θ)
    g .+= bufs[k]
    return v + _mv_sum!(g, Base.tail(subs), bufs, θ, k + 1)
end
_mv_etas!(::Tuple{}, θ) = nothing
function _mv_etas!(subs::Tuple, θ)
    sub = subs[1]
    map((pw, p) -> _eta!(pw, p, θ), sub.ws, sub.preds)
    return _mv_etas!(Base.tail(subs), θ)
end
_mv_pullbacks!(g, ::Tuple{}, θ) = 0
_mv_pullbacks!(g, subs::Tuple, θ) =
    _pullbacks!(g, subs[1].preds, subs[1].ws, θ) + _mv_pullbacks!(g, Base.tail(subs), θ)

function logdensity_and_gradient!(g::AbstractVector{T}, mv::MVDensity{S,T}, θ::AbstractVector{T}) where {S,T}
    fill!(g, zero(T))
    mv.rescor || mv.cross || !isempty(mv.marg) || return _mv_sum!(g, mv.subs, mv.gbuf, θ, 1)
    isempty(mv.pinned) && return mv.rescor ? _mv_rescor!(g, mv, θ) : _mv_phased!(g, mv, θ)
    θm = copyto!(mv.obuf, θ)
    for (k, c) in zip(mv.pinned, mv.pinvals)
        θm[k] = c
    end
    v = mv.rescor ? _mv_rescor!(g, mv, θm) : _mv_phased!(g, mv, θm)
    for k in mv.pinned                       # decoupled standard-normal coordinates
        g[k] = -θ[k]
        v -= θ[k]^2 / 2
    end
    return v
end

# no residual correlations but a group-term ID across responses: every response's
# forward pass and observation loop first, then every pullback, so the shared
# block's first member (forward) and last member (reverse pass) see all members
_mv_fwd_obs!(::Tuple{}, θ) = 0
function _mv_fwd_obs!(subs::Tuple, θ)
    sub = subs[1]
    etas = map((pw, p) -> _eta!(pw, p, θ), sub.ws, sub.preds)
    ll = sub.prior_only ? (foreach(pw -> fill!(pw.w, zero(eltype(pw.w))), sub.ws); 0) : _obs_loop!(sub, etas)
    return ll + _mv_fwd_obs!(Base.tail(subs), θ)
end
function _mv_phased!(g::AbstractVector{T}, mv::MVDensity{S,T}, θ::AbstractVector{T}) where {S,T}
    ll = _mv_fwd_obs!(mv.subs, θ)
    isempty(mv.marg) || (ll += _mv_marg!(g, mv, θ))      # after every obs loop, before the pullbacks
    return ll + _mv_pullbacks!(g, mv.subs, θ)
end

function _mv_rescor!(g::AbstractVector{T}, mv::MVDensity{S,T}, θ::AbstractVector{T}) where {S,T}
    K, N = length(mv.subs), mv.N
    student = mv.nuidx > 0
    nu = student ? 1 + exp(θ[mv.nuidx]) : one(T)
    gnu = zero(T)
    _mv_etas!(mv.subs, θ)
    for k in 1:K
        if !mv.sigvec[k]
            mv.lsig[k][1] = θ[mv.sigidx[k]]
            mv.wsig[k][1] = zero(T)
        end
    end
    L, A, s, v, a = mv.L, mv.A, mv.s, mv.v, mv.a
    _cholcorr!(L, view(θ, mv.ridx))
    fill!(A, zero(T))
    ll = zero(T)
    @inbounds for n in 1:N
        for k in 1:K
            ls = mv.lsig[k][mv.sigvec[k] ? n : 1]
            s[k] = (mv.Y[k][n] - mv.mus[k][n]) * exp(-ls)
            ll -= ls
        end
        for i in 1:K                       # v = L \ s
            t = s[i]
            for j in 1:(i - 1)
                t -= L[i, j] * v[j]
            end
            v[i] = t / L[i, i]
        end
        for i in K:-1:1                    # a = L' \ v
            t = v[i]
            for j in (i + 1):K
                t -= L[j, i] * a[j]
            end
            a[i] = t / L[i, i]
        end
        Q = zero(T)
        for k in 1:K
            Q += v[k] * v[k]
        end
        # multivariate t: ll = -(nu + K)/2 log(1 + Q/nu) (+ nu-only terms below);
        # d ll / d Q = -c/2 with c = (nu + K)/(nu + Q), so every adjoint of the
        # gaussian case (c = 1, ll = -Q/2) is scaled by c
        if student
            c = (nu + K) / (nu + Q)
            ll -= (nu + K) / 2 * log1p(Q / nu)
            gnu += -log1p(Q / nu) / 2 + (nu + K) * Q / (2 * nu * (nu + Q))
        else
            c = one(T)
            ll -= Q / 2
        end
        for k in 1:K
            mv.wmus[k][n] = c * a[k] * exp(-mv.lsig[k][mv.sigvec[k] ? n : 1])
            d = c * a[k] * s[k] - 1
            mv.sigvec[k] ? (mv.wsig[k][n] = d) : (mv.wsig[k][1] += d)
        end
        for i in 1:K, j in 1:i
            A[i, j] += c * a[i] * v[j]
        end
    end
    if student
        ll += N * (loggamma((nu + K) / 2) - loggamma(nu / 2) - K * log(nu) / 2)
        gnu += N * ((digamma((nu + K) / 2) - digamma(nu / 2)) / 2 - K / (2 * nu))
        u = θ[mv.nuidx]
        pv, pd = prior_kernel_grad(mv.nu_prior, nu)
        g[mv.nuidx] += (gnu + pd) * exp(u) + 1          # nu = 1 + exp(u), +1 log-Jacobian
        ll += pv + u
    end
    for i in 1:K
        ll -= N * log(L[i, i])
        A[i, i] -= N / L[i, i]
    end
    lp = _mv_pullbacks!(g, mv.subs, θ)
    lp += _corr_value_grad!(mv.gc, A, view(θ, mv.ridx), K, mv.lkj)
    @inbounds for (j, k) in enumerate(mv.ridx)
        g[k] += mv.gc[j]
    end
    return ll + lp
end

function LogDensityProblems.logdensity_and_gradient(mv::MVDensity{S,T}, θ::AbstractVector{T}) where {S,T}
    g = Vector{T}(undef, mv.dim)
    return logdensity_and_gradient!(g, mv, θ), g
end

# generic path: plain evaluation, any number type (the AD reference)
function LogDensityProblems.logdensity(mv::MVDensity, θraw::AbstractVector)
    mv.rescor || return sum(LogDensityProblems.logdensity(sub, θraw) for sub in mv.subs) +
                        (isempty(mv.marg) ? 0 : _mv_marg_generic(mv, _pinned_generic(mv, θraw)))
    K, N = length(mv.subs), mv.N
    θ = Vector{promote_type(eltype(θraw), Float64)}(θraw)
    lp = zero(eltype(θ))
    for (k, c) in zip(mv.pinned, mv.pinvals)
        θ[k] = c
        lp -= θraw[k]^2 / 2
    end
    student = mv.nuidx > 0
    nu = student ? 1 + exp(θ[mv.nuidx]) : one(eltype(θ))
    if student
        lp += prior_kernel(mv.nu_prior, nu) + θ[mv.nuidx] +
              N * (loggamma((nu + K) / 2) - loggamma(nu / 2) - K * log(nu) / 2)
    end
    for sub in mv.subs, p in sub.preds
        lp += _prior_generic(p, θ)
    end
    mu = [_eta_generic(sub.preds[mv.kmu[k]], θ, N) for (k, sub) in enumerate(mv.subs)]
    ls = [_eta_generic(sub.preds[mv.ksig[k]], θ, N) for (k, sub) in enumerate(mv.subs)]
    L, lj = _cholcorr(θ[mv.ridx], K)
    lp += lj + lkj_logpdf_chol(L, mv.lkj)
    for n in 1:N
        sig = [exp(_at(ls[k], n)) for k in 1:K]
        s = [(mv.subs[k].y[n] - _at(mu[k], n)) / sig[k] for k in 1:K]
        v = LowerTriangular(L) \ s
        Q = sum(abs2, v)
        lp += (student ? -(nu + K) / 2 * log1p(Q / nu) : -Q / 2) - sum(log, sig) - sum(log(L[i, i]) for i in 1:K)
    end
    return lp
end

# --- names and constrained values --------------------------------------------------

function _rescor_names(resp)
    out = String[]
    for k in 2:length(resp), j in 1:(k - 1)
        push!(out, "rescor__$(resp[j])__$(resp[k])")
    end
    return out
end

# a student-rescor model reports the shared `nu`, not the responses' own
_mv_keep(mv::MVDensity, sub, r) = mv.nuidx > 0 ? variables(sub; resp=r) .!= "nu_$(r)" : trues(length(variables(sub)))

function variables(mv::MVDensity)
    out = String[]
    for (sub, r) in zip(mv.subs, mv.resp)
        append!(out, variables(sub; resp=r)[_mv_keep(mv, sub, r)])
    end
    mv.rescor && append!(out, _rescor_names(mv.resp))
    mv.nuidx > 0 && push!(out, "nu")
    return out
end

function constrain(mv::MVDensity, θ::AbstractVector)
    out = Float64[]
    for (sub, r) in zip(mv.subs, mv.resp)
        append!(out, constrain(sub, θ)[_mv_keep(mv, sub, r)])
    end
    if mv.rescor
        L = first(_cholcorr(θ[mv.ridx], length(mv.subs)))
        R = L * L'
        for k in 2:size(R, 1), j in 1:(k - 1)
            push!(out, R[j, k])
        end
    end
    mv.nuidx > 0 && push!(out, 1 + exp(θ[mv.nuidx]))
    return out
end

# --- building ------------------------------------------------------------------------

"""
    build_mv_model(formula, data; family=gaussian(), prior=PriorSpec[], T=Float64)
        -> (MVDensity, Vector{ModelInfo})

The multivariate counterpart of [`build_model`](@ref). `family` is one family
for every response or a vector with one per response; priors apply to every
response unless given `resp = "name"`.
"""
function build_mv_model(formula, data; family=gaussian(), prior=PriorSpec[], T::Type{<:AbstractFloat}=Float64,
                        sample_prior=:no, mi::Symbol=:auto)
    mvf = _mvformula(formula)
    K = length(mvf.forms)
    K >= 2 || throw(ArgumentError("a multivariate model needs at least two responses"))
    fams = family isa Union{AbstractVector,Tuple} ? collect(family) : fill(family, K)
    length(fams) == K || throw(ArgumentError("give one family, or one per response ($K)"))
    fams = [f isa Family ? f : brmsfamily(f) for f in fams]
    resp = _resp_name.(mvf.forms)
    allunique(resp) || throw(ArgumentError("responses must have distinct names"))
    user = prior isa PriorSpec ? [prior] : collect(PriorSpec, prior)
    for r in user
        r.resp == "" || r.resp in resp || r.class === :rescor ||
            throw(ArgumentError("prior for unknown response `$(r.resp)` (responses: $(join(resp, ", ")))"))
    end

    # rows complete in EVERY response's variables, as brms does by default
    cols = Tables.columntable(data)
    keep = trues(_nrows(cols))
    # variables imputed by mi() may be missing
    mivars = Set{String}(f.response.label for f in mvf.forms if get(f.extra, :mi, false))
    for f in mvf.forms, a in _all_atoms(f)
        a.label in mivars && continue
        keep .&= _complete(_eval_atom(a, cols, f.mod))
    end
    if !all(keep)
        @warn "Rows containing missing values were excluded from the model." n_excluded = count(!, keep)
    end
    sub = map(c -> c[keep], cols)

    allgauss = all(f -> f isa Gaussian{IdentityLink}, fams)
    allstud = all(f -> f isa Student{IdentityLink}, fams)
    rescor = mvf.rescor === nothing ? (allgauss || allstud) : mvf.rescor
    # y | subset(s): that response's rows only (brms); everything about it --
    # centring, default priors, group levels -- comes from those rows
    hassub = [haskey(f.extra, :subset) for f in mvf.forms]
    if any(hassub)
        mvf.rescor === true && throw(ArgumentError("subset() cannot be combined with residual correlations (brms)"))
        # (mi() across responses with different rows: brms's mi(x, idx) + index(), linked below)
    end
    subrows(k) = hassub[k] ? findall(v -> v isa Bool ? v : !iszero(v),
                                     _eval_atom(mvf.forms[k].extra[:subset], sub, mvf.forms[k].mod)) : nothing
    rescor = rescor && !any(hassub)
    if rescor
        allgauss || allstud ||
            throw(ArgumentError("residual correlations (rescor) need all-gaussian or all-student (identity) responses"))
        for f in mvf.forms
            f.weights === nothing && f.cens === nothing && isempty(f.extra) && !f.nl ||
                throw(ArgumentError("rescor models do not support weights, cens, trunc, se or non-linear formulas yet"))
        end
    end

    next = Ref(1)
    reg = Dict{Tuple{String,String},Vector{Any}}()       # group-term IDs, across responses
    # missing values: integrated out where exact / quadrature-accurate (mimarg.jl)
    mimodes, mimarg = _mi_plan(mvf.forms, fams, resp, mi)
    built = map(1:K) do k
        uk = PriorSpec[r for r in user if r.resp == "" || r.resp == resp[k]]
        rk = subrows(k)
        dk = rk === nothing ? sub : map(c -> c[rk], sub)
        ld, info = build_model(mvf.forms[k], dk; family=fams[k], prior=uk, T=T, sample_prior=sample_prior, next=next,
                               shared_ids=reg, resp=resp[k], mi_mode=mimodes[k], mi_marg=mimarg[k])
        for (j, r) in enumerate(info.priors)
            info.priors[j] = PriorSpec(r.prior, r.class, r.coef, r.group, r.dpar, r.source, r.lb, r.ub, resp[k])
        end
        (ld, info)
    end
    take(n) = (r = next[]:(next[] + n - 1); next[] += n; r)
    _finish_shared!(nothing, reg, nothing, nothing, nothing, nothing, T, take)
    # mi(x): point each term at the imputed values of x's own response
    if !isempty(mivars)
        rescor && throw(ArgumentError("mi() models need set_rescor(false), as in brms"))
        any(mem -> length(unique(m.resp for m in mem)) > 1, values(reg)) &&
            throw(ArgumentError("mi() together with group-term IDs across responses is not supported yet"))
        for (ld, _) in built, p in ld.preds
            p isa LinPred || continue
            for m in p.mi
                k = findfirst(==(m.var), resp)
                k === nothing && throw(ArgumentError("mi($(m.var)): `$(m.var)` needs its own formula `$(m.var) | mi() ~ ...`"))
                src = built[k][1]
                if !isempty(m.idxv)
                    # mi(x, idx): row n uses x's row whose index() value is idx[n] (brms idxl)
                    fk = mvf.forms[k]
                    haskey(fk.extra, :index) ||
                        throw(ArgumentError("mi($(m.var), idx): `$(m.var)`'s formula needs `| index(...)` (brms)"))
                    rk = subrows(k)
                    xidx = _eval_atom(fk.extra[:index], rk === nothing ? sub : map(c -> c[rk], sub), fk.mod)
                    allunique(xidx) || throw(ArgumentError("index(): values must be unique"))
                    loc = Dict(v => i for (i, v) in enumerate(xidx))
                    pos = map(m.idxv) do v
                        haskey(loc, v) || throw(ArgumentError("mi($(m.var), idx): idx value `$v` is not an index() value of `$(m.var)`"))
                        loc[v]
                    end
                    kpos = Dict(n => j for (j, n) in enumerate(src.Jmi))
                    for (n, p) in enumerate(pos)
                        m.x[n] = haskey(kpos, p) ? oftype(m.x[n], NaN) : src.y[p]
                    end
                    empty!(m.Jmi); append!(m.Jmi, [n for (n, p) in enumerate(pos) if haskey(kpos, p)])
                    m.kmap[] = [kpos[pos[n]] for n in m.Jmi]
                else
                    src.Jmi == m.Jmi || throw(ArgumentError("mi($(m.var)): the missing rows do not match " *
                                                            "(responses with different rows need mi(x, idx) and index(), as brms)"))
                end
                m.ymi[] = src.ymi
                m.bnd[] = _mi_bounds(src.family)
            end
        end
    end
    cross = any(mem -> length(unique(m.resp for m in mem)) > 1, values(reg))
    cross && any(hassub) &&
        throw(ArgumentError("group-term IDs across responses together with subset() are not supported yet"))
    cross && any(!isempty(ld.ordered) for (ld, _) in built) &&
        throw(ArgumentError("group-term IDs across responses with ordered (mixture) intercepts are not supported"))
    lkj = 1.0
    ridx = 1:0
    priors_rescor = PriorSpec[]
    if rescor
        row = findfirst(r -> r.class === :rescor, user)
        lkj = row === nothing ? 1.0 : user[row].prior.eta
        push!(priors_rescor, PriorSpec(LKJ(lkj), :rescor, "", "", "", row === nothing ? :default : :user))
        ridx = next[]:(next[] + K * (K - 1) ÷ 2 - 1)
        next[] += length(ridx)
    end
    nuidx = 0; nu_prior = flat(); alias = Int[]
    if rescor && allstud
        nuidx = next[]; next[] += 1
        row = findfirst(r -> r.class === :nu && r.resp == "", user)
        p0 = row === nothing ? _default_prior(fams[1], :nu, :mu, Float64.(built[1][1].y)) : user[row].prior
        push!(priors_rescor, PriorSpec(p0, :nu, "", "", "", row === nothing ? :default : :user))
        nu_prior = with_bound(p0, 1.0)
        for (ld, _) in built
            k = findfirst(p -> p isa ScalarDpar && p.dpar === :nu, ld.preds)
            k === nothing && throw(ArgumentError("student rescor: the responses' nu cannot have a formula"))
            push!(alias, ld.preds[k].idx)
        end
    end
    dim = next[] - 1
    # integrated-out missing values: generated entries past the joint θ
    ymis = [ld.ymi for (ld, _) in built]
    gnext = dim
    for (k, (ld, _)) in enumerate(built)
        (isempty(ld.Jmi) || mimodes[k] !== :drop) && continue
        ymis[k] = (gnext + 1):(gnext + length(ld.Jmi)); gnext += length(ld.Jmi)
    end
    for (ld, _) in built, p in ld.preds
        p isa LinPred || continue
        foreach(m -> (m.ymi[] = ymis[findfirst(==(m.var), resp)]), p.mi)
    end
    links = _marg_links([ld for (ld, _) in built], mimarg, resp)
    # every response indexes the joint θ: give each its full dimension; a
    # student response's own nu is pinned (it takes the shared nu's draws)
    subs = Tuple(begin
                     pin = vcat(ld.pinned, filter(in(alias), [p.idx for p in ld.preds if p isa ScalarDpar]))
                     pv = vcat(ld.pinvals, zeros(eltype(ld.pinvals), length(pin) - length(ld.pinned)))
                     GLMDensity(ld.family, ld.preds, ld.y, ld.trials, ld.weights, dim, ld.lconst;
                                prior_only=ld.prior_only, ordered=ld.ordered, pinned=pin, pinvals=pv,
                                Jmi=ld.Jmi, ymi=ymis[k])
                 end for (k, (ld, _)) in enumerate(built))
    infos = [info for (_, info) in built]
    isempty(priors_rescor) || append!(infos[1].priors, priors_rescor)
    return MVDensity(subs, resp, rescor, ridx, lkj, dim; nuidx, nu_prior, nu_alias=alias, marg=links), infos
end

# --- the fit ----------------------------------------------------------------------------

"""
    BrmsMVFit

A fitted multivariate model. `fit["y"]` is response `y`'s own [`BrmsFit`](@ref)
(sharing the joint draws), so every univariate tool — summaries, predictions,
LOO, plots — works per response. The joint chain uses brms's multivariate
names (`b_y_x`, `sigma_y`, `rescor__y__y2`).
"""
struct BrmsMVFit{D<:MVDensity}
    density::D
    resp::Vector{String}
    fits::Vector{BrmsFit}
    chain::FlexiChains.FlexiChain{String}
    draws::Array{Float64,3}
    sampler::String
    iter::Int
    warmup::Int
    thin::Int
    sampling_time::Vector{Float64}
    call::Dict{Symbol,Any}
end

function Base.getindex(fit::BrmsMVFit, r::AbstractString)
    k = findfirst(==(r), fit.resp)
    k === nothing && throw(KeyError("no response `$r` (responses: $(join(fit.resp, ", ")))"))
    return fit.fits[k]
end
nchains(fit::BrmsMVFit) = size(fit.draws, 3)
ndraws(fit::BrmsMVFit) = size(fit.draws, 2) * size(fit.draws, 3)
variables(fit::BrmsMVFit) = variables(fit.density)
draws(fit::BrmsMVFit, name::AbstractString) = Matrix{Float64}(fit.chain[FlexiChains.Parameter(String(name))])
Base.show(io::IO, fit::BrmsMVFit) = print(io, "BrmsMVFit(", join(fit.resp, ", "), fit.density.rescor ? "; rescor" : "", ")")

function _brm_mv(formula, data; family=gaussian(), prior=PriorSpec[], chains::Int=4, iter::Int=2000,
                 warmup::Int=iter ÷ 2, thin::Int=1, seed=nothing, sampler=:nuts, delta=0.8,
                 max_depth::Int=10, metric::Symbol=:diag, init=2.0, threads::Bool=true,
                 T::Type{<:AbstractFloat}=Float64, sample_prior=:no, mi::Symbol=:auto,
                 algorithm::Symbol=:sampling, ndraws::Int=1000)
    iter > warmup || throw(ArgumentError("iter must exceed warmup"))
    sampler === :nuts || throw(ArgumentError("multivariate models use the package's NUTS (sampler = :nuts)"))
    mv, infos = build_mv_model(formula, data; family=family, prior=prior, T=T, sample_prior=sample_prior, mi=mi)
    algorithm === :sampling || ((iter, warmup, thin) = (ndraws, 0, 1))
    draws_, results, name = algorithm === :sampling ?
        _sample_core(mv, T; chains, iter, warmup, thin, seed, sampler, delta, max_depth, metric, init, threads) :
        _approx_draws(mv, algorithm; seed, ndraws, init)
    draws_ = _mi_generate(mv, draws_, seed === nothing ? Random.default_rng() : Random.Xoshiro(seed + 104729))
    times = [r.time for r in results]
    chain = _make_chain(mv, draws_, results)
    views = BrmsFit[BrmsFit(infos[k], mv.subs[k], _make_chain(mv.subs[k], draws_, results), draws_, name,
                            iter, warmup, thin, times, Dict{Symbol,Any}(:multivariate => true))
                    for k in eachindex(mv.subs)]
    _warn_divergences(results)
    call = Dict{Symbol,Any}(:formula => formula, :data => data, :family => family, :prior => prior,
                            :chains => chains, :iter => iter, :warmup => warmup, :thin => thin, :seed => seed,
                            :metric => metric, :T => T)
    return BrmsMVFit(mv, mv.resp, views, chain, draws_, name, iter, warmup, thin, times, call)
end

# --- predictions -------------------------------------------------------------------------
#
# Per response these are the univariate functions on fit["y"]. Without `resp`
# the responses are stacked as brms does: (draws x observations x responses).
# With residual correlations, posterior_predict and log_lik are JOINT over a
# row's responses (correlated residuals), as brms.

_mv_stack(f, fit::BrmsMVFit, resp; kw...) =
    resp === nothing ? (_mv_same_rows(fit); cat((f(v; kw...) for v in fit.fits)...; dims=3)) : f(fit[string(resp)]; kw...)

# subset() gives the responses different rows: nothing can be stacked or summed by row
_mv_same_rows(fit::BrmsMVFit) = allequal(v.density.N for v in fit.fits) ||
    throw(ArgumentError("the responses use different rows (subset()); pass `resp = ` for one response"))

posterior_epred(fit::BrmsMVFit; resp=nothing, kw...) = _mv_stack(posterior_epred, fit, resp; kw...)
posterior_linpred(fit::BrmsMVFit; resp=nothing, kw...) = _mv_stack(posterior_linpred, fit, resp; kw...)

# mean, log sigma and the residual Cholesky factor of every draw (rescor models)
function _mv_joint_parts(fit::BrmsMVFit; newdata=nothing, kw...)
    mu = [posterior_epred(v; newdata, kw...) for v in fit.fits]
    ls = [posterior_linpred(v; newdata, dpar=:sigma, kw...) for v in fit.fits]
    D = reshape(fit.draws, size(fit.draws, 1), :)
    K = length(fit.fits)
    nus = fit.density.nuidx > 0 ? 1 .+ exp.(D[fit.density.nuidx, :]) : fill(Inf, size(D, 2))
    Ls = [first(_cholcorr(D[fit.density.ridx, s], K)) for s in axes(D, 2)]
    return mu, ls, Ls, nus
end

function posterior_predict(fit::BrmsMVFit; resp=nothing, newdata=nothing, rng=Random.default_rng(), kw...)
    fit.density.rescor || return _mv_stack(posterior_predict, fit, resp; newdata, rng, kw...)
    mu, ls, Ls, nus = _mv_joint_parts(fit; newdata, kw...)
    S, N = size(mu[1]); K = length(mu)
    Y = Array{Float64,3}(undef, S, N, K)
    for s in 1:S, n in 1:N
        e = Ls[s] * randn(rng, K)
        # multivariate t: a gaussian draw scaled by sqrt(nu / chi2_nu)
        isfinite(nus[s]) && (e .*= sqrt(nus[s] / rand(rng, Distributions.Chisq(nus[s]))))
        for k in 1:K
            Y[s, n, k] = mu[k][s, n] + exp(ls[k][s, n]) * e[k]
        end
    end
    resp === nothing && return Y
    return Y[:, :, findfirst(==(string(resp)), fit.resp)]
end

function log_lik(fit::BrmsMVFit; newdata=nothing, resp=nothing, kw...)
    if !isempty(fit.density.marg)
        newdata === nothing ||
            throw(ArgumentError("log_lik for new data with integrated-out missing predictors is not supported yet (fit with mi = :sample)"))
        LLs = _marg_patch_ll!([log_lik(v; kw...) for v in fit.fits], fit)
        resp === nothing || return LLs[findfirst(==(string(resp)), fit.resp)]
        _mv_same_rows(fit)
        return sum(LLs)
    end
    resp === nothing || return log_lik(fit[string(resp)]; newdata, kw...)
    # without residual correlations the responses are independent given the
    # parameters: a row's log-likelihood is the sum over its responses (brms)
    _mv_same_rows(fit)
    fit.density.rescor || return sum(log_lik(v; newdata, kw...) for v in fit.fits)
    mu, ls, Ls, nus = _mv_joint_parts(fit; newdata, kw...)
    S, N = size(mu[1]); K = length(mu)
    Yobs = if newdata === nothing
        [Float64.(sub.y) for sub in fit.density.subs]
    else
        cols = Tables.columntable(newdata)
        [Float64.(_eval_atom(v.info.formula.response, cols, v.info.formula.mod)) for v in fit.fits]
    end
    LL = Matrix{Float64}(undef, S, N)
    c = K * log(2π) / 2
    for s in 1:S
        L = LowerTriangular(Ls[s]); ldet = sum(log, diag(Ls[s]))
        for n in 1:N
            r = [(Yobs[k][n] - mu[k][s, n]) * exp(-ls[k][s, n]) for k in 1:K]
            v = L \ r
            Q = sum(abs2, v); nu = nus[s]
            kern = isfinite(nu) ? loggamma((nu + K) / 2) - loggamma(nu / 2) - K * log(nu * π) / 2 -
                                  (nu + K) / 2 * log1p(Q / nu) : -Q / 2 - c
            LL[s, n] = kern - sum(ls[k][s, n] for k in 1:K) - ldet
        end
    end
    return LL
end

# --- display -----------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", fit::BrmsMVFit)
    _print_header(io, fit, "MV(" * join((family_name(v.info.family) for v in fit.fits), ", ") * ")", "",
                  [v.info.formula.text for v in fit.fits], fit.density.N)
    rows = SummaryRow[]
    for (r, v) in zip(fit.resp, fit.fits)
        println(io)
        println(io, _sty(io, "━━ Response " * r * " ━━"; color=_PAL.head, bold=true))
        append!(rows, _print_sections(io, v; footer=false))
    end
    if fit.density.rescor
        names = _rescor_names(fit.resp)
        append!(rows, _print_section(io, fit, "Residual Correlations",
                                     [(nothing, names, [replace(n[9:end], "__" => ",") for n in names])]))
    end
    _print_footer(io, fit, rows)
    return nothing
end
