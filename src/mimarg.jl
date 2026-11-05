# =============================================================================
# Missing values integrated out instead of sampled (`mi = :auto`, the default;
# `mi = :sample` keeps brms's behaviour of one parameter per missing value).
#
# 1. A missing response that no mi() term uses (`y | mi()` in a univariate
#    model, or a response nobody imputes from): its density integrates to one,
#    so the row simply leaves the likelihood (weight 0). Exact.
# 2. A missing predictor mi(x) whose own model `x | mi() ~ ...` is gaussian
#    (identity), used linearly in the mu formula of ONE response y: at a row
#    where x is missing and y observed, x_n ~ N(m_n, s_n^2) from x's model and
#    eta_y = e0 + b x_n, so with several missing predictors in the row
#        eta_y = e0 + S + sqrt(V) u,  S = sum_j b_j m_j,  V = sum_j b_j^2 s_j^2,  u ~ N(0, 1)
#    - y gaussian (identity): y_n ~ N(e0 + S, sigma^2 + V)   (exact)
#    - any other family: Gauss-Hermite quadrature over u (40 nodes), with the
#      family's own per-observation gradient at each node:
#        ll = log sum_k w_k exp(l_k),  pi_k = w_k exp(l_k - ll)
#        d/d e0 = D1 = sum pi_k g_k,   d/d V = sum pi_k g_k u_k / (2 sqrt V)
#    The chain rule then reaches b_j (m_j D1 + 2 b_j s_j^2 dV), x's mu (b_j D1)
#    and x's log sigma (2 b_j^2 s_j^2 dV). x's own likelihood drops the row.
# Anything else (a student x model, x used by several responses or outside mu,
# chained mi(), a sampled y) keeps the sampled form.
#
# The imputed values are then drawn after sampling, per posterior draw, as
# generated quantities past θ (`Ymi_x[k]`, as brms names its parameters):
# from x's conditional given y (gaussian: exactly, by Matheron's update;
# otherwise by sampling-importance-resampling from x's model), or from the
# posterior predictive for a dropped response. Predictions read them there.
# =============================================================================

struct MargMi
    ky::Int                          # the response whose likelihood integrates
    rows::Vector{Int}                # its rows with missing integrated predictors
    terms::Vector{Vector{Int}}       # per row: positions in y's mu `.mi` that are missing
    kx::Vector{Int}                  # per mi term of y's mu: the x response (0: not integrated)
    wts::Vector{Float64}             # the rows' own weights
    gauss::Bool                      # y gaussian (identity): closed form
    nodes::Vector{Float64}           # Gauss-Hermite for N(0, 1)
    gw::Vector{Float64}
end

# probabilists' Gauss-Hermite: nodes / weights for E[f(u)], u ~ N(0, 1) (Golub-Welsch)
function _gh_normal(K::Int)
    e = eigen(SymTridiagonal(zeros(K), sqrt.(1.0:(K - 1))))
    return e.values, e.vectors[1, :] .^ 2
end

# families whose per-observation gradient is a tuple of numbers in eta order
_marg_family_ok(f) = !(f isa Union{_Ord,Disc,Categorical,_MultiCat,GenExtremeValue,TimeCov,Mixture,Cox})

# x's mean and log sigma at row n (analytic path: from the workspaces)
@inline _marg_mx(xs::GLMDensity, n) = xs.ws[1].eta[n]
@inline _marg_lsx(xs::GLMDensity, θ, n) = xs.preds[2] isa ScalarDpar ? θ[xs.preds[2].idx] : xs.ws[2].eta[n]

# the k-th sub-density of a (heterogeneous) tuple, unrolled so every call is
# static (a runtime index into the tuple would be an abstract type: untrimmable)
@inline _on_sub(f::F, subs::Tuple, k::Int) where {F} = k == 1 ? f(subs[1]) : _on_sub(f, Base.tail(subs), k - 1)
@inline _on_sub(f::F, ::Tuple{}, k::Int) where {F} = throw(ArgumentError("no sub-density $k"))

function _mv_marg!(g::AbstractVector{T}, mv, θ) where {T}
    ll = zero(T)
    for L in mv.marg
        ll += _on_sub(ys -> _marg_link!(g, L, ys, mv.subs, θ), mv.subs, L.ky)::T
    end
    return ll
end

function _marg_link!(g::AbstractVector{T}, L::MargMi, ys::GLMDensity, subs, θ) where {T}
    mis = ys.preds[1].mi
    ηy = map((pw, p) -> p isa ScalarDpar ? θ[p.idx] : pw.eta, ys.ws, ys.preds)
    ll = zero(T)
    naux = length(ηy) - 1
    gaux = zeros(T, naux)
    for (r, n) in enumerate(L.rows)
        wt = T(L.wts[r])
        S = zero(T); V = zero(T)
        for j in L.terms[r]
            b = θ[mis[j].bsp]
            mx, lsx = _on_sub(xs -> (T(_marg_mx(xs, n)), T(_marg_lsx(xs, θ, n))), subs, L.kx[j])::Tuple{T,T}
            S += b * mx; V += b^2 * exp(2lsx)
        end
        e0 = ηy[1][n]
        if L.gauss
            ls = _at(ηy[2], n); σ2 = exp(2ls); v = σ2 + V; res = ys.y[n] - e0 - S
            ll += wt * (-log(v) / 2 - res^2 / (2v))
            D1 = res / v; dV = (res^2 / v - 1) / (2v)
            _tc_addw!(ys.ws[2].w, ys.preds[2], n, wt * dV * 2σ2)
        else
            aux = map(e -> _at(e, n), Base.tail(ηy))
            sV = sqrt(max(V, floatmin(T)))
            xd = _xd(ys, n)
            # streaming log-sum-exp over the nodes, with the posterior-weighted
            # gradient sums rescaled whenever the running maximum moves
            m = T(-Inf); S0 = zero(T); S1 = zero(T); Su = zero(T)
            fill!(gaux, zero(T))
            for k in eachindex(L.nodes)
                v, gk = _ll_grad(ys.family, ys.y[n], xd, (e0 + S + sV * L.nodes[k], aux...))
                l = T(log(L.gw[k]) + v)
                if l > m
                    c = exp(m - l); S0 *= c; S1 *= c; Su *= c; gaux .*= c; m = l
                end
                e = exp(l - m); g1 = T(gk[1])
                S0 += e; S1 += e * g1; Su += e * g1 * L.nodes[k]
                for d in 1:naux
                    gaux[d] += e * T(gk[d + 1])
                end
            end
            ll += wt * (m + log(S0))
            D1 = S1 / S0
            dV = Su / S0 / (2sV)
            for d in 1:naux
                _marg_addaux!(ys, d + 1, n, wt * gaux[d] / S0)
            end
        end
        ys.ws[1].w[n] += wt * D1
        # (fresh single-assignment bindings for the closure: D1 / dV are set in
        # both branches above, and a captured multiply-assigned variable is boxed)
        let D1 = T(D1), dV = T(dV)
            for j in L.terms[r]
                b = θ[mis[j].bsp]
                mx, s2 = _on_sub(subs, L.kx[j]) do xs
                    mxj = T(_marg_mx(xs, n)); s2j = T(exp(2 * _marg_lsx(xs, θ, n)))
                    xs.ws[1].w[n] += wt * b * D1
                    _tc_addw!(xs.ws[2].w, xs.preds[2], n, wt * dV * 2 * b^2 * s2j)
                    (mxj, s2j)
                end::Tuple{T,T}
                g[mis[j].bsp] += wt * (mx * D1 + dV * 2 * b * s2)
            end
        end
    end
    return ll
end
# add to the d-th predictor's derivative at row n (statically, over the preds tuple)
_marg_addaux!(ys, d, n, v) = _on_pred(ys.preds, ys.ws, d, (p, pw) -> _tc_addw!(pw.w, p, n, v))
@inline _on_pred(ps::Tuple, ws::Tuple, d::Int, f::F) where {F} =
    d == 1 ? f(ps[1], ws[1]) : _on_pred(Base.tail(ps), Base.tail(ws), d - 1, f)
@inline _on_pred(::Tuple{}, ::Tuple{}, d::Int, f::F) where {F} = nothing

# generic path (any number type): each link's rows' marginal log-likelihoods
# (unweighted, without data constants) from plain etas
function _marg_rows_ll(mv, L::MargMi, θ)
    ys = mv.subs[L.ky]
    mis = ys.preds[1].mi
    ηy = map(p -> _eta_generic(p, θ, ys.N), ys.preds)
    ηx = Dict(k => map(p -> _eta_generic(p, θ, mv.subs[k].N), mv.subs[k].preds) for k in unique(filter(>(0), L.kx)))
    return map(enumerate(L.rows)) do (r, n)
        S = zero(eltype(θ)); V = zero(eltype(θ))
        for j in L.terms[r]
            b = θ[mis[j].bsp]; ex = ηx[L.kx[j]]
            S += b * _at(ex[1], n); V += b^2 * exp(2 * _at(ex[2], n))
        end
        e0 = _at(ηy[1], n)
        if L.gauss
            s2 = exp(2 * _at(ηy[2], n)) + V
            -log(s2) / 2 - (ys.y[n] - e0 - S)^2 / (2s2)
        else
            aux = map(e -> _at(e, n), Base.tail(ηy))
            sV = sqrt(V)
            logsumexp([log(L.gw[k]) + _ll(ys.family, ys.y[n], _xd(ys, n), (e0 + S + sV * L.nodes[k], aux...))
                       for k in eachindex(L.nodes)])
        end
    end
end
_mv_marg_generic(mv, θ) = sum(L -> sum(L.wts .* _marg_rows_ll(mv, L, θ); init=zero(eltype(θ))), mv.marg; init=zero(eltype(θ)))

# log_lik at the integrated rows: the MARGINAL density p(y_n | θ) (with the
# data constants and the row's weight), which is what PSIS-LOO needs here
function _marg_patch_ll!(LLs, fit)
    mv = fit.density
    D = reshape(fit.draws, size(fit.draws, 1), :)
    for L in mv.marg
        ys = mv.subs[L.ky]
        c = [_ll_const_obs(ys.family, ys.y[n], isempty(ys.trials) ? 0 : ys.trials[n], _xd(ys, n)) for n in L.rows]
        for s in axes(D, 2)
            v = _marg_rows_ll(mv, L, view(D, 1:mv.dim, s))
            LLs[L.ky][s, L.rows] .= L.wts .* (v .+ c)
        end
    end
    return LLs
end

# --- generated draws of the integrated-out values ---------------------------------

"""
    _mi_generate(dens, draws, rng) -> draws (extended)

Append, to every posterior draw, the integrated-out missing values (the
entries `ymi` past θ that `Ymi_<resp>[k]` names): a dropped response's from its
posterior predictive, an integrated predictor's from its conditional given the
response it enters.
"""
function _mi_generate(dens, draws::Array{Float64,3}, rng)
    subs = dens isa GLMDensity ? Any[dens] : collect(Any, dens.subs)
    gens = [k for (k, s) in enumerate(subs) if !isempty(s.Jmi) && !_mi_sampled(s)]
    isempty(gens) && return draws
    dim = size(draws, 1)
    hi = maximum(last(subs[k].ymi) for k in gens)
    out = vcat(draws, zeros(hi - dim, size(draws, 2), size(draws, 3)))
    links = dens isa GLMDensity ? MargMi[] : dens.marg
    pos = Dict(k => Dict(n => j for (j, n) in enumerate(subs[k].Jmi)) for k in gens)
    for c in axes(draws, 3), i in axes(draws, 2)
        θ = draws[:, i, c]
        etas = Dict(k => map(p -> _eta_generic(p, θ, subs[k].N), subs[k].preds) for k in gens)
        for k in gens
            s = subs[k]
            for (j, n) in enumerate(s.Jmi)
                eta = map(e -> _at(e, n), etas[k])
                out[s.ymi[j], i, c] = _rand(rng, s.family, eta, isempty(s.trials) ? 0 : s.trials[n])
            end
        end
        for L in links
            ys = subs[L.ky]; mis = ys.preds[1].mi
            ηy = map(p -> _eta_generic(p, θ, ys.N), ys.preds)
            for n in L.rows
                r = findfirst(==(n), L.rows); J = L.terms[r]
                b = [θ[mis[j].bsp] for j in J]
                mx = [_at(etas[L.kx[j]][1], n) for j in J]
                sx = [exp(_at(etas[L.kx[j]][2], n)) for j in J]
                e0 = _at(ηy[1], n)
                x = if L.gauss
                    # Matheron: x = x0 + D b (y - y0) / (sigma^2 + b'Db), x0, y0 a joint prior draw
                    σ = exp(_at(ηy[2], n))
                    x0 = mx .+ sx .* randn(rng, length(J))
                    y0 = e0 + dot(b, x0) + σ * randn(rng)
                    x0 .+ sx .^ 2 .* b .* (ys.y[n] - y0) ./ (σ^2 + sum(b .^ 2 .* sx .^ 2))
                else
                    aux = map(e -> _at(e, n), Base.tail(ηy))
                    M = 256
                    X = mx .+ sx .* randn(rng, length(J), M)
                    lw = [_ll(ys.family, ys.y[n], _xd(ys, n), (e0 + dot(b, view(X, :, m)), aux...)) for m in 1:M]
                    w = exp.(lw .- maximum(lw))
                    X[:, _wsample(rng, w)]
                end
                for (q, j) in enumerate(J)
                    k = L.kx[j]
                    out[subs[k].ymi[pos[k][n]], i, c] = x[q]
                end
            end
        end
    end
    return out
end
_wsample(rng, w) = (u = rand(rng) * sum(w); c = 0.0; for (m, v) in enumerate(w); c += v; c >= u && return m; end; length(w))

# --- deciding what integrates out (multivariate models) ----------------------------

_mi_vars(rhs) = [replace(brms_rename(a.label), "_" => "", "." => "") for a in rhs.mi]

# per response: :sample (brms), :drop (integrated: a response nobody imputes from,
# or an integrated predictor), and the mi() variables each response integrates
function _mi_plan(forms, fams, resp, mode::Symbol)
    K = length(forms)
    modes = fill(:sample, K); marg = [Set{String}() for _ in 1:K]
    mode in (:auto, :sample) || throw(ArgumentError("mi must be :auto or :sample"))
    mode === :sample && return modes, marg
    for k in 1:K
        get(forms[k].extra, :mi, false) || continue
        v = resp[k]
        users = [j for j in 1:K if j != k && any(d -> v in _mi_vars(d.rhs), forms[j].dpars)]
        if isempty(users)
            modes[k] = :drop
        elseif length(users) == 1
            j = users[1]
            ok = fams[k] isa Gaussian{IdentityLink} &&
                 all(d -> isempty(d.rhs.mi), forms[k].dpars) &&                      # no chained mi()
                 count(==(v), _mi_vars(forms[j].dpars[1].rhs)) == 1 &&               # once, in mu ...
                 !any(d -> v in _mi_vars(d.rhs), forms[j].dpars[2:end]) &&           # ... only
                 forms[j].dpars[1].dpar === :mu && !forms[j].nl && _marg_family_ok(fams[j])
            ok && (modes[k] = :drop; push!(marg[j], v))
        end
    end
    # a response whose own missing values are SAMPLED cannot integrate its predictors
    for j in 1:K
        (get(forms[j].extra, :mi, false) && modes[j] === :sample && !isempty(marg[j])) || continue
        for v in marg[j]
            modes[findfirst(==(v), resp)] = :sample
        end
        empty!(marg[j])
    end
    return modes, marg
end

# the integration links of a multivariate build: per response with integrated
# mi() terms, its rows (observed y, some integrated predictor missing), which
# terms are missing in each, and the weights those rows had (now 0 in the loop)
function _marg_links(lds, marg, resp)
    links = MargMi[]
    nodes, gw = _gh_normal(40)
    for (j, ys) in enumerate(lds)
        isempty(marg[j]) && continue
        mis = ys.preds[1].mi
        kx = [m.marg ? findfirst(==(m.var), resp) : 0 for m in mis]
        ydrop = Set(ys.Jmi)
        miss = [Set(m.Jmi) for m in mis]
        rows = sort(unique(n for (t, m) in enumerate(mis) if m.marg for n in m.Jmi if !(n in ydrop)))
        terms = [[t for (t, m) in enumerate(mis) if m.marg && n in miss[t]] for n in rows]
        if isempty(ys.weights)
            resize!(ys.weights, ys.N); fill!(ys.weights, one(eltype(ys.weights)))
        end
        wts = Float64.(ys.weights[rows])
        ys.weights[rows] .= 0
        push!(links, MargMi(j, rows, terms, kx, wts, ys.family isa Gaussian{IdentityLink}, nodes, gw))
    end
    return links
end

# θ with the constant() pins applied (generic path)
function _pinned_generic(mv, θraw)
    isempty(mv.pinned) && return θraw
    θ = Vector{promote_type(eltype(θraw), eltype(mv.pinvals))}(θraw)
    for (k, c) in zip(mv.pinned, mv.pinvals)
        θ[k] = c
    end
    return θ
end
