# =============================================================================
# Residual covariance structures (brms): ar/ma/arma(..., cov = TRUE), cosy(),
# fcor(M) and sar(M, type = "lag" | "error"), for gaussian and student
# responses (identity link).
#
# Time structures (ar/ma/arma cov, cosy): brms orders the data by (gr, time);
# each group's residuals are multivariate normal (multi_student_t) with
# covariance diag(s) R diag(s), R the leading nobs x nobs block of ONE
# correlation matrix -- the k-th observation of a group is time k, gaps in
# `time` are ignored (brms only uses the time values for the ordering):
#   ar1:    R_ij = ar^|i-j| / (1 - ar^2)
#   ma1:    R_ii = 1 + ma^2,  R_i,i+-1 = ma
#   arma11: R_ii = (1 + ma^2 + 2 ar ma) / (1 - ar^2),
#           R_ij = (1 + ar ma)(ar + ma) ar^(|i-j|-1) / (1 - ar^2)
#   cosy:   R_ii = 1,  R_ij = cosy
# fcor: one group, R = M (data, its Cholesky factor computed once).
# sar: one group, scalar sigma (brms normal_lagsar / normal_errorsar):
#   ll = -N log sigma + sum log(1 - rho eig(W)) - |r|^2 / (2 sigma^2),
#   r = (I - rho W) y - mu (lag) or (I - rho W)(y - mu) (error).
#
# Gradient (time structures, fcor), per group of size n, with z = (y - mu)/s,
# alpha = R^-1 z, q = z' alpha and c = (nu + n) / (nu + q) (c = 1 gaussian):
#     d/d mu_j = c alpha_j / s_j,     d/d log s_j = -1 + c z_j alpha_j
# The structure parameters (and student's nu) enter through log det R, q and
# the multi_student_t normaliser only, so they are carried as dual numbers
# through one Cholesky of the largest R: the leading block of a Cholesky
# factor is the factor of the leading block, so every group size is served
# by one factorisation and a cumulative sum of 2 log L_kk.
#
# The structure parameters are scalar distributional parameters of the
# wrapper family (as gev's xi): priors, bounds, names, constrain and the
# samplers need nothing new.
# =============================================================================


# the structure parameters, in θ order after the base family's
function _ac_dpars(t::TimeCov)
    t.kind === :arma_cov && return ((t.p > 0 ? (:ar,) : ())..., (t.q > 0 ? (:ma,) : ())...)
    t.kind === :cosy && return (:cosy,)
    t.kind === :unstr && return (:cortime,)
    t.kind === :lagsar && return (:lagsar,)
    t.kind === :errorsar && return (:errorsar,)
    return ()
end
const _AC_CLASSES = (:ar, :ma, :cosy, :lagsar, :errorsar)
_nbase(t::TimeCov) = length(dpars(t.fam))
_is_sar(t::TimeCov) = t.kind in (:lagsar, :errorsar)
_is_student(t::TimeCov) = t.fam isa Student

# delegation: everything but the likelihood is the base family's
link(t::TimeCov) = link(t.fam)
dpars(t::TimeCov) = (dpars(t.fam)..., _ac_dpars(t)...)
function aux_bounds(t::TimeCov, d::Symbol)
    d in (:ar, :ma) && return (:logit, -1.0)
    d === :cosy && return (:logit, 0.0)
    d in (:lagsar, :errorsar) && return (:logit, minimum(t.eig))
    return aux_bounds(t.fam, d)
end
function aux_ub(t::TimeCov, d::Symbol)
    d in (:ar, :ma, :cosy) && return 1.0
    d in (:lagsar, :errorsar) && return maximum(t.eig)
    return aux_ub(t.fam, d)
end
dpar_link_name(t::TimeCov, d::Symbol) = d in _AC_CLASSES ? "identity" : dpar_link_name(t.fam, d)
prior_link(t::TimeCov) = prior_link(t.fam)
mu_link(t::TimeCov) = mu_link(t.fam)
uses_trials(t::TimeCov) = false
family_name(t::TimeCov) = family_name(t.fam)
_check_response(t::TimeCov, y, tr) = _check_response(t.fam, y, tr)
_ll_const(t::TimeCov, y, tr) = _ll_const(t.fam, y, tr)
_mean(t::TimeCov, eta, tr) = _mean(t.fam, eta, tr)
_default_loc_scale(t::TimeCov, y) = _default_loc_scale(t.fam, y)
_default_prior(t::TimeCov, class::Symbol, dpar::Symbol, y) =
    class in _AC_CLASSES ? flat() : _default_prior(t.fam, class, dpar, y)
# the observations are not independent: nothing may use a per-observation density
_ll(t::TimeCov, y, x, eta) =
    throw(ArgumentError("$(t.kind) models have no per-observation likelihood; use log_lik(fit)"))

# brms names: ar[1], ma[1]; the others as they are
_scalar_label(d::Symbol) = d in (:ar, :ma) ? "$(d)[1]" : string(d)

# --- building ---------------------------------------------------------------------

function _timecov_family(fam, ac, vals, y, rows, N0)
    fam isa Union{Gaussian{IdentityLink},Student{IdentityLink}} ||
        throw(ArgumentError("autocorrelation structures with a covariance formulation are supported for " *
                            "gaussian and student (identity link) responses so far, not $(family_name(fam))"))
    ta, ga, p, q, kind, Mex = ac
    N = length(y)
    if kind in (:arma_cov, :cosy, :unstr)
        gv = ga === nothing ? fill(1, N) : vals[ga]
        tv = ta === nothing ? collect(1:N) : vals[ta]
        # brms order_data: order(gr, time)
        perm = sortperm(collect(1:N); by=i -> (gv[i], tv[i]))
        beg = [1]
        for k in 2:N
            a, b = perm[k - 1], perm[k]
            isequal(gv[a], gv[b]) && isequal(tv[a], tv[b]) &&
                throw(ArgumentError("time points within groups must be unique"))
            isequal(gv[a], gv[b]) || push!(beg, k)
        end
        push!(beg, N + 1)
        if kind === :unstr
            # one correlation over the time LEVELS (brms extract_levels: sorted); each
            # group uses the rows / columns of the levels it has
            lv = sort(unique(tv))
            tlev = [findfirst(isequal(tv[perm[k]]), lv) for k in 1:N]
            return TimeCov(fam, kind, 0, 0, perm, beg, length(lv), zeros(0, 0), Float64[], Float64[], 0.0,
                           tlev, [_level_name(l) for l in lv])
        end
        return TimeCov(fam, kind, p, q, perm, beg, maximum(diff(beg)), zeros(0, 0), Float64[], Float64[], 0.0)
    end
    M = Float64.(_lookup_data2(Mex))
    # brms drops the rows / columns of observations removed for missing values
    size(M, 1) == N0 != N && (M = M[rows, rows])
    size(M) == (N, N) || throw(ArgumentError("$(kind)(M): M must be $N x $N (one row per observation)"))
    if kind === :fcor
        issymmetric(M) || throw(ArgumentError("fcor(M): M must be symmetric"))
        C = cholesky(Symmetric(M); check=false)
        issuccess(C) || throw(ArgumentError("fcor(M): M must be positive definite"))
        return TimeCov(fam, :fcor, 0, 0, collect(1:N), [1, N + 1], N, Matrix(C.L), Float64[], Float64[], logdet(C))
    end
    ev = eigvals(M)
    all(v -> abs(imag(v)) < 1e-8, ev) ||
        throw(ArgumentError("sar(M): M has complex eigenvalues (brms needs real ones; use a row-standardised symmetric-structure M)"))
    eig = real.(ev)
    return TimeCov(fam, kind, 0, 0, collect(1:N), [1, N + 1], N, M, eig, kind === :lagsar ? M * y : Float64[], 0.0)
end

# --- the correlation matrix (generic in the number type) ----------------------------

function _tc_R(t::TimeCov, ρ, n)
    E = typeof(float(ρ[1]))
    R = Matrix{E}(undef, n, n)
    if t.kind === :cosy
        c = ρ[1]
        for j in 1:n, i in 1:n
            R[i, j] = i == j ? one(E) : c
        end
    elseif t.q == 0                                  # ar1
        a = ρ[1]; f = inv(1 - a^2)
        pw = _tc_powers(a, n)
        for j in 1:n, i in 1:n
            R[i, j] = f * pw[abs(i - j) + 1]
        end
    elseif t.p == 0                                  # ma1
        m = ρ[1]
        for j in 1:n, i in 1:n
            d = abs(i - j)
            R[i, j] = d == 0 ? 1 + m^2 : d == 1 ? m : zero(E)
        end
    else                                             # arma11
        a, m = ρ[1], ρ[2]; f = inv(1 - a^2); γ = (1 + a * m) * (a + m)
        pw = _tc_powers(a, n)
        for j in 1:n, i in 1:n
            d = abs(i - j)
            R[i, j] = d == 0 ? (1 + m^2 + 2a * m) * f : γ * pw[d] * f
        end
    end
    return R
end

# a^0 .. a^(n-1) by multiplication: a dual's a^0 via `^` has a NaN derivative at a = 0
function _tc_powers(a, n)
    pw = Vector{typeof(float(a))}(undef, max(n, 1))
    pw[1] = one(a)
    for k in 2:n
        pw[k] = pw[k - 1] * a
    end
    return pw
end

# lower Cholesky factor of R (nothing if not positive definite: a rejected proposal)
function _tc_chol(t::TimeCov, ρ)
    t.kind === :fcor && return t.M
    C = cholesky(Symmetric(_tc_R(t, ρ, t.nmax), :L); check=false)
    issuccess(C) || return nothing
    return Matrix(C.L)
end

# --- analytic path ------------------------------------------------------------------

_tc_nd(t::TimeCov) = length(_ac_dpars(t)) + _is_student(t)

function _obs_loop!(ld::GLMDensity{F,P,W,T}, etas) where {F<:TimeCov,P,W,T}
    foreach(pw -> fill!(pw.w, zero(T)), ld.ws)
    _is_sar(ld.family) && return _sar_loop!(ld, etas)
    ld.family.kind === :unstr && return _unstr_loop!(ld, etas)
    # a literal branch, so the dual width is a compile-time constant (trimmable)
    K = _tc_nd(ld.family)
    return K == 0 ? _tc_loop!(ld, etas, Val(0)) : K == 1 ? _tc_loop!(ld, etas, Val(1)) :
           K == 2 ? _tc_loop!(ld, etas, Val(2)) : _tc_loop!(ld, etas, Val(3))
end

# add an observation's derivative (per-observation predictor or scalar)
_tc_addw!(w, ::LinPred, i, v) = (@inbounds w[i] += v; nothing)
_tc_addw!(w, ::ScalarDpar, i, v) = (@inbounds w[1] += v; nothing)

function _tc_loop!(ld::GLMDensity{F,P,W,T}, etas, ::Val{K}) where {F,P,W,T,K}
    fam = ld.family
    nb = _nbase(fam); na = length(_ac_dpars(fam)); stud = _is_student(fam)
    D = ForwardDiff.Dual{Nothing,T,K}
    seed(v, k) = D(v, ForwardDiff.Partials(ntuple(j -> T(j == k), Val(K))))
    # structure parameters on their natural scale (value, d value / d u), as duals
    tr = [_dpar_transform(ld.preds[nb + k]::ScalarDpar, etas[nb + k]) for k in 1:na]
    ρd = D[seed(tr[k][1], k) for k in 1:na]          # a Vector: no runtime-length tuple
    nut = stud ? _dpar_transform(ld.preds[3], etas[3]) : (one(T), one(T), zero(T), zero(T))
    νd = stud ? seed(nut[1], K) : D(one(T))       # nu's partial is the last slot
    L = _tc_chol(fam, ρd)
    L === nothing && return T(NaN)
    total = _tc_groups!(ld, etas, L, νd, stud)
    for k in 1:na
        ld.ws[nb + k].w[1] = ForwardDiff.partials(total)[k] * tr[k][2]
    end
    stud && (ld.ws[3].w[1] = ForwardDiff.partials(total)[K] * nut[2])
    return ForwardDiff.value(total)
end

# the group sweep; L's element type is a dual (time structures) or Float64 (fcor)
function _tc_groups!(ld::GLMDensity{F,P,W,T}, etas, L::AbstractMatrix{E}, νd, stud) where {F,P,W,T,E}
    fam, y = ld.family, ld.y
    nmax = fam.nmax
    V = promote_type(E, typeof(νd))
    cld = Vector{E}(undef, nmax)                    # cld[n] = log det of the leading n x n block
    acc = zero(E)
    for k in 1:nmax
        acc += 2 * log(L[k, k])
        cld[k] = acc
    end
    z = Vector{T}(undef, nmax); s = Vector{T}(undef, nmax)
    w = Vector{E}(undef, nmax); α = Vector{T}(undef, nmax)
    ν = ForwardDiff.value(νd)
    wmu, wsg = ld.ws[1].w, ld.ws[2].w
    total = zero(V)
    @inbounds for g in 1:(length(fam.beg) - 1)
        b0 = fam.beg[g] - 1; n = fam.beg[g + 1] - fam.beg[g]
        sumls = zero(T)
        for k in 1:n
            i = fam.perm[b0 + k]
            ls = _at(etas[2], i)
            s[k] = exp(ls)
            z[k] = (y[i] - _at(etas[1], i)) / s[k]
            sumls += ls
        end
        # w = L^-1 z (duals), alpha = L^-T value(w) = R^-1 z (values)
        qd = zero(E)
        for k in 1:n
            a = convert(E, z[k])
            for j in 1:(k - 1)
                a -= L[k, j] * w[j]
            end
            w[k] = a / L[k, k]
            qd += w[k]^2
        end
        for k in n:-1:1
            a = ForwardDiff.value(w[k])
            for j in (k + 1):n
                a -= ForwardDiff.value(L[j, k]) * α[j]
            end
            α[k] = a / ForwardDiff.value(L[k, k])
        end
        lpg = if stud
            loggamma((νd + n) / 2) - loggamma(νd / 2) - n * log(νd) / 2 - cld[n] / 2 - (νd + n) / 2 * log1p(qd / νd)
        else
            -cld[n] / 2 - qd / 2
        end
        total += lpg - sumls
        c = stud ? (ν + n) / (ν + ForwardDiff.value(qd)) : one(T)
        for k in 1:n
            i = fam.perm[b0 + k]
            wmu[i] = c * α[k] / s[k]
            _tc_addw!(wsg, ld.preds[2], i, -1 + c * z[k] * α[k])
        end
    end
    return total
end

# student (brms student_t_lagsar / errorsar): with Q = |r|^2 / sigma^2,
#   ll = lgamma((nu + N)/2) - lgamma(nu/2) - N/2 log nu - N log sigma + log det
#        - (nu + N)/2 log1p(Q / nu)
# and the gaussian derivatives in r scaled by c = (nu + N) / (nu + Q).
function _sar_loop!(ld::GLMDensity{F,P,W,T}, etas) where {F,P,W,T}
    fam, y, N = ld.family, ld.y, ld.N
    nb = _nbase(fam); stud = _is_student(fam)
    ls = etas[2]
    ρ, dρu = _dpar_transform(ld.preds[nb + 1], etas[nb + 1])
    # brms bounds rho by W's extreme eigenvalues, which (for a W that is not
    # row-standardised) still allows 1 - rho eig <= 0: Stan rejects those
    all(λ -> ρ * λ < 1, fam.eig) || return T(NaN)
    is2 = exp(-2ls)
    r = Vector{T}(undef, N)
    lag = fam.kind === :lagsar
    if lag
        @inbounds for i in 1:N
            r[i] = y[i] - ρ * fam.Wy[i] - _at(etas[1], i)
        end
        side = fam.Wy
    else
        e = [y[i] - _at(etas[1], i) for i in 1:N]
        side = fam.M * e                          # W e
        r .= e .- ρ .* side
    end
    rr = sum(abs2, r)
    Q = rr * is2
    c = one(T)
    if stud
        ν, dνu = _dpar_transform(ld.preds[3], etas[3])
        c = (ν + N) / (ν + Q)
        ll = loggamma((ν + N) / 2) - loggamma(ν / 2) - N * log(ν) / 2 - N * ls - (ν + N) / 2 * log1p(Q / ν)
        dν = (digamma((ν + N) / 2) - digamma(ν / 2) - N / ν - log1p(Q / ν)) / 2 + (ν + N) * Q / (2ν * (ν + Q))
        ld.ws[3].w[1] = dν * dνu
    else
        ll = -N * ls - Q / 2
    end
    dρ = c * dot(r, side) * is2
    for λ in fam.eig
        ll += log1p(-ρ * λ)
        dρ -= λ / (1 - ρ * λ)
    end
    wmu = ld.ws[1].w
    if lag
        wmu .= c .* r .* is2
    else
        Wtr = fam.M' * r
        @inbounds for i in 1:N
            wmu[i] = c * (r[i] - ρ * Wtr[i]) * is2
        end
    end
    ld.ws[2].w[1] = -N + c * Q
    ld.ws[nb + 1].w[1] = dρ * dρu
    return ll
end

# --- unstr: an LKJ correlation over the time levels -------------------------------
# Per group (levels J): R_J = R[J, J], alpha = R_J^-1 z, q = z' alpha; as for the
# other structures d/d mu_j = c alpha_j / s_j, d/d log s_j = -1 + c z_j alpha_j, and
# d ll / d R_J = -R_J^-1 / 2 + c alpha alpha' / 2 is scattered into G = d ll / d R;
# R = L L' gives d ll / d L = (G + G') L, which the LKJ block's own reverse pass
# (_corr_value_grad!) takes from its workspace.
function _unstr_loop!(ld::GLMDensity{F,P,W,T}, etas) where {F,P,W,T}
    fam, y = ld.family, ld.y
    nb = _nbase(fam); stud = _is_student(fam)
    D = fam.nmax
    pw = ld.ws[nb + 1]
    L = reshape(pw.eta, D, D)
    R = L * L'
    G = zeros(T, D, D)
    ν, dνu = stud ? _dpar_transform(ld.preds[3]::ScalarDpar, _at(etas[3], 1)) : (one(T), zero(T))
    gν = zero(T); total = zero(T)
    wmu, wsg = ld.ws[1].w, ld.ws[2].w
    for g_ in 1:(length(fam.beg) - 1)
        r = fam.beg[g_]:(fam.beg[g_ + 1] - 1); n = length(r)
        J = fam.tlev[r]; obs = fam.perm[r]
        lsv = T[_at(etas[2], i) for i in obs]
        s = exp.(lsv)
        z = T[(y[obs[k]] - _at(etas[1], obs[k])) / s[k] for k in 1:n]
        C = cholesky(Symmetric(R[J, J]); check=false)
        issuccess(C) || return T(NaN)
        α = C \ z; q = dot(z, α); ldR = logdet(C)
        if stud
            c = (ν + n) / (ν + q)
            total += loggamma((ν + n) / 2) - loggamma(ν / 2) - n * log(ν) / 2 - ldR / 2 - (ν + n) / 2 * log1p(q / ν)
            gν += (digamma((ν + n) / 2) - digamma(ν / 2) - n / ν - log1p(q / ν)) / 2 + (ν + n) * q / (2ν * (ν + q))
        else
            c = one(T)
            total += -ldR / 2 - q / 2
        end
        total -= sum(lsv)
        G[J, J] .+= (c / 2) .* (α * α') .- inv(C) ./ 2
        for k in 1:n
            i = obs[k]
            wmu[i] = c * α[k] / s[k]
            _tc_addw!(wsg, ld.preds[2], i, -1 + c * z[k] * α[k])
        end
    end
    reshape(pw.w, D, D) .= (G .+ G') * L
    stud && (ld.ws[3].w[1] = gν * dνu)
    return total
end

# --- generic path (any number type: the AD reference) ---------------------------

function _tc_ll_generic(ld::GLMDensity, etas)
    fam, y, N = ld.family, ld.y, ld.N
    nb = _nbase(fam)
    fam.kind === :unstr && return _unstr_generic(ld, etas)
    ρ = ntuple(k -> _natural(ld.preds[nb + k], etas[nb + k]), length(_ac_dpars(fam)))
    mu = [_at(etas[1], i) for i in 1:N]
    ν = _is_student(fam) ? _natural(ld.preds[3], etas[3]) : nothing
    if _is_sar(fam)
        ls = etas[2]
        W = fam.M
        all(λ -> ρ[1] * λ < 1, fam.eig) || return oftype(float(ρ[1]), NaN)
        r = fam.kind === :lagsar ? y .- ρ[1] .* (W * y) .- mu : (y .- mu) .- ρ[1] .* (W * (y .- mu))
        Q = sum(abs2, r) * exp(-2ls)
        lp = -N * ls + sum(log1p.(-ρ[1] .* fam.eig))
        return ν === nothing ? lp - Q / 2 :
               lp + loggamma((ν + N) / 2) - loggamma(ν / 2) - N * log(ν) / 2 - (ν + N) / 2 * log1p(Q / ν)
    end
    if fam.kind === :fcor
        L = fam.M
    else
        C = cholesky(Symmetric(_tc_R(fam, ρ, fam.nmax), :L); check=false)
        issuccess(C) || return oftype(float(ρ[1]), NaN)
        L = Matrix(C.L)
    end
    lp = zero(promote_type(eltype(mu), eltype(L), ν === nothing ? Bool : typeof(ν)))
    for g in 1:(length(fam.beg) - 1)
        obs = fam.perm[fam.beg[g]:(fam.beg[g + 1] - 1)]
        n = length(obs)
        lsv = [_at(etas[2], i) for i in obs]
        z = (y[obs] .- mu[obs]) ./ exp.(lsv)
        Ln = LowerTriangular(L[1:n, 1:n])
        q = sum(abs2, Ln \ z)
        ldR = 2 * sum(log, diag(Ln))
        lp += -sum(lsv) - ldR / 2
        lp += ν === nothing ? -q / 2 :
              loggamma((ν + n) / 2) - loggamma(ν / 2) - n * log(ν) / 2 - (ν + n) / 2 * log1p(q / ν)
    end
    return lp
end

function _unstr_generic(ld::GLMDensity, etas)
    fam, y = ld.family, ld.y
    nb = _nbase(fam)
    L = etas[nb + 1].L; R = L * L'
    ν = _is_student(fam) ? _natural(ld.preds[3], etas[3]) : nothing
    lp = zero(eltype(R))
    for g_ in 1:(length(fam.beg) - 1)
        r = fam.beg[g_]:(fam.beg[g_ + 1] - 1); n = length(r)
        J = fam.tlev[r]; obs = fam.perm[r]
        lsv = [_at(etas[2], i) for i in obs]
        z = [(y[i] - _at(etas[1], i)) for i in obs] ./ exp.(lsv)
        Lg = cholesky(Symmetric(R[J, J])).L
        q = sum(abs2, Lg \ z); ldR = 2 * sum(log, diag(Lg))
        lp += -sum(lsv) - ldR / 2
        lp += ν === nothing ? -q / 2 :
              loggamma((ν + n) / 2) - loggamma(ν / 2) - n * log(ν) / 2 - (ν + n) / 2 * log1p(q / ν)
    end
    return lp
end

function LogDensityProblems.logdensity(ld::GLMDensity{F}, θraw::AbstractVector) where {F<:TimeCov}
    θ, ljo = _order_generic(ld.ordered, θraw)
    if !isempty(ld.pinned)                       # constant() priors, as the default method
        θ = Vector{promote_type(eltype(θ), eltype(ld.pinvals))}(θ)
        for (k, c) in zip(ld.pinned, ld.pinvals)
            θ[k] = c
            ljo -= θraw[k]^2 / 2
        end
    end
    etas = map(p -> _eta_generic(p, θ, ld.N), ld.preds)
    lp = ljo + (ld.prior_only ? 0 : _tc_ll_generic(ld, etas))
    for p in ld.preds
        lp += _prior_generic(p, θ)
    end
    return lp
end

# --- post-processing ------------------------------------------------------------------
# brms: posterior_predict draws each group jointly; log_lik is the pointwise
# CONDITIONAL density of y_i given the rest of its group (Buerkner, Gabry &
# Vehtari 2021), which PSIS-LOO needs for non-factorising models:
#   g = C^-1 e, cbar = diag(C^-1),  y_i | y_-i ~ N(y_i - g_i / cbar_i, 1 / cbar_i)
# (student: df nu + n - 1 and the variance scaled by (nu + beta1)/(nu + n - 1)).
# Training data only: new data would need the time / spatial structure of the
# new observations.

# one draw on the natural scale: (mu, s, nu, rho)
function _tc_draw(ld::GLMDensity, θ)
    fam = ld.family; nb = _nbase(fam)
    etas = map(p -> _eta_generic(p, θ, ld.N), ld.preds)
    mu = [Float64(_at(etas[1], i)) for i in 1:ld.N]
    s = [exp(Float64(_at(etas[2], i))) for i in 1:ld.N]
    ν = _is_student(fam) ? Float64(_natural(ld.preds[3], etas[3])) : Inf
    ρ = fam.kind === :unstr ? (Matrix{Float64}(etas[nb + 1].L * etas[nb + 1].L'),) :
        ntuple(k -> Float64(_natural(ld.preds[nb + k], etas[nb + k])), length(_ac_dpars(fam)))
    return mu, s, ν, ρ
end

# (observations, lower Cholesky factor of their residual covariance) per group;
# for sar the factor of B^-1 B^-T sigma^2 is not needed: see _tc_sar_B
function _tc_cov_chols(fam::TimeCov, s, ρ)
    out = Tuple{Vector{Int},Matrix{Float64}}[]
    if fam.kind === :unstr
        R = ρ[1]
        for g in 1:(length(fam.beg) - 1)
            r = fam.beg[g]:(fam.beg[g + 1] - 1); obs = fam.perm[r]; J = fam.tlev[r]
            push!(out, (obs, s[obs] .* Matrix(cholesky(Symmetric(R[J, J])).L)))
        end
        return out
    end
    Lfull = fam.kind === :fcor ? fam.M : Matrix(cholesky(Symmetric(_tc_R(fam, ρ, fam.nmax), :L)).L)
    for g in 1:(length(fam.beg) - 1)
        obs = fam.perm[fam.beg[g]:(fam.beg[g + 1] - 1)]
        n = length(obs)
        push!(out, (obs, s[obs] .* Lfull[1:n, 1:n]))          # diag(s) L
    end
    return out
end
_tc_sar_B(fam::TimeCov, ρ) = I - ρ[1] .* fam.M

function _tc_loo_ll!(out, s_idx, obs, y, e, Cinv, ν)
    g = Cinv * e
    n = length(obs)
    for (k, i) in enumerate(obs)
        cb = Cinv[k, k]
        yloo = y[i] - g[k] / cb
        if isinf(ν)
            out[s_idx, i] = -log(2π) / 2 + log(cb) / 2 - (y[i] - yloo)^2 * cb / 2
        else
            # brms student_t_cov_factor's beta1: e_-i' C_-i^-1 e_-i
            b1 = dot(e, g) - g[k]^2 / cb
            df = ν + n - 1
            sd = sqrt((ν + b1) / df / cb)
            out[s_idx, i] = Distributions.logpdf(Distributions.TDist(df), (y[i] - yloo) / sd) - log(sd)
        end
    end
end

# the time structure of NEW rows (brms: the new data's own groups and times;
# unstr's time levels must be training levels)
function _tc_newfam(fit, newdata)
    fam0 = fit.density.family
    f = fit.info.formula
    ac = f.dpars[1].rhs.ar
    cols = Tables.columntable(newdata)
    N = _nrows(cols)
    vals = Dict{Atom,AbstractVector}()
    ac[1] === nothing || (vals[ac[1]] = _eval_atom(ac[1], cols, f.mod))
    ac[2] === nothing || (vals[ac[2]] = _eval_atom(ac[2], cols, f.mod))
    fn = _timecov_family(fam0.fam, ac, vals, zeros(N), 1:N, N)
    fam0.kind === :unstr || return fn
    lv = [findfirst(==(l), fam0.tlabels) for l in fn.tlabels]
    any(isnothing, lv) && throw(ArgumentError("unstr(): new data has time points the model was not fitted with"))
    return TimeCov(fn.fam, fn.kind, 0, 0, fn.perm, fn.beg, fam0.nmax, fn.M, fn.eig, fn.Wy, 0.0,
                   Int[lv[k] for k in fn.tlev], fam0.tlabels)
end

function _tc_draws(fit, newdata, what::Symbol; rng=Random.default_rng(), re_formula=nothing,
                   allow_new_levels=false, sample_new_levels=:uncertainty, kw...)
    ld = fit.density; fam = ld.family
    D = reshape(fit.draws, size(fit.draws, 1), :)
    if newdata === nothing
        y = Float64.(ld.y); N = ld.N
        means = θ -> _tc_draw(ld, θ)
    else
        fam.kind in (:arma_cov, :cosy, :unstr) ||
            throw(ArgumentError("predictions for new data are not supported with $(fam.kind) terms (its matrix belongs to the training rows)"))
        preds, N, _ = _prediction_preds(fit, newdata; re_formula, allow_new_levels)
        opts = _PredOpts(rng, Symbol(sample_new_levels))
        fam = _tc_newfam(fit, newdata)
        f = fit.info.formula
        y = what === :log_lik ? Float64.(_eval_atom(f.response, Tables.columntable(newdata), f.mod)) : zeros(N)
        means = θ -> begin
            etas = map(p -> _eta_pred(p, θ, N, opts), preds)
            _, _, ν, ρ = _tc_draw(ld, θ)                      # structure / nu: from the draw alone
            ([Float64(_at(etas[1], i)) for i in 1:N], [exp(Float64(_at(etas[2], i))) for i in 1:N], ν, ρ)
        end
    end
    out = Matrix{Float64}(undef, size(D, 2), N)
    for si in axes(D, 2)
        mu, s, ν, ρ = means(view(D, :, si))
        # student draws: a common chi-square scale per group (multi_student_t)
        scale() = isinf(ν) ? 1.0 : sqrt(ν / rand(rng, Distributions.Chisq(ν)))
        if _is_sar(fam)
            B = _tc_sar_B(fam, ρ)
            lag = fam.kind === :lagsar
            if what === :epred
                out[si, :] .= lag ? B \ mu : mu
            elseif what === :predict
                ε = s[1] * scale() .* randn(rng, N)
                out[si, :] .= lag ? B \ (mu .+ ε) : mu .+ B \ ε
            else
                e = lag ? y .- B \ mu : y .- mu
                _tc_loo_ll!(out, si, 1:N, y, e, Symmetric(B' * B) ./ s[1]^2, ν)
            end
            continue
        end
        if what === :epred
            out[si, :] .= mu
            continue
        end
        for (obs, Lc) in _tc_cov_chols(fam, s, ρ)
            if what === :predict
                out[si, obs] .= mu[obs] .+ scale() .* (Lc * randn(rng, length(obs)))
            else
                Ci = inv(cholesky(Symmetric(Lc * Lc')))
                _tc_loo_ll!(out, si, obs, y, y[obs] .- mu[obs], Ci, ν)
            end
        end
    end
    return out
end

# =============================================================================
# Latent residual autocorrelation (brms, non-gaussian families with
# ar/ma/arma(cov = TRUE) or cosy()): a linear-predictor term
#     err[group] = sderr * L_n zerr[group],   zerr ~ N(0, 1),   L = chol(R(rho))
# (brms scale_time_err; R as for the gaussian structures above, the same
# (gr, time) ordering, zerr stored in that order). Gradient: with ebar = d ll /
# d err, d/d zerr = sderr L' ebar, d/d log sderr = ebar' err, and d/d rho by
# dual numbers carried through one Cholesky of R, as in _tc_loop!.
# =============================================================================

_lac_rho(t::LatentAC, θ) = ntuple(k -> first(_btrans(t.lb[k], t.ub[k], θ[t.rho[k]])), length(t.rho))

# the residuals in data order (generic in the number type; NaN if R is not PD)
function _lac_err(t::LatentAC, θ)
    tc = t.tc
    s = exp(θ[t.sd]); ρ = _lac_rho(t, θ)
    E = promote_type(eltype(θ), typeof(float(ρ[1])))
    err = zeros(E, length(t.z))
    C = cholesky(Symmetric(_tc_R(tc, ρ, tc.nmax), :L); check=false)
    issuccess(C) || return fill(E(NaN), length(t.z))
    L = Matrix(C.L)
    for g in 1:(length(tc.beg) - 1)
        r = tc.beg[g]:(tc.beg[g + 1] - 1); n = length(r)
        v = LowerTriangular(L[1:n, 1:n]) * θ[t.z[r]]
        err[tc.perm[r]] .= s .* v
    end
    return err
end

function _lac_prior(t::LatentAC, θ)
    u = θ[t.sd]
    lp = prior_kernel(t.sd_prior, exp(u)) + u - sum(abs2, θ[t.z]) / 2
    for k in eachindex(t.rho)
        x, _, lj, _ = _btrans(t.lb[k], t.ub[k], θ[t.rho[k]])
        lp += prior_kernel(t.rho_priors[k], x) + lj
    end
    return lp
end
_lac_generic(t::LatentAC, θ) = (_lac_err(t, θ), _lac_prior(t, θ))

# --- analytic path: concretely typed throughout (these loops are compiled for
# every LinPred, even with no lac term, so nothing here may be non-inferable:
# the structure-parameter count K goes through a literal Val branch, and all
# arithmetic is in eltype(θ) -- never a dual on this path)
function _lac_forward!(η::AbstractVector{T}, t::LatentAC, θ::AbstractVector{T}) where {T}
    K = length(t.rho)
    K == 1 ? _lac_forward!(η, t, θ, Val(1)) : _lac_forward!(η, t, θ, Val(2))
    return η
end
function _lac_forward!(η::AbstractVector{T}, t::LatentAC, θ::AbstractVector{T}, ::Val{K}) where {T,K}
    tc = t.tc
    s = exp(θ[t.sd])
    ρ = ntuple(k -> T(first(_btrans(t.lb[k], t.ub[k], θ[t.rho[k]]))), Val(K))
    C = cholesky(Symmetric(_tc_R(tc, ρ, tc.nmax)::Matrix{T}, :L); check=false)
    if !issuccess(C)
        fill!(η, T(NaN)); return nothing                # a rejected proposal
    end
    L = C.L
    @inbounds for g_ in 1:(length(tc.beg) - 1)
        r0 = tc.beg[g_] - 1; n = tc.beg[g_ + 1] - tc.beg[g_]
        for i in 1:n
            v = zero(T)
            for j in 1:i
                v += L[i, j] * θ[t.z[r0 + j]]
            end
            η[tc.perm[r0 + i]] += s * v
        end
    end
    return nothing
end

function _lac_pullback!(g::AbstractVector{T}, t::LatentAC, w::AbstractVector{T}, θ) where {T}
    K = length(t.rho)
    lp = K == 1 ? _lac_pullback!(g, t, w, θ, Val(1), T) : _lac_pullback!(g, t, w, θ, Val(2), T)
    return lp::T
end
function _lac_pullback!(g, t::LatentAC, w, θ, ::Val{K}, ::Type{T}) where {K,T}
    tc = t.tc
    D = ForwardDiff.Dual{Nothing,T,K}
    u = θ[t.sd]; s = exp(u)
    tr = ntuple(k -> _btrans(t.lb[k], t.ub[k], θ[t.rho[k]]), Val(K))
    ρd = ntuple(k -> D(tr[k][1], ForwardDiff.Partials(ntuple(j -> T(j == k), Val(K)))), Val(K))
    L = _tc_chol(tc, ρd)::Union{Nothing,Matrix{D}}
    L === nothing && return T(NaN)
    gsd = zero(T); gρ = zeros(T, K)
    for g_ in 1:(length(tc.beg) - 1)
        r = tc.beg[g_]:(tc.beg[g_ + 1] - 1); n = length(r)
        @inbounds for i in 1:n
            # v_i = sum_{j <= i} L_ij z_j (dual in rho); ebar_i = w at observation perm[r[i]]
            v = zero(D)
            for j in 1:i
                v += L[i, j] * θ[t.z[r[j]]]
            end
            eb = w[tc.perm[r[i]]]
            gsd += eb * ForwardDiff.value(v)
            p = ForwardDiff.partials(v)
            for k in 1:K
                gρ[k] += s * eb * p[k]
            end
            # d/d z_j += s * L_ij * ebar_i  (j <= i)
            for j in 1:i
                g[t.z[r[j]]] += s * ForwardDiff.value(L[i, j]) * eb
            end
        end
    end
    vs, ds = prior_kernel_grad(t.sd_prior, s)
    g[t.sd] += s * gsd + s * ds + 1                 # + 1: log-Jacobian of exp
    lp = vs + u
    for k in 1:K
        x, dx, lj, dlj = tr[k]
        vk, dk = prior_kernel_grad(t.rho_priors[k], x)
        g[t.rho[k]] += (gρ[k] + dk) * dx + dlj
        lp += vk + lj
    end
    @inbounds for k in t.z
        g[k] -= θ[k]
        lp -= θ[k]^2 / 2
    end
    return lp
end
