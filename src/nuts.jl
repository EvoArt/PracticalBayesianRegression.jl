# =============================================================================
# An allocation-free NUTS, following Stan's `base_nuts` transition line by
# line (multinomial sampling, generalised U-turn criterion with the two extra
# checks across merged subtrees, divergence at Delta H > 1000), with Stan's
# windowed adaptation (75 / 25-doubling / 50 buffers, diagonal or dense metric
# with the 5-sample regularisation, dual averaging restarted after each metric
# update).
#
# The dense metric (Stan's dense_e) pays a d x d matvec per leapfrog step on
# top of the gradient, which is cheap next to an N-observation gradient, and
# in the sampler sweep it was what made AdvancedHMC's dense NUTS beat the
# diagonal one on multilevel models (posterior correlation between an
# intercept and its group effects / sd).
#
# Why not AdvancedHMC: it is the same algorithm, but on small models its
# per-iteration overhead (AbstractMCMC's step loop, a fresh phase point and
# transition object per step, dynamic dispatch) was most of the run time —
# profiled on gaussian K = 3, N = 1000: AdvancedHMC 0.078 s per chain against
# 0.043 s of gradient work, brms 0.08 s. Here every buffer a transition needs
# is allocated once per chain (one set per tree depth), so a transition
# allocates nothing and the only per-step cost is the gradient.
#
# Also generic in the number type (Float32 sampling path), dependency-free
# beyond Random (the trimmable path), and it records Stan's per-draw
# diagnostics: accept_stat__, stepsize__, treedepth__, n_leapfrog__,
# divergent__, energy__.
# =============================================================================

const _MAX_DELTA_H = 1000.0

# integration state: position, momentum, gradient, log density
struct _PS{T}
    x::Vector{T}
    p::Vector{T}
    g::Vector{T}
end
_PS{T}(d) where {T} = _PS{T}(zeros(T, d), zeros(T, d), zeros(T, d))

# per-depth scratch for the recursive tree builder (Stan's locals)
struct _Level{T}
    p_init_end::Vector{T}
    ps_init_end::Vector{T}       # p_sharp = M^{-1} p
    rho_init::Vector{T}
    p_final_beg::Vector{T}
    ps_final_beg::Vector{T}
    rho_final::Vector{T}
    rho_ext::Vector{T}
    prop_x::Vector{T}            # the final subtree's proposal
    prop_g::Vector{T}
end
_Level{T}(d) where {T} = _Level{T}((zeros(T, d) for _ in 1:9)...)

mutable struct _NutsState{T,F,R}
    gradf!::F
    rng::R
    d::Int
    z::_PS{T}                    # current integration state
    lp::T                        # its log density
    minv::Vector{T}              # inverse diagonal metric
    dense::Bool                  # dense metric: Minv = U'U, minv unused
    Minv::Matrix{T}              # inverse dense metric (0 x 0 when diagonal)
    U::Matrix{T}                 # its upper Cholesky factor
    tmp::Vector{T}
    eps::T
    max_depth::Int
    levels::Vector{_Level{T}}
    # proposal of the current subtree (written by the leaves)
    prop_x::Vector{T}
    prop_g::Vector{T}
    prop_lp::T
    n_leapfrog::Int
    sum_metro::T
    divergent::Bool
    H0::T
    # transition-level buffers
    fwd::_PS{T}
    bck::_PS{T}
    samp_x::Vector{T}
    samp_g::Vector{T}
    p_fwd_bck::Vector{T}; ps_fwd_bck::Vector{T}; p_fwd_fwd::Vector{T}; ps_fwd_fwd::Vector{T}
    p_bck_fwd::Vector{T}; ps_bck_fwd::Vector{T}; p_bck_bck::Vector{T}; ps_bck_bck::Vector{T}
    rho::Vector{T}; rho_fwd::Vector{T}; rho_bck::Vector{T}; rho_ext::Vector{T}
end

function _NutsState(gradf!::F, x0::Vector{T}, rng, max_depth; dense::Bool=false) where {F,T}
    d = length(x0)
    z = _PS{T}(d)
    copyto!(z.x, x0)
    lp = _safe_grad!(gradf!, z.g, z.x)
    v() = zeros(T, d)
    m = dense ? d : 0
    eye = Matrix{T}(I, m, m)
    return _NutsState(gradf!, rng, d, z, T(lp), ones(T, d), dense, eye, copy(eye), v(), one(T), max_depth,
                      [_Level{T}(d) for _ in 1:(max_depth + 1)], v(), v(), T(lp), 0, zero(T), false, zero(T),
                      _PS{T}(d), _PS{T}(d), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v())
end

@inline _dot(a, b) = (s = zero(eltype(a)); @inbounds @simd for i in eachindex(a); s += a[i] * b[i]; end; s)

# the three places the metric enters: p_sharp = M^{-1} p, the kinetic
# energy p' M^{-1} p / 2, and the momentum draw p ~ N(0, M)
@inline function _psharp!(out, s::_NutsState, p)
    if s.dense
        mul!(out, s.Minv, p)
    else
        @inbounds @simd for i in 1:s.d
            out[i] = s.minv[i] * p[i]
        end
    end
    return out
end
@inline _kin(s::_NutsState, p) = _dot(p, _psharp!(s.tmp, s, p)) / 2
function _draw_p!(p, s::_NutsState{T}) where {T}
    if s.dense
        # Minv = U'U, so p = U^{-1} u with u ~ N(0, I) has covariance M
        @inbounds for i in 1:s.d
            p[i] = randn(s.rng, T)
        end
        ldiv!(UpperTriangular(s.U), p)
    else
        @inbounds for i in 1:s.d
            p[i] = randn(s.rng, T) / sqrt(s.minv[i])
        end
    end
    return p
end
@inline _criterion(ps_minus, ps_plus, rho) = _dot(ps_plus, rho) > 0 && _dot(ps_minus, rho) > 0
@inline _lse(a, b) = a == -Inf ? b : b == -Inf ? a : (m = max(a, b); m + log(exp(a - m) + exp(b - m)))

# Stan's reject: a domain error while evaluating the density at a proposal
# (log of a negative number, a shape parameter <= 0, a matrix that is not
# positive definite) means the point is outside where the model is defined.
# Stan reports it as an informational "proposal rejected" and carries on; so
# do we, by returning -Inf: H is then infinite, the step counts as divergent,
# the subtree stops and the point is never accepted. Only domain-type errors
# are caught, as in Stan — anything else (a MethodError, a BoundsError) is a
# bug and still propagates. Families should still return -Inf themselves
# where they can (cheaper than an exception); this is the safety net.
@inline function _safe_grad!(f::F, g::AbstractVector{T}, x) where {F,T}
    try
        return T(f(g, x))
    catch e
        e isa Union{DomainError,PosDefException,SingularException} || rethrow()
        fill!(g, zero(T))
        return T(-Inf)
    end
end

# one leapfrog step of the current state in place; returns the new log density
@inline function _leap!(s::_NutsState, eps)
    z = s.z
    @inbounds @simd for i in 1:s.d
        z.p[i] += eps / 2 * z.g[i]
    end
    ps = _psharp!(s.tmp, s, z.p)
    @inbounds @simd for i in 1:s.d
        z.x[i] += eps * ps[i]
    end
    lp = _safe_grad!(s.gradf!, z.g, z.x)
    @inbounds @simd for i in 1:s.d
        z.p[i] += eps / 2 * z.g[i]
    end
    s.lp = lp
    return lp
end

# Stan's build_tree. Writes the subtree's proposal into (prop_x, prop_g,
# prop_lp), its begin/end momenta and p_sharps, adds its momentum to `rho`,
# and logsumexps its weight into lsw[]. Returns validity.
function _build!(s::_NutsState{T}, depth, ps_beg, ps_end, rho, p_beg, p_end, sgn, lsw::Base.RefValue{T}) where {T}
    if depth == 0
        lp = _leap!(s, sgn * s.eps)
        s.n_leapfrog += 1
        z = s.z
        _psharp!(ps_beg, s, z.p)
        h = -lp + _dot(z.p, ps_beg) / 2
        isnan(h) && (h = T(Inf))
        h - s.H0 > _MAX_DELTA_H && (s.divergent = true)
        lsw[] = _lse(lsw[], s.H0 - h)
        s.sum_metro += s.H0 - h > 0 ? one(T) : exp(s.H0 - h)
        copyto!(s.prop_x, z.x); copyto!(s.prop_g, z.g); s.prop_lp = lp
        @inbounds for i in 1:s.d
            ps_end[i] = ps_beg[i]
            rho[i] += z.p[i]
            p_beg[i] = z.p[i]
            p_end[i] = z.p[i]
        end
        return !s.divergent
    end
    L = s.levels[depth]
    # initial subtree
    lsw_init = Ref(T(-Inf))
    fill!(L.rho_init, 0)
    _build!(s, depth - 1, ps_beg, L.ps_init_end, L.rho_init, p_beg, L.p_init_end, sgn, lsw_init) || return false
    # keep the initial subtree's proposal while the final one is built
    copyto!(L.prop_x, s.prop_x); copyto!(L.prop_g, s.prop_g); lp_init = s.prop_lp
    lsw_final = Ref(T(-Inf))
    fill!(L.rho_final, 0)
    _build!(s, depth - 1, L.ps_final_beg, ps_end, L.rho_final, L.p_final_beg, p_end, sgn, lsw_final) || return false
    # multinomial choice between the two subtrees' proposals
    lsw_sub = _lse(lsw_init[], lsw_final[])
    lsw[] = _lse(lsw[], lsw_sub)
    take_final = lsw_final[] > lsw_sub || rand(s.rng) < exp(lsw_final[] - lsw_sub)
    if !take_final
        copyto!(s.prop_x, L.prop_x); copyto!(s.prop_g, L.prop_g); s.prop_lp = lp_init
    end
    # rho_subtree = rho_init + rho_final, added to rho
    @inbounds for i in 1:s.d
        L.rho_ext[i] = L.rho_init[i] + L.rho_final[i]
        rho[i] += L.rho_ext[i]
    end
    persist = _criterion(ps_beg, ps_end, L.rho_ext)
    @inbounds for i in 1:s.d
        L.rho_ext[i] = L.rho_init[i] + L.p_final_beg[i]
    end
    persist &= _criterion(ps_beg, L.ps_final_beg, L.rho_ext)
    @inbounds for i in 1:s.d
        L.rho_ext[i] = L.rho_final[i] + L.p_init_end[i]
    end
    persist &= _criterion(L.ps_init_end, ps_end, L.rho_ext)
    return persist
end

@inline _copyps!(a::_PS, b::_PS) = (copyto!(a.x, b.x); copyto!(a.p, b.p); copyto!(a.g, b.g); a)

# one NUTS transition from (z, lp); returns (treedepth, n_leapfrog,
# accept_stat, divergent, energy)
function _transition!(s::_NutsState{T}) where {T}
    z, d = s.z, s.d
    _draw_p!(z.p, s)
    lp0 = s.lp
    _copyps!(s.fwd, z); _copyps!(s.bck, z)
    copyto!(s.samp_x, z.x); copyto!(s.samp_g, z.g); lp_samp = lp0
    _psharp!(s.rho_ext, s, z.p)
    @inbounds for i in 1:d
        ps = s.rho_ext[i]
        s.p_fwd_bck[i] = z.p[i]; s.ps_fwd_bck[i] = ps
        s.p_fwd_fwd[i] = z.p[i]; s.ps_fwd_fwd[i] = ps
        s.p_bck_fwd[i] = z.p[i]; s.ps_bck_fwd[i] = ps
        s.p_bck_bck[i] = z.p[i]; s.ps_bck_bck[i] = ps
        s.rho[i] = z.p[i]
    end
    lsw = zero(T)
    s.H0 = -lp0 + _dot(z.p, s.ps_fwd_bck) / 2
    s.n_leapfrog = 0; s.sum_metro = zero(T); s.divergent = false
    depth = 0
    while depth < s.max_depth
        fill!(s.rho_fwd, 0); fill!(s.rho_bck, 0)
        lsw_sub = Ref(T(-Inf))
        if rand(s.rng) > 0.5
            # extend forward from the forward end
            _copyps!(z, s.fwd); s.lp = NaN
            copyto!(s.rho_bck, s.rho)
            copyto!(s.p_bck_fwd, s.p_fwd_bck); copyto!(s.ps_bck_fwd, s.ps_fwd_bck)
            valid = _build!(s, depth, s.ps_fwd_bck, s.ps_fwd_fwd, s.rho_fwd, s.p_fwd_bck, s.p_fwd_fwd, 1, lsw_sub)
            _copyps!(s.fwd, z)
        else
            _copyps!(z, s.bck)
            copyto!(s.rho_fwd, s.rho)
            copyto!(s.p_fwd_bck, s.p_bck_fwd); copyto!(s.ps_fwd_bck, s.ps_bck_fwd)
            valid = _build!(s, depth, s.ps_bck_fwd, s.ps_bck_bck, s.rho_bck, s.p_bck_fwd, s.p_bck_bck, -1, lsw_sub)
            _copyps!(s.bck, z)
        end
        valid || break
        depth += 1
        if lsw_sub[] > lsw || rand(s.rng) < exp(lsw_sub[] - lsw)
            copyto!(s.samp_x, s.prop_x); copyto!(s.samp_g, s.prop_g); lp_samp = s.prop_lp
        end
        lsw = _lse(lsw, lsw_sub[])
        @inbounds for i in 1:d
            s.rho[i] = s.rho_bck[i] + s.rho_fwd[i]
        end
        persist = _criterion(s.ps_bck_bck, s.ps_fwd_fwd, s.rho)
        @inbounds for i in 1:d
            s.rho_ext[i] = s.rho_bck[i] + s.p_fwd_bck[i]
        end
        persist &= _criterion(s.ps_bck_bck, s.ps_fwd_bck, s.rho_ext)
        @inbounds for i in 1:d
            s.rho_ext[i] = s.rho_fwd[i] + s.p_bck_fwd[i]
        end
        persist &= _criterion(s.ps_bck_fwd, s.ps_fwd_fwd, s.rho_ext)
        persist || break
    end
    copyto!(z.x, s.samp_x); copyto!(z.g, s.samp_g); s.lp = lp_samp
    acc = s.n_leapfrog > 0 ? s.sum_metro / s.n_leapfrog : zero(T)
    return depth, s.n_leapfrog, acc, s.divergent, s.H0
end

# --- adaptation ------------------------------------------------------------------

mutable struct _DualAvg
    mu::Float64; lebar::Float64; hbar::Float64; k::Int; delta::Float64
end
_DualAvg(eps, delta) = _DualAvg(log(10eps), 0.0, 0.0, 0, delta)
function _da_update!(a::_DualAvg, accept)
    a.k += 1
    m = a.k
    eta = 1 / (m + 10)
    a.hbar = (1 - eta) * a.hbar + eta * (a.delta - accept)
    le = a.mu - sqrt(m) / 0.05 * a.hbar
    w = m^(-0.75)
    a.lebar = w * le + (1 - w) * a.lebar
    return exp(le)
end

# Stan's init_stepsize: double or halve until the one-step acceptance crosses 0.8
function _init_stepsize!(s::_NutsState{T}) where {T}
    x0 = copy(s.z.x); g0 = copy(s.z.g); lp0 = s.lp
    function accept(eps)
        copyto!(s.z.x, x0); copyto!(s.z.g, g0); s.lp = lp0
        _draw_p!(s.z.p, s)
        h0 = -lp0 + _kin(s, s.z.p)
        lp = _leap!(s, eps)
        h = -lp + _kin(s, s.z.p)
        return isfinite(h) ? h0 - h : T(-Inf)
    end
    eps = s.eps
    (iszero(eps) || isnan(eps) || eps > 1e7) && return s.eps
    dir = accept(eps) > log(0.8) ? 1 : -1
    for _ in 1:200
        # as Stan: retest the current size with fresh momentum, then move
        dh = accept(eps)
        dir == 1 && !(dh > log(0.8)) && break
        dir == -1 && !(dh < log(0.8)) && break
        eps = dir == 1 ? 2eps : eps / 2
        (eps > 1e7 || eps < 1e-12) && break
    end
    copyto!(s.z.x, x0); copyto!(s.z.g, g0); s.lp = lp0
    s.eps = T(eps)
    return s.eps
end

"""
    nuts_chain(gradf!, x0, nkeep, nwarm; delta=0.8, max_depth=10, thin=1, metric=:diag, rng)

Run one chain of the package's NUTS on `gradf!(g, x) -> logdensity`.
`metric` is `:diag` (Stan's diag_e) or `:dense` (dense_e). Returns
`(draws, stats)`: a dim x nkeep matrix and a Dict of Stan-named per-draw
diagnostics.
"""
function nuts_chain(gradf!::F, x0::Vector{T}, nkeep::Int, nwarm::Int; delta=0.8, max_depth::Int=10, thin::Int=1,
                    metric::Symbol=:diag, rng=Random.default_rng()) where {F,T}
    # `gradf!::F ... where {F}` is load-bearing: `gradf!` is only passed on
    # here, never called, and Julia does not specialise on a function argument
    # it only passes through — the state was then `_NutsState{T,Function}` and
    # every `_transition!` a dynamic dispatch (unresolved under juliac --trim).
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    dense = metric === :dense
    s = _NutsState(gradf!, x0, rng, max_depth; dense=dense)
    isfinite(s.lp) || throw(ArgumentError("the log density is not finite at the initial values"))
    _init_stepsize!(s)
    da = _DualAvg(Float64(s.eps), Float64(delta))
    # Stan's windows: init_buffer 75, term_buffer 50, base window 25 doubling
    ib, tb, w0 = 75, 50, 25
    if nwarm < 20
        ib, tb, w0 = nwarm, 0, 0
    elseif ib + tb + w0 > nwarm
        ib = round(Int, 0.15nwarm); tb = round(Int, 0.1nwarm); w0 = nwarm - ib - tb
    end
    win_end = ib + w0
    win_size = w0
    n_w = 0; wmean = zeros(Float64, s.d); wm2 = zeros(Float64, s.d)
    wcov = zeros(Float64, dense ? s.d : 0, dense ? s.d : 0); wδ = zeros(Float64, dense ? s.d : 0)
    for it in 1:nwarm
        _, _, acc, _, _ = _transition!(s)
        s.eps = T(_da_update!(da, Float64(acc)))
        if w0 > 0 && ib < it <= nwarm - tb
            n_w += 1
            if dense
                # Welford covariance: M2 += (x - mean_old) (x - mean_new)'
                @inbounds for i in 1:s.d
                    wδ[i] = Float64(s.z.x[i]) - wmean[i]
                    wmean[i] += wδ[i] / n_w
                end
                @inbounds for j in 1:s.d
                    xj = Float64(s.z.x[j]) - wmean[j]
                    for i in 1:s.d
                        wcov[i, j] += wδ[i] * xj
                    end
                end
            else
                @inbounds for i in 1:s.d
                    xi = Float64(s.z.x[i])
                    δ = xi - wmean[i]
                    wmean[i] += δ / n_w
                    wm2[i] += δ * (xi - wmean[i])
                end
            end
            if it == win_end
                # new metric (Stan's regularisation toward 1e-3), restart step size
                if dense
                    sh = n_w / (n_w + 5.0)
                    C = sh .* (wcov ./ max(n_w - 1, 1))
                    for i in 1:s.d
                        C[i, i] += 1e-3 * (5.0 / (n_w + 5.0))
                    end
                    C = (C .+ C') ./ 2
                    s.Minv .= T.(C)
                    s.U .= T.(cholesky(Symmetric(C)).U)
                    fill!(wcov, 0)
                else
                    @inbounds for i in 1:s.d
                        v = n_w > 1 ? wm2[i] / (n_w - 1) : 1.0
                        s.minv[i] = T((n_w / (n_w + 5.0)) * v + 1e-3 * (5.0 / (n_w + 5.0)))
                    end
                end
                n_w = 0; fill!(wmean, 0); fill!(wm2, 0)
                _init_stepsize!(s)
                da = _DualAvg(Float64(s.eps), Float64(delta))
                win_size *= 2
                next_end = win_end + win_size
                # stretch the last window to the terminal buffer
                if next_end + 2win_size > nwarm - tb
                    next_end = nwarm - tb
                end
                win_end = next_end
            end
        end
    end
    nwarm > 0 && (s.eps = T(exp(da.lebar)))
    draws = Matrix{Float64}(undef, s.d, nkeep)
    st = Dict(k => Vector{Float64}(undef, nkeep) for k in
              ("lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__", "energy__"))
    for k in 1:nkeep
        local dep, nl, acc, div, en
        for _ in 1:thin
            dep, nl, acc, div, en = _transition!(s)
        end
        draws[:, k] .= s.z.x
        st["lp__"][k] = s.lp; st["accept_stat__"][k] = acc; st["stepsize__"][k] = s.eps
        st["treedepth__"][k] = dep; st["n_leapfrog__"][k] = nl; st["divergent__"][k] = div; st["energy__"][k] = en
    end
    return draws, st
end
