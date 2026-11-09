# =============================================================================
# The regularised horseshoe prior on population-level effects, as brms's
# `horseshoe(df, scale_global, df_global, scale_slab, df_slab, autoscale)`
# (Piironen & Vehtari 2017; brms's Stan code):
#
#     b[j]   = zb[j] * s[j],      s[j] = tau * lambda[j] * sqrt(c2 / (c2 + tau^2 lambda[j]^2))
#     zb     ~ N(0, 1)            lambda ~ half-t(df, 0, 1)
#     tau    ~ half-t(df_global, 0, scale_global * sigma)      (sigma if autoscale)
#     c2     = scale_slab^2 * slab,   slab ~ inv_gamma(df_slab / 2, df_slab / 2)
#
# Closed-form partials (q = sqrt(c2 / (c2 + a)), a = tau^2 lambda^2):
#     ds/dlambda = tau q^3,   ds/dtau = lambda q^3,   ds/dc2 = s a / (2 c2 (c2 + a))
# The global scale's prior involves sigma, so with autoscale its gradient also
# flows into log sigma.
# =============================================================================

"""
    HorseshoePrior

brms's `horseshoe()` prior specification (class b, all coefficients of one
distributional parameter).
"""
struct HorseshoePrior
    df::Float64
    scale_global::Float64
    df_global::Float64
    scale_slab::Float64
    df_slab::Float64
    par_ratio::Float64          # NaN unless given: then scale_global = par_ratio / sqrt(N)
    autoscale::Bool
end

horseshoe(df=1; scale_global=1, df_global=1, scale_slab=2, df_slab=4, par_ratio=NaN, autoscale=true) =
    HorseshoePrior(df, scale_global, df_global, scale_slab, df_slab, Float64(par_ratio), autoscale)

Base.show(io::IO, h::HorseshoePrior) =
    print(io, "horseshoe($(_g(h.df)), scale_global = $(_g(h.scale_global)), df_global = $(_g(h.df_global)), ",
          "scale_slab = $(_g(h.scale_slab)), df_slab = $(_g(h.df_slab)))")
to_prior(p::HorseshoePrior) = p

struct HSTerm
    on::Bool
    df::Float64
    scale_global::Float64
    df_global::Float64
    scale_slab::Float64
    df_slab::Float64
    glob::Int                   # θ index of log tau
    slab::Int                   # θ index of log slab
    loc::UnitRange{Int}         # θ indices of log lambda
    sigma::Base.RefValue{Int}   # θ index of log sigma for autoscale (set after build), 0 = none
end
HSTerm() = HSTerm(false, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 1:0, Ref(0))

# global scale of tau's prior, and log sigma for the gradient
@inline function _hs_gscale(h::HSTerm, θ)
    h.sigma[] == 0 && return oftype(float(θ[h.glob]), h.scale_global)
    return _pv(θ[h.glob], h.scale_global) * exp(θ[h.sigma[]])
end

function _hs_scales(h::HSTerm, θ)
    tau = exp(θ[h.glob])
    c2 = _pv(tau, h.scale_slab)^2 * exp(θ[h.slab])
    return [(lam = exp(θ[k]); tau * lam * sqrt(c2 / (c2 + tau^2 * lam^2))) for k in h.loc]
end

# fills bvals = zb .* s (analytic path)
function _hs_bvals!(bvals, h::HSTerm, b::UnitRange{Int}, θ)
    tau = exp(θ[h.glob])
    c2 = _pv(tau, h.scale_slab)^2 * exp(θ[h.slab])
    @inbounds for (j, k) in enumerate(b)
        lam = exp(θ[h.loc[j]])
        bvals[j] = θ[k] * tau * lam * sqrt(c2 / (c2 + tau^2 * lam^2))
    end
    return bvals
end

# g[b] holds d loglik / d b on entry; replaces it with d / d zb and adds the
# gradients of the horseshoe hyperparameters; returns the prior value
function _hs_pullback!(g, h::HSTerm, b::UnitRange{Int}, θ)
    T = eltype(g)
    ut, us = θ[h.glob], θ[h.slab]
    tau = exp(ut)
    slab = exp(us)
    ss2 = _pv(tau, h.scale_slab)^2
    c2 = ss2 * slab
    lp = zero(T)
    dtau = zero(T); dc2 = zero(T)
    nu = _pv(tau, h.df)
    @inbounds for (j, k) in enumerate(b)
        ul = θ[h.loc[j]]
        lam = exp(ul)
        a = tau^2 * lam^2
        D = c2 + a
        q = sqrt(c2 / D)
        s = tau * lam * q
        zb = θ[k]
        db = g[k]
        g[k] = db * s - zb
        lp -= zb * zb / 2
        ds = db * zb
        q3 = q^3
        # lambda: half-t(nu, 0, 1) kernel, log transform
        dl = ds * tau * q3 - (nu + 1) * lam / (nu + lam^2)
        g[h.loc[j]] += dl * lam + 1
        lp += -(nu + 1) / 2 * log1p(lam^2 / nu) + ul
        dtau += ds * lam * q3
        dc2 += ds * s * a / (2 * c2 * D)
    end
    # tau: half-t(df_global, 0, scale), scale = scale_global (* sigma)
    sc = _hs_gscale(h, θ)
    ng = _pv(tau, h.df_global)
    r2 = (tau / sc)^2
    lp += -(ng + 1) / 2 * log1p(r2 / ng) - log(sc) + ut
    g[h.glob] += (dtau - (ng + 1) * tau / (ng * sc^2 + tau^2)) * tau + 1
    if h.sigma[] != 0
        g[h.sigma[]] += (ng + 1) * tau^2 / (ng * sc^2 + tau^2) - 1
    end
    # slab: inv_gamma(df_slab/2, df_slab/2) on `slab`
    al = _pv(tau, h.df_slab) / 2
    lp += -(al + 1) * us - al / slab + us
    g[h.slab] += (dc2 * ss2 - (al + 1) / slab + al / slab^2) * slab + 1
    return lp
end

# generic path: the same prior terms, any number type
function _hs_prior_generic(h::HSTerm, b::UnitRange{Int}, θ)
    x0 = θ[h.glob]
    nu, ng, al = _pv(x0, h.df), _pv(x0, h.df_global), _pv(x0, h.df_slab) / 2
    lp = zero(eltype(θ))
    for (j, k) in enumerate(b)
        lp -= θ[k]^2 / 2
        ul = θ[h.loc[j]]
        lp += -(nu + 1) / 2 * log1p(exp(2ul) / nu) + ul
    end
    ut, us = θ[h.glob], θ[h.slab]
    sc = _hs_gscale(h, θ)
    lp += -(ng + 1) / 2 * log1p((exp(ut) / sc)^2 / ng) - log(sc) + ut
    lp += -(al + 1) * us - al / exp(us) + us
    return lp
end
