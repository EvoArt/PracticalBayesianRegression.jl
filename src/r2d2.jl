# =============================================================================
# The R2D2 prior on population-level effects, as brms's
# `R2D2(mean_R2, prec_R2, cons_D2, autoscale)` (Zhang et al. 2022):
#
#     b[j] = zb[j] * sqrt(phi[j] * tau2),   tau2 = sigma^2 * R2 / (1 - R2)  (sigma if autoscale)
#     zb ~ N(0, 1),  R2 ~ Beta(mean_R2 * prec_R2, (1 - mean_R2) * prec_R2),
#     phi ~ Dirichlet(cons_D2)  (a simplex over the coefficients)
#
# R2 is sampled on the logit scale, phi through Stan's stick-breaking
# simplex (special.jl's transform and its closed-form reverse pass).
#     ds/dphi[j] = s[j] / (2 phi[j]),  ds/dtau2 = s / (2 tau2),
#     dtau2/dR2 = sigma^2 / (1 - R2)^2, dtau2/dlog sigma = 2 tau2
# =============================================================================

struct R2D2Prior
    mean_R2::Float64
    prec_R2::Float64
    cons_D2::Float64
    autoscale::Bool
end
R2D2(mean_R2=0.5, prec_R2=2, cons_D2=0.5; autoscale=true) = R2D2Prior(mean_R2, prec_R2, cons_D2, autoscale)
Base.show(io::IO, p::R2D2Prior) = print(io, "R2D2($(_g(p.mean_R2)), $(_g(p.prec_R2)), $(_g(p.cons_D2)))")
to_prior(p::R2D2Prior) = p

struct R2D2Term
    on::Bool
    a::Float64                  # Beta parameters of R2
    b::Float64
    cons::Float64               # Dirichlet concentration
    r2::Int                     # θ index of logit R2
    phi::UnitRange{Int}         # K - 1 stick-breaking parameters
    sigma::Base.RefValue{Int}   # θ index of log sigma (autoscale), 0 = none, -1 = to be set
end
R2D2Term() = R2D2Term(false, 0.0, 0.0, 0.0, 0, 1:0, Ref(0))

@inline _r2d2_sig2(t::R2D2Term, θ) = t.sigma[] == 0 ? one(float(θ[t.r2])) : exp(2θ[t.sigma[]])

function _r2d2_scales(t::R2D2Term, θ, K)
    R2 = logistic(θ[t.r2])
    tau2 = _r2d2_sig2(t, θ) * R2 / (1 - R2)
    phi, _ = _stick(θ[t.phi], K)
    return sqrt.(phi .* tau2)
end

function _r2d2_bvals!(bvals, t::R2D2Term, b::UnitRange{Int}, θ)
    s = _r2d2_scales(t, θ, length(b))
    @inbounds for (j, k) in enumerate(b)
        bvals[j] = θ[k] * s[j]
    end
    return bvals
end

# g[b] holds d loglik / d b on entry; returns the prior value
function _r2d2_pullback!(g, t::R2D2Term, b::UnitRange{Int}, θ)
    T = eltype(g)
    K = length(b)
    u = θ[t.r2]
    R2 = logistic(u)
    sig2 = _r2d2_sig2(t, θ)
    tau2 = sig2 * R2 / (1 - R2)
    phi = zeros(T, K); rem = zeros(T, K); z = zeros(T, max(K - 1, 0))
    _stick!(phi, rem, z, view(θ, t.phi))
    lp = zero(T)
    dphi = zeros(T, K)
    dtau2 = zero(T)
    @inbounds for (j, k) in enumerate(b)
        s = sqrt(phi[j] * tau2)
        zb = θ[k]
        db = g[k]
        g[k] = db * s - zb
        lp -= zb * zb / 2
        ds = db * zb
        dphi[j] = ds * s / (2phi[j])
        dtau2 += ds * s / (2tau2)
    end
    # phi: Dirichlet(cons) + stick-breaking Jacobian, chained into its parameters
    gu = zeros(T, length(t.phi))
    lp += _stick_grad!(gu, dphi, phi, rem, z, t.cons)
    @inbounds for (n, k) in enumerate(t.phi)
        g[k] += gu[n]
    end
    # R2: Beta(a, b) kernel + logit Jacobian
    ta, tb = _pv(u, t.a), _pv(u, t.b)
    dR2 = dtau2 * sig2 / (1 - R2)^2 + (ta - 1) / R2 - (tb - 1) / (1 - R2)
    g[t.r2] += dR2 * R2 * (1 - R2) + (1 - 2R2)
    lp += (ta - 1) * log(R2) + (tb - 1) * log1p(-R2) + log(R2) + log1p(-R2)
    t.sigma[] != 0 && (g[t.sigma[]] += dtau2 * 2tau2)
    return lp
end

function _r2d2_prior_generic(t::R2D2Term, b::UnitRange{Int}, θ)
    K = length(b)
    lp = zero(eltype(θ))
    for k in b
        lp -= θ[k]^2 / 2
    end
    phi, lj = _stick(θ[t.phi], K)
    R2 = logistic(θ[t.r2])
    lp += lj + (_pv(R2, t.cons) - 1) * sum(log.(phi))
    lp += (_pv(R2, t.a) - 1) * log(R2) + (_pv(R2, t.b) - 1) * log1p(-R2) + log(R2) + log1p(-R2)
    return lp
end
