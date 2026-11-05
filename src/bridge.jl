# =============================================================================
# Marginal likelihoods by bridge sampling (brms `bridge_sampler`, from the
# bridgesampling package: Meng & Wong's iterative estimator with a normal
# proposal fitted to half the posterior draws), and `bayes_factor` /
# `post_prob` from them.
#
# The sampling density drops every constant (data constants, prior
# normalisers, truncation constants, standard-normal constants); a marginal
# likelihood needs them all. `_log_normconst` adds them back, walking the
# model's parameters; structures whose constants are not tracked yet raise an
# error rather than give a wrong answer. Flat (improper) priors make the
# marginal likelihood undefined, as brms warns: they are refused.
# =============================================================================

# log of LKJ's normaliser 1 / c_K(eta) (Lewandowski, Kurowicka & Joe 2009):
#   c_K = 2^(sum_k (2 eta - 2 + K - k)(K - k)) prod_k B(eta + (K - k - 1)/2, eta + (K - k - 1)/2)^(K - k)
function _lkj_lognorm(K::Int, η)
    lc = 0.0
    for k in 1:(K - 1)
        a = η + (K - k - 1) / 2
        lc += (2η - 2 + K - k) * (K - k) * log(2) + (K - k) * logbeta(a, a)
    end
    return -lc
end

function _prior_const(pr::ScalarPrior, what)
    isflat(pr) && throw(ArgumentError("bridge_sampler: the prior of $what is flat (improper), so the marginal " *
                                      "likelihood is undefined; give it a proper prior"))
    isconstant(pr) && throw(ArgumentError("bridge_sampler: constant() priors are not supported"))
    return pr.lnorm + pr.lconst
end

const _HALF_LOG_2PI = log(2π) / 2

_pred_const(p, ld) = throw(ArgumentError("bridge_sampler: models with $(nameof(typeof(p))) blocks are not supported yet"))
_pred_const(p::ScalarDpar, ld) = _prior_const(p.prior, string(p.dpar))
function _pred_const(p::Thresholds, ld)
    return sum(pr -> _prior_const(pr, "a threshold"), p.priors; init=0.0)
end
function _pred_const(p::LinPred, ld)
    (isempty(p.mo) && isempty(p.me) && isempty(p.gp) && isempty(p.mi) && isempty(p.car) && isempty(p.lac) &&
     !_has_ar(p.ar) && !p.hs.on && !p.r2d2.on) ||
        throw(ArgumentError("bridge_sampler: mo / me / gp / mi / car / autocorrelation / horseshoe / R2D2 terms are not supported yet"))
    c = p.intercept > 0 ? _prior_const(p.int_prior, "the intercept") : 0.0
    for pr in p.b_priors
        c += _prior_const(pr, "a coefficient")
    end
    for gb in p.groups
        _owns(gb) || continue
        _student(gb) && throw(ArgumentError("bridge_sampler: student-t group effects are not supported yet"))
        for pr in gb.sd_priors
            c += _prior_const(pr, "a group-level sd")
        end
        _has_corr(gb) && (c += _lkj_lognorm(_q(gb), gb.lkj))
        c -= _HALF_LOG_2PI * length(gb.z)
    end
    for sb in p.smooths
        c += _prior_const(sb.sds_prior, "a smooth's sds") - _HALF_LOG_2PI * length(sb.zs)
    end
    return c
end

"""
    _log_normconst(ld) -> C

The constant the sampling density drops: logdensity(ld, θ) + C is the full log
of p(y | θ) p(θ) |dθ/du| on the unconstrained scale.
"""
function _log_normconst(ld::GLMDensity)
    (isempty(ld.ordered) && isempty(ld.pinned) && !_mi_sampled(ld) && isempty(ld.Jmi)) ||
        throw(ArgumentError("bridge_sampler: ordered mixtures, constant() priors and missing values are not supported yet"))
    ld.family isa Union{TimeCov,Mixture,_MultiCat,Cox,GenExtremeValue} &&
        throw(ArgumentError("bridge_sampler: the $(family_name(ld.family)) family is not supported yet"))
    return Float64(ld.lconst) + sum(p -> _pred_const(p, ld), ld.preds; init=0.0)
end
function _log_normconst(mv::MVDensity)
    (mv.rescor || !isempty(mv.marg) || mv.cross) &&
        throw(ArgumentError("bridge_sampler: residual correlations, integrated missing values and cross-response IDs are not supported yet"))
    return sum(_log_normconst, mv.subs)
end

"""
    bridge_sampler(fit; maxiter = 1000, tol = 1e-10, rng = Xoshiro(1)) -> log marginal likelihood

brms / bridgesampling's estimator: a multivariate normal proposal fitted to
the first half of the (unconstrained) draws, the iterative Meng & Wong scheme
on the second half against as many proposal draws.
"""
function bridge_sampler(fit; maxiter::Int=1000, tol=1e-10, rng=Random.Xoshiro(1))
    dens = fit.density
    C = _log_normconst(dens)
    dim = LogDensityProblems.dimension(dens)
    U = reshape(fit.draws, size(fit.draws, 1), :)[1:dim, :]
    S = size(U, 2)
    S >= 40 || throw(ArgumentError("bridge_sampler needs more draws"))
    i1 = 1:2:S; i2 = 2:2:S                    # interleaved halves (chains mixed)
    μ = vec(mean(U[:, i1]; dims=2))
    Σ = Symmetric(cov(U[:, i1]; dims=2) + 1e-10I)
    G = Distributions.MvNormal(μ, Σ)
    post = U[:, i2]; N1 = size(post, 2)
    prop = rand(rng, G, N1); N2 = N1
    logq(θ) = LogDensityProblems.logdensity(dens, θ) + C
    l1 = [logq(view(post, :, i)) - Distributions.logpdf(G, view(post, :, i)) for i in 1:N1]
    l2 = [logq(view(prop, :, j)) - Distributions.logpdf(G, view(prop, :, j)) for j in 1:N2]
    l2[.!isfinite.(l2)] .= -Inf
    lstar = median(l1)
    s1 = N1 / (N1 + N2); s2 = N2 / (N1 + N2)
    e1 = exp.(l1 .- lstar); e2 = exp.(l2 .- lstar)
    r = 1.0; logr = 0.0
    for _ in 1:maxiter
        num = mean(e2 ./ (s1 .* e2 .+ s2 * r))
        den = mean(1 ./ (s1 .* e1 .+ s2 * r))
        rn = num / den
        lrn = log(rn)
        done = abs(lrn - logr) < tol
        r, logr = rn, lrn
        done && break
    end
    return logr + lstar
end

"""
    bayes_factor(fit1, fit2) -> (bf, log_bf)

brms `bayes_factor`: the ratio of the bridge-sampled marginal likelihoods.
"""
function bayes_factor(f1, f2; kw...)
    l = bridge_sampler(f1; kw...) - bridge_sampler(f2; kw...)
    return (bf=exp(l), log_bf=l)
end

"""
    post_prob(fits...; prior_prob = equal) -> Vector

brms `post_prob`: posterior model probabilities from bridge-sampled marginal likelihoods.
"""
function post_prob(fits...; prior_prob=fill(1 / length(fits), length(fits)), kw...)
    l = [bridge_sampler(f; kw...) for f in fits] .+ log.(prior_prob)
    w = exp.(l .- maximum(l))
    return w ./ sum(w)
end
