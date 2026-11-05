# =============================================================================
# brms families added late (dual-number gradients, see dualfamilies.jl):
#
# zero_inflated_beta_binomial (brms 2.21+): mu (logit), phi (log), zi (logit), trials
#     y = 0:  log(zi + (1 - zi) BB(0 | n, mu, phi))
#     y > 0:  log(1 - zi) + log BB(y | n, mu, phi)
#     BB(y) = choose(n, y) B(y + mu phi, n - y + (1 - mu) phi) / B(mu phi, (1 - mu) phi)
# The binomial coefficient is a data constant (dropped from the sampling
# density, added back by log_lik) except inside the y = 0 mixture, where it is 1.
# =============================================================================

struct ZeroInflatedBetaBinomial{L<:Link} <: DualFamily end

"""
    zero_inflated_beta_binomial(; link = :logit)

brms's `zero_inflated_beta_binomial()` (mu, phi, zi); the response needs `| trials(n)`.
"""
function zero_inflated_beta_binomial(; link=nothing)
    l = link === nothing ? :logit : Symbol(link)
    l in (:logit, :probit, :cloglog, :probit_approx, :cauchit, :softit, :identity, :log) ||
        throw(ArgumentError("zero_inflated_beta_binomial: unsupported link `$l`"))
    return ZeroInflatedBetaBinomial{typeof(_link(l))}()
end
link(::ZeroInflatedBetaBinomial{L}) where {L} = L()
family_name(::ZeroInflatedBetaBinomial) = "zero_inflated_beta_binomial"
dpars(::ZeroInflatedBetaBinomial) = (:mu, :phi, :zi)
uses_trials(::ZeroInflatedBetaBinomial) = true
_xdata(::ZeroInflatedBetaBinomial, y, trials) = Float64.(trials)
_check_response(::ZeroInflatedBetaBinomial, y, t) =
    _chk(all(i -> isinteger(y[i]) && 0 <= y[i] <= t[i], eachindex(y)), "binomial responses must be integers in 0:trials")
_ll_const(::ZeroInflatedBetaBinomial, y, t) = y > 0 ? logabsbinomial(Int(t), Int(y))[1] : 0.0

_lbeta_(a, b) = loggamma(a) + loggamma(b) - loggamma(a + b)
function _ll(f::ZeroInflatedBetaBinomial, y, n, eta)
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2]); zi = logistic(eta[3])
    a, b = mu * phi, (1 - mu) * phi
    lbb = _lbeta_(y + a, n - y + b) - _lbeta_(a, b)          # without choose(n, y)
    return y == 0 ? log(zi + (1 - zi) * exp(lbb)) : log1p(-zi) + lbb
end
function _mean(f::ZeroInflatedBetaBinomial, eta, t)
    return (1 - logistic(eta[3])) * t * linkinv(link(f), eta[1])
end
function _rand(rng, f::ZeroInflatedBetaBinomial, eta, t)
    rand(rng) < logistic(eta[3]) && return 0.0
    mu = linkinv(link(f), eta[1]); phi = exp(eta[2])
    p = rand(rng, Distributions.Beta(mu * phi, (1 - mu) * phi))
    return Float64(rand(rng, Distributions.Binomial(Int(t), p)))
end
