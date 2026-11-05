# =============================================================================
# Link functions.
#
# A family's mean parameter lives on the scale of its link; `linkinv` maps
# the linear predictor to the mean and `mu_eta` is d mean / d eta, which is
# all the generic (non-canonical-link) route in families.jl needs to chain a
# mean-scale derivative back to the linear predictor.
#
# Canonical links (logit for bernoulli, log for poisson, ...) do NOT go
# through here on the hot path: their derivatives simplify (y - mu) and the
# families specialise on them. Probit and cloglog get their own stable
# specialisations for the binary families too, since the generic route would
# take log(Phi(eta)) of an underflowed probability in the tails.
# =============================================================================

abstract type Link end
struct IdentityLink <: Link end
struct LogLink <: Link end
struct LogitLink <: Link end
struct ProbitLink <: Link end
struct CloglogLink <: Link end
struct SqrtLink <: Link end
struct InverseLink <: Link end
struct InverseSquaredLink <: Link end      # brms's "1/mu^2" (inverse.gaussian)
struct TanHalfLink <: Link end             # brms's "tan_half" (von_mises): mu = 2 atan(eta)
# brms 2.23's further links (inverses as in its Stan code):
struct SoftplusLink <: Link end            # mu = log1p(exp(eta))
struct SquareplusLink <: Link end          # mu = (eta + sqrt(eta^2 + 4)) / 2
struct ProbitApproxLink <: Link end        # mu = Stan's Phi_approx(eta) = logistic(0.07056 eta^3 + 1.5976 eta)
struct CauchitLink <: Link end             # mu = atan(eta) / pi + 1/2
struct SoftitLink <: Link end              # mu = s / (1 + s), s = log1p(exp(eta)) (brms inv_softit)

link_name(::IdentityLink) = "identity"
link_name(::LogLink) = "log"
link_name(::LogitLink) = "logit"
link_name(::ProbitLink) = "probit"
link_name(::CloglogLink) = "cloglog"
link_name(::SqrtLink) = "sqrt"
link_name(::InverseLink) = "inverse"
link_name(::InverseSquaredLink) = "1/mu^2"
link_name(::TanHalfLink) = "tan_half"
link_name(::SoftplusLink) = "softplus"
link_name(::SquareplusLink) = "squareplus"
link_name(::ProbitApproxLink) = "probit_approx"
link_name(::CauchitLink) = "cauchit"
link_name(::SoftitLink) = "softit"

const _LINKS = Dict{Symbol,Link}(
    :identity => IdentityLink(), :log => LogLink(), :logit => LogitLink(),
    :probit => ProbitLink(), :cloglog => CloglogLink(), :sqrt => SqrtLink(),
    :inverse => InverseLink(), :inverse_squared => InverseSquaredLink(),
    Symbol("1/mu^2") => InverseSquaredLink(), :tan_half => TanHalfLink(),
    :softplus => SoftplusLink(), :squareplus => SquareplusLink(), :probit_approx => ProbitApproxLink(),
    :cauchit => CauchitLink(), :softit => SoftitLink(),
)
function _link(s)
    l = get(_LINKS, Symbol(s), nothing)
    l === nothing && throw(ArgumentError("unknown link `$s` (known: $(join(sort(collect(keys(_LINKS))), ", ")))"))
    return l
end

@inline linkinv(::IdentityLink, e) = e
@inline linkinv(::LogLink, e) = exp(e)
@inline linkinv(::LogitLink, e) = logistic(e)
@inline linkinv(::ProbitLink, e) = normcdf(e)
@inline linkinv(::CloglogLink, e) = -expm1(-exp(e))
@inline linkinv(::SqrtLink, e) = e * e
@inline linkinv(::InverseLink, e) = inv(e)
@inline linkinv(::InverseSquaredLink, e) = inv(sqrt(e))
@inline linkinv(::TanHalfLink, e) = 2 * atan(e)
@inline linkinv(::SoftplusLink, e) = log1pexp(e)
@inline linkinv(::SquareplusLink, e) = (e + sqrt(e * e + 4)) / 2
@inline linkinv(::ProbitApproxLink, e) = logistic(0.07056 * e^3 + 1.5976 * e)
@inline linkinv(::CauchitLink, e) = atan(e) / π + 0.5
@inline linkinv(::SoftitLink, e) = (s = log1pexp(e); s / (1 + s))

@inline mu_eta(::IdentityLink, e) = one(e)
@inline mu_eta(::LogLink, e) = exp(e)
@inline mu_eta(::LogitLink, e) = (s = logistic(e); s * (1 - s))
@inline mu_eta(::ProbitLink, e) = exp(-e * e / 2) / sqrt(2π)
@inline mu_eta(::CloglogLink, e) = exp(e - exp(e))
@inline mu_eta(::SqrtLink, e) = 2e
@inline mu_eta(::InverseLink, e) = -inv(e * e)
@inline mu_eta(::InverseSquaredLink, e) = -inv(2 * e * sqrt(e))
@inline mu_eta(::TanHalfLink, e) = 2 / (1 + e * e)
@inline mu_eta(::SoftplusLink, e) = logistic(e)
@inline mu_eta(::SquareplusLink, e) = (1 + e / sqrt(e * e + 4)) / 2
@inline mu_eta(::ProbitApproxLink, e) = (a = 0.07056 * e^3 + 1.5976 * e; s = logistic(a); s * (1 - s) * (3 * 0.07056 * e^2 + 1.5976))
@inline mu_eta(::CauchitLink, e) = 1 / (π * (1 + e * e))
@inline mu_eta(::SoftitLink, e) = (s = log1pexp(e); logistic(e) / (1 + s)^2)

# --- normal cdf helpers, stable in the tails ---------------------------------
const _INVSQRT2 = 1 / sqrt(2)
@inline normcdf(x) = erfc(-x * _INVSQRT2) / 2
@inline normlogpdf(x) = -x * x / 2 - 0.5 * log(2π)
# log Phi(x): for x < -1 use the scaled complementary error function, which
# does not underflow (Phi(-40) is 1e-350 but its log is fine)
@inline function normlogcdf(x)
    if x < -1
        return log(erfcx(-x * _INVSQRT2) / 2) - x * x / 2
    else
        return log1p(-erfc(x * _INVSQRT2) / 2)
    end
end
# inverse Mills ratio phi(x) / Phi(x)
@inline mills(x) = exp(normlogpdf(x) - normlogcdf(x))

# safe log for mean-scale routes whose link can leave the support
# (identity link on a positive mean): -Inf is a rejected proposal, a
# DomainError would kill the chain
@inline _slog(x) = x > 0 ? log(x) : oftype(float(x), -Inf)
