# =============================================================================
# brms's distribution functions (dstudent_t, pskew_normal, rwiener, ...), with
# brms's argument order and parameterisations, for scalars (broadcast them with
# a dot: `dskew_normal.(x, 0, 1, 2)`). Densities take `log = false`;
# distribution functions `lower_tail = true, log_p = false`, as R's
# `lower.tail` / `log.p`; random draws an `rng` keyword. Built on Distributions.jl
# where it has the distribution, otherwise on the family code the models use
# (the same log densities, cdfs and samplers).
# =============================================================================

const _DD = Distributions
_dout(lp, logd::Bool) = logd ? lp : exp(lp)
_pout(lF, lS, lower_tail::Bool, log_p::Bool) = (v = lower_tail ? lF : lS; log_p ? v : exp(v))
# quantile by bisection on a cdf (monotone), for the families without a closed form
function _qbisect(F, p; lo=-1.0, hi=1.0)
    while F(lo) > p; lo = 2lo - hi; end
    while F(hi) < p; hi = 2hi - lo; end
    for _ in 1:200
        m = (lo + hi) / 2
        F(m) < p ? (lo = m) : (hi = m)
        hi - lo <= 1e-12 * max(1.0, abs(m)) && break
    end
    return (lo + hi) / 2
end
_qp(p, lower_tail, log_p) = (q = log_p ? exp(p) : p; lower_tail ? q : 1 - q)

# --- student_t(df, mu, sigma) --------------------------------------------------------
_st(df, mu, sigma) = mu + sigma * _DD.TDist(df)
"""
    dstudent_t(x, df, mu = 0, sigma = 1; log = false)
    pstudent_t(q, df, mu = 0, sigma = 1; lower_tail = true, log_p = false)
    qstudent_t(p, df, mu = 0, sigma = 1)
    rstudent_t(df, mu = 0, sigma = 1; rng)

brms's location-scale student-t distribution.
"""
dstudent_t(x, df, mu=0, sigma=1; log::Bool=false) = _dout(_DD.logpdf(_st(df, mu, sigma), x), log)
pstudent_t(q, df, mu=0, sigma=1; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_DD.logcdf(_st(df, mu, sigma), q), _DD.logccdf(_st(df, mu, sigma), q), lower_tail, log_p)
qstudent_t(p, df, mu=0, sigma=1; lower_tail::Bool=true, log_p::Bool=false) = _DD.quantile(_st(df, mu, sigma), _qp(p, lower_tail, log_p))
rstudent_t(df, mu=0, sigma=1; rng=Random.default_rng()) = rand(rng, _st(df, mu, sigma))
@doc (@doc dstudent_t) pstudent_t
@doc (@doc dstudent_t) qstudent_t
@doc (@doc dstudent_t) rstudent_t

# --- skew_normal (brms: mean mu, sd sigma, skewness alpha) ---------------------------
const _snfam = SkewNormal{IdentityLink}()
_sneta(mu, sigma, alpha) = (Float64(mu), Base.log(Float64(sigma)), Float64(alpha))
"""
    dskew_normal(x, mu = 0, sigma = 1, alpha = 0; log = false)
    pskew_normal(q, mu = 0, sigma = 1, alpha = 0; lower_tail = true, log_p = false)
    qskew_normal(p, mu = 0, sigma = 1, alpha = 0)
    rskew_normal(mu = 0, sigma = 1, alpha = 0; rng)

brms's skew normal, parameterised by its mean `mu`, standard deviation
`sigma` and skewness `alpha` (as the `skew_normal()` family).
"""
dskew_normal(x, mu=0, sigma=1, alpha=0; log::Bool=false) = _dout(_ll(_snfam, x, 0.0, _sneta(mu, sigma, alpha)), log)
pskew_normal(q, mu=0, sigma=1, alpha=0; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_cc_lcdf(_snfam, q, _sneta(mu, sigma, alpha)), _cc_lccdf(_snfam, q, _sneta(mu, sigma, alpha)), lower_tail, log_p)
qskew_normal(p, mu=0, sigma=1, alpha=0; lower_tail::Bool=true, log_p::Bool=false) =
    _qbisect(v -> pskew_normal(v, mu, sigma, alpha), _qp(p, lower_tail, log_p); lo=mu - 10sigma, hi=mu + 10sigma)
rskew_normal(mu=0, sigma=1, alpha=0; rng=Random.default_rng()) = _rand(rng, _snfam, _sneta(mu, sigma, alpha), 0)
@doc (@doc dskew_normal) pskew_normal
@doc (@doc dskew_normal) qskew_normal
@doc (@doc dskew_normal) rskew_normal

# --- exgaussian (brms: mean mu, sigma, beta = exponential scale) --------------------
const _exgfam = ExGaussian{IdentityLink}()
_exgeta(mu, sigma, beta) = (Float64(mu), Base.log(Float64(sigma)), Base.log(Float64(beta)))
"""
    dexgaussian(x, mu, sigma, beta; log = false)
    pexgaussian(q, mu, sigma, beta; lower_tail = true, log_p = false)
    rexgaussian(mu, sigma, beta; rng)

brms's exponentially modified gaussian: `mu` is the mean, `beta` the scale of
the exponential part.
"""
dexgaussian(x, mu, sigma, beta; log::Bool=false) = _dout(_ll(_exgfam, x, 0.0, _exgeta(mu, sigma, beta)), log)
pexgaussian(q, mu, sigma, beta; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_cc_lcdf(_exgfam, q, _exgeta(mu, sigma, beta)), _cc_lccdf(_exgfam, q, _exgeta(mu, sigma, beta)), lower_tail, log_p)
rexgaussian(mu, sigma, beta; rng=Random.default_rng()) = _rand(rng, _exgfam, _exgeta(mu, sigma, beta), 0)
@doc (@doc dexgaussian) pexgaussian
@doc (@doc dexgaussian) rexgaussian

# --- frechet(loc, scale, shape) --------------------------------------------------------
_fr(loc, scale, shape) = loc + _DD.Frechet(shape, scale)
"""
    dfrechet(x, loc = 0, scale = 1, shape = 1; log = false)
    pfrechet(q, loc = 0, scale = 1, shape = 1; lower_tail = true, log_p = false)
    qfrechet(p, loc = 0, scale = 1, shape = 1)
    rfrechet(loc = 0, scale = 1, shape = 1; rng)

brms's Fréchet distribution with location, scale and shape.
"""
dfrechet(x, loc=0, scale=1, shape=1; log::Bool=false) = _dout(_DD.logpdf(_fr(loc, scale, shape), x), log)
pfrechet(q, loc=0, scale=1, shape=1; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_DD.logcdf(_fr(loc, scale, shape), q), _DD.logccdf(_fr(loc, scale, shape), q), lower_tail, log_p)
qfrechet(p, loc=0, scale=1, shape=1; lower_tail::Bool=true, log_p::Bool=false) = _DD.quantile(_fr(loc, scale, shape), _qp(p, lower_tail, log_p))
rfrechet(loc=0, scale=1, shape=1; rng=Random.default_rng()) = rand(rng, _fr(loc, scale, shape))
@doc (@doc dfrechet) pfrechet
@doc (@doc dfrechet) qfrechet
@doc (@doc dfrechet) rfrechet

# --- gen_extreme_value(mu, sigma, xi) ---------------------------------------------------
_gev(mu, sigma, xi) = _DD.GeneralizedExtremeValue(mu, sigma, xi)
"""
    dgen_extreme_value(x, mu = 0, sigma = 1, xi = 0; log = false)
    pgen_extreme_value(q, mu = 0, sigma = 1, xi = 0; lower_tail = true, log_p = false)
    qgen_extreme_value(p, mu = 0, sigma = 1, xi = 0)
    rgen_extreme_value(mu = 0, sigma = 1, xi = 0; rng)

The generalised extreme value distribution, as brms's.
"""
dgen_extreme_value(x, mu=0, sigma=1, xi=0; log::Bool=false) = _dout(_DD.logpdf(_gev(mu, sigma, xi), x), log)
pgen_extreme_value(q, mu=0, sigma=1, xi=0; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_DD.logcdf(_gev(mu, sigma, xi), q), _DD.logccdf(_gev(mu, sigma, xi), q), lower_tail, log_p)
qgen_extreme_value(p, mu=0, sigma=1, xi=0; lower_tail::Bool=true, log_p::Bool=false) = _DD.quantile(_gev(mu, sigma, xi), _qp(p, lower_tail, log_p))
rgen_extreme_value(mu=0, sigma=1, xi=0; rng=Random.default_rng()) = rand(rng, _gev(mu, sigma, xi))
@doc (@doc dgen_extreme_value) pgen_extreme_value
@doc (@doc dgen_extreme_value) qgen_extreme_value
@doc (@doc dgen_extreme_value) rgen_extreme_value

# --- inverse gaussian(mu, shape) --------------------------------------------------------
"""
    dinv_gaussian(x, mu = 1, shape = 1; log = false)
    pinv_gaussian(q, mu = 1, shape = 1; lower_tail = true, log_p = false)
    rinv_gaussian(mu = 1, shape = 1; rng)

The inverse gaussian distribution (mean `mu`, shape `shape`), as brms's.
"""
dinv_gaussian(x, mu=1, shape=1; log::Bool=false) = _dout(_DD.logpdf(_DD.InverseGaussian(mu, shape), x), log)
pinv_gaussian(q, mu=1, shape=1; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_DD.logcdf(_DD.InverseGaussian(mu, shape), q), _DD.logccdf(_DD.InverseGaussian(mu, shape), q), lower_tail, log_p)
rinv_gaussian(mu=1, shape=1; rng=Random.default_rng()) = rand(rng, _DD.InverseGaussian(mu, shape))
@doc (@doc dinv_gaussian) pinv_gaussian
@doc (@doc dinv_gaussian) rinv_gaussian

# --- shifted lognormal(meanlog, sdlog, shift) -------------------------------------------
_sln(meanlog, sdlog, shift) = shift + _DD.LogNormal(meanlog, sdlog)
"""
    dshifted_lnorm(x, meanlog = 0, sdlog = 1, shift = 0; log = false)
    pshifted_lnorm(q, meanlog = 0, sdlog = 1, shift = 0; lower_tail = true, log_p = false)
    qshifted_lnorm(p, meanlog = 0, sdlog = 1, shift = 0)
    rshifted_lnorm(meanlog = 0, sdlog = 1, shift = 0; rng)

The shifted lognormal distribution, as brms's.
"""
dshifted_lnorm(x, meanlog=0, sdlog=1, shift=0; log::Bool=false) = _dout(_DD.logpdf(_sln(meanlog, sdlog, shift), x), log)
pshifted_lnorm(q, meanlog=0, sdlog=1, shift=0; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_DD.logcdf(_sln(meanlog, sdlog, shift), q), _DD.logccdf(_sln(meanlog, sdlog, shift), q), lower_tail, log_p)
qshifted_lnorm(p, meanlog=0, sdlog=1, shift=0; lower_tail::Bool=true, log_p::Bool=false) =
    _DD.quantile(_sln(meanlog, sdlog, shift), _qp(p, lower_tail, log_p))
rshifted_lnorm(meanlog=0, sdlog=1, shift=0; rng=Random.default_rng()) = rand(rng, _sln(meanlog, sdlog, shift))
@doc (@doc dshifted_lnorm) pshifted_lnorm
@doc (@doc dshifted_lnorm) qshifted_lnorm
@doc (@doc dshifted_lnorm) rshifted_lnorm

# --- asymmetric laplace(mu, sigma, quantile) ----------------------------------------------
const _alfam = AsymLaplace{IdentityLink}()
_aleta(mu, sigma, q) = (Float64(mu), Base.log(Float64(sigma)), Base.log(q / (1 - q)))
"""
    dasym_laplace(x, mu = 0, sigma = 1, quantile = 0.5; log = false)
    pasym_laplace(q, mu = 0, sigma = 1, quantile = 0.5; lower_tail = true, log_p = false)
    qasym_laplace(p, mu = 0, sigma = 1, quantile = 0.5)
    rasym_laplace(mu = 0, sigma = 1, quantile = 0.5; rng)

brms's asymmetric Laplace distribution (`mu` is its `quantile`-th quantile).
"""
dasym_laplace(x, mu=0, sigma=1, quantile=0.5; log::Bool=false) = _dout(_ll(_alfam, x, 0.0, _aleta(mu, sigma, quantile)), log)
pasym_laplace(q, mu=0, sigma=1, quantile=0.5; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_cc_lcdf(_alfam, q, _aleta(mu, sigma, quantile)), _cc_lccdf(_alfam, q, _aleta(mu, sigma, quantile)), lower_tail, log_p)
function qasym_laplace(p, mu=0, sigma=1, quantile=0.5; lower_tail::Bool=true, log_p::Bool=false)
    u = _qp(p, lower_tail, log_p); q = quantile
    return u < q ? mu + sigma / (1 - q) * Base.log(u / q) : mu - sigma / q * Base.log((1 - u) / (1 - q))
end
rasym_laplace(mu=0, sigma=1, quantile=0.5; rng=Random.default_rng()) = _rand(rng, _alfam, _aleta(mu, sigma, quantile), 0)
@doc (@doc dasym_laplace) pasym_laplace
@doc (@doc dasym_laplace) qasym_laplace
@doc (@doc dasym_laplace) rasym_laplace

# --- von mises(mu, kappa) ------------------------------------------------------------------
"""
    dvon_mises(x, mu, kappa; log = false)
    pvon_mises(q, mu, kappa; lower_tail = true, log_p = false)
    rvon_mises(mu, kappa; rng)

The von Mises distribution on (-pi, pi), as brms's (the cdf is Stan's algorithm,
through log I0).
"""
dvon_mises(x, mu, kappa; log::Bool=false) = _dout(_DD.logpdf(_DD.VonMises(mu, kappa), x), log)
function pvon_mises(q, mu, kappa; lower_tail::Bool=true, log_p::Bool=false)
    F = _vm_cdf(Float64(q), Float64(mu), Float64(kappa))
    return _pout(Base.log(F), Base.log1p(-F), lower_tail, log_p)
end
rvon_mises(mu, kappa; rng=Random.default_rng()) = rand(rng, _DD.VonMises(mu, kappa))
@doc (@doc dvon_mises) pvon_mises
@doc (@doc dvon_mises) rvon_mises

# --- beta_binomial(size, mu, phi) ------------------------------------------------------------
_bb(size, mu, phi) = _DD.BetaBinomial(size, mu * phi, (1 - mu) * phi)
"""
    dbeta_binomial(x, size, mu, phi; log = false)
    pbeta_binomial(q, size, mu, phi; lower_tail = true, log_p = false)
    rbeta_binomial(size, mu, phi; rng)

brms's beta-binomial with mean `mu` and precision `phi`.
"""
dbeta_binomial(x, size, mu, phi; log::Bool=false) = _dout(_DD.logpdf(_bb(size, mu, phi), x), log)
pbeta_binomial(q, size, mu, phi; lower_tail::Bool=true, log_p::Bool=false) =
    _pout(_DD.logcdf(_bb(size, mu, phi), q), _DD.logccdf(_bb(size, mu, phi), q), lower_tail, log_p)
rbeta_binomial(size, mu, phi; rng=Random.default_rng()) = rand(rng, _bb(size, mu, phi))
@doc (@doc dbeta_binomial) pbeta_binomial
@doc (@doc dbeta_binomial) rbeta_binomial

# --- zero-inflated and hurdle distributions -------------------------------------------------
# zero-inflated: P(0) = zi + (1 - zi) P_base(0); hurdle: P(0) = hu, else (1 - hu) P_base(x) / (1 - P_base(0))
function _zi_logpdf(base, zi, x)
    x == 0 && return Base.log(zi + (1 - zi) * _DD.pdf(base, 0))
    return Base.log1p(-zi) + _DD.logpdf(base, x)
end
function _zi_cdf(base, zi, q)
    q < 0 && return 0.0
    return zi + (1 - zi) * _DD.cdf(base, q)
end
function _hu_logpdf(base, hu, x)
    x == 0 && return Base.log(hu)
    return Base.log1p(-hu) + _DD.logpdf(base, x) - Base.log1p(-_DD.cdf(base, 0))
end
function _hu_cdf(base, hu, q; discrete::Bool=true)
    q < 0 && return 0.0
    F0 = discrete ? _DD.cdf(base, 0) : 0.0
    return hu + (1 - hu) * (_DD.cdf(base, q) - F0) / (1 - F0)
end
_pq(F, lower_tail, log_p) = (v = lower_tail ? F : 1 - F; log_p ? Base.log(v) : v)
_nb(mu, shape) = _DD.NegativeBinomial(shape, shape / (mu + shape))

"""
    dzero_inflated_poisson(x, lambda, zi; log = false), pzero_inflated_poisson(q, lambda, zi; ...)
    dzero_inflated_negbinomial(x, mu, shape, zi; ...), pzero_inflated_negbinomial(q, mu, shape, zi; ...)
    dzero_inflated_binomial(x, size, prob, zi; ...), pzero_inflated_binomial(q, size, prob, zi; ...)
    dzero_inflated_beta_binomial(x, size, mu, phi, zi; ...), pzero_inflated_beta_binomial(q, size, mu, phi, zi; ...)
    dzero_inflated_beta(x, shape1, shape2, zi; ...), pzero_inflated_beta(q, shape1, shape2, zi; ...)

brms's zero-inflated distributions: an extra zero with probability `zi`.
"""
dzero_inflated_poisson(x, lambda, zi; log::Bool=false) = _dout(_zi_logpdf(_DD.Poisson(lambda), zi, x), log)
pzero_inflated_poisson(q, lambda, zi; lower_tail::Bool=true, log_p::Bool=false) = _pq(_zi_cdf(_DD.Poisson(lambda), zi, q), lower_tail, log_p)
dzero_inflated_negbinomial(x, mu, shape, zi; log::Bool=false) = _dout(_zi_logpdf(_nb(mu, shape), zi, x), log)
pzero_inflated_negbinomial(q, mu, shape, zi; lower_tail::Bool=true, log_p::Bool=false) = _pq(_zi_cdf(_nb(mu, shape), zi, q), lower_tail, log_p)
dzero_inflated_binomial(x, size, prob, zi; log::Bool=false) = _dout(_zi_logpdf(_DD.Binomial(size, prob), zi, x), log)
pzero_inflated_binomial(q, size, prob, zi; lower_tail::Bool=true, log_p::Bool=false) = _pq(_zi_cdf(_DD.Binomial(size, prob), zi, q), lower_tail, log_p)
dzero_inflated_beta_binomial(x, size, mu, phi, zi; log::Bool=false) = _dout(_zi_logpdf(_bb(size, mu, phi), zi, x), log)
pzero_inflated_beta_binomial(q, size, mu, phi, zi; lower_tail::Bool=true, log_p::Bool=false) = _pq(_zi_cdf(_bb(size, mu, phi), zi, q), lower_tail, log_p)
function dzero_inflated_beta(x, shape1, shape2, zi; log::Bool=false)
    return _dout(x == 0 ? Base.log(zi) : Base.log1p(-zi) + _DD.logpdf(_DD.Beta(shape1, shape2), x), log)
end
pzero_inflated_beta(q, shape1, shape2, zi; lower_tail::Bool=true, log_p::Bool=false) =
    _pq(q < 0 ? 0.0 : zi + (1 - zi) * _DD.cdf(_DD.Beta(shape1, shape2), q), lower_tail, log_p)
for f in (:pzero_inflated_poisson, :dzero_inflated_negbinomial, :pzero_inflated_negbinomial, :dzero_inflated_binomial,
          :pzero_inflated_binomial, :dzero_inflated_beta_binomial, :pzero_inflated_beta_binomial, :dzero_inflated_beta,
          :pzero_inflated_beta)
    @eval @doc (@doc dzero_inflated_poisson) $f
end

"""
    dhurdle_poisson(x, lambda, hu; log = false), phurdle_poisson(q, lambda, hu; ...)
    dhurdle_negbinomial(x, mu, shape, hu; ...), phurdle_negbinomial(q, mu, shape, hu; ...)
    dhurdle_gamma(x, shape, scale, hu; ...), phurdle_gamma(q, shape, scale, hu; ...)
    dhurdle_lognormal(x, mu, sigma, hu; ...), phurdle_lognormal(q, mu, sigma, hu; ...)

brms's hurdle distributions: zero with probability `hu`, otherwise the base
distribution (truncated at zero for the counts).
"""
dhurdle_poisson(x, lambda, hu; log::Bool=false) = _dout(_hu_logpdf(_DD.Poisson(lambda), hu, x), log)
phurdle_poisson(q, lambda, hu; lower_tail::Bool=true, log_p::Bool=false) = _pq(_hu_cdf(_DD.Poisson(lambda), hu, q), lower_tail, log_p)
dhurdle_negbinomial(x, mu, shape, hu; log::Bool=false) = _dout(_hu_logpdf(_nb(mu, shape), hu, x), log)
phurdle_negbinomial(q, mu, shape, hu; lower_tail::Bool=true, log_p::Bool=false) = _pq(_hu_cdf(_nb(mu, shape), hu, q), lower_tail, log_p)
dhurdle_gamma(x, shape, scale, hu; log::Bool=false) =
    _dout(x == 0 ? Base.log(hu) : Base.log1p(-hu) + _DD.logpdf(_DD.Gamma(shape, scale), x), log)
phurdle_gamma(q, shape, scale, hu; lower_tail::Bool=true, log_p::Bool=false) =
    _pq(_hu_cdf(_DD.Gamma(shape, scale), hu, q; discrete=false), lower_tail, log_p)
dhurdle_lognormal(x, mu, sigma, hu; log::Bool=false) =
    _dout(x == 0 ? Base.log(hu) : Base.log1p(-hu) + _DD.logpdf(_DD.LogNormal(mu, sigma), x), log)
phurdle_lognormal(q, mu, sigma, hu; lower_tail::Bool=true, log_p::Bool=false) =
    _pq(_hu_cdf(_DD.LogNormal(mu, sigma), hu, q; discrete=false), lower_tail, log_p)
for f in (:phurdle_poisson, :dhurdle_negbinomial, :phurdle_negbinomial, :dhurdle_gamma, :phurdle_gamma,
          :dhurdle_lognormal, :phurdle_lognormal)
    @eval @doc (@doc dhurdle_poisson) $f
end

# --- multivariate --------------------------------------------------------------------------
"""
    ddirichlet(x, alpha; log = false), rdirichlet(alpha; rng)
    dmulti_normal(x, mu, Sigma; log = false), rmulti_normal(mu, Sigma; rng)
    dmulti_student_t(x, df, mu, Sigma; log = false), rmulti_student_t(df, mu, Sigma; rng)
    dlogistic_normal(x, mu, Sigma; refcat = 1, log = false), rlogistic_normal(mu, Sigma; refcat = 1, rng)

brms's multivariate distributions for one vector `x`. The logistic normal: the
log ratios of `x` to its reference category are multivariate normal (`mu`,
`Sigma` of length / size K - 1).
"""
ddirichlet(x, alpha; log::Bool=false) = _dout(_DD.logpdf(_DD.Dirichlet(alpha), x), log)
rdirichlet(alpha; rng=Random.default_rng()) = rand(rng, _DD.Dirichlet(alpha))
dmulti_normal(x, mu, Sigma; log::Bool=false) = _dout(_DD.logpdf(_DD.MvNormal(mu, Sigma), x), log)
rmulti_normal(mu, Sigma; rng=Random.default_rng()) = rand(rng, _DD.MvNormal(mu, Sigma))
dmulti_student_t(x, df, mu, Sigma; log::Bool=false) = _dout(_DD.logpdf(_DD.MvTDist(df, mu, Matrix(Sigma)), x), log)
rmulti_student_t(df, mu, Sigma; rng=Random.default_rng()) = rand(rng, _DD.MvTDist(df, mu, Matrix(Sigma)))
function dlogistic_normal(x, mu, Sigma; refcat::Int=1, log::Bool=false)
    r = [Base.log(x[k] / x[refcat]) for k in eachindex(x) if k != refcat]
    return _dout(_DD.logpdf(_DD.MvNormal(mu, Sigma), r) - sum(Base.log, x), log)
end
function rlogistic_normal(mu, Sigma; refcat::Int=1, rng=Random.default_rng())
    z = rand(rng, _DD.MvNormal(mu, Sigma))
    e = exp.(vcat(z[1:(refcat - 1)], 0.0, z[refcat:end]))
    return e ./ sum(e)
end
for f in (:rdirichlet, :dmulti_normal, :rmulti_normal, :dmulti_student_t, :rmulti_student_t, :dlogistic_normal, :rlogistic_normal)
    @eval @doc (@doc ddirichlet) $f
end

# --- wiener (alpha boundary separation, tau non-decision time, beta bias, delta drift) ----------
"""
    dwiener(x, alpha, tau, beta, delta; resp = 1, log = false)
    rwiener(alpha, tau, beta, delta; rng) -> (q = response time, resp = 1 upper / 0 lower)

brms's Wiener diffusion first-passage time distribution (the `wiener()`
family's density, Navarro & Fuss series; draws by simulation).
"""
function dwiener(x, alpha, tau, beta, delta; resp=1, log::Bool=false)
    lp = resp == 1 ? _wiener_lower(Float64(x), -Float64(delta), Float64(alpha), Float64(tau), 1 - Float64(beta)) :
                     _wiener_lower(Float64(x), Float64(delta), Float64(alpha), Float64(tau), Float64(beta))
    return _dout(lp, log)
end
function rwiener(alpha, tau, beta, delta; rng=Random.default_rng(), dt=1e-4)
    x = alpha * beta; s = 0.0; sq = sqrt(dt)
    while 0 < x < alpha
        x += delta * dt + sq * randn(rng); s += dt
    end
    return (q=tau + s, resp=x >= alpha ? 1 : 0)
end
@doc (@doc dwiener) rwiener
