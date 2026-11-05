# One shared synthetic data set with a column for every family's response.
function make_testdata(; N=300, seed=1)
    rng = Xoshiro(seed)
    x = randn(rng, N); z = randn(rng, N)
    g = rand(rng, ["a", "b", "c", "d", "e", "f"], N)
    h = rand(rng, ["u", "v"], N)
    gi = Dict(l => i for (i, l) in enumerate(sort(unique(g))))
    re = randn(rng, 6)
    yg = 1 .+ 0.5x .+ [re[gi[v]] for v in g] .+ randn(rng, N)
    yp = [rand(rng) < 0.5 ? 0 : rand(rng, 0:7) for _ in 1:N]
    yb = Float64.(rand(rng, N) .< 0.4)
    n = fill(10, N); yn = [rand(rng, 0:10) for _ in 1:N]
    ya = exp.(0.3 .+ 0.2x) .* rand(rng, N) .+ 0.05
    yu = clamp.(rand(rng, N), 0.01, 0.99)
    yh = [rand(rng) < 0.3 ? 0.0 : v for v in ya]
    yo = rand(rng, 1:4, N)
    cc = rand(rng, [-1, 0, 0, 1, 2], N)                 # censoring codes
    ccs = [("left", "none", "right", "interval")[(c == -1 ? 1 : c + 2)] for c in cc]
    yg2 = yg .+ 0.1 .+ rand(rng, N)
    ynl = 2.0 .* exp.(-0.7 .* abs.(x)) .+ 0.1 .* randn(rng, N)
    se = 0.2 .+ rand(rng, N)
    mlev = rand(rng, 0:3, N)
    ordf = rand(rng, ["lo", "mid", "hi"], N)
    tt = repeat(1:(N ÷ 6), 6)[1:N]
    expo = 0.5 .+ 2 .* rand(rng, N)
    ya2 = ya .+ 0.1 .+ rand(rng, N)
    w = rand(rng, N) .+ 0.5
    # drawn last, so adding them left every earlier column unchanged
    yz0 = [rand(rng) < 0.2 ? 0.0 : v for v in yu]
    yz01 = [(r = rand(rng); r < 0.1 ? 0.0 : r < 0.2 ? 1.0 : v) for v in yu]
    yang = [mod2pi(2 * atan(0.3 + 0.4 * xi) + 0.6 * randn(rng) + π) - π for xi in x]
    yhc = [rand(rng) < 0.25 ? 0 : v for v in yo]
    ca = rand(rng, 0:6, N); cb = rand(rng, 0:6, N); cc3 = rand(rng, 0:6, N)      # multinomial counts
    pr = [(r = rand(rng, 3) .+ 0.2; r ./ sum(r)) for _ in 1:N]
    pa = getindex.(pr, 1); pb = getindex.(pr, 2); pc = getindex.(pr, 3)
    ccox = rand(rng, [0, 0, 0, 1, -1], N)                     # cox censoring codes
    g2 = rand(rng, ["a", "b", "c", "d", "e", "f", "q"], N)      # a second membership factor
    gh = [string(h[i], g[i]) for i in 1:N]                       # g nested in h
    h2 = rand(rng, 0:2, N)                                       # a coarse covariate (exact GP tests)
    mlev1 = mlev .+ 1                                             # CAR locations 1..4
    yrt = 0.35 .+ ya ./ 3; dd = yb                                # wiener: response times, boundaries
    # unstr(): groups of 4 rows, time = row mod 5 (each group a different subset of 5 levels)
    ug = ["u$(i ÷ 4)" for i in 0:(N - 1)]; ut = ["t$(i % 5 + 1)" for i in 0:(N - 1)]
    # censored / truncated counts: interval upper bounds, left/right codes for the
    # binomial (no room above n), a response >= 1 for zero truncation
    yp2 = yp .+ rand(rng, 1:3, N); ccb = rand(rng, [-1, 0, 0, 1], N); ypp = yp .+ 1
    return (; x, z, g, h, yg, yp, yb, n, yn, ya, yu, yh, yo, cc, ccs, yg2, ya2, ynl, se, expo, mlev, ordf, tt, w,
            yz0, yz01, yang, yhc, g2, gh, ca, cb, cc3, pa, pb, pc, ccox, h2, mlev1, yrt, dd, ug, ut, yp2, ccb, ypp)
end
const TD = make_testdata()
# a ring of four locations for the car() tests
const WCAR = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]
# gr(g, cov = ACOV): a known covariance among g's six levels (sorted a..f)
const ACOV = [exp(-abs(i - j) / 2) for i in 1:6, j in 1:6]
# fcor(): a known residual covariance (tridiagonal, positive definite) over the rows of TD;
# sar(): a row-standardised ring over the rows of TD (eigenvalues in [-1, 1])
const MFCOR = [i == j ? 1.0 : abs(i - j) == 1 ? 0.3 : 0.0 for i in 1:300, j in 1:300]
const WSAR = [abs(i - j) == 1 || abs(i - j) == 299 ? 0.5 : 0.0 for i in 1:300, j in 1:300]
# custom_family: brms's beta_binomial2 vignette example (trials through vint)
_lbeta(a, b) = PracticalBayesianRegression.loggamma(a) + PracticalBayesianRegression.loggamma(b) - PracticalBayesianRegression.loggamma(a + b)
const CUSTOM_BB = custom_family("beta_binomial2"; dpars=(:mu, :phi), links=(:logit, :log), type=:int,
    ll=(y, mu, phi, n) -> PracticalBayesianRegression.loggamma(n + 1) - PracticalBayesianRegression.loggamma(y + 1) -
                          PracticalBayesianRegression.loggamma(n - y + 1) + _lbeta(y + mu * phi, n - y + (1 - mu) * phi) -
                          _lbeta(mu * phi, (1 - mu) * phi))
const CUSTOM_NORMAL = custom_family("normal2"; dpars=(:mu, :sigma), links=(:identity, :log),
    ll=(y, mu, sigma) -> -log(sigma) - (y - mu)^2 / (2sigma^2), rand=(rng, mu, sigma) -> mu + sigma * randn(rng),
    mean=(mu, sigma) -> mu)

# (name, formula, family) covering every structural feature of the density
const GRAD_CASES = [
    ("gauss", @bf(yg ~ x + (1 | g)), gaussian()),
    ("gauss_cor", @bf(yg ~ x * h + (1 + x | g)), gaussian()),
    ("gauss_cor3", @bf(yg ~ x + (1 + x + z | g) + (1 || h)), gaussian()),
    ("student", @bf(yg ~ x + log(abs(z) + 1)), student()),
    ("distrib", @bf(yg ~ x + (1 | g), sigma ~ z + (1 | g)), gaussian()),
    ("distrib_student", @bf(yg ~ x, nu ~ z), student()),
    ("bern", @bf(yb ~ x + z + (1 + x || g)), bernoulli()),
    ("binom", bf("yn | trials(n) ~ x + (1|g:h)"), binomial()),
    ("pois", @bf(yp ~ x + offset(z) + (1 | g)), poisson()),
    ("negbin", @bf(yp ~ x + (1 | g)), negbinomial()),
    ("negbin_distrib", @bf(yp ~ x, shape ~ z), negbinomial()),
    ("gamma", @bf(ya ~ x + h), gammafamily()),
    ("weighted", @bf(yg | weights(w) ~ x + g), gaussian()),
    ("nointercept", @bf(yg ~ 0 + g + x), gaussian()),
    ("zero_plus_intercept", @bf(yg ~ 0 + Intercept + x + (1 | g)), gaussian()),
    ("gauss_loglink", @bf(ya ~ x), gaussian(; link=:log)),
    ("lognormal", @bf(ya ~ x + (1 | g)), lognormal()),
    ("probit", @bf(yb ~ x + (1 | g)), bernoulli(; link=:probit)),
    ("cloglog", @bf(yb ~ x), bernoulli(; link=:cloglog)),
    ("binom_probit", @bf(yn | trials(n) ~ x), binomial(; link=:probit)),
    ("beta_binomial", @bf(yn | trials(n) ~ x + (1 | g)), beta_binomial()),
    ("pois_sqrt", @bf(yp ~ 1 + z), poisson(; link=:sqrt)),
    # brms 2.23's further links (bounded / positive inverses, so any θ is in the support)
    ("bern_cauchit", @bf(yb ~ x + (1 | g)), bernoulli(; link=:cauchit)),
    ("bern_softit", @bf(yb ~ x), bernoulli(; link=:softit)),
    ("binom_probit_approx", @bf(yn | trials(n) ~ x), binomial(; link=:probit_approx)),
    ("beta_cauchit", @bf(yu ~ x), betafamily(; link=:cauchit)),
    ("pois_softplus", @bf(yp ~ x + (1 | g)), poisson(; link=:softplus)),
    ("negbin_squareplus", @bf(yp ~ z), negbinomial(; link=:squareplus)),
    ("gamma_softplus", @bf(ya ~ x), gammafamily(; link=:softplus)),
    ("gauss_squareplus", @bf(ya ~ x, sigma ~ z), gaussian(; link=:squareplus)),
    ("student_softit", @bf(yu ~ x), student(; link=:softit)),
    ("lognormal_inverse", @bf(ya2 ~ x), lognormal(; link=:inverse)),
    ("zoib_softit", @bf(yz01 ~ x), zero_one_inflated_beta(; link=:softit)),
    ("von_mises_identity", @bf(yang ~ x), von_mises(; link=:identity)),
    ("invgauss_softplus", @bf(ya ~ x), inverse_gaussian(; link=:softplus)),
    ("cumulative_cauchit", @bf(yo ~ x + (1 | g)), cumulative(; link=:cauchit)),
    ("cumulative_cloglog", @bf(yo ~ x), cumulative(; link=:cloglog)),
    ("sratio_probit_approx", @bf(yo ~ x), sratio(; link=:probit_approx)),
    ("cratio_cauchit", @bf(yo ~ z), cratio(; link=:cauchit)),
    ("acat_softit", @bf(yo ~ x), acat(; link=:softit)),
    # ordinal discrimination (brms disc ~ ...)
    ("cumulative_disc", @bf(yo ~ x, disc ~ z), cumulative()),
    ("cumulative_disc_gr", @bf(yo ~ x + (1 | g), disc ~ 0 + z + (1 | h)), cumulative(; link=:probit)),
    ("sratio_disc", @bf(yo ~ x, disc ~ z), sratio()),
    ("cratio_disc", @bf(yo ~ z, disc ~ x), cratio(; link=:cloglog)),
    ("acat_disc", @bf(yo ~ x, disc ~ z), acat()),
    ("acat_disc_probit", @bf(yo ~ x, disc ~ z), acat(; link=:probit)),
    ("acat_disc_cs", @bf(yo ~ x + cs(z), disc ~ z), acat()),
    ("hurdle_cumulative_disc", @bf(yhc ~ x, hu ~ z, disc ~ z), hurdle_cumulative()),
    ("geometric", @bf(yp ~ x), geometric()),
    ("weibull", @bf(ya ~ x + (1 | g)), weibull()),
    ("exponential", @bf(ya ~ x), exponential()),
    ("beta", @bf(yu ~ x + (1 | g)), betafamily()),
    ("beta_distrib", @bf(yu ~ x, phi ~ z), betafamily()),
    ("zip", @bf(yp ~ x + (1 | g)), zero_inflated_poisson()),
    ("zip_distrib", @bf(yp ~ x, zi ~ z), zero_inflated_poisson()),
    ("zinb", @bf(yp ~ x), zero_inflated_negbinomial()),
    ("zib", @bf(yn | trials(n) ~ x), zero_inflated_binomial()),
    ("hurdle_pois", @bf(yp ~ x + (1 | g)), hurdle_poisson()),
    ("hurdle_nb", @bf(yp ~ x, hu ~ z), hurdle_negbinomial()),
    ("hurdle_gamma", @bf(yh ~ x), hurdle_gamma()),
    ("hurdle_lognormal", @bf(yh ~ x), hurdle_lognormal()),
    ("cumulative", @bf(yo ~ x + (1 | g)), cumulative()),
    ("cumulative_probit", @bf(yo ~ x + z), cumulative(; link=:probit)),
    ("sratio", @bf(yo ~ x + (1 | g)), sratio()),
    ("cratio_cloglog", @bf(yo ~ x + z), cratio(; link=:cloglog)),
    ("acat", @bf(yo ~ x + (1 | g)), acat()),
    ("acat_probit", @bf(yo ~ x), acat(; link=:probit)),
    ("sratio_cs", @bf(yo ~ cs(x) + z + (1 | g)), sratio()),
    ("acat_cs", @bf(yo ~ cs(x + z)), acat()),
    ("cens_gauss", @bf(yg | cens(cc, yg2) ~ x + (1 | g)), gaussian()),
    ("cens_lognormal", @bf(ya | cens(ccs, ya2) ~ x), lognormal()),
    ("cens_weibull", @bf(ya | cens(cc, ya2) ~ x + (1 | g)), weibull()),
    ("cens_exponential", @bf(ya | cens(cc, ya2) ~ x), exponential()),
    ("cens_distrib", @bf(yg | cens(cc, yg2) ~ x, sigma ~ z), gaussian()),
    # count families: cdfs as sums of the pmf (dual-number derivatives)
    ("cens_pois", @bf(yp | cens(cc, yp2) ~ x + (1 | g)), poisson()),
    ("cens_negbin", @bf(yp | cens(cc, yp2) ~ x), negbinomial()),
    ("cens_negbin_distrib", @bf(yp | cens(cc, yp2) ~ x, shape ~ z), negbinomial()),
    ("cens_geometric", @bf(yp | cens(cc, yp2) ~ x), geometric()),
    ("cens_negbin2", @bf(yp | cens(cc, yp2) ~ x), negbinomial2()),
    ("cens_binom", @bf(yn | trials(n) + cens(ccb) ~ x), binomial()),
    ("cens_pois_sqrt", @bf(yp | cens(cc, yp2) ~ z), poisson(; link=:sqrt)),
    ("categorical", @bf(yo ~ x + (1 | g)), categorical()),
    ("categorical_own", @bf(yo ~ x, mu3 ~ z), categorical()),
    ("se", @bf(yg | se(se) ~ x + (1 | g)), gaussian()),
    ("se_sigma", @bf(yg | se(se, sigma = TRUE) ~ x), gaussian()),
    ("rate", @bf(yp | rate(expo) ~ x), poisson()),
    ("trunc_gauss", @bf(yg | trunc(lb = -10, ub = 12) ~ x), gaussian()),
    ("trunc_lognormal_lb", @bf(ya | trunc(lb = 0.01) ~ x), lognormal()),
    ("trunc_weibull_ub", @bf(ya | trunc(ub = 20) ~ x), weibull()),
    ("trunc_pois_zero", @bf(ypp | trunc(lb = 1) ~ x + (1 | g)), poisson()),
    ("trunc_negbin_both", @bf(yp | trunc(lb = 0, ub = 20) ~ x), negbinomial()),
    ("trunc_negbin_ub_distrib", @bf(yp | trunc(ub = 12) ~ x, shape ~ z), negbinomial()),
    ("trunc_binom_lb", @bf(yn | trials(n) + trunc(lb = 0, ub = 10) ~ x), binomial()),
    ("mo", @bf(yg ~ x + mo(mlev) + (1 | g)), gaussian()),
    ("mo_two", @bf(yp ~ mo(mlev) + mo(ordf)), poisson()),
    ("mo_inter", @bf(yg ~ x + mo(mlev) + mo(mlev):z), gaussian()),
    ("mo_inter2", @bf(yp ~ mo(mlev) * mo(ordf) + (1 | g)), poisson()),
    ("mo_factor", @bf(yg ~ x + mo(mlev):h + (1 | g)), gaussian()),
    ("mo_factor_num", @bf(yp ~ mo(mlev):h:z), poisson()),
    ("ar1", @bf(yg ~ x + ar(time = tt, gr = g, p = 1)), gaussian()),
    ("ar2", @bf(yg ~ x + ar(tt, g, 2)), student()),
    ("ma1", @bf(yg ~ x + ma(time = tt, gr = g, q = 1)), gaussian()),
    ("arma21", @bf(yg ~ x + arma(tt, g, 2, 1)), student()),
    ("ar_plus_ma", @bf(yg ~ x + ar(tt, g, 1) + ma(tt, g, 2)), gaussian()),
    # gr(g, dist = "student"): student-t group effects (brms df / udf scales)
    ("gr_student", @bf(yg ~ x + (1 | gr(g, dist = "student"))), gaussian()),
    ("gr_cov", @bf(yg ~ x + (1 | gr(g, cov = ACOV))), gaussian()),
    ("gr_cov_cor", @bf(yp ~ x + (1 + x | gr(g, cov = ACOV)) + (1 | h)), poisson()),
    ("gr_student_cor", @bf(yp ~ x + (1 + x | gr(g, dist = "student")) + (1 | h)), poisson()),
    # thres(): a fixed number of thresholds, and one set per level of a factor
    ("thres_n", @bf(yo | thres(4) ~ x), cumulative()),
    ("thres_gr", @bf(yo | thres(gr = h) ~ x + (1 | g)), cumulative()),
    ("thres_gr_sratio", @bf(yo | thres(gr = h) ~ x), sratio()),
    ("thres_gr_acat", @bf(yo | thres(3, gr = h) ~ x), acat(; link=:probit)),
    # residual covariance structures (timecov.jl); time = row order within g
    ("ar_cov", @bf(yg ~ x + ar(gr = g, cov = TRUE)), gaussian()),
    ("ma_cov_het", @bf(yg ~ x + ma(gr = g, cov = TRUE), sigma ~ z), gaussian()),
    ("arma_cov_student", @bf(yg ~ x + arma(gr = g, cov = TRUE) + (1 | h)), student()),
    ("cosy", @bf(yg ~ x + cosy(gr = g)), gaussian()),
    ("cosy_student_het", @bf(yg ~ x + cosy(gr = h), sigma ~ z), student()),
    ("fcor", @bf(yg ~ x + fcor(MFCOR)), gaussian()),
    ("unstr", @bf(yg ~ x + unstr(time = ut, gr = ug)), gaussian()),
    ("unstr_student_het", @bf(yg ~ x + unstr(ut, ug), sigma ~ z), student()),
    ("fcor_student_het", @bf(yg ~ x + fcor(MFCOR), sigma ~ z), student()),
    ("sar_lag", @bf(yg ~ x + sar(WSAR)), gaussian()),
    ("sar_error", @bf(yg ~ x + sar(WSAR, type = "error")), gaussian()),
    ("sar_lag_student", @bf(yg ~ x + sar(WSAR, type = "lag")), student()),
    ("wiener", @bf(yrt | dec(dd) ~ x), wiener()),
    ("zi_beta_binomial", bf("yn | trials(n) ~ x + (1|g)", "zi ~ z"), zero_inflated_beta_binomial()),
    ("wiener_distrib", @bf(yrt | dec(dd) ~ x, bs ~ z, bias ~ x), wiener()),
    # custom families
    ("custom_bb", @bf(yn | vint(n) ~ x + (1 | g)), CUSTOM_BB),
    ("custom_bb_distrib", @bf(yn | vint(n) ~ x, phi ~ z), CUSTOM_BB),
    ("custom_normal", @bf(yg ~ x + (1 | g)), CUSTOM_NORMAL),
    # non-gaussian: brms's latent residuals err = sderr * L zerr in mu
    ("ar_cov_pois", @bf(yp ~ x + ar(gr = g, cov = TRUE)), poisson()),
    ("arma_cov_negbin", @bf(yp ~ x + arma(gr = g, cov = TRUE)), negbinomial()),
    ("cosy_bern", @bf(yb ~ x + cosy(gr = h)), bernoulli()),
    ("skew_normal", @bf(yg ~ x + (1 | g)), skew_normal()),
    ("exgaussian", @bf(ya ~ x), exgaussian()),
    ("inverse_gaussian", @bf(ya ~ x), inverse_gaussian(; link=:log)),
    ("frechet", @bf(ya ~ x), frechet()),
    ("asym_laplace", @bf(yg ~ x + (1 | g)), asym_laplace()),
    ("asym_laplace_distrib", @bf(yg ~ x, quantile ~ z), asym_laplace()),
    ("von_mises", @bf(yang ~ x), von_mises()),
    ("zero_inflated_beta", @bf(yz0 ~ x, zi ~ z), zero_inflated_beta()),
    ("zero_one_inflated_beta", @bf(yz01 ~ x + (1 | g)), zero_one_inflated_beta()),
    ("discrete_weibull", @bf(yp ~ x), discrete_weibull()),
    ("negbinomial2", @bf(yp ~ x + (1 | g)), negbinomial2()),
    ("com_poisson", @bf(yp ~ x), com_poisson()),
    ("multinomial", @bf(cbind(ca, cb, cc3) ~ x + (1 | g)), multinomial()),
    ("cox", @bf(ya | cens(ccox) ~ x + (1 | g)), cox()),
    ("gev", @bf(yg ~ x, sigma ~ z), gen_extreme_value()),
    ("xbeta", @bf(yz01 ~ x, kappa ~ z), xbeta()),
    ("gev_scalar", @bf(yg ~ x + (1 | g)), gen_extreme_value()),
    ("cox_events", @bf(ya ~ x + z), cox()),
    ("dirichlet", @bf(cbind(pa, pb, pc) ~ x, phi ~ z), dirichlet()),
    ("logistic_normal", @bf(cbind(pa, pb, pc) ~ x + (1 | g)), logistic_normal()),
    ("hurdle_cumulative", @bf(yhc ~ x + (1 | g)), hurdle_cumulative()),
    ("hurdle_cumulative_probit", @bf(yhc ~ x, hu ~ z), hurdle_cumulative(; link=:probit)),
    ("mixture", @bf(yg ~ x), mixture(gaussian(), gaussian())),
    ("shared_id", @bf(yg ~ x + (1 + x | p | g), sigma ~ z + (1 | p | g)), gaussian()),
    ("me", @bf(yg ~ me(x, se) + z + (1 | g)), gaussian()),
    # (an exact GP over the 300 distinct x values is numerically near-singular with brms's
    # 1e-12 jitter: both gradient paths lose digits there; a well-conditioned one here)
    ("gp_exact", @bf(yg ~ x + gp(mlev)), gaussian()),
    ("car_icar", @bf(yg ~ x + car(WCAR, gr = mlev1, type = "icar")), gaussian()),
    ("car_bym2", @bf(yp ~ x + car(WCAR, gr = mlev1, type = "bym2")), poisson()),
    ("car_escar", @bf(yg ~ x + car(WCAR, gr = mlev1)), gaussian()),
    ("car_esicar", @bf(yg ~ car(WCAR, gr = mlev1, type = "esicar")), gaussian()),
    ("gp_exact_2d", @bf(yp ~ gp(mlev, h2)), poisson()),
    ("gp_approx", @bf(yp ~ z + gp(x, k = 8, c = 5/4)), poisson()),
    ("gp_2d_approx", @bf(yg ~ gp(x, z, k = 4)), gaussian()),
    ("gp_matern52", @bf(yg ~ x + gp(mlev, cov = "matern52")), gaussian()),
    ("gp_matern32_2d", @bf(yp ~ gp(mlev, h2, cov = "matern32")), poisson()),
    ("gp_exponential_approx", @bf(yp ~ z + gp(x, k = 8, cov = "exponential")), poisson()),
    ("gp_matern52_2d_approx", @bf(yg ~ gp(x, z, k = 4, cov = "matern52")), gaussian()),
    ("me_pois", @bf(yp ~ me(z, se)), poisson()),
    # brms mecor (default): correlated latents for two or more me() terms
    ("me_cor", @bf(yg ~ me(x, se) + me(z, se) + (1 | g)), gaussian()),
    ("me_cor3", @bf(yp ~ me(x, se) + me(z, se) + me(ya, se)), poisson()),
    ("me_nocor", @bf(yg ~ me(x, se) + me(z, se)) + set_mecor(false), gaussian()),
    ("shared_id_zi", @bf(yp ~ x + (1 | q | g), zi ~ (1 | q | g)), zero_inflated_poisson()),
    ("shared_id_once", @bf(yg ~ x + (1 | p | g)), gaussian()),
    ("gr_by", @bf(yg ~ x + (1 + x | gr(gh, by = h))), gaussian()),
    ("mm", @bf(yp ~ x + (1 | mm(g, g2))), poisson()),
    ("mm_weights", @bf(yg ~ x + (1 + x | mm(g, g2, weights = cbind(w, expo)))), gaussian()),
    ("mixture_theta", @bf(yg ~ x, theta2 ~ z), mixture(gaussian(), gaussian())),
    ("mixture_pois", @bf(yp ~ x + (1 | g)), mixture(poisson(), poisson())),
    ("mixture_mixed", @bf(yg ~ x, mu2 ~ z, sigma2 ~ x), mixture(gaussian(), student())),
    ("mixture3", @bf(yg ~ 1), mixture(gaussian(), gaussian(), gaussian())),
    ("shifted_lognormal", @bf(ya ~ x + (1 | g)), shifted_lognormal()),
    ("smooth", @bf(yg ~ z + s(x) + (1 | g)), gaussian()),
    ("smooth_2d", @bf(yg ~ s(x, z, k = 12)), gaussian()),
    ("smooth_t2", @bf(yp ~ t2(x, z)), poisson()),
    ("smooth_cr", @bf(yg ~ s(x, bs = "cr") + (1 | g)), gaussian()),
    ("smooth_by_numeric", @bf(yp ~ s(x, by = z, k = 6)), poisson()),
    ("smooth_ps", @bf(yg ~ s(x, bs = "ps", k = 8)), gaussian()),
    ("smooth_cc", @bf(yp ~ z + s(x, bs = "cc")), poisson()),
    ("smooth_ts", @bf(yg ~ s(x, bs = "ts")), gaussian()),
    ("smooth_cs", @bf(yp ~ z + s(x, bs = "cs", k = 6)), poisson()),
    ("smooth_re", @bf(yg ~ x + s(g, bs = "re")), gaussian()),
    ("smooth_re_slope", @bf(yp ~ s(z, h, bs = "re") + s(g, h, bs = "re")), poisson()),
    ("smooth_by", @bf(yg ~ h + s(x, by = h, k = 6)), gaussian()),
    ("poly", @bf(yp ~ poly(x, 3) + poly(z, 2, raw = TRUE)), poisson()),
    ("rsplines", @bf(yg ~ ns(x, 4) + bs(z, df = 5, degree = 2) + (1 | g)), gaussian()),
    ("smooth_k", @bf(yp ~ s(x, k = 6) + s(z)), poisson()),
    ("smooth_distrib", @bf(yg ~ s(x), sigma ~ s(z, k = 5)), gaussian()),
]
const NL_PRIORS = [prior("normal(1, 2)"; nlpar=:a), prior("normal(0, 2)"; nlpar=:b)]
const NL_CASES = [
    ("nl_basic", @bf(ynl ~ a * exp(-b * abs(x)), a ~ 1, b ~ 1, nl = true), gaussian()),
    ("nl_groups", @bf(ynl ~ a * exp(-b * abs(x)), a ~ 1 + (1 | g), b ~ 1 + z, nl = true), gaussian()),
    ("nl_distrib", @bf(ynl ~ a + b * x, a ~ 1, b ~ 1, sigma ~ z, nl = true), gaussian()),
    ("nl_string", bf("ynl ~ a * exp(-b * abs(x))", "a ~ 1", "b ~ 1"; nl=true), gaussian()),
]

# bounded coefficients: lower, upper, both
const BOUND_PRIORS = [prior("normal(0, 2)"; class=:b, coef="x", lb=0),
                      prior("normal(0, 2)"; class=:b, coef="z", ub=0.5),
                      prior("normal(0, 2)"; class=:b, coef="hv", lb=-1, ub=1)]
