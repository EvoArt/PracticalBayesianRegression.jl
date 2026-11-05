# Each family's hand-written derivative against ForwardDiff of its value
# function, and its normalised log-likelihood against Distributions.jl.
@testset "families" begin
    D = Distributions
    lg(x) = log(x / (1 - x))
    σ(x) = 1 / (1 + exp(-x))
    nbd(m, φ) = D.NegativeBinomial(φ, φ / (m + φ))
    hurdle(base, hu, y) = y == 0 ? log(hu) : log1p(-hu) + D.logpdf(base, y) - log1p(-D.pdf(base, 0))
    zinf(base, zi, y) = y == 0 ? log(zi + (1 - zi) * D.pdf(base, 0)) : log1p(-zi) + D.logpdf(base, y)
    Φ(x) = D.cdf(D.Normal(), x)
    clog(x) = 1 - exp(-exp(x))

    # (family, eta, y, trials, reference log-likelihood as a function of eta)
    cases = Any[
        (gaussian(), (0.3, log(1.7)), 1.1, 0, e -> D.logpdf(D.Normal(e[1], exp(e[2])), 1.1)),
        (gaussian(; link=:log), (0.3, log(1.7)), 1.1, 0, e -> D.logpdf(D.Normal(exp(e[1]), exp(e[2])), 1.1)),
        (student(), (0.3, log(1.7), log(3.0)), -0.4, 0,
            e -> D.logpdf(e[1] + exp(e[2]) * D.TDist(1 + exp(e[3])), -0.4)),
        (lognormal(), (0.2, log(0.6)), 1.4, 0, e -> D.logpdf(D.LogNormal(e[1], exp(e[2])), 1.4)),
        (bernoulli(), (0.4,), 1.0, 0, e -> D.logpdf(D.Bernoulli(σ(e[1])), 1)),
        (bernoulli(; link=:probit), (-0.7,), 1.0, 0, e -> D.logpdf(D.Bernoulli(Φ(e[1])), 1)),
        (bernoulli(; link=:probit), (2.1,), 0.0, 0, e -> D.logpdf(D.Bernoulli(Φ(e[1])), 0)),
        (bernoulli(; link=:cloglog), (0.3,), 0.0, 0, e -> D.logpdf(D.Bernoulli(clog(e[1])), 0)),
        (bernoulli(; link=:cloglog), (-1.2,), 1.0, 0, e -> D.logpdf(D.Bernoulli(clog(e[1])), 1)),
        (binomial(), (-0.2,), 4.0, 10, e -> D.logpdf(D.Binomial(10, σ(e[1])), 4)),
        (binomial(; link=:probit), (-0.2,), 4.0, 10, e -> D.logpdf(D.Binomial(10, Φ(e[1])), 4)),
        (binomial(; link=:cloglog), (-0.2,), 4.0, 10, e -> D.logpdf(D.Binomial(10, clog(e[1])), 4)),
        (beta_binomial(), (0.3, log(4.0)), 3.0, 12,
            e -> D.logpdf(D.BetaBinomial(12, σ(e[1]) * exp(e[2]), (1 - σ(e[1])) * exp(e[2])), 3)),
        (poisson(), (1.2,), 3.0, 0, e -> D.logpdf(D.Poisson(exp(e[1])), 3)),
        (poisson(; link=:identity), (2.2,), 3.0, 0, e -> D.logpdf(D.Poisson(e[1]), 3)),
        (poisson(; link=:sqrt), (1.3,), 3.0, 0, e -> D.logpdf(D.Poisson(e[1]^2), 3)),
        (negbinomial(), (1.1, log(2.5)), 2.0, 0, e -> D.logpdf(nbd(exp(e[1]), exp(e[2])), 2)),
        (negbinomial(), (1.1, log(2.5)), 40.0, 0, e -> D.logpdf(nbd(exp(e[1]), exp(e[2])), 40)),
        (negbinomial(; link=:identity), (3.1, log(2.5)), 2.0, 0, e -> D.logpdf(nbd(e[1], exp(e[2])), 2)),
        (geometric(), (0.7,), 2.0, 0, e -> D.logpdf(D.Geometric(1 / (1 + exp(e[1]))), 2)),
        (gammafamily(), (0.5, log(3.0)), 1.4, 0, e -> D.logpdf(D.Gamma(exp(e[2]), exp(e[1]) / exp(e[2])), 1.4)),
        (gammafamily(; link=:identity), (1.5, log(3.0)), 1.4, 0, e -> D.logpdf(D.Gamma(exp(e[2]), e[1] / exp(e[2])), 1.4)),
        (gammafamily(; link=:inverse), (0.6, log(3.0)), 1.4, 0, e -> D.logpdf(D.Gamma(exp(e[2]), (1 / e[1]) / exp(e[2])), 1.4)),
        (weibull(), (0.5, log(1.8)), 1.3, 0,
            e -> D.logpdf(D.Weibull(exp(e[2]), exp(e[1] - PG.loggamma(1 + 1 / exp(e[2])))), 1.3)),
        (exponential(), (0.5,), 1.3, 0, e -> D.logpdf(D.Exponential(exp(e[1])), 1.3)),
        (betafamily(), (0.4, log(6.0)), 0.3, 0,
            e -> D.logpdf(D.Beta(σ(e[1]) * exp(e[2]), (1 - σ(e[1])) * exp(e[2])), 0.3)),
        (zero_inflated_poisson(), (0.8, -0.5), 0.0, 0, e -> zinf(D.Poisson(exp(e[1])), σ(e[2]), 0)),
        (zero_inflated_poisson(), (0.8, -0.5), 3.0, 0, e -> zinf(D.Poisson(exp(e[1])), σ(e[2]), 3)),
        (zero_inflated_negbinomial(), (0.8, log(2.0), -0.5), 0.0, 0, e -> zinf(nbd(exp(e[1]), exp(e[2])), σ(e[3]), 0)),
        (zero_inflated_negbinomial(), (0.8, log(2.0), -0.5), 5.0, 0, e -> zinf(nbd(exp(e[1]), exp(e[2])), σ(e[3]), 5)),
        (zero_inflated_binomial(), (-0.3, 0.2), 0.0, 8, e -> zinf(D.Binomial(8, σ(e[1])), σ(e[2]), 0)),
        (zero_inflated_binomial(), (-0.3, 0.2), 3.0, 8, e -> zinf(D.Binomial(8, σ(e[1])), σ(e[2]), 3)),
        (hurdle_poisson(), (0.8, -0.5), 0.0, 0, e -> hurdle(D.Poisson(exp(e[1])), σ(e[2]), 0)),
        (hurdle_poisson(), (0.8, -0.5), 3.0, 0, e -> hurdle(D.Poisson(exp(e[1])), σ(e[2]), 3)),
        (hurdle_negbinomial(), (0.8, log(2.0), -0.5), 4.0, 0, e -> hurdle(nbd(exp(e[1]), exp(e[2])), σ(e[3]), 4)),
        (hurdle_negbinomial(), (0.8, log(2.0), -0.5), 0.0, 0, e -> hurdle(nbd(exp(e[1]), exp(e[2])), σ(e[3]), 0)),
        (hurdle_gamma(), (0.5, log(3.0), 0.3), 1.4, 0,
            e -> log1p(-σ(e[3])) + D.logpdf(D.Gamma(exp(e[2]), exp(e[1]) / exp(e[2])), 1.4)),
        (hurdle_gamma(), (0.5, log(3.0), 0.3), 0.0, 0, e -> log(σ(e[3]))),
        (hurdle_lognormal(), (0.2, log(0.6), 0.3), 1.4, 0,
            e -> log1p(-σ(e[3])) + D.logpdf(D.LogNormal(e[1], exp(e[2])), 1.4)),
    ]
    sn(e) = (d = e[3] / sqrt(1 + e[3]^2); om = exp(e[2]) / sqrt(1 - 2 / π * d^2);
             D.SkewNormal(e[1] - om * d * sqrt(2 / π), om, e[3]))
    append!(cases, Any[
        (skew_normal(), (0.4, log(1.3), 2.0), 1.1, 0, e -> D.logpdf(sn(e), 1.1)),
        (inverse_gaussian(; link=:log), (0.3, log(2.0)), 1.4, 0, e -> D.logpdf(D.InverseGaussian(exp(e[1]), exp(e[2])), 1.4)),
        (inverse_gaussian(), (0.6, log(2.0)), 1.4, 0, e -> D.logpdf(D.InverseGaussian(1 / sqrt(e[1]), exp(e[2])), 1.4)),
        (frechet(), (0.5, log(2.5)), 1.7, 0,
            e -> D.logpdf(D.Frechet(1 + exp(e[2]), exp(e[1] - PG.loggamma(1 - 1 / (1 + exp(e[2]))))), 1.7)),
        (PG.ShiftedLognormal{PG.IdentityLink}(0.8), (0.1, log(0.5), -0.3), 1.9, 0,
            e -> D.logpdf(D.LogNormal(e[1], exp(e[2])), 1.9 - 0.8 / (1 + exp(-e[3])))),
    ])
    # later brms families; references written from brms's Stan functions
    zb(e, y) = D.logpdf(D.Beta(σ(e[1]) * exp(e[2]), (1 - σ(e[1])) * exp(e[2])), y)
    al(e, y) = (q = σ(e[3]); u = (y - e[1]) / exp(e[2]); log(q * (1 - q)) - e[2] - (u < 0 ? u * (q - 1) : u * q))
    vm(e, y) = exp(e[2]) * cos(y - 2atan(e[1])) - log(2π) - PG._logI0(exp(e[2]))
    dw(e, y) = log(σ(e[1])^(y^exp(e[2])) - σ(e[1])^((y + 1)^exp(e[2])))
    append!(cases, Any[
        (asym_laplace(), (0.3, log(1.4), 0.6), 1.2, 0, e -> al(e, 1.2)),
        (asym_laplace(), (0.3, log(1.4), -1.1), -2.0, 0, e -> al(e, -2.0)),
        # (Distributions' VonMises goes through besselix, which ForwardDiff cannot
        # differentiate: the gradient reference uses our own log I0; the value is
        # checked against Distributions below)
        (von_mises(), (0.4, log(2.5)), 1.1, 0, e -> vm(e, 1.1)),
        (von_mises(), (-1.5, log(0.3)), -2.9, 0, e -> vm(e, -2.9)),
        (zero_inflated_beta(), (0.2, log(4.0), -0.7), 0.0, 0, e -> log(σ(e[3]))),
        (zero_inflated_beta(), (0.2, log(4.0), -0.7), 0.35, 0, e -> log1p(-σ(e[3])) + zb(e, 0.35)),
        (zero_one_inflated_beta(), (0.2, log(4.0), -0.7, 0.4), 0.0, 0, e -> log(σ(e[3])) + log1p(-σ(e[4]))),
        (zero_one_inflated_beta(), (0.2, log(4.0), -0.7, 0.4), 1.0, 0, e -> log(σ(e[3])) + log(σ(e[4]))),
        (zero_one_inflated_beta(), (0.2, log(4.0), -0.7, 0.4), 0.6, 0, e -> log1p(-σ(e[3])) + zb(e, 0.6)),
        (discrete_weibull(), (0.9, log(1.3)), 0.0, 0, e -> dw(e, 0.0)),
        (discrete_weibull(), (0.9, log(1.3)), 3.0, 0, e -> dw(e, 3.0)),
        (negbinomial2(), (log(2.5), log(0.7)), 4.0, 0,
            e -> D.logpdf(D.NegativeBinomial(exp(-e[2]), exp(-e[2]) / (exp(-e[2]) + exp(e[1]))), 4)),
    ])
    for (fam, eta, y, t, ref) in cases
        xv = PG._xdata(fam, [y], [t])
        x = isempty(xv) ? 0.0 : xv[1]
        label = "$(PG.family_name(fam)) [$(PG.link_name(PG.link(fam)))] y=$y"
        @testset "$label" begin
            # normalised value against Distributions.jl
            @test PG._ll(fam, y, x, eta) + PG._ll_const(fam, y, t) ≈ ref(collect(eta)) atol = 1e-9
            # closed-form derivative, called the way the density calls it
            v, g = PG._ll_grad(fam, y, x, eta)
            @test v ≈ PG._ll(fam, y, x, eta) atol = 1e-10
            gad = ForwardDiff.gradient(e -> PG._ll(fam, y, x, Tuple(e)), collect(eta))
            @test collect(g) ≈ gad atol = 1e-8
            # and against the reference itself
            @test collect(g) ≈ ForwardDiff.gradient(ref, collect(eta)) atol = 1e-7
        end
    end
    @testset "com_poisson and hurdle_cumulative" begin
        fc = com_poisson()
        for (e, y) in (((log(2.5), log(0.7)), 3.0), ((log(0.4), log(1.6)), 0.0), ((log(8.0), log(1.3)), 6.0))
            # normalised: P sums to one (exact series for the reference)
            ps = [exp(PG._ll(fc, Float64(k), 0.0, e)) for k in 0:400]
            # (brms's asymptotic log Z for larger mu is accurate, not exact)
            @test sum(ps) ≈ 1 atol = 2e-3          # brms asymptotic log Z: ~1e-3 at mu = 8, nu = 1.3
            @test PG._mean(fc, e, 0) ≈ sum(k * ps[k + 1] for k in 0:400) / sum(ps) atol = 1e-6
            _, g = PG._ll_grad(fc, y, 0.0, e)
            @test collect(g) ≈ ForwardDiff.gradient(t -> PG._ll(fc, y, 0.0, Tuple(t)), collect(e)) atol = 1e-8
        end
        # nu = 1 is the Poisson
        @test PG._ll(fc, 3.0, 0.0, (log(2.5), 0.0)) ≈ D.logpdf(D.Poisson(2.5), 3) atol = 1e-10
        fh = hurdle_cumulative()
        tau = [-1.0, 0.2, 1.5]
        for y in 0:4, m in (-0.4, 0.9)
            eta = (m, 0.3, PG.Thr(tau))
            ref = y == 0 ? log(σ(0.3)) : log1p(-σ(0.3)) + PG._ll(cumulative(), Float64(y), 0.0, (m, PG.Thr(tau)))
            @test PG._ll(fh, Float64(y), 0.0, eta) ≈ ref atol = 1e-12
            _, (gm, gh, tg) = PG._ll_grad(fh, Float64(y), 0.0, eta)
            gt = zeros(3); PG._put!(gt, PG.Thresholds(1:3, PG.ScalarPrior[]), 1, tg)
            ad = ForwardDiff.gradient(e -> PG._ll(fh, Float64(y), 0.0, (e[1], e[2], PG.Thr(e[3:5]))), [m; 0.3; tau])
            @test [gm; gh; gt] ≈ ad atol = 1e-9
        end
        @test sum(PG._hc_probs(fh, (0.2, 0.3, PG.Thr(tau)))) ≈ 1
    end

    @testset "xbeta" begin
        f = xbeta(); e = (0.2, log(5.0), log(0.1))
        # a density on [0, 1] with point masses at 0 and 1: total probability one
        ys = range(1e-6, 1 - 1e-6; length=20001)
        mass = sum(exp(PG._ll(f, y, 0.0, e)) for y in ys) * step(ys) + exp(PG._ll(f, 0.0, 0.0, e)) + exp(PG._ll(f, 1.0, 0.0, e))
        @test mass ≈ 1 atol = 1e-3
        for y in (0.0, 0.35, 1.0)
            _, g = PG._ll_grad(f, y, 0.0, e)
            h = 1e-6
            fd = [(PG._ll(f, y, 0.0, Tuple(collect(e) .+ h .* (1:3 .== j))) - PG._ll(f, y, 0.0, Tuple(collect(e) .- h .* (1:3 .== j)))) / 2h for j in 1:3]
            @test collect(g) ≈ fd rtol = 1e-5 atol = 1e-6
        end
        r = [PG._rand(Xoshiro(k), f, e, 0) for k in 1:20000]
        @test abs(mean(r) - PG._mean(f, e, 0)) < 0.01
    end

    @testset "gen_extreme_value" begin
        # brms's density against Distributions.jl's GeneralizedExtremeValue
        for (mu, ls, xi, y) in ((0.3, log(1.2), 0.25, 1.4), (-0.2, log(0.7), -0.3, 0.1), (0.0, 0.0, 0.0, 0.5))
            @test PG._gev_ll(y, mu, ls, xi) ≈ D.logpdf(D.GeneralizedExtremeValue(mu, exp(ls), xi), y) atol = 1e-10
        end
        xi, dt, dmin, dmax = PG._gev_xi(0.4, -1.5, 2.0)
        @test -1 / 2.0 < xi < 1 / 1.5
        h = 1e-6
        @test dt ≈ (PG._gev_xi(0.4 + h, -1.5, 2.0)[1] - PG._gev_xi(0.4 - h, -1.5, 2.0)[1]) / 2h rtol = 1e-6
        @test dmin ≈ (PG._gev_xi(0.4, -1.5 + h, 2.0)[1] - PG._gev_xi(0.4, -1.5 - h, 2.0)[1]) / 2h rtol = 1e-6
        @test dmax ≈ (PG._gev_xi(0.4, -1.5, 2.0 + h)[1] - PG._gev_xi(0.4, -1.5, 2.0 - h)[1]) / 2h rtol = 1e-6
    end

    @testset "cox: M/I-spline baseline hazard" begin
        y = collect(range(0.5, 6.0; length=40))
        bb = PG.BhazBasis(y)
        Zb, Zc = PG._bhaz_matrices(bb, y)
        @test size(Zb) == (40, 5)
        # each M-spline integrates to one over the boundary knots; I = its integral
        xs = range(bb.lo, bb.hi; length=20001); h = step(xs)
        M = reduce(vcat, (permutedims(first(PG._bhaz_rows(bb, x))) for x in xs))
        @test vec(sum(M; dims=1)) .* h ≈ ones(5) atol = 1e-3
        for (j, x) in enumerate((1.3, 3.0, 5.2))
            k = searchsortedlast(xs, x)
            @test last(PG._bhaz_rows(bb, x)) ≈ vec(sum(M[1:k, :]; dims=1)) .* h atol = 2e-3
        end
        fc = cox()
        for code in (0.0, 1.0, -1.0)
            e = (0.4, (0.3, 0.8))
            v, (gm, gb) = PG._ll_grad(fc, 2.0, code, e)
            ad = ForwardDiff.gradient(t -> PG._ll(fc, 2.0, code, (t[1], (t[2], t[3]))), [0.4, 0.3, 0.8])
            @test [gm, gb.dh, gb.dH] ≈ ad atol = 1e-10
        end
    end

    @testset "multinomial and dirichlet" begin
        fm = PG._with_matrix(multinomial(), ["a", "b", "c"], [2.0 3.0 5.0])
        x = (2.0, 3.0, 5.0); eta = (0.3, -0.5)
        p = [1, exp(0.3), exp(-0.5)] ./ (1 + exp(0.3) + exp(-0.5))
        @test PG._ll(fm, 10.0, x, eta) + PG._ll_const_obs(fm, 10.0, 0, x) ≈ D.logpdf(D.Multinomial(10, p), [2, 3, 5]) atol = 1e-10
        _, g = PG._ll_grad(fm, 10.0, x, eta)
        @test collect(g) ≈ ForwardDiff.gradient(e -> PG._ll(fm, 10.0, x, Tuple(e)), collect(eta)) atol = 1e-10
        fd = PG._with_matrix(dirichlet(), ["a", "b", "c"], [0.2 0.3 0.5])
        xd = log.((0.2, 0.3, 0.5)); ed = (0.3, -0.5, log(4.0))
        @test PG._ll(fd, 1.0, xd, ed) ≈ D.logpdf(D.Dirichlet(4.0 .* p), [0.2, 0.3, 0.5]) atol = 1e-10
        @test PG.dpars(fd) == (:mub, :muc, :phi)
        # logistic_normal against Distributions' MvNormal of the log-ratios
        fl = PG._with_matrix(PG.logistic_normal(), ["a", "b", "c"], [0.2 0.3 0.5])
        xl = PG._xdata(fl, [1.0], Int[])[1]
        L = [1.0 0.0; 0.4 sqrt(1 - 0.16)]
        el = (0.1, -0.3, log(0.8), log(1.3), PG.LnCorL(L))
        Σ = Diagonal([0.8, 1.3]) * L * L' * Diagonal([0.8, 1.3])
        ref = D.logpdf(D.MvNormal([0.1, -0.3], Σ), [xl[1], xl[2]]) - sum(log, [0.2, 0.3, 0.5])
        @test PG._ll(fl, 1.0, xl, el) + PG._ll_const_obs(fl, 1.0, 0, xl) ≈ ref atol = 1e-10
    end

    @testset "mixtures" begin
        mixref(comps, lth, y) = D.logpdf(D.MixtureModel(comps, exp.(lth)), y)
        cases = Any[
            # (family, eta, y, reference)
            (mixture(gaussian(), gaussian()), (-1.0, log(0.6), 1.5, log(1.2), PG.LogTheta(log.([0.3, 0.7]))), 0.4,
             e -> mixref([D.Normal(e[1], exp(e[2])), D.Normal(e[3], exp(e[4]))], log.([0.3, 0.7]), 0.4)),
            (PG._with_theta_formulas(mixture(gaussian(), student()), true), (-1.0, log(0.6), 1.5, log(1.2), log(3.0), 0.8), 2.9,
             e -> mixref([D.Normal(e[1], exp(e[2])), e[3] + exp(e[4]) * D.TDist(1 + exp(e[5]))],
                         [-log1p(exp(e[6])), e[6] - log1p(exp(e[6]))], 2.9)),
            (mixture(poisson(), poisson()), (log(0.8), log(6.0), PG.LogTheta(log.([0.45, 0.55]))), 3.0,
             e -> mixref([D.Poisson(exp(e[1])), D.Poisson(exp(e[2]))], log.([0.45, 0.55]), 3)),
        ]
        for (fam, eta, y, ref) in cases
            nreal = count(e -> e isa Real, eta)
            @test PG._ll(fam, y, 0.0, eta) ≈ ref(collect(Float64, eta[1:nreal])) atol = 1e-10
            v, g = PG._ll_grad(fam, y, 0.0, eta)
            @test v ≈ PG._ll(fam, y, 0.0, eta) atol = 1e-12
            ad = ForwardDiff.gradient(e -> PG._ll(fam, y, 0.0, (e..., eta[(nreal + 1):end]...)), collect(Float64, eta[1:nreal]))
            @test collect(g[1:nreal]) ≈ ad atol = 1e-9
            # d ll / d log theta = responsibilities (sum to one)
            eta[end] isa PG.LogTheta && @test sum(g[end].r) ≈ 1
        end
        @test PG.dpars(mixture(gaussian(), student())) == (:mu1, :sigma1, :mu2, :sigma2, :nu2, :theta)
        @test mixture(gaussian(), gaussian()).order && !mixture(gaussian(), student()).order
        @test PG.brmsfamily("mixture(gaussian, student)") isa PG.Mixture
    end

    @testset "densities without a Distributions.jl equivalent are normalised" begin
        fa = asym_laplace(); ea = (0.3, log(0.8), 0.9)
        # left tail decays at rate (1 - q) / sigma ~ 0.36 here: integrate far out
        ys = range(-90, 30; length=400001)
        for (eta, y) in (((0.4, log(2.5)), 1.1), ((-1.5, log(0.3)), -2.9))
            @test PG._ll(von_mises(), y, 0.0, eta) ≈ D.logpdf(D.VonMises(2atan(eta[1]), exp(eta[2])), y) atol = 1e-10
        end
        @test sum(exp(PG._ll(fa, y, 0.0, ea)) for y in ys) * step(ys) ≈ 1 atol = 1e-4
        # the brms mean formula, and draws that match it
        @test sum(y * exp(PG._ll(fa, y, 0.0, ea)) for y in ys) * step(ys) ≈ PG._mean(fa, ea, 0) atol = 1e-3
        r = [PG._rand(Xoshiro(k), fa, ea, 0) for k in 1:20000]
        @test abs(mean(r) - PG._mean(fa, ea, 0)) < 0.05
        fd = discrete_weibull(); ed = (1.2, log(0.8))
        @test sum(exp(PG._ll(fd, Float64(y), 0.0, ed)) for y in 0:2000) ≈ 1 atol = 1e-9
        @test sum(y * exp(PG._ll(fd, Float64(y), 0.0, ed)) for y in 0:2000) ≈ PG._mean(fd, ed, 0) atol = 1e-8
        rd = [PG._rand(Xoshiro(k), fd, ed, 0) for k in 1:20000]
        @test abs(mean(rd) - PG._mean(fd, ed, 0)) < 0.1
        @test PG._mean(zero_one_inflated_beta(), (0.0, 1.0, 0.0, 0.0), 0) ≈ 0.5
    end

    @testset "exgaussian (no Distributions.jl equivalent)" begin
        # integrates to one and has mean mu
        fam = exgaussian(); eta = (1.2, log(0.4), log(0.7))
        ys = range(-4, 15; length=40001); h = step(ys)
        dens = [exp(PG._ll(fam, y, 0.0, eta)) for y in ys]
        @test sum(dens) * h ≈ 1 atol = 1e-4
        @test sum(ys .* dens) * h ≈ 1.2 atol = 1e-3
        # hand-derived value and gradient, including both tails of the erfc
        # argument z (y far below the mean: z >> 0; far above: z << 0)
        for y in (-30.0, -2.0, 0.5, 2.0, 9.0, 60.0), e0 in (eta, (0.0, log(2.5), log(0.05)), (-1.0, log(0.02), log(3.0)))
            v, g = PG._ll_grad(fam, y, 0.0, e0)
            @test v ≈ PG._ll(fam, y, 0.0, e0) rtol = 1e-10
            @test collect(g) ≈ ForwardDiff.gradient(e -> PG._ll(fam, y, 0.0, Tuple(e)), collect(e0)) rtol = 1e-8 atol = 1e-10
        end
    end

    @testset "cumulative" begin
        tau = [-1.0, 0.2, 1.5]
        for L in (:logit, :probit), y in 1:4, m in (-0.4, 0.9)
            fam = cumulative(; link=L)
            v = PG._ll(fam, Float64(y), 0.0, (m, PG.Thr(tau)))
            # P(y = k) = F(tau_k - m) - F(tau_{k-1} - m)
            Fc(x) = L === :logit ? 1 / (1 + exp(-x)) : D.cdf(D.Normal(), x)
            c = [0.0; Fc.(tau .- m); 1.0]
            @test v ≈ log(c[y + 1] - c[y]) atol = 1e-10
            # closed form: d/d m and the scattered d/d tau
            _, (gm, tg) = PG._ll_grad(fam, Float64(y), 0.0, (m, PG.Thr(tau)))
            gt = zeros(3); PG._put!(gt, PG.Thresholds(1:3, PG.ScalarPrior[]), 1, tg)
            ad = ForwardDiff.gradient(e -> PG._ll(fam, Float64(y), 0.0, (e[1], PG.Thr(e[2:4]))), [m; tau])
            @test [gm; gt] ≈ ad atol = 1e-9
        end
    end

    @testset "sratio / cratio / acat" begin
        tau = [-0.8, 0.3, -0.1]          # unordered is legal for these families
        Fl(L, x) = L === :logit ? 1 / (1 + exp(-x)) : L === :probit ? D.cdf(D.Normal(), x) : 1 - exp(-exp(x))
        function ref(fn, L, y, m)
            F = x -> Fl(L, x)
            K1 = length(tau)
            if fn === :sratio
                p = [F(tau[k] - m) * prod(1 - F(tau[j] - m) for j in 1:k-1; init=1.0) for k in 1:K1]
                push!(p, prod(1 - F(tau[j] - m) for j in 1:K1))
            elseif fn === :cratio
                p = [(1 - F(m - tau[k])) * prod(F(m - tau[j]) for j in 1:k-1; init=1.0) for k in 1:K1]
                push!(p, prod(F(m - tau[j]) for j in 1:K1))
            else   # brms acat_<link>: prod_{j<k} q_j prod_{j>=k} (1 - q_j), normalised
                q = [F(m - t) for t in tau]
                p = [prod(q[1:k-1]; init=1.0) * prod(1 .- q[k:K1]; init=1.0) for k in 1:K1+1]
                p ./= sum(p)
            end
            return log(p[y])
        end
        for fn in (:sratio, :cratio, :acat), L in (:logit, :probit, :cloglog), y in 1:4, m in (-0.4, 0.9)
            fam = PG.brmsfamily(fn; link=L)
            v = PG._ll(fam, Float64(y), 0.0, (m, PG.Thr(tau)))
            @test v ≈ ref(fn, L, y, m) atol = 1e-10
            _, (gm, tg) = PG._ll_grad(fam, Float64(y), 0.0, (m, PG.Thr(tau)))
            gt = zeros(3); PG._put!(gt, PG.Thresholds(1:3, PG.ScalarPrior[], false), 1, 2.0 * tg)
            ad = ForwardDiff.gradient(e -> PG._ll(fam, Float64(y), 0.0, (e[1], PG.Thr(e[2:4]))), [m; tau])
            @test [gm; gt ./ 2] ≈ ad atol = 1e-9
            @test sum(PG._cum_probs(fam, m, tau)) ≈ 1 atol = 1e-12
        end
    end

    @testset "censoring" begin
        wb(e) = D.Weibull(exp(e[2]), exp(e[1] - PG.loggamma(1 + 1 / exp(e[2]))))
        cases_trunc = [
            (gaussian(), [0.3, log(1.7)], 1.1, e -> D.Normal(e[1], exp(e[2]))),
            (lognormal(), [0.2, log(0.6)], 1.4, e -> D.LogNormal(e[1], exp(e[2]))),
            (weibull(), [0.5, log(1.8)], 1.3, wb),
            (exponential(), [0.5], 1.3, e -> D.Exponential(exp(e[1]))),
        ]
        cases = [
            (gaussian(), [0.3, log(1.7)], 1.1, 2.0, e -> D.Normal(e[1], exp(e[2]))),
            (lognormal(), [0.2, log(0.6)], 1.4, 2.3, e -> D.LogNormal(e[1], exp(e[2]))),
            (weibull(), [0.5, log(1.8)], 1.3, 2.1, wb),
            (exponential(), [0.5], 1.3, 2.1, e -> D.Exponential(exp(e[1]))),
        ]
        for (base, eta, y, y2, dist) in cases, code in (-1, 0, 1, 2)
            fam = PG.Censored(base, [code], [y2])
            x = PG._xdata(fam, [y], [0])[1]
            ref = e -> code == 0 ? D.logpdf(dist(e), y) : code == 1 ? D.logccdf(dist(e), y) :
                       code == -1 ? D.logcdf(dist(e), y) : log(D.cdf(dist(e), y2) - D.cdf(dist(e), y))
            @testset "$(PG.family_name(base)) code=$code" begin
                @test PG._ll(fam, y, x, Tuple(eta)) + PG._ll_const_obs(fam, y, 0, x) ≈ ref(eta) atol = 1e-9
                _, g = PG._ll_grad(fam, y, x, Tuple(eta))
                @test collect(g) ≈ ForwardDiff.gradient(ref, eta) atol = 1e-8
            end
        end
        @test PG.cens_code("right") == 1 && PG.cens_code(true) == 1 && PG.cens_code("interval") == 2
        # truncation: log f(y) - log(F(ub) - F(lb)) against Distributions.truncated
        for (base, eta, y, dist) in cases_trunc
            for (lb, ub) in ((0.3, Inf), (-Inf, 4.0), (0.3, 4.0))
                fam = PG.Truncated(base, [lb], [ub])
                x = PG._xdata(fam, [y], [0])[1]
                Fb(e, v) = isfinite(v) ? D.cdf(dist(e), v) : (v > 0 ? 1.0 : 0.0)
                ref = e -> D.logpdf(dist(e), y) - log(Fb(e, ub) - Fb(e, lb))
                @test PG._ll(fam, y, x, Tuple(eta)) + PG._ll_const(fam, y, 0) ≈ ref(eta) atol = 1e-9
                _, g = PG._ll_grad(fam, y, x, Tuple(eta))
                @test collect(g) ≈ ForwardDiff.gradient(ref, eta) atol = 1e-8
            end
        end
        @test_throws ArgumentError build_model(@bf(yp | cens(cc) ~ x), TD; family=poisson())
    end

    # probit tails stay finite where Phi underflows
    v, g = PG._ll_grad(bernoulli(; link=:probit), 1.0, 0.0, (-40.0,))
    @test isfinite(v) && isfinite(g[1])

    @test_throws ArgumentError build_model(@bf(yg ~ x), TD; family=bernoulli())
    @test_throws ArgumentError build_model(@bf(yg ~ x, shape ~ z), TD; family=gaussian())
    @test_throws ArgumentError poisson(; link=:logit)
end

@testset "custom_family" begin
    # a custom normal is the built-in gaussian (up to the dropped constant)
    lc, _ = build_model(@bf(yg ~ x), TD; family=CUSTOM_NORMAL, prior=[prior("normal(0, 5)"; class=:sigma)])
    lgs, _ = build_model(@bf(yg ~ x), TD; family=gaussian(), prior=[prior("normal(0, 5)"; class=:sigma)])
    θ1, θ2 = [0.3, 0.1, -0.2], [1.0, 0.5, 0.4]
    @test LDP.logdensity(lc, θ1) - LDP.logdensity(lgs, θ1) ≈ LDP.logdensity(lc, θ2) - LDP.logdensity(lgs, θ2) atol = 1e-8
    @test "sigma" in variables(lc) && PG.family_name(lc.family) == "normal2"
    @test_throws ArgumentError custom_family("bad"; dpars=(:phi,), ll=(y, p) -> 0.0)
    @test_throws ArgumentError build_model(@bf(yg | vreal(x) ~ x), TD)        # vreal needs a custom family
    fc = brm(@bf(yn | vint(n) ~ x), TD; family=CUSTOM_BB, chains=1, iter=400, warmup=200, seed=1)
    @test "phi" in variables(fc) && all(isfinite, log_lik(fc))
    fcn = brm(@bf(yg ~ x), TD; family=CUSTOM_NORMAL, chains=1, iter=300, warmup=150, seed=1)
    @test size(posterior_predict(fcn)) == (150, length(TD.yg)) && posterior_epred(fcn) ≈ posterior_linpred(fcn)
end

@testset "wiener density" begin
    # the lower-boundary density integrates to the absorption probability
    #   P(lower) = (exp(-2 v a) - exp(-2 v a w)) / (exp(-2 v a) - 1)
    for (v, a, w) in ((0.0, 1.0, 0.5), (0.8, 1.3, 0.4), (-1.2, 0.9, 0.7))
        τ = 0.2
        ts = range(τ + 1e-6, τ + 30; length=300_000); h = step(ts)
        lo = sum(exp(PG._wiener_lower(t, v, a, τ, w)) for t in ts) * h
        up = sum(exp(PG._wiener_lower(t, -v, a, τ, 1 - w)) for t in ts) * h
        pl = v == 0 ? 1 - w : (exp(-2v * a) - exp(-2v * a * w)) / (exp(-2v * a) - 1)
        @test lo ≈ pl atol = 1e-3
        @test lo + up ≈ 1 atol = 1e-3
    end
    # far tails stay finite (each series is taken relative to its leading term)
    @test isfinite(PG._wiener_logf0(400.0, 0.3)) && isfinite(PG._wiener_logf0(1e-3, 0.9))
    @test PG._wiener_logf0(400.0, 0.3) ≈ log(π * sin(π * 0.3)) - π^2 * 400 / 2 rtol = 1e-10
    # small- and large-time series agree where both are accurate
    @test PG._wiener_logf0(0.3, 0.4) ≈ log(sum(k * exp(-k^2 * π^2 * 0.3 / 2) * sin(k * π * 0.4) for k in 1:200) * π) rtol = 1e-8
    ld, _ = build_model(@bf(yrt | dec(dd) ~ x), TD; family=wiener())
    @test all(in(variables(ld)), ["bs", "ndt", "bias"])
    @test_throws ArgumentError build_model(@bf(yrt ~ x), TD; family=wiener())       # needs dec()
end

@testset "zero_inflated_beta_binomial against Distributions" begin
    f = zero_inflated_beta_binomial()
    for (y, n, e) in ((0, 10, (0.3, 1.1, -0.5)), (4, 10, (-0.2, 0.4, 0.8)), (10, 10, (1.0, 2.0, -1.0)))
        mu, phi, zi = PG.logistic(e[1]), exp(e[2]), PG.logistic(e[3])
        bb = Distributions.BetaBinomial(n, mu * phi, (1 - mu) * phi)
        ref = y == 0 ? log(zi + (1 - zi) * Distributions.pdf(bb, 0)) : log(1 - zi) + Distributions.logpdf(bb, y)
        @test PG._ll(f, y, n, e) + PG._ll_const(f, y, n) ≈ ref rtol = 1e-10
    end
end

@testset "gp covariance functions (brms 2.23)" begin
    # spectral densities: brms's Stan functions, literally
    stan_spd(cov, D, s, l, lam) = cov === :matern52 ?
        s^2 * (2^D * π^(D / 2) * PG.SpecialFunctions.gamma((D + 5) / 2) * 5^2.5) / (0.75 * sqrt(π)) * l^D * (5 + l^2 * lam)^(-(D + 5) / 2) :
        cov === :matern32 ?
        s^2 * (2^D * π^(D / 2) * PG.SpecialFunctions.gamma((D + 3) / 2) * 3^1.5) / (0.5 * sqrt(π)) * l^D * (3 + l^2 * lam)^(-(D + 3) / 2) :
        cov === :exponential ?
        s^2 * (2^D * π^(D / 2) * PG.SpecialFunctions.gamma((D + 1) / 2)) / sqrt(π) * l^D * (1 + l^2 * lam)^(-(D + 1) / 2) :
        s^2 * sqrt(2π)^D * l^D * exp(-l^2 * lam / 2)
    for cov in (:exp_quad, :matern52, :matern32, :exponential), D in (1, 2), lam in (0.3, 4.0)
        s, l = 1.3, 0.7
        lq, dlq = PG._gp_logsq(cov, D, l, lam)
        @test 2 * (log(s) + lq) ≈ log(stan_spd(cov, D, s, l, lam)) rtol = 1e-12
        h = 1e-6
        @test dlq ≈ (first(PG._gp_logsq(cov, D, l * exp(h), lam)) - first(PG._gp_logsq(cov, D, l * exp(-h), lam))) / 2h rtol = 1e-6
        k, dk = PG._gp_k(cov, 0.37, s^2, l)
        @test dk ≈ (first(PG._gp_k(cov, 0.37, s^2, l * exp(h))) - first(PG._gp_k(cov, 0.37, s^2, l * exp(-h)))) / 2h rtol = 1e-6
        @test first(PG._gp_k(cov, 0.0, s^2, l)) ≈ s^2
    end
    @test_throws ArgumentError bf("y ~ gp(x, cov = \"matern12\")")
end

@testset "bridge sampling" begin
    # LKJ normaliser: K = 2 by quadrature over the one correlation
    for η in (1.0, 2.5)
        rs = range(-1 + 1e-7, 1 - 1e-7; length=200_001)
        @test PG._lkj_lognorm(2, η) ≈ -log(sum((1 - r^2)^(η - 1) for r in rs) * step(rs)) rtol = 1e-4
    end
    # the constant makes logdensity + C the full log p(y | θ) p(θ) |J|: check against
    # an independent formula on y ~ normal(mu, sigma), mu ~ normal(0, 5), sigma ~ half-normal(0, 2)
    yy = TD.yg[1:40]
    pri = [prior("normal(0, 5)"; class=:Intercept), prior("normal(0, 2)"; class=:sigma)]
    ld, _ = build_model(@bf(yy ~ 1), (; yy); prior=pri)
    C = PG._log_normconst(ld)
    for θ in ([0.3, -0.2], [1.1, 0.4])
        μ, σ = θ[1], exp(θ[2])
        full = sum(Distributions.logpdf.(Distributions.Normal(μ, σ), yy)) + Distributions.logpdf(Distributions.Normal(0, 5), μ) +
               Distributions.logpdf(Distributions.truncated(Distributions.Normal(0, 2); lower=0), σ) + θ[2]
        @test LDP.logdensity(ld, θ) + C ≈ full rtol = 1e-10
    end
    # the bridge estimate against 2-d quadrature of that integrand
    fb = brm(@bf(yy ~ 1), (; yy); prior=pri, chains=4, iter=2000, warmup=1000, seed=3)
    grid_m = range(-2, 4; length=601); grid_l = range(-1.5, 1.5; length=601)
    lv = [LDP.logdensity(ld, [m, l]) + C for m in grid_m, l in grid_l]
    truth = maximum(lv) + log(sum(exp.(lv .- maximum(lv))) * step(grid_m) * step(grid_l))
    @test bridge_sampler(fb) ≈ truth atol = 0.05
    # flat priors (the default on b) make the marginal likelihood undefined: refused
    ff = brm(@bf(yy ~ xx), (; yy, xx=TD.x[1:40]); prior=pri, chains=1, iter=200, warmup=100, seed=1)
    @test_throws ArgumentError bridge_sampler(ff)
end

@testset "mo() interactions (brms layout and names)" begin
    ld, _ = build_model(@bf(yg ~ x + mo(mlev) * z), TD)
    v = variables(ld)
    @test "bsp_momlev" in v && "bsp_momlev:z" in v && "b_z" in v
    @test "simo_momlev1[1]" in v && "simo_momlev:z1[3]" in v          # one simplex per term (brms simo_1, simo_2)
    l2, _ = build_model(@bf(yp ~ mo(mlev):mo(ordf)), TD; family=poisson())
    @test "bsp_momlev:moordf" in variables(l2) && "simo_momlev:moordf2[2]" in variables(l2)
    # new data reproduces the training predictor
    f = brm(@bf(yg ~ x + mo(mlev) * z), TD; chains=1, iter=200, warmup=100, seed=2)
    @test posterior_linpred(f; newdata=(; x=TD.x[1:7], mlev=TD.mlev[1:7], z=TD.z[1:7])) ≈ posterior_linpred(f)[:, 1:7]
    # mo(x):f, f a factor: brms's full dummy coding, one term (bsp + simplex) per level
    l3, _ = build_model(@bf(yg ~ x + mo(mlev):h), TD)
    v3 = variables(l3)
    @test "bsp_momlev:hu" in v3 && "bsp_momlev:hv" in v3 && "simo_momlev:hv1[3]" in v3 && !("b_hv" in v3)
    f3 = brm(@bf(yg ~ x + mo(mlev):h), TD; chains=1, iter=200, warmup=100, seed=2)
    nd = (; x=TD.x[1:9], mlev=TD.mlev[1:9], h=TD.h[1:9])
    @test posterior_linpred(f3; newdata=nd) ≈ posterior_linpred(f3)[:, 1:9]
end

@testset "count cdfs (censoring / truncation) against Distributions" begin
    D = Distributions
    lp(f, a...) = getfield(PG, f)(a...)
    m = 2.5
    for y in (0, 3, 7)
        @test lp(:_disc_lcdf, poisson(), y, 0.0, (log(m),), 0) ≈ D.logcdf(D.Poisson(m), y)
        @test lp(:_disc_lccdf, poisson(), y, 0.0, (log(m),), 0) ≈ D.logccdf(D.Poisson(m), y)
    end
    # far upper tail: summed directly, not 1 - F
    @test lp(:_disc_lccdf, poisson(), 30, 0.0, (log(m),), 0) ≈ D.logccdf(D.Poisson(m), 30) rtol = 1e-9
    @test lp(:_disc_lint, poisson(), 1, 4, 0.0, (log(m),), 0) ≈ log(D.cdf(D.Poisson(m), 4) - D.cdf(D.Poisson(m), 1))
    @test lp(:_disc_lcdf, poisson(), -1, 0.0, (log(m),), 0) == -Inf
    @test lp(:_disc_lccdf, poisson(), -1, 0.0, (log(m),), 0) == 0
    sh = 1.7
    nb = D.NegativeBinomial(sh, sh / (sh + m))
    @test lp(:_disc_lcdf, negbinomial(), 4, 0.0, (log(m), log(sh)), 0) ≈ D.logcdf(nb, 4)
    @test lp(:_disc_lccdf, negbinomial(), 9, 0.0, (log(m), log(sh)), 0) ≈ D.logccdf(nb, 9)
    pr = 0.3
    @test lp(:_disc_lcdf, binomial(), 3, 10.0, (PG.logit(pr),), 10) ≈ D.logcdf(D.Binomial(10, pr), 3)
    @test lp(:_disc_lccdf, binomial(), 8, 10.0, (PG.logit(pr),), 10) ≈ D.logccdf(D.Binomial(10, pr), 8)
    # the truncated poisson's normalised density sums to one over its support
    ld, _ = build_model(@bf(ypp | trunc(lb = 1) ~ 1), (; ypp=collect(1.0:30.0)); family=poisson())
    fam = ld.family
    @test sum(exp(PG._ll(fam, Float64(k), (1.0, 0.0, 0.0, 0.0), (log(m),)) + PG._ll_const(fam, Float64(k), 0)) for k in 1:60) ≈ 1
end

@testset "s(..., bs = \"re\") (mgcv random-effect smooth, brms layout)" begin
    lv = sort(unique(TD.g))
    t = PG.res([TD.g], Union{Nothing,Vector{String}}[lv]; label="sg")
    Xs, Zs = PG.smooth_basis(t, [TD.g])
    @test size(Xs, 2) == 0 && Zs[1] == [TD.g[i] == l ? 1.0 : 0.0 for i in eachindex(TD.g), l in lv]
    # a numeric covariate times the dummies, scaled by ||X||_inf = max |x| (smoothCon)
    t2 = PG.res([TD.x, TD.g], Union{Nothing,Vector{String}}[nothing, lv])
    _, Z2 = PG.smooth_basis(t2, [TD.x, TD.g])
    @test Z2[1] ≈ [TD.g[i] == l ? TD.x[i] : 0.0 for i in eachindex(TD.g), l in lv] ./ maximum(abs.(TD.x))
    # two factors: ~ g:h - 1, the first varying fastest
    t3 = PG.res([TD.g, TD.h], Union{Nothing,Vector{String}}[lv, ["u", "v"]])
    _, Z3 = PG.smooth_basis(t3, [TD.g, TD.h])
    @test size(Z3[1], 2) == 12 && Z3[1][1, findfirst(==(TD.g[1]), lv) + 6 * (TD.h[1] == "v")] == 1
    ld, _ = build_model(@bf(yg ~ x + s(g, bs = "re")), TD)
    @test "sds_sg_1" in variables(ld) && "s_sg_1[6]" in variables(ld) && !any(startswith("bs_"), variables(ld))
    f = brm(@bf(yg ~ x + s(g, bs = "re")), TD; chains=1, iter=200, warmup=100, seed=2)
    @test posterior_linpred(f; newdata=(; x=TD.x[1:9], g=TD.g[1:9])) ≈ posterior_linpred(f)[:, 1:9]
end

@testset "me() latents correlated by default (brms mecor)" begin
    ld, info = build_model(@bf(yg ~ me(x, se) + me(z, se)), TD)
    v = variables(ld)
    @test "corme__mex__mez" in v && "meanme_mex" in v && "sdme_mez" in v
    @test any(r -> r.class === :corme && r.prior isa PG.LKJ, info.priors)
    l0, _ = build_model(@bf(yg ~ me(x, se) + me(z, se)) + set_mecor(false), TD)
    @test !any(startswith("corme"), variables(l0)) && l0.dim == ld.dim - 1
    l3, _ = build_model(@bf(yg ~ me(x, se) + me(z, se) + me(ya, se)), TD)
    @test count(startswith("corme"), variables(l3)) == 3
    # one me() term: nothing to correlate
    l1, _ = build_model(@bf(yg ~ me(x, se) + z), TD)
    @test !any(startswith("corme"), variables(l1))
    f = brm(@bf(yg ~ me(x, se) + me(z, se)), TD; chains=1, iter=200, warmup=100, seed=3)
    @test size(posterior_epred(f)) == (100, length(TD.x))
end
