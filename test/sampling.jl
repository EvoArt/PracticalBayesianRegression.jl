# Posterior correctness by sampling, against references that do not depend
# on this package's own code.
@testset "sampling" begin
    rng = Xoshiro(11)
    N = 400
    X = randn(rng, N, 2)
    y = 0.5 .+ X * [1.0, -0.5] .+ 0.7 .* randn(rng, N)
    d = (; y, x1=X[:, 1], x2=X[:, 2])

    # Flat priors on b and a weak intercept prior: the posterior mean of the
    # slopes is the OLS estimate (marginal posterior is Student-t centred
    # there), to well within Monte Carlo error.
    fit = brm(@bf(y ~ x1 + x2), d; chains=2, iter=1500, warmup=500, seed=3)
    A = hcat(ones(N), X)
    ols = A \ y
    for (k, name) in enumerate(["b_Intercept", "b_x1", "b_x2"])
        x = draws(fit, name)
        se = std(vec(x)) / sqrt(1000)   # conservative: ESS is ~2000 here
        @test abs(mean(x) - ols[k]) < 5se
    end
    # residual sd near the OLS estimate
    s_ols = sqrt(sum(abs2, y - A * ols) / (N - 3))
    @test abs(mean(draws(fit, "sigma")) - s_ols) < 0.03

    r2 = bayes_R2(fit)
    @test 0.5 < r2.estimate < 1
    rows = posterior_summary(fit)
    @test all(r -> r.rhat < 1.02, rows)
    @test length(fixef(fit)) == 3

    # the REPL summary renders (PrettyTables) with the brms sections
    txt = sprint(show, MIME"text/plain"(), fit)
    @test occursin("Regression Coefficients", txt) && occursin("x1", txt) && occursin("sigma", txt)

    @testset "AdvancedHMC agrees with the package NUTS" begin
        fa = brm(@bf(y ~ x1 + x2), d; chains=2, iter=1500, warmup=500, seed=5, sampler=:advancedhmc)
        for v in ("b_x1", "b_x2", "sigma")
            a, b = vec(draws(fa, v)), vec(draws(fit, v))
            @test abs(mean(a) - mean(b)) < 5 * sqrt(var(a) / 500 + var(b) / 500)
        end
        st = fit.chain[PracticalBayesianRegression.FlexiChains.Extra("treedepth__")]
        @test all(1 .<= st .<= 10)
    end

    @testset "ARMA: brms's Stan loop, and recovery" begin
        # literal transcription of brms's model block (Err matrix, J_lag) on
        # rows that are already in series order; our version gets them shuffled
        nser, len = 3, 40
        tt = repeat(1:len, nser); gg = repeat(["s$k" for k in 1:nser]; inner=len)
        base = 0.3 .* randn(rng, nser * len); yy = randn(rng, nser * len)
        arc, mac = [0.4, -0.2], [0.5]
        function stan_loop(base, yy)
            # (not `N`: a closure assigning `N` would overwrite the testset's own)
            nobs = length(yy); Kar, Kma = 2, 1; ml = 2
            Err = zeros(nobs + 1, ml); err = zeros(nobs); mu = copy(base)
            Jlag = [n == nobs || gg[n + 1] != gg[n] ? 0 : min(ml, tt[n]) for n in 1:nobs]
            for n in 1:nobs
                mu[n] += sum(Err[n, i] * mac[i] for i in 1:Kma)
                err[n] = yy[n] - mu[n]
                for i in 1:Jlag[n]
                    Err[n + 1, i] = err[n + 1 - i]
                end
                mu[n] += sum(Err[n, i] * arc[i] for i in 1:Kar)
            end
            return mu
        end
        perm = randperm(rng, length(yy))
        prev, order = PG._ar_prev(Float64.(tt[perm]), gg[perm], 2)
        a = PG.ARTerm{Float64}(2, 1:2, 1, 3:3, prev, order, yy[perm], PG.ScalarPrior[], PG.ScalarPrior[])
        @test PG._ar_generic(base[perm], a, [arc; mac]) ≈ stan_loop(base, yy)[perm]

        # ARMA(1,1) recovery on a long series
        # simulated from the model's own recursion: m = base + ma * err_lag,
        # eta = m + ar * err_lag, err = y - m
        n = 1500; xs = randn(rng, n); ys = zeros(n); err = zeros(n)
        for t in 1:n
            m = 1.0 + 0.5xs[t] + (t > 1 ? 0.4 * err[t - 1] : 0.0)
            η = m + (t > 1 ? 0.6 * err[t - 1] : 0.0)
            ys[t] = η + 0.5 * randn(rng)
            err[t] = ys[t] - m
        end
        # (ma is unbounded in brms's cov = FALSE form, so the likelihood also has the
        # non-invertible |ma| > 1 mode; start chains near 0, as with Stan's init = 0.5)
        fa = brm(@bf(y ~ x + arma(t, p = 1, q = 1)), (; y=ys, x=xs, t=collect(1:n)); chains=2, iter=1000, warmup=500,
                 seed=4, init=0.5)
        @test abs(mean(draws(fa, "ar[1]")) - 0.6) < 0.1
        @test abs(mean(draws(fa, "ma[1]")) - 0.4) < 0.12
        @test abs(mean(draws(fa, "b_x")) - 0.5) < 0.05
    end

    @testset "residual covariance structures (cov = TRUE, cosy, sar)" begin
        # AR(1) residuals in 10 series of 30, rows shuffled (the order is rebuilt from gr, time)
        ns, len = 10, 30
        tc = repeat(1:len, ns); gc = repeat(["s$k" for k in 1:ns]; inner=len)
        ec = zeros(ns * len)
        for k in 1:ns, t in 1:len
            i = (k - 1) * len + t
            ec[i] = (t == 1 ? randn(rng) / sqrt(1 - 0.36) : 0.6 * ec[i - 1] + randn(rng))
        end
        xc = randn(rng, ns * len); yc = 1 .+ 0.5 .* xc .+ 0.7 .* ec
        pm = randperm(rng, ns * len)
        dc = (; y=yc[pm], x=xc[pm], t=tc[pm], g=gc[pm])
        fc = brm(@bf(y ~ x + ar(time = t, gr = g, cov = TRUE)), dc; chains=2, iter=800, warmup=400, seed=5)
        @test abs(mean(draws(fc, "ar[1]")) - 0.6) < 0.1
        @test abs(mean(draws(fc, "sigma")) - 0.7) < 0.1
        LL = log_lik(fc)
        @test size(LL) == (800, ns * len) && all(isfinite, LL)
        @test size(posterior_predict(fc; rng=Xoshiro(1))) == (800, ns * len)
        @test posterior_epred(fc) ≈ posterior_linpred(fc)
        # new data: the structure over the new rows' own groups / times (brms)
        @test posterior_epred(fc; newdata=dc) ≈ posterior_epred(fc)
        @test log_lik(fc; newdata=dc) ≈ LL
        @test size(posterior_predict(fc; newdata=(; x=[0.0, 0.5, 1.0], t=[1, 2, 3], g=["new", "new", "new"]),
                                     allow_new_levels=true)) == (800, 3)
        @test_throws ArgumentError build_model(@bf(y ~ x + ar(time = t, gr = g, cov = TRUE), ar ~ x), dc)
        @test_throws ArgumentError build_model(@bf(y ~ x + ar(time = t, cov = TRUE)), dc)   # repeated times
        @test_throws ArgumentError build_model(@bf(y ~ x + ar(time = t, gr = g, p = 2, cov = TRUE)), dc)
        # sar lag: posterior_epred is (I - rho W)^-1 mu
        fs = brm(@bf(y ~ x + sar(WS, type = "lag")), dc; data2=(; WS=WSAR), chains=1, iter=300, warmup=150, seed=6)
        D = PG.draws(fs, "lagsar"); B0 = PG.draws(fs, "b_Intercept"); B1 = PG.draws(fs, "b_x")
        @test posterior_epred(fs)[1, :] ≈ (I - D[1] .* WSAR) \ (B0[1] .+ B1[1] .* dc.x)
        @test all(isfinite, log_lik(fs))
    end

    @testset "gr(g, dist = \"student\")" begin
        fst = brm(@bf(yg ~ x + (1 | gr(g, dist = "student"))), TD; chains=1, iter=400, warmup=200, seed=7)
        @test "df_g" in variables(fst) && all(>(1), draws(fst, "df_g"))
        @test any(p -> p.class === :df && p.group == "g", fst.info.priors)
        E = posterior_epred(fst; newdata=(; x=[0.0], g=["zz"]), allow_new_levels=true, sample_new_levels=:gaussian)
        @test size(E) == (200, 1) && all(isfinite, E)
        @test_throws ArgumentError bf("y ~ x + (1 | gr(g, dist = \"cauchy\"))")
    end

    @testset "algorithm = laplace / meanfield / fullrank / pathfinder" begin
        fN = brm(@bf(yg ~ x + z), TD; chains=2, iter=1000, warmup=500, seed=1)
        for alg in (:laplace, :meanfield, :fullrank, :pathfinder)
            fa = brm(@bf(yg ~ x + z), TD; algorithm=alg, seed=2)
            @test size(fa.draws) == (fa.density.dim, 1000, 1) && fa.sampler == string(alg)
            for n in ("b_Intercept", "b_x", "sigma")
                a, b = vec(draws(fa, n)), vec(draws(fN, n))
                @test abs(mean(a) - mean(b)) < 0.5 * std(b)            # a near-normal posterior
                alg === :meanfield || @test std(a) ≈ std(b) rtol = 0.25
            end
        end
        @test_throws ArgumentError brm(@bf(yg ~ x), TD; algorithm=:nonsense)
        # multivariate models take the same route
        fm = brm(@bf(yg ~ x) + @bf(ya ~ z) + set_rescor(false), TD; algorithm=:laplace, seed=3, ndraws=200)
        @test size(fm.draws, 2) == 200 && abs(mean(draws(fm, "b_yg_x")) - mean(draws(fN, "b_x"))) < 0.1
    end

    @testset "constant() priors and sample_prior = :yes" begin
        pc = [prior("constant(0.5)"; class=:b, coef="x1"), prior("constant(0.8)"; class=:sigma)]
        ldc, _ = PG.build_model(@bf(y ~ x1 + x2 + (1 | g)), (; d..., g=rand(rng, ["a", "b", "c"], N)); prior=pc)
        θ = 0.3 .* randn(Xoshiro(4), ldc.dim)
        v, gr = LDP.logdensity_and_gradient(ldc, θ)
        @test gr ≈ ForwardDiff.gradient(t -> LDP.logdensity(ldc, t), θ) rtol = 1e-8 atol = 1e-8
        fc = brm(@bf(y ~ x1 + x2), d; prior=pc, chains=2, iter=600, warmup=300, seed=3)
        @test all(==(0.5), draws(fc, "b_x1")) && all(≈(0.8), draws(fc, "sigma"))
        @test std(vec(draws(fc, "b_x2"))) > 0
        @test sprint(show, PG.to_prior("constant(2)")) == "constant(2)"

        fp = brm(@bf(y ~ x1 + x2), d; prior=[prior("normal(0, 1)"; class=:b)], sample_prior=:yes,
                 chains=2, iter=1500, warmup=500, seed=5)
        pb = vec(draws(fp, "prior_b"))
        @test abs(mean(pb)) < 0.1 && abs(std(pb) - 1) < 0.1
        @test all(>(0), draws(fp, "prior_sigma")) && "prior_Intercept" in [string(k.name) for k in keys(fp.chain) if k isa PG.FlexiChains.Parameter]
        h = hypothesis(fp, "x2 = 0")
        # x2 = -0.5 with N = 400: the posterior density at 0 is ~0, so strong evidence against
        @test isfinite(h.evid_ratio) && h.evid_ratio < 0.01
        h0 = hypothesis(fp, "x2 + 0.5 = 0")                  # near the truth: evidence for
        @test h0.evid_ratio > 1
    end

    @testset "cox recovers its coefficient" begin
        nc = 600
        xc = randn(rng, nc)
        tt0 = -log.(rand(rng, nc)) ./ (0.5 .* exp.(0.7 .* xc))     # exponential hazard 0.5 e^{0.7 x}
        cens = tt0 .> 4.0                                          # administrative censoring at 4
        tt = min.(tt0, 4.0)
        fcx = brm(@bf(t | cens(c) ~ x), (; t=tt, c=Int.(cens), x=xc); family=cox(), chains=2, iter=800, warmup=400, seed=4)
        @test abs(mean(draws(fcx, "b_x")) - 0.7) < 0.1
        @test "sbhaz[5]" in variables(fcx)
        @test all(isfinite, log_lik(fcx))
    end

    @testset "gr(by =) and mm() names" begin
        ld, _ = PG.build_model(@bf(yg ~ x + (1 + x | gr(gh, by = h))), TD)
        vs = variables(ld)
        @test "sd_gh__Intercept:hu" in vs && "sd_gh__x:hv" in vs && "cor_gh__Intercept:hu__x:hu" in vs
        @test any(startswith("r_gh[ua,"), vs)
        @test length(vs) == length(unique(vs)) == length(PG.constrain(ld, zeros(ld.dim)))
        lm, _ = PG.build_model(@bf(yp ~ x + (1 | mm(g, g2))), TD; family=poisson())
        @test "sd_mmgg2__Intercept" in variables(lm) && "r_mmgg2[q,Intercept]" in variables(lm)
        f = brm(@bf(yg ~ x + (1 | gr(gh, by = h))), TD; chains=1, iter=300, warmup=150, seed=2)
        @test size(posterior_epred(f; newdata=(; x=TD.x[1:6], gh=TD.gh[1:6], h=TD.h[1:6]))) == (150, 6)
        @test posterior_epred(f; newdata=(; x=TD.x[1:6], gh=TD.gh[1:6], h=TD.h[1:6])) ≈ posterior_epred(f)[:, 1:6]
        @test length(ranef(f)["gh"]["Intercept"]) == length(unique(TD.gh))
    end

    @testset "gp(): exact and approximate" begin
        ngp = 60
        xg = sort(4 .* rand(rng, ngp)); yg = sin.(2 .* xg) .+ 0.2 .* randn(rng, ngp)
        for spec in (@bf(y ~ gp(x)), @bf(y ~ gp(x, k = 15)))
            fg = brm(spec, (; y=yg, x=xg); chains=2, iter=500, warmup=250, seed=8)
            E = vec(mean(posterior_epred(fg); dims=1))
            @test maximum(abs.(E .- sin.(2 .* xg))) < 0.35
            @test "sdgp_gpx" in variables(fg) && "lscale_gpx" in variables(fg)
            En = vec(mean(posterior_epred(fg; newdata=(; x=[1.0, 2.5])); dims=1))
            @test all(abs.(En .- sin.(2 .* [1.0, 2.5])) .< 0.35)
        end
        # brms's tuned length-scale prior: inv_gamma with 1% below/above the distance range
        p = PG._lscale_prior(reshape(collect(range(0, 1; length=20)), :, 1))
        IG = Distributions.InverseGamma(p.a, p.b)
        @test Distributions.cdf(IG, 1 / 19) ≈ 0.01 atol = 1e-4
        @test Distributions.ccdf(IG, 1.0) ≈ 0.01 atol = 1e-4
        # Cholesky reverse pass against finite differences
        A = [4.0 1 0.5; 1 3 0.2; 0.5 0.2 2]
        Lbar = LowerTriangular(randn(Xoshiro(1), 3, 3))
        S = PG._chol_rev(Matrix(cholesky(A).L), Matrix(Lbar))
        dA = randn(Xoshiro(2), 3, 3); dA = dA + dA'
        h = 1e-6
        fd = sum(Matrix(Lbar) .* (Matrix(cholesky(A + h * dA).L) - Matrix(cholesky(A - h * dA).L))) / 2h
        @test sum(S .* dA) ≈ fd rtol = 1e-5
    end

    @testset "me(): measurement error" begin
        nme = 500
        xt = randn(rng, nme); sdx = fill(0.5, nme); xo = xt .+ sdx .* randn(rng, nme)
        ym = 1 .+ 0.8 .* xt .+ 0.3 .* randn(rng, nme)
        fme = brm(@bf(y ~ me(x, sdx)), (; y=ym, x=xo, sdx); chains=2, iter=1000, warmup=500, seed=6)
        @test "bsp_mexsdx" in variables(fme) && "sdme_mex" in variables(fme)
        # attenuation is corrected: naive OLS slope ~ 0.8 / 1.25, me() recovers ~ 0.8
        @test abs(mean(draws(fme, "bsp_mexsdx")) - 0.8) < 0.1
        @test abs(mean(draws(fme, "sdme_mex")) - 1) < 0.15
        @test size(posterior_epred(fme; newdata=(; x=[0.0, 1.0], sdx=[0.5, 0.5]))) == (PG.ndraws(fme), 2)
    end

    @testset "shared group IDs (1 | p | g) across dpars" begin
        Jg = 40; ng = 30
        a = randn(rng, Jg); b = 0.7 .* a .+ sqrt(1 - 0.49) .* randn(rng, Jg)    # cor(mu effect, log-sigma effect) = 0.7
        gidx = repeat(1:Jg, inner=ng); xs = randn(rng, Jg * ng)
        ys = [0.5 + 0.3xs[i] + 0.8a[gidx[i]] + exp(-0.5 + 0.4b[gidx[i]]) * randn(rng) for i in eachindex(xs)]
        dd = (; y=ys, x=xs, g=string.("g", lpad.(gidx, 2, "0")))
        fs = brm(@bf(y ~ x + (1 | p | g), sigma ~ (1 | p | g)), dd; chains=2, iter=1000, warmup=500, seed=3)
        vs = variables(fs)
        @test "cor_g__Intercept__sigma_Intercept" in vs && "sd_g__sigma_Intercept" in vs && "r_g__sigma[g01,Intercept]" in vs
        @test count(startswith("cor_g"), vs) == 1
        @test mean(draws(fs, "cor_g__Intercept__sigma_Intercept")) > 0.3
        @test abs(mean(draws(fs, "sd_g__Intercept")) - 0.8) < 0.25
        # three-way: names of the joint correlation follow brms
        ld, _ = PG.build_model(@bf(yg ~ x + (1 + x | p | g), sigma ~ z + (1 | p | g)), TD)
        nm = variables(ld)
        @test all(n -> n in nm, ["cor_g__Intercept__x", "cor_g__Intercept__sigma_Intercept", "cor_g__x__sigma_Intercept"])
        @test length(nm) == length(unique(nm)) == length(PG.constrain(ld, zeros(ld.dim)))
    end

    @testset "mixture recovers its components" begin
        nmix = 800
        comp = rand(rng, nmix) .< 0.3
        ym = [c ? -2 + 0.5randn(rng) : 2 + randn(rng) for c in comp]
        fm = brm(@bf(y ~ 1), (; y=ym); family=mixture(gaussian(), gaussian()), chains=2, iter=1000, warmup=500, seed=2)
        m1, m2 = vec(draws(fm, "b_mu1_Intercept")), vec(draws(fm, "b_mu2_Intercept"))
        @test all(m1 .< m2)                       # brms's order = "mu" identification
        @test abs(mean(m1) + 2) < 0.15 && abs(mean(m2) - 2) < 0.15
        @test abs(mean(draws(fm, "theta1")) - 0.3) < 0.05
        @test abs(mean(draws(fm, "sigma1")) - 0.5) < 0.1
        @test abs(mean(posterior_epred(fm)) - mean(ym)) < 0.1
        @test occursin("theta1", sprint(show, MIME"text/plain"(), fm))
    end

    @testset "dense metric" begin
        # a correlated, badly scaled Gaussian: the dense metric should learn
        # the covariance, and draws should match it
        Σ = [4.0 1.9 0.0; 1.9 1.0 0.0; 0.0 0.0 0.01]
        P = inv(Σ)
        gradf! = (g, x) -> (g .= -(P * x); -dot(x, P * x) / 2)
        dr, st = PracticalBayesianRegression.nuts_chain(gradf!, [1.0, -1.0, 0.1], 4000, 1000; metric=:dense, rng=Xoshiro(2))
        @test isapprox(cov(dr'), Σ; rtol=0.15, atol=0.02)
        @test maximum(abs.(mean(dr; dims=2))) < 0.15
        # with the right metric the target is isotropic: short trees
        @test mean(st["treedepth__"]) < 3
        drd, std_ = PracticalBayesianRegression.nuts_chain(gradf!, [1.0, -1.0, 0.1], 4000, 1000; rng=Xoshiro(2))
        @test mean(std_["n_leapfrog__"]) > mean(st["n_leapfrog__"])
        fdn = brm(@bf(y ~ x1 + x2), d; chains=2, iter=1500, warmup=500, seed=6, metric=:dense)
        @test abs(mean(draws(fdn, "b_x1")) - ols[2]) < 0.02
        @test fdn.sampler == "nuts (dense metric)"
        @test_throws ArgumentError PracticalBayesianRegression.nuts_chain(gradf!, zeros(3), 10, 10; metric=:full)
    end

    @testset "unknown sampler symbol" begin
        @test_throws ArgumentError brm(@bf(y ~ x1 + x2), d; chains=1, iter=20, warmup=10, sampler=:static_nuts)
    end

    @testset "non-linear model recovers its parameters" begin
        fnl = brm(@bf(ynl ~ a * exp(-b * abs(x)), a ~ 1, b ~ 1, nl = true), TD; prior=NL_PRIORS,
                  chains=2, iter=1000, warmup=500, seed=7)
        @test abs(mean(draws(fnl, "b_a_Intercept")) - 2.0) < 0.1
        @test abs(mean(draws(fnl, "b_b_Intercept")) - 0.7) < 0.1
        nd = (; x=[0.0, 1.0])
        E = posterior_epred(fnl; newdata=nd)
        @test abs(mean(E[:, 1]) - 2.0) < 0.1
    end

    @testset "multi-variable smooths" begin
        n2 = 300
        x1 = rand(rng, n2); x2 = rand(rng, n2)
        fsurf(a, b) = sin(3a) * cos(3b)
        y2 = fsurf.(x1, x2) .+ 0.1 .* randn(rng, n2)
        for spec in (@bf(y ~ s(x1, x2, k = 20)), @bf(y ~ t2(x1, x2)))
            fs = brm(spec, (; y=y2, x1, x2); chains=2, iter=800, warmup=400, seed=3)
            E = vec(mean(posterior_epred(fs); dims=1))
            @test sqrt(mean(abs2, E .- fsurf.(x1, x2))) < 0.1
            En = vec(mean(posterior_epred(fs; newdata=(; x1=[0.2, 0.7], x2=[0.5, 0.3])); dims=1))
            @test all(abs.(En .- fsurf.([0.2, 0.7], [0.5, 0.3])) .< 0.2)
        end
        ld, _ = PG.build_model(@bf(yg ~ t2(x, z)), TD)
        @test count(startswith("sds_t2xz_"), variables(ld)) == 3            # brms: three t2 penalties
        @test count(startswith("b_t2xz_"), variables(ld)) == 3
        lb, _ = PG.build_model(@bf(yg ~ h + s(x, by = h)), TD)
        @test "sds_sx:hu_1" in variables(lb) && "b_sx:hv_1" in variables(lb)
        # 1-d kernels match mgcv's constants
        @test PG._tps_eta(2.0, 1, 2) ≈ 8 / 12
        @test PG._tps_eta(2.0, 2, 2) ≈ 4 * log(2) / (8π)
    end

    @testset "smooth recovers a curve" begin
        xs = range(-2, 3; length=300)
        ys = sin.(xs) .+ 0.2 .* randn(rng, 300)
        fsm = brm(@bf(y ~ s(x)), (; y=ys, x=collect(xs)); chains=2, iter=1000, warmup=500, seed=9)
        E = vec(mean(posterior_epred(fsm); dims=1))
        @test maximum(abs.(E .- sin.(xs))) < 0.2
        @test "sds_sx_1" in variables(fsm) && "b_sx_1" in variables(fsm)
        En = posterior_epred(fsm; newdata=(; x=[0.0, 1.5]))
        @test all(abs.(vec(mean(En; dims=1)) .- sin.([0.0, 1.5])) .< 0.2)
        # posterior_smooths: the term alone = epred minus the intercept (gaussian, identity link)
        Ps = posterior_smooths(fsm; smooth="s(x)")
        @test Ps .+ vec(draws(fsm, "b_Intercept")) ≈ posterior_epred(fsm)
        cs = conditional_smooths(fsm; smooth="s(x)", resolution=20)
        @test length(cs.x) == 20 && all(cs.lower .<= cs.estimate .<= cs.upper)
        @test_throws ArgumentError posterior_smooths(fsm; smooth="s(z)")
    end

    @testset "multilevel model recovers its structure" begin
        J = 15
        g = rand(rng, 1:J, N)
        a = 0.8 .* randn(rng, J)
        y2 = 1 .+ 0.5 .* X[:, 1] .+ a[g] .+ 0.5 .* randn(rng, N)
        d2 = (; y=y2, x=X[:, 1], g=string.("g", lpad.(g, 2, "0")))
        f2 = brm(@bf(y ~ x + (1 | g)), d2; chains=2, iter=1500, warmup=750, seed=5)
        @test abs(mean(draws(f2, "b_x")) - 0.5) < 0.1
        @test 0.4 < mean(draws(f2, "sd_g__Intercept")) < 1.5
        @test abs(mean(draws(f2, "sigma")) - 0.5) < 0.06
    end
end
