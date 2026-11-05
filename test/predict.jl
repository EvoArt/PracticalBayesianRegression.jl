@testset "predictions" begin
    fit = brm(@bf(yp ~ x + (1 | g)), TD; family=poisson(), chains=2, iter=400, warmup=200, seed=1)
    S = PG.ndraws(fit)
    N = length(TD.x)
    E = posterior_epred(fit)
    @test size(E) == (S, N) && all(>(0), E)
    LP = posterior_linpred(fit)
    @test exp.(LP) ≈ E
    Y = posterior_predict(fit; rng=Xoshiro(1))
    @test size(Y) == (S, N) && all(isinteger, Y)
    # log_lik is normalised: matches Distributions.jl's logpdf
    LL = log_lik(fit)
    @test LL[1, 1] ≈ Distributions.logpdf(Distributions.Poisson(E[1, 1]), TD.yp[1])

    # newdata reproduces the training predictions for the same rows
    nd = (; x=TD.x[1:5], g=TD.g[1:5], yp=TD.yp[1:5])
    @test posterior_epred(fit; newdata=nd) ≈ E[:, 1:5]
    @test log_lik(fit; newdata=nd) ≈ LL[:, 1:5]
    # ordinal: epred is category probabilities summing to one
    fo = brm(@bf(yo ~ x), TD; family=cumulative(), chains=1, iter=300, warmup=150, seed=2)
    P = posterior_epred(fo)
    @test size(P) == (150, N, 4)
    @test all(s -> isapprox(s, 1; atol=1e-10), sum(P; dims=3))
    @test "b_Intercept[3]" in variables(fo)
    # thres(gr = h): one threshold vector per level of h, brms names b_Intercept[u,1]
    fgt = brm(@bf(yo | thres(gr = h) ~ x), TD; family=cumulative(), chains=1, iter=300, warmup=150, seed=2)
    @test "b_Intercept[u,1]" in variables(fgt) && "b_Intercept[v,3]" in variables(fgt)
    @test all(s -> isapprox(s, 1; atol=1e-10), sum(posterior_epred(fgt); dims=3))
    @test posterior_epred(fgt; newdata=(; x=TD.x[1:5], h=TD.h[1:5])) ≈ posterior_epred(fgt)[:, 1:5, :]
    @test_throws ArgumentError posterior_epred(fgt; newdata=(; x=[0.0], h=["w"]))
    fmn = brm(@bf(cbind(ca, cb, cc3) ~ x), TD; family=multinomial(), chains=1, iter=300, warmup=150, seed=2)
    @test "b_mucb_x" in variables(fmn) && "b_mucc3_Intercept" in variables(fmn)
    Em = posterior_epred(fmn)
    @test size(Em) == (150, N, 3)
    @test vec(sum(Em; dims=3)) ≈ repeat(TD.ca .+ TD.cb .+ TD.cc3; inner=150) rtol = 1e-10
    Ym = posterior_predict(fmn; rng=Xoshiro(3))
    @test all(sum(Ym; dims=3)[:, :, 1] .== (TD.ca .+ TD.cb .+ TD.cc3)')
    fdi = brm(@bf(cbind(pa, pb, pc) ~ x), TD; family=dirichlet(), chains=1, iter=300, warmup=150, seed=2)
    @test all(s -> isapprox(s, 1; atol=1e-8), sum(posterior_epred(fdi); dims=3))
    fcs = brm(@bf(yo ~ cs(x) + z), TD; family=acat(), chains=1, iter=300, warmup=150, seed=2)
    @test "bcs_x[1]" in variables(fcs) && "bcs_x[3]" in variables(fcs) && !("b_x" in variables(fcs))
    Pn = posterior_epred(fcs; newdata=(; x=TD.x[1:5], z=TD.z[1:5]))
    @test Pn ≈ posterior_epred(fcs)[:, 1:5, :]
    @test_throws ArgumentError PG.build_model(@bf(yo ~ cs(x)), TD; family=cumulative())
    for fam in (sratio(), cratio(), acat())
        fs = brm(@bf(yo ~ x), TD; family=fam, chains=1, iter=300, warmup=150, seed=2)
        Ps = posterior_epred(fs)
        @test size(Ps) == (150, N, 4) && all(s -> isapprox(s, 1; atol=1e-10), sum(Ps; dims=3))
        @test all(in(1:4), posterior_predict(fs))
    end
    fc = brm(@bf(yo ~ x), TD; family=categorical(), chains=1, iter=300, warmup=150, seed=2)
    Pc = posterior_epred(fc)
    @test size(Pc) == (150, N, 4) && all(s -> isapprox(s, 1; atol=1e-10), sum(Pc; dims=3))
    @test "b_mu4_x" in variables(fc)

    # unseen group level: an error unless allowed
    @test_throws ArgumentError posterior_epred(fit; newdata=(; x=[0.0], g=["zz"]))
    for mode in (:uncertainty, :gaussian, :old_levels)
        En = posterior_epred(fit; newdata=(; x=[0.0, 0.0], g=["zz", "a"]), allow_new_levels=true,
                             sample_new_levels=mode)
        @test size(En) == (S, 2) && all(isfinite, En)
    end
    # re_formula = :none drops group effects: eta = intercept + b * x
    E0 = posterior_linpred(fit; re_formula=:none)
    bi, bx = vec(draws(fit, "b_Intercept")), vec(draws(fit, "b_x"))
    @test E0[:, 3] ≈ bi .+ bx .* TD.x[3]

    # moment matching: an outlier with high Pareto k, checked against exact LOO
    rngm = Xoshiro(5)
    xo = randn(rngm, 40); yo2 = 1 .+ 0.5 .* xo .+ 0.3 .* randn(rngm, 40); yo2[7] += 3.5
    fo2 = brm(@bf(y ~ x), (; y=yo2, x=xo); chains=4, iter=2000, warmup=1000, seed=9)
    l0 = loo(fo2); lm = loo(fo2; moment_match=true)
    @test lm.pareto_k[7] <= l0.pareto_k[7]
    keep = setdiff(1:40, 7)
    fex = brm(@bf(y ~ x), (; y=yo2[keep], x=xo[keep]); chains=4, iter=2000, warmup=1000, seed=10)
    llex = log_lik(fex; newdata=(; y=yo2[7:7], x=xo[7:7]))
    exact = PG.logsumexp(vec(llex)) - log(length(llex))
    @test abs(lm.pointwise[7] - exact) <= abs(l0.pointwise[7] - exact) + 0.05

    # LOO / WAIC / R2 / ranef run and are sane
    l = loo(fit); w = waic(fit)
    @test isfinite(l.elpd) && length(l.pareto_k) == N
    @test abs(l.elpd - w.elpd) < 5
    cmp = loo_compare("ri" => fit, "fe" => brm(@bf(yp ~ x), TD; family=poisson(), chains=2, iter=400, warmup=200, seed=1))
    @test cmp[1].elpd_diff == 0
    re = ranef(fit)
    @test length(re["g"]["Intercept"]) == 6
    gc = group_coef(fit)
    @test gc["g"]["x"][1].estimate ≈ mean(draws(fit, "b_x"))   # x does not vary by g

    # brms-style summaries, loo_R2, update, kfold
    fs = fitted_summary(fit)
    @test fs.Estimate ≈ vec(mean(E; dims=1)) && length(fs.lower) == N
    # LOO-R2 can be slightly negative for a near-useless model (this response is noise)
    @test -0.5 < loo_R2(fit; ndraws=500).estimate <= 1
    f2 = update(fit; chains=1, iter=300, warmup=150)
    @test PG.nchains(f2) == 1
    kf = kfold(fit; K=3, chains=1, iter=300, warmup=150)
    @test isfinite(kf.elpd) && length(kf.pointwise) == N

    # hypothesis and conditional_effects
    h = hypothesis(fit, "x > 0")
    @test h.post_prob == mean(vec(draws(fit, "b_x")) .> 0)
    h2 = hypothesis(fit, ["x = 0", "Intercept - x > 0"])
    @test length(h2) == 2 && isnan(h2[1].evid_ratio)
    ce = conditional_effects(fit, :x; resolution=20)
    @test length(ce.x) == 20 && all(ce.lower .<= ce.estimate .<= ce.upper)
    ce2 = conditional_effects(fit, :x => :g; resolution=10)
    @test length(ce2.x) == 60
    @test occursin("Hypothesis", sprint(show, MIME"text/plain"(), h2))

    # prior-only fits need proper priors
    @test_throws ArgumentError brm(@bf(yg ~ x), TD; sample_prior=:only, chains=1, iter=20, warmup=10)
    fp = brm(@bf(yg ~ x), TD; sample_prior=:only, prior=[prior("normal(0, 1)"; class=:b)],
             chains=2, iter=1000, warmup=500, seed=3)
    @test abs(mean(draws(fp, "b_x"))) < 0.15 && 0.85 < std(vec(draws(fp, "b_x"))) < 1.15

    # several fits: brm_multiple pools chains; model weights favour the better model
    d1 = (; y=TD.yg, x=TD.x); d2 = (; y=TD.yg .+ 0.01 .* randn(Xoshiro(9), N), x=TD.x)
    fmul = brm_multiple(@bf(y ~ x), [d1, d2]; chains=1, iter=300, warmup=150, seed=4)
    @test PG.nchains(fmul) == 2 && size(fmul.draws, 3) == 2 && length(vec(draws(fmul, "b_x"))) == 300
    fx = brm(@bf(yg ~ x), TD; chains=1, iter=400, warmup=200, seed=5)
    f0 = brm(@bf(yg ~ 1), TD; chains=1, iter=400, warmup=200, seed=5)
    for wt in (:stacking, :pseudobma, :loo)
        w = model_weights(fx, f0; weights=wt)
        @test sum(w) ≈ 1 && w[1] > 0.9
    end
    pa = posterior_average(fx, f0; variables=["b_Intercept", "sigma"], ndraws=500)
    @test length(pa["sigma"]) == 500
    @test size(pp_average(fx, f0; ndraws=300)) == (300, N)
    @test size(predictive_interval(fx)) == (N, 2) && size(predictive_error(fx)) == (200, N)
    lo, hi = posterior_interval(fx)["b_x"]
    @test lo < mean(draws(fx, "b_x")) < hi

    @testset "brms helpers: get_prior, VarCorr, loo_*, pp_mixture, accessors" begin
        gp = get_prior(@bf(yg ~ x + (1 + x | g)), TD)
        @test gp isa PriorTable && any(r -> r.class === :cor && r.group == "g", gp) && any(r -> r.class === :sigma, gp)
        @test occursin("student_t", sprint(show, MIME"text/plain"(), gp))
        vp = validate_prior(@bf(yg ~ x), TD; prior=[prior("normal(0, 1)"; class=:b)])
        @test any(r -> r.class === :b && r.source === :user, vp)
        @test length(default_prior(@bf(yp ~ x), TD; family=poisson())) == length(get_prior(@bf(yp ~ x), TD; family=poisson()))
        gpm = get_prior(@bf(yg ~ x) + @bf(yp ~ z), TD; family=[gaussian(), poisson()])
        @test Set(r.resp for r in gpm) == Set(["yg", "yp"])

        fvc = brm(@bf(yg ~ x + (1 + x | g)), TD; chains=1, iter=300, warmup=150, seed=4)
        vc = VarCorr(fvc)
        @test [r.variable for r in vc["g"].sd] == ["Intercept", "x"]
        @test vc["g"].sd[2].estimate ≈ mean(draws(fvc, "sd_g__x"))
        @test length(vc["g"].cor) == 1 && length(vc["g"].cov) == 1
        @test vc["g"].cov[1].estimate ≈
              mean(draws(fvc, "cor_g__Intercept__x") .* draws(fvc, "sd_g__Intercept") .* draws(fvc, "sd_g__x"))
        @test vc["residual__"].sd[1].estimate ≈ mean(draws(fvc, "sigma"))

        # PSIS weights are near uniform for a well-fitting model: LOO means ~ posterior means
        le = loo_epred(fx)
        @test length(le) == N && maximum(abs.(le .- vec(mean(posterior_epred(fx); dims=1)))) < 0.1
        q = loo_predict(fx; type=:quantile, probs=[0.1, 0.9], rng=Xoshiro(2))
        @test size(q) == (N, 2) && all(q[:, 1] .< q[:, 2])
        @test all(>(0), loo_linpred(fx; type=:var))
        @test PG._wquant([3.0, 1.0, 2.0], fill(1 / 3, 3), 0.5) ≈ 1.5     # loo's interpolation rule

        fmx = brm(@bf(yg ~ 1), TD; family=mixture(gaussian(), gaussian()), chains=1, iter=300, warmup=150, seed=3)
        P = pp_mixture(fmx)
        @test size(P) == (PG.ndraws(fmx), N, 2) && all(isapprox.(sum(P; dims=3), 1))
        @test size(pp_mixture(fmx; summary=true)) == (N, 2)
        @test_throws ArgumentError pp_mixture(fx)

        fpy = brm(@bf(yg ~ x), TD; sample_prior=:yes, prior=[prior("normal(0, 1)"; class=:b)],
                  chains=1, iter=300, warmup=150, seed=6)
        pd = prior_draws(fpy)
        @test haskey(pd, "prior_b") && abs(std(vec(pd["prior_b"])) - 1) < 0.2
        @test_throws ArgumentError prior_draws(fx)

        np = nuts_params(fx)
        @test haskey(np, "divergent__") && size(np["treedepth__"]) == (200, 1) && !haskey(np, "lp__")
        @test size(log_posterior(fx)) == (200, 1)
        @test haskey(rhat(fx), "b_x") && 0 < neff_ratio(fx)["b_x"]
        @test niterations(fx) == 200 && nchains(fx) == 1 && ndraws(fx) == 200
    end
end
