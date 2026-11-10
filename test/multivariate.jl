@testset "multivariate models" begin
    rng = Xoshiro(21)
    N = 250
    x = randn(rng, N); z = randn(rng, N); g = rand(rng, ["a", "b", "c", "d", "e"], N)
    e1 = randn(rng, N); e2 = 0.6 .* e1 .+ sqrt(1 - 0.36) .* randn(rng, N)     # residual correlation 0.6
    y1 = 1 .+ 0.5 .* x .+ 0.7 .* e1
    y2 = -0.5 .+ 0.8 .* z .+ 1.2 .* e2
    yc = [rand(rng) < 0.5 ? 0 : rand(rng, 0:5) for _ in 1:N]
    d = (; y1, y2, yc, x, z, g)

    # closed-form gradient against ForwardDiff of the generic joint density
    function check(mv)
        for seed in 1:3
            θ = 0.4 .* randn(Xoshiro(seed), mv.dim)
            gr = zeros(mv.dim)
            v = logdensity_and_gradient!(gr, fresh_workspace(mv), θ)
            @test gr ≈ ForwardDiff.gradient(t -> LDP.logdensity(mv, t), θ) rtol = 1e-8 atol = 1e-8
            θ2 = θ .+ 0.1
            v2 = logdensity_and_gradient!(gr, fresh_workspace(mv), θ2)
            @test v2 - v ≈ LDP.logdensity(mv, θ2) - LDP.logdensity(mv, θ) atol = 1e-8
        end
    end
    mv, infos = build_mv_model(@bf(y1 ~ x) + @bf(y2 ~ z), d)
    @test mv.rescor && length(mv.ridx) == 1          # rescor on by default for gaussian responses
    check(mv)
    mv3, _ = build_mv_model(@bf(y1 ~ x + (1 | g), sigma ~ z) + @bf(y2 ~ z) + @bf(yc ~ x), d;
                            family=[gaussian(), gaussian(), poisson()])
    @test !mv3.rescor                                 # a poisson response turns it off by default
    check(mv3)
    @test_throws ArgumentError build_mv_model(@bf(y1 ~ x) + @bf(yc ~ x) + set_rescor(true), d;
                                              family=[gaussian(), poisson()])
    mv4, _ = build_mv_model(@bf(y1 ~ x + (1 | g), sigma ~ z) + @bf(y2 ~ z) + @bf(yc ~ x) + set_rescor(false), d;
                            family=[gaussian(), gaussian(), poisson()])
    check(mv4)
    mv5, _ = build_mv_model(@bf(y1 ~ x + (1 | g), sigma ~ z) + @bf(y2 ~ z + (1 | g)), d)
    check(mv5)

    # student rescor: one shared nu (brms multi_student_t)
    mvt, _ = build_mv_model(@bf(y1 ~ x) + @bf(y2 ~ z + (1 | g)), d; family=student())
    @test mvt.rescor && mvt.nuidx > 0 && length(mvt.nu_alias) == 2
    check(mvt)
    @test "nu" in variables(mvt) && !("nu_y1" in variables(mvt))
    @test length(variables(mvt)) == length(PG.constrain(mvt, zeros(mvt.dim)))
    # constant() priors on the rescor path
    mvc, _ = build_mv_model(@bf(y1 ~ x) + @bf(y2 ~ z), d; prior=[prior("constant(0.3)"; class=:b, coef="x", resp="y1")])
    check(mvc)

    # mi(): x is missing in places, imputed by its own model, and used to predict y
    xm = copy(x) .+ 0.0; xm = Vector{Union{Missing,Float64}}(xm); xm[[3, 17, 40, 41, 100]] .= missing
    ymi = 0.5 .+ 0.8 .* x .+ 0.3 .* randn(rng, N)
    dmi = (; ymi, xm, z)
    mvm, _ = build_mv_model(@bf(ymi ~ mi(xm) + z) + @bf(xm | mi() ~ z) + set_rescor(false), dmi)
    check(mvm)
    @test "bsp_ymi_mixm" in variables(mvm) && "Ymi_xm[5]" in variables(mvm)
    fmi = brm(@bf(ymi ~ mi(xm) + z) + @bf(xm | mi() ~ z) + set_rescor(false), dmi; chains=2, iter=800, warmup=400, seed=6)
    @test abs(mean(draws(fmi, "bsp_ymi_mixm")) - 0.8) < 0.1
    # imputed values sit near the true x (they are informed by y)
    @test abs(mean(draws(fmi, "Ymi_xm[1]")) - x[3]) < 0.6

    # missing values integrated out (mi = :auto, the default) vs sampled (mi = :sample, brms)
    @test length(mvm.marg) == 1 && mvm.marg[1].gauss && mvm.marg[1].rows == [3, 17, 40, 41, 100]
    mvs, _ = build_mv_model(@bf(ymi ~ mi(xm) + z) + @bf(xm | mi() ~ z) + set_rescor(false), dmi; mi=:sample)
    @test isempty(mvs.marg) && mvs.dim == mvm.dim + 5           # five Ymi parameters when sampled
    check(mvs)
    # mi(x, idx) + index(): x observed on a subset of rows; each y row uses the x row whose
    # index equals its idx (brms's nhanes example)
    N2 = 60; zz = randn(Xoshiro(5), N2)
    xx = Vector{Union{Missing,Float64}}(zz .+ 0.3 .* randn(Xoshiro(6), N2)); xx[[4, 11]] .= missing
    subx = [i > 2 for i in 1:N2]; id = collect(1:N2); idx = rand(Xoshiro(7), 3:N2, N2)
    yy = 0.5 .+ 0.8 .* coalesce.(xx, 0.0)[idx] .+ 0.3 .* randn(Xoshiro(8), N2)
    dix = (; yy, xx, zz, subx, id, idx)
    mvi, _ = build_mv_model(@bf(yy ~ mi(xx, idx)) + @bf(xx | mi() + subset(subx) + index(id) ~ zz) + set_rescor(false), dix)
    check(mvi)
    m1 = only(mvi.subs[1].preds[1].mi)
    @test m1.Jmi == findall(in((4, 11)), idx) && all(isnan, m1.x[m1.Jmi])
    @test all(m1.x[n] == xx[idx[n]] for n in eachindex(idx) if !(idx[n] in (4, 11)))
    @test_throws ArgumentError build_mv_model(@bf(yy ~ mi(xx, idx)) + @bf(xx | mi() + subset(subx) ~ zz) + set_rescor(false), dix)
    # sampled imputation of a bounded family: brms's Ymi <lower = 0> (gamma), (0, 1) (beta)
    xg = Vector{Union{Missing,Float64}}(exp.(0.3 .* x)); xg[[3, 17, 40]] .= missing
    xu = Vector{Union{Missing,Float64}}(1 ./ (1 .+ exp.(-x))); xu[[5, 9]] .= missing
    dgm = (; ymi, xg, xu, z)
    mvg, _ = build_mv_model(@bf(ymi ~ mi(xg) + mi(xu) + z) + @bf(xg | mi() ~ z) + @bf(xu | mi() ~ z) + set_rescor(false),
                            dgm; family=[gaussian(), gammafamily(), betafamily()])
    check(mvg)
    @test "Ymi_xg[3]" in variables(mvg) && "Ymi_xu[2]" in variables(mvg)
    cg = constrain(mvg, 3 .* randn(Xoshiro(2), mvg.dim)); vg = variables(mvg)
    @test all(>(0), cg[startswith.(vg, "Ymi_xg")]) && all(v -> 0 < v < 1, cg[startswith.(vg, "Ymi_xu")])
    fms = brm(@bf(ymi ~ mi(xm) + z) + @bf(xm | mi() ~ z) + set_rescor(false), dmi; chains=2, iter=800, warmup=400,
              seed=6, mi=:sample)
    for n in ("bsp_ymi_mixm", "b_ymi_z", "sigma_ymi", "b_xm_z", "sigma_xm")
        a, b = vec(draws(fmi, n)), vec(draws(fms, n))
        @test abs(mean(a) - mean(b)) < 4 * sqrt(var(a) / 200 + var(b) / 200)      # same posterior
    end
    @test all(isfinite, log_lik(fmi)) && size(log_lik(fmi)) == (800, N)
    # a bernoulli outcome: Gauss-Hermite quadrature; two missing predictors in the same rows
    yb = Float64.(rand(rng, N) .< 1 ./ (1 .+ exp.(-(0.3 .+ 1.2 .* x))))
    zm = Vector{Union{Missing,Float64}}(z .+ 0.0); zm[[3, 17, 60]] .= missing
    db = (; yb, xm, zm, z, w=randn(rng, N))
    mvb, _ = build_mv_model(@bf(yb ~ mi(xm) + mi(zm)) + @bf(xm | mi() ~ w) + @bf(zm | mi() ~ w) + set_rescor(false), db;
                            family=[bernoulli(), gaussian(), gaussian()])
    @test length(mvb.marg) == 1 && !mvb.marg[1].gauss && length(mvb.marg[1].rows) == 6
    @test count(t -> length(t) == 2, mvb.marg[1].terms) == 2      # rows 3 and 17 miss both
    check(mvb)
    # a student x model cannot be integrated: sampled as brms
    mvt, _ = build_mv_model(@bf(ymi ~ mi(xm) + z) + @bf(xm | mi() ~ z) + set_rescor(false), dmi;
                            family=[gaussian(), student()])
    @test isempty(mvt.marg) && mvt.dim == mvs.dim + 1
    # univariate y | mi(): the missing responses leave the likelihood; draws are predictive
    yu = Vector{Union{Missing,Float64}}(ymi .+ 0.0); yu[[5, 9]] .= missing
    ldu, _ = build_model(@bf(yu | mi() ~ z), (; yu, z))
    @test !PG._mi_sampled(ldu) && ldu.ymi == (ldu.dim + 1):(ldu.dim + 2) && iszero(ldu.weights[[5, 9]])
    @test count(startswith("Ymi["), variables(ldu)) == 2
    fu = brm(@bf(yu | mi() ~ z), (; yu, z); chains=1, iter=400, warmup=200, seed=3)
    @test "Ymi[2]" in variables(fu)
    yd = vec(draws(fu, "Ymi[1]"))                      # posterior predictive draws at row 5
    @test abs(mean(yd) - mean(posterior_epred(fu)[:, 5])) < 4 * std(yd) / sqrt(length(yd))
    # any family when integrated out (brms allows mi() for every continuous family)
    yg = Vector{Union{Missing,Float64}}(exp.(0.3 .* z .+ 0.2 .* ymi ./ std(ymi))); yg[[4, 7]] .= missing
    ldg, _ = build_model(@bf(yg | mi() ~ z), (; yg, z); family=gammafamily())
    @test !PG._mi_sampled(ldg) && iszero(ldg.weights[[4, 7]])
    θg = 0.3 .* randn(ldg.dim)
    @test isfinite(LDP.logdensity(ldg, θg))
    fg = brm(@bf(yg | mi() ~ z), (; yg, z); family=gammafamily(), chains=1, iter=300, warmup=150, seed=3)
    @test all(>(0), draws(fg, "Ymi[1]"))
    # sampled (brms's Ymi <lower = 0>): exp transform + Jacobian, d ll / d y by a dual pass
    lgs, _ = build_model(@bf(yg | mi() ~ z), (; yg, z); family=gammafamily(), mi_mode=:sample)
    @test PG._mi_sampled(lgs)
    θs = 0.3 .* randn(Xoshiro(4), lgs.dim); gs = zeros(lgs.dim)
    @test logdensity_and_gradient!(gs, fresh_workspace(lgs), θs) ≈ LDP.logdensity(lgs, θs)
    @test gs ≈ ForwardDiff.gradient(t -> LDP.logdensity(lgs, t), θs) rtol = 1e-8 atol = 1e-8
    @test_throws ArgumentError build_model(@bf(yp | mi() ~ z), (; yp=Vector{Union{Missing,Float64}}([missing; 1.0 .* (1:N-1)]), z);
                                           family=poisson(), mi_mode=:sample)

    # a group-term ID spanning responses: one correlated block across y1 and y2
    mvx, _ = build_mv_model(@bf(y1 ~ x + (1 | p | g)) + @bf(y2 ~ z + (1 + z | p | g)) + set_rescor(false), d)
    @test mvx.cross
    check(mvx)
    vx = variables(mvx)
    @test "cor_g__y1_Intercept__y2_Intercept" in vx && "cor_g__y2_Intercept__y2_z" in vx
    @test count(startswith("cor_g"), vx) == 3
    mvxr, _ = build_mv_model(@bf(y1 ~ x + (1 | p | g)) + @bf(y2 ~ z + (1 | p | g)), d)   # with rescor
    check(mvxr)

    # brms's multivariate names
    vs = variables(mv5)
    # (brms puts the response after the dpar: combine_prefix gives "sigma_y1")
    for n in ("b_y1_Intercept", "b_y1_x", "b_sigma_y1_z", "b_sigma_y1_Intercept", "b_y2_z", "sigma_y2",
              "sd_g__y1_Intercept", "r_g__y1[a,Intercept]", "r_g__y2[e,Intercept]", "rescor__y1__y2",
              "Intercept_y1", "Intercept_sigma_y1")
        @test n in vs
    end
    @test length(vs) == length(unique(vs)) == length(PG.constrain(mv5, zeros(mv5.dim)))

    # per-response priors (brms `resp =`)
    _, inf2 = build_mv_model(@bf(y1 ~ x) + @bf(y2 ~ z), d; prior=[prior("normal(0, 0.5)"; class=:b, resp="y2")])
    @test any(r -> r.class === :b && r.source === :user && r.resp == "y2", inf2[2].priors)
    @test !any(r -> r.class === :b && r.source === :user, inf2[1].priors)
    @test_throws ArgumentError build_mv_model(@bf(y1 ~ x) + @bf(y2 ~ z), d; prior=[prior("normal(0, 1)"; resp="nope")])

    # mvbind form
    f = @bf(mvbind(y1, y2) ~ x) + set_rescor(false)
    @test f isa MVFormula && length(f.forms) == 2 && f.rescor == false

    # recovery, predictions, LOO
    fit = brm(@bf(y1 ~ x) + @bf(y2 ~ z), d; chains=2, iter=1000, warmup=500, seed=3)
    @test fit isa BrmsMVFit
    @test abs(mean(draws(fit, "rescor__y1__y2")) - 0.6) < 0.1
    @test abs(mean(draws(fit, "b_y1_x")) - 0.5) < 0.1
    @test abs(mean(draws(fit, "sigma_y2")) - 1.2) < 0.15
    S = PG.ndraws(fit)
    @test size(posterior_epred(fit)) == (S, N, 2)
    @test size(posterior_epred(fit; resp="y2")) == (S, N)
    Y = posterior_predict(fit; rng=Xoshiro(1))
    @test size(Y) == (S, N, 2)
    # joint predictive draws carry the residual correlation
    r = [cor(Y[s, :, 1] .- vec(mean(Y[:, :, 1]; dims=1)), Y[s, :, 2] .- vec(mean(Y[:, :, 2]; dims=1))) for s in 1:50]
    @test 0.4 < mean(r) < 0.8
    LL = log_lik(fit)
    @test size(LL) == (S, N) && all(isfinite, LL)
    @test isfinite(loo(fit).elpd)
    @test fit["y1"] isa BrmsFit && "b_x" in variables(fit["y1"])
    @test occursin("Residual Correlations", sprint(show, MIME"text/plain"(), fit))
    ft = brm(@bf(y1 ~ x) + @bf(y2 ~ z), d; family=student(), chains=2, iter=800, warmup=400, seed=4)
    @test abs(mean(draws(ft, "rescor__y1__y2")) - 0.6) < 0.12
    @test mean(draws(ft, "nu")) > 5                     # gaussian data: large nu
    @test vec(draws(ft["y1"], "nu")) ≈ vec(draws(ft, "nu"))   # the views see the shared nu
    @test all(isfinite, log_lik(ft)) && size(posterior_predict(ft; rng=Xoshiro(2))) == (PG.ndraws(ft), N, 2)

    # y | subset(s): each response on its own rows (brms)
    sub = [isodd(i) for i in 1:N]
    fs = brm(@bf(y1 | subset(sub) ~ x) + @bf(y2 ~ z), merge(d, (; sub)); chains=1, iter=400, warmup=200, seed=3)
    @test !fs.density.rescor                           # subset turns residual correlations off
    @test fs["y1"].density.N == count(sub) && fs["y2"].density.N == N
    @test size(log_lik(fs; resp="y1")) == (200, count(sub))
    @test_throws ArgumentError log_lik(fs)
    @test_throws ArgumentError posterior_epred(fs)
    @test abs(mean(draws(fs, "b_y1_x")) - 0.5) < 0.2
    @test_throws ArgumentError build_model(@bf(y1 | subset(sub) ~ x), merge(d, (; sub)))
    @test_throws ArgumentError brm(@bf(y1 | subset(sub) ~ x) + @bf(y2 ~ z) + set_rescor(true), merge(d, (; sub)))
end
