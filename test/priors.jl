@testset "priors" begin
    @testset "parsing" begin
        p = PG.to_prior("student_t(3, 0, 2.5)")
        @test p.kind == PG.PK_STUDENT && (p.a, p.b, p.c) == (3.0, 0.0, 2.5)
        @test PG.to_prior("normal(-1, 2)").a == -1.0
        @test PG.isflat(PG.to_prior(""))
        @test PG.to_prior(Distributions.Normal(0, 3)).b == 3.0
        @test PG.to_prior(Distributions.Exponential(2.0)).a == 0.5   # rate
        @test PG.to_prior(3.0 * Distributions.TDist(4) + 1.0).kind == PG.PK_STUDENT
        @test PG.to_prior("lkj(2)") isa PG.LKJ
        @test sprint(show, PG.to_prior("inv_gamma(0.4, 0.3)")) == "inv_gamma(0.4, 0.3)"
        for s in ("uniform(-1, 3)", "logistic(0, 2)", "double_exponential(0, 1)", "skew_normal(0, 1, 4)")
            @test sprint(show, PG.to_prior(s)) == s
        end
        @test PG.to_prior(Distributions.Laplace(0, 2)).kind == PG.PK_LAPLACE
        @test PG.with_bound(PG.uniform(-1, 3), 0.0).lconst ≈ log(4 / 3)
    end

    @testset "kernels and closed-form derivatives" begin
        D = Distributions
        cases = [
            (PG.normal(1, 2), D.Normal(1, 2), 0.7),
            (PG.student_t(3, 0.5, 2.5), 0.5 + 2.5 * D.TDist(3), -1.3),
            (PG.cauchy(0, 1.5), D.Cauchy(0, 1.5), 2.0),
            (PG.exponential(2), D.Exponential(0.5), 0.8),
            (PG.gamma(2, 0.1), D.Gamma(2, 10), 3.0),
            (PG.inv_gamma(0.4, 0.3), D.InverseGamma(0.4, 0.3), 1.2),
            (PG.lognormal(0, 1), D.LogNormal(0, 1), 0.6),
            (PG.uniform(-1, 3), D.Uniform(-1, 3), 0.4),
            (PG.logistic_prior(0.5, 2), D.Logistic(0.5, 2), -1.1),
            (PG.double_exponential(0.2, 1.5), D.Laplace(0.2, 1.5), 1.3),
            (PG.skew_normal_prior(0.3, 1.2, -2.0), D.SkewNormal(0.3, 1.2, -2.0), 0.1),
        ]
        for (p, d, x) in cases
            # normalised density equals Distributions.jl's
            @test PG.prior_logpdf(p, x) ≈ D.logpdf(d, x) atol = 1e-10
            # derivative of the kernel equals AD of the kernel
            _, dk = PG.prior_kernel_grad(p, x)
            @test dk ≈ ForwardDiff.derivative(t -> PG.prior_kernel(p, t), x) atol = 1e-10
        end
        # truncation constant for a half-student prior is log 2
        h = PG.with_bound(PG.student_t(3, 0, 2.5), 0.0)
        @test h.lconst ≈ log(2)
    end

    @testset "defaults match brms 2.23 get_prior" begin
        # median 3.5, mad = 1.4826 * 2 = 2.97 -> 3.0 > 2.5
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        @test sprint(show, PG._default_prior(gaussian(), :Intercept, :mu, y)) == "student_t(3, 3.5, 3)"
        @test sprint(show, PG._default_prior(bernoulli(), :Intercept, :mu, y)) == "student_t(3, 0, 2.5)"
        @test sprint(show, PG._default_prior(negbinomial(), :shape, :mu, y)) == "inv_gamma(0.4, 0.3)"
        @test sprint(show, PG._default_prior(gammafamily(), :shape, :mu, y)) == "gamma(0.01, 0.01)"
        @test sprint(show, PG._default_prior(student(), :nu, :mu, y)) == "gamma(2, 0.1)"
        big = 10 .* y
        # scale grows with mad(y) once it exceeds 2.5
        @test PG._default_prior(gaussian(), :sd, :mu, big).c == round(PG._rmad(big); digits=1)
    end

    @testset "user priors override defaults, most specific wins" begin
        ps = [prior("normal(0, 1)"; class=:b), prior("normal(0, 5)"; class=:b, coef="x")]
        ld, info = build_model(@bf(yg ~ x + z), TD; prior=ps)
        bp = ld.preds[1].b_priors
        @test bp[1].b == 5.0   # x
        @test bp[2].b == 1.0   # z
        @test all(r -> r.source === :user, filter(r -> r.class === :b, info.priors))
    end
end
