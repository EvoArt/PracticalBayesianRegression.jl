# The analytic path (logdensity_and_gradient!) against the generic path
# (LogDensityProblems.logdensity) and ForwardDiff of it, on every structural
# feature. This is the test that licenses the closed-form gradient.
@testset "closed-form gradients" begin
    rng = Xoshiro(3)
    @testset "$name" for (name, f, fam) in GRAD_CASES
        ld, _ = build_model(f, TD; family=fam, prior=[prior("normal(0, 2)"; class=:b)])
        for _ in 1:3
            θ = 0.5 .* randn(rng, ld.dim)
            v, g = LDP.logdensity_and_gradient(ld, θ)
            @test v ≈ LDP.logdensity(ld, θ) rtol = 1e-10
            @test g ≈ ForwardDiff.gradient(t -> LDP.logdensity(ld, t), θ) rtol = 1e-8 atol = 1e-8
        end
        @test length(variables(ld)) == length(constrain(ld, zeros(ld.dim)))
        # (a duplicated name means two terms built for one feature: e.g. a latent and a
        # covariance autocorrelation structure at once)
        @test allunique(variables(ld))
    end

    @testset "non-linear $name" for (name, f, fam) in NL_CASES
        ld, _ = Base.invokelatest(build_model, f, TD; family=fam, prior=NL_PRIORS)
        for _ in 1:3
            θ = 0.5 .* randn(rng, ld.dim)
            v, g = Base.invokelatest(LDP.logdensity_and_gradient, ld, θ)
            @test v ≈ Base.invokelatest(LDP.logdensity, ld, θ) rtol = 1e-10
            @test g ≈ Base.invokelatest(ForwardDiff.gradient, t -> LDP.logdensity(ld, t), θ) rtol = 1e-8 atol = 1e-8
        end
        @test "b_a_Intercept" in variables(ld)
    end
    @test_throws ArgumentError build_model(@bf(ynl ~ a * x, a ~ 1, nl = true), TD)   # flat prior on a

    @testset "stick-breaking simplex reverse pass" begin
        for K in 2:5, alpha in (1.0, 2.5)
            u = randn(rng, K - 1); a = randn(rng, K)
            simo = zeros(K); rem = zeros(K); z = zeros(K - 1)
            PG._stick!(simo, rem, z, u)
            @test sum(simo) ≈ 1 && all(>(0), simo)
            gu = zeros(K - 1)
            v = PG._stick_grad!(gu, a, simo, rem, z, alpha)
            f = function (t)
                s, lj = PG._stick(t, K)
                return sum(a .* s) + lj + (alpha - 1) * sum(log.(s))
            end
            @test v ≈ f(u) - sum(a .* simo) atol = 1e-10
            @test gu ≈ ForwardDiff.gradient(f, u) atol = 1e-10
        end
    end

    @testset "horseshoe prior" begin
        for (f, fam) in ((@bf(yg ~ x + z + h + (1 | g)), gaussian()), (@bf(yp ~ x + z), poisson()),
                         (@bf(yg ~ x + z, sigma ~ x), gaussian()))
            ld, info = build_model(f, TD; family=fam, prior=[prior("horseshoe(1, scale_global = 0.5)")])
            ld2, _ = build_model(f, TD; family=fam, prior=[prior("R2D2(0.5, 2, 0.7)")])
            for _ in 1:3
                θ = 0.5 .* randn(rng, ld2.dim)
                v, g = LDP.logdensity_and_gradient(ld2, θ)
                @test v ≈ LDP.logdensity(ld2, θ) rtol = 1e-10
                @test g ≈ ForwardDiff.gradient(t -> LDP.logdensity(ld2, t), θ) rtol = 1e-8 atol = 1e-8
            end
            @test "R2D2_R2" in variables(ld2)
            for _ in 1:3
                θ = 0.5 .* randn(rng, ld.dim)
                v, g = LDP.logdensity_and_gradient(ld, θ)
                @test v ≈ LDP.logdensity(ld, θ) rtol = 1e-10
                @test g ≈ ForwardDiff.gradient(t -> LDP.logdensity(ld, t), θ) rtol = 1e-8 atol = 1e-8
            end
            @test "hs_global" in variables(ld)
        end
    end

    @testset "bounded coefficients" begin
        ld, _ = build_model(@bf(yg ~ x + z + h + (1 | g)), TD; prior=BOUND_PRIORS)
        for _ in 1:3
            θ = 0.5 .* randn(rng, ld.dim)
            v, g = LDP.logdensity_and_gradient(ld, θ)
            @test v ≈ LDP.logdensity(ld, θ) rtol = 1e-10
            @test g ≈ ForwardDiff.gradient(t -> LDP.logdensity(ld, t), θ) rtol = 1e-8 atol = 1e-8
            c = constrain(ld, θ)
            names = variables(ld)
            @test c[findfirst(==("b_x"), names)] > 0
            @test c[findfirst(==("b_z"), names)] < 0.5
            @test -1 < c[findfirst(==("b_hv"), names)] < 1
        end
        # the truncation constant of a two-sided bound
        p = PG.with_bound(PG.normal(0, 2), -1.0, 1.0)
        @test p.lconst ≈ -log(Distributions.cdf(Distributions.Normal(0, 2), 1) - Distributions.cdf(Distributions.Normal(0, 2), -1))
    end

    @testset "correlation block, q = 2..5" begin
        for q in 2:5
            m = q * (q - 1) ÷ 2
            u = randn(rng, m); A = randn(rng, q, q); gc = zeros(m)
            v = PG._corr_value_grad!(gc, A, u, q, 1.7)
            v2, g2 = PG._corr_grad(A, u, q, 1.7)
            @test v ≈ v2 atol = 1e-12
            @test gc ≈ g2 atol = 1e-10
            # the transform gives a valid correlation Cholesky factor
            L, _ = PG._cholcorr(u, q)
            @test diag(L * L') ≈ ones(q)
        end
    end

    @testset "Float32 path" begin
        ld, _ = build_model(@bf(yg ~ x + (1 | g)), TD; T=Float32)
        θ = 0.3f0 .* randn(rng, Float32, ld.dim)
        v, g = LDP.logdensity_and_gradient(ld, θ)
        @test v isa Float32 && eltype(g) == Float32
        ld64, _ = build_model(@bf(yg ~ x + (1 | g)), TD)
        v64, g64 = LDP.logdensity_and_gradient(ld64, Float64.(θ))
        @test v ≈ v64 rtol = 1e-4
        @test g ≈ g64 rtol = 1e-3 atol = 1e-3
    end

    @testset "no allocation in the gradient" begin
        ld, _ = build_model(@bf(yg ~ x + (1 + x | g)), TD)
        θ = 0.3 .* randn(rng, ld.dim); g = zeros(ld.dim)
        logdensity_and_gradient!(g, ld, θ)
        @test (@allocated logdensity_and_gradient!(g, ld, θ)) == 0
    end
end
