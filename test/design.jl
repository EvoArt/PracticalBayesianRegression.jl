# Expected columns/names are R's model.matrix output for the same formulas
# (worked out from R's TermCode rules; spot-checked against brms).
@testset "design matrices (R coding rules)" begin
    @testset "bs() / ns() follow R's splines package" begin
        x = collect(range(1.0, 10.0; length=37))
        # no interior knots: bs(x, df = 3) is the cubic Bernstein basis minus its first column
        B3 = PG._spline_block(PG.SplineBasis(x, false, 3, 3), x)
        u = (x .- 1) ./ 9
        @test B3 ≈ hcat(3 .* u .* (1 .- u) .^ 2, 3 .* u .^ 2 .* (1 .- u), u .^ 3) atol = 1e-12
        # a natural spline with no interior knots is linear, and without the
        # intercept column it vanishes at the lower boundary
        N1 = PG._spline_block(PG.SplineBasis(x, true, 1, 3), x)
        @test size(N1, 2) == 1
        @test N1[:, 1] ≈ N1[end, 1] .* u atol = 1e-12
        xs = sort(vcat(x, [2.2, 2.2, 7.9]))          # ties, as quantile knots may land on data
        for (nat, df, deg) in ((false, 5, 3), (false, 6, 2), (true, 4, 3), (true, 2, 3))
            sb = PG.SplineBasis(xs, nat, df, deg)
            B = PG._spline_block(sb, xs)
            @test size(B) == (length(xs), df) && PG._basis_ncol(sb) == df
            # the underlying B-splines are a partition of unity on [lo, hi]
            @test all(v -> sum(PG._bsplines(sb.knots, v, sb.ord)) ≈ 1, xs)
            # derivative recursion agrees with finite differences
            for v in (1.7, 5.05, 9.6), d in 1:2
                h = 1e-5
                fd = (PG._bsplines(sb.knots, v + h, sb.ord, d - 1) .- PG._bsplines(sb.knots, v - h, sb.ord, d - 1)) ./ 2h
                @test PG._bsplines(sb.knots, v, sb.ord, d) ≈ fd atol = 1e-5
            end
            # extrapolation is continuous at both boundaries
            @test PG._spline_block(sb, [0.999999, 10.000001]) ≈ PG._spline_block(sb, [1.0, 10.0]) atol = 1e-4
            if nat
                # zero second derivative at the boundaries (the ns constraint)
                C = reduce(vcat, [PG._bsplines(sb.knots, b, 4, 2)[2:end]' for b in (1.0, 10.0)])
                @test C * sb.Q[:, 3:end] ≈ zeros(2, df) atol = 1e-10
                # linear beyond the boundary
                E = PG._spline_block(sb, [11.0, 12.0, 13.0])
                @test E[3, :] - E[2, :] ≈ E[2, :] - E[1, :] atol = 1e-10
            end
        end
        # in a formula: brms names, predictions on new data, input checks
        f = brm(@bf(yg ~ ns(x, 3) + bs(z, df = 4)), TD; chains=1, iter=200, warmup=100, seed=1)
        @test "b_nsx31" in variables(f) && "b_bszdfEQ44" in variables(f)
        @test posterior_epred(f; newdata=(; x=TD.x[1:4], z=TD.z[1:4])) ≈ posterior_epred(f)[:, 1:4]
        @test_throws ArgumentError PG.build_model(@bf(yg ~ bs(x, df = 2)), TD)
        @test_throws ArgumentError PG.build_model(@bf(yg ~ ns(x, knots = 3)), TD)
    end

    @testset "poly() matches R's stats::poly" begin
        # R 4.4: x <- c(0.3, 1.2, 2.0, 2.9, 4.1, 5.5); poly(x, 2)
        x = [0.3, 1.2, 2.0, 2.9, 4.1, 5.5]
        pb = PG.PolyBasis(x, 2, false)
        P = PG._poly_block(pb, x)
        # orthonormal columns, orthogonal to the constant (R's defining properties)
        @test P' * P ≈ I(2) atol = 1e-12
        @test all(abs.(sum(P; dims=1)) .< 1e-12)
        # column 1 is the standardised centred x, with R's sign
        @test P[:, 1] ≈ (x .- mean(x)) ./ sqrt(sum(abs2, x .- mean(x)))
        # column 2 is the monic quadratic residual, positive leading coefficient
        r = (x .- mean(x)) .^ 2; A = hcat(ones(6), x)
        q = r - A * (A \ r)
        @test P[:, 2] ≈ q ./ sqrt(sum(abs2, q))
        # new data via the recurrence = refitting the same polynomials there
        xn = [0.0, 3.3]
        Pn = PG._poly_block(pb, xn)
        @test Pn[:, 1] ≈ (xn .- mean(x)) ./ sqrt(sum(abs2, x .- mean(x)))
        cq = A \ r
        qn = (xn .- mean(x)) .^ 2 .- hcat(ones(2), xn) * cq
        @test Pn[:, 2] ≈ qn ./ sqrt(sum(abs2, q))
        @test PG._poly_block(PG.PolyBasis(x, 3, true), x) ≈ hcat(x, x .^ 2, x .^ 3)
        # in a formula: brms names and newdata predictions
        f = brm(@bf(yg ~ poly(x, 2)), TD; chains=1, iter=200, warmup=100, seed=1)
        @test "b_polyx21" in variables(f) && "b_polyx22" in variables(f)
        @test posterior_epred(f; newdata=(; x=TD.x[1:4])) ≈ posterior_epred(f)[:, 1:4]
    end

    d = (; y=1.0:6.0, x=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
         g=["a", "b", "c", "a", "b", "c"], h=["u", "v", "u", "v", "u", "v"])
    names(f) = variables(first(build_model(f, d)))

    # treatment coding, first level dropped
    @test names(@bf(y ~ x + g))[1:4] == ["b_Intercept", "b_x", "b_gb", "b_gc"]
    # no intercept: first factor gets full coding, the next gets contrasts
    @test names(@bf(y ~ 0 + g + h))[1:4] == ["b_ga", "b_gb", "b_gc", "b_hv"]
    # factor x factor interaction with both margins: contrasts on both,
    # first factor varies fastest
    @test filter(n -> startswith(n, "b_"), names(@bf(y ~ g * h))) ==
          ["b_Intercept", "b_gb", "b_gc", "b_hv", "b_gb:hv", "b_gc:hv"]
    # continuous x factor: g is contrast-coded when the margin `x` came
    # earlier, fully coded when it did not
    @test filter(n -> startswith(n, "b_"), names(@bf(y ~ x + x:g))) ==
          ["b_Intercept", "b_x", "b_x:gb", "b_x:gc"]
    @test filter(n -> startswith(n, "b_"), names(@bf(y ~ x:g))) ==
          ["b_Intercept", "b_x:ga", "b_x:gb", "b_x:gc"]
    # function terms are renamed brms-style
    @test "b_logx" in names(@bf(y ~ log(x)))
    @test "b_IxE2" in names(@bf(y ~ I(x^2)))

    # centring: the reported b_Intercept is on the uncentred scale
    ld, _ = build_model(@bf(y ~ x), d)
    θ = [0.3, 2.0, 0.0]
    c = constrain(ld, θ)
    @test c[1] ≈ 0.3 - mean(d.x) * 2.0
    @test c[2] == 2.0

    # unknown levels in new data are an error, not silently the reference level
    @test_throws ArgumentError PG._atom_block(["z"], false, PG.AtomInfo(Any["a", "b"]))
end

@testset "0 + Intercept (brms: the intercept as an ordinary coefficient)" begin
    ld, info = build_model(@bf(yg ~ 0 + Intercept + x), TD)
    v = variables(ld)
    @test "b_Intercept" in v && "b_x" in v && length(filter(startswith("b_"), v)) == 2
    # class b (flat by default), and the design is not centred (no formula intercept)
    @test any(r -> r.class === :b && r.coef == "Intercept", info.priors)
    @test !any(r -> r.class === :Intercept, info.priors)
    @test all(iszero, ld.preds[1].means)
    lp = ld.preds[1]
    θ = randn(ld.dim)
    c = constrain(ld, θ)
    bI, bx = c[findfirst(==("b_Intercept"), v)], c[findfirst(==("b_x"), v)]
    @test PG._eta_generic(lp, θ, ld.N)[1:5] ≈ bI .+ bx .* TD.x[1:5]
end

@testset "bf(..., center = false)" begin
    ld, info = build_model(@bf(yg ~ x + z, center = false), TD)
    @test all(iszero, ld.preds[1].means) && any(r -> r.class === :Intercept, info.priors)
    l2, _ = build_model(@bf(yg ~ x + z), TD)
    @test !all(iszero, l2.preds[1].means)
    @test bf("yg ~ x"; center=false).extra[:center] == false
end
