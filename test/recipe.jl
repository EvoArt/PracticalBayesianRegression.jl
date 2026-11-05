# Model recipes (recipe.jl) and the pieces of a model binary that run without
# juliac (binary.jl, csvio.jl). The gate: for every supported GRAD_CASE, a
# recipe built from the test data must reproduce build_model EXACTLY on new
# data (other rows, other N) -- the property a refit binary relies on. A
# change to build_model's rules (layout order, coding, centring, default
# priors) that recipe.jl does not mirror fails here.
@testset "model recipes" begin
    PB2 = [prior("normal(0, 2)"; class=:b)]
    sub(d, r) = map(v -> v isa AbstractVector && length(v) == length(d.x) ? v[r] : v, d)
    rng = Xoshiro(7)
    newsets = [sub(TD, 1:180), sub(TD, rand(rng, 1:300, 400))]
    nsupported = 0
    @testset "$name" for (name, f, fam) in GRAD_CASES
        rec = atoms = nothing
        try
            rec, atoms = model_recipe(f, TD; family=fam, prior=PB2)
        catch e
            # unsupported features are refused by name, never silently
            @test e isa ArgumentError && occursin("does not support yet", e.msg)
            continue
        end
        nsupported += 1
        for d2 in newsets
            ref, _ = build_model(f, d2; family=fam, prior=PB2)
            ld2, _ = assemble(rec, PG.frontend_atomvals(atoms, bf(f), d2))
            @test PG._density_mismatch(ref, ld2) === nothing
        end
    end
    @test nsupported >= 50

    @testset "CSV and generated atom code" begin
        f = @bf(yg ~ x + log(abs(z) + 1) + I(x^2) + h + offset(z) + (1 + x | g), sigma ~ z)
        rec, atoms = model_recipe(f, TD)
        cols = Set{Symbol}(keys(TD))
        src, columns = PG._atoms_source(atoms, rec.cat, cols)
        @test "g (factor)" in columns && "x (numeric)" in columns
        PG._check_generated_atoms(src, rec, atoms, bf(f), TD)     # throws on any difference
        # the CSV path end to end: data -> CSV text -> DataTable -> generated atoms -> density
        m = Module(:_RecipeTest)
        Core.eval(m, :(import PracticalBayesianRegression as PBR))
        Base.include_string(m, src)
        tab = PG.parse_csv(PG._table_csv(TD))
        ld2, rows = assemble(rec, Base.invokelatest(Base.invokelatest(getfield, m, :_eval_atoms), tab))
        ref, _ = build_model(f, TD)
        @test length(rows) == 300
        @test PG._density_mismatch(ref, ld2) === nothing
        # quoting, NA, TRUE/FALSE, CRLF
        t = PG.parse_csv("a,b,c\r\n1,\"x, y\",TRUE\r\nNA,\"say \"\"hi\"\"\",FALSE\r\n")
        @test t.raw[2] == ["x, y", "say \"hi\""]
        @test t.num[3] == [1.0, 0.0] && t.miss[1] == [false, true]
        @test_throws ArgumentError PG._numeric_col(PG.parse_csv("a\n1\nfoo\n"), 1)
    end

    @testset "binary driver pieces" begin
        o, cols = PG._parse_binary_args(["--data", "d.csv", "--chains=2", "--iter", "300", "--seed", "5"])
        @test !cols && o.data == "d.csv" && o.chains == 2 && o.iter == 300 && o.warmup == 150 && o.seed == 5
        @test_throws ArgumentError PG._parse_binary_args(["--iter", "300"])          # no --data
        @test_throws ArgumentError PG._parse_binary_args(["--data", "d", "--bogus"])
        # the binary's sampler == brm's, draw for draw
        ld, _ = build_model(@bf(yg ~ x + (1 | g)), TD)
        o, _ = PG._parse_binary_args(["--data", "d", "--chains", "2", "--iter", "300", "--seed", "3", "--no-threads"])
        dr, ndiv, _ = PG._binary_sample(ld, o)
        fit = brm(@bf(yg ~ x + (1 | g)), TD; chains=2, iter=300, seed=3, threads=false)
        D = PG._binary_constrain(ld, dr)
        @test D[:, :, 1] ≈ Matrix{Float64}(fit.chain[PG.FlexiChains.Parameter(variables(ld)[1])])
        # Rhat / ESS of independent normal draws
        x = randn(Xoshiro(1), 1000, 4)
        rhat, bulk, tail = PG._diagnostics(x)
        @test 0.99 < rhat < 1.01 && 3000 < bulk < 5000 && 2000 < tail < 5000
    end
end
