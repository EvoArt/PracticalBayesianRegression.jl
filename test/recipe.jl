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
        # what a binary writes (draws with sampler stats, unconstrained draws)
        # rebuilds the fit brm would have made: brm(::CompiledModel, data)'s path
        # without launching a process
        dr, _, times, stats = PG._binary_sample(ld, o)
        dir = mktempdir()
        dfile, ufile = joinpath(dir, "draws.csv"), joinpath(dir, "theta.csv")
        open(io -> PG.write_draws_csv(io, variables(ld), D, String[s for s in PG._STAT_NAMES], stats), dfile, "w")
        open(io -> PG.write_draws_csv(io, ["theta[$k]" for k in 1:ld.dim], permutedims(dr, (2, 3, 1))), ufile, "w")
        dt = read_csv(dfile)
        @test dt.names[1:9] == ["chain", "iteration", "lp__", "accept_stat__", "stepsize__", "treedepth__",
                                "n_leapfrog__", "divergent__", "energy__"]
        f = bf("yg ~ x + (1 | g)")
        ld2, info = build_model(f, TD)
        cm = PG.CompiledModel("m", dir, "", ["yg~x+(1|g)"], "gaussian", "identity", PriorSpec[],
                              ["g (factor)", "x (numeric)", "yg (numeric)"], String[], Dict{String,Any}())
        fit2 = PG._fit_from_binary(cm, f, gaussian(), TD, ld2, info, dt, read_csv(ufile), times;
                                   chains=2, iter=300, warmup=150, seed=3, delta=0.8, max_depth=10,
                                   metric=:diag, init=2.0, threads=false)
        @test fit2.draws == fit.draws
        @test nuts_params(fit2)["divergent__"] == nuts_params(fit)["divergent__"]
        @test posterior_summary(fit2)[1].estimate ≈ posterior_summary(fit)[1].estimate
        @test fit2.call[:compiled] === cm
        rm(dir; recursive=true)
        # Rhat / ESS of independent normal draws
        x = randn(Xoshiro(1), 1000, 4)
        rhat, bulk, tail = PG._diagnostics(x)
        @test 0.99 < rhat < 1.01 && 3000 < bulk < 5000 && 2000 < tail < 5000
    end
end

# The model store (store.jl) without juliac: lookup keys, model.toml round
# trip, listing. Building and running a real binary is trimtest/storecheck.jl.
@testset "model store" begin
    @test PG._split_formulas("y~x+s(x,k=5), sigma~z") == ["y~x+s(x,k=5)", "sigma~z"]
    @test PG._split_formulas("y~x") == ["y~x"]
    d = (y=randn(Xoshiro(1), 40), x=randn(Xoshiro(2), 40), g=repeat(["a", "b"], 20))
    P = [prior("normal(0, 2)"; class=:b)]
    k1 = PG._model_identity(@bf(y ~ x + (1 | g)), d, gaussian(), P)[5]
    # the same model written differently has the same key; any change does not
    @test PG._model_identity("y ~ x + (1|g)", d, "gaussian", [prior("normal(0,2)"; class="b")])[5] == k1
    @test PG._model_identity(@bf(y ~ x + (1 | g)), d, student(), P)[5] != k1
    @test PG._model_identity(@bf(y ~ x + (1 | g)), d, gaussian(), PriorSpec[])[5] != k1
    @test PG._model_identity(@bf(y ~ x + (1 | g)), merge(d, (g=repeat([1.0, 2.0], 20),)), gaussian(), P)[5] != k1

    store = mktempdir()
    dir = joinpath(store, "m1")
    mkpath(joinpath(dir, "bin"))
    meta = Dict{String,Any}("name" => "m1", "exe" => "bin/m1", "formula" => "y~x+(1|g)",
                            "formulas" => ["y~x+(1|g)"], "family" => "gaussian", "link" => "identity",
                            "columns" => ["g (factor)", "x (numeric)", "y (numeric)"], "keys" => [k1],
                            "os" => PG._store_os(), "arch" => string(Sys.ARCH), "created" => "2026-11-09 12:00:00",
                            "prior" => [Dict{String,Any}("prior" => "normal(0, 2)", "class" => "b", "coef" => "",
                                                         "group" => "", "dpar" => "", "lb" => "", "ub" => "")])
    PG._write_model_toml(joinpath(dir, "model.toml"), meta)
    # a half-finished build and a foreign-OS model are not listed
    mkpath(joinpath(store, ".build_m2_x"))
    other = joinpath(store, "m3"); mkpath(other)
    PG._write_model_toml(joinpath(other, "model.toml"), merge(meta, Dict{String,Any}("name" => "m3", "os" => "plan9")))
    s = compiled_models(; store=store)
    @test length(s) == 1 && s["m1"].name == "m1"
    cm = compiled_model("m1"; store=store)
    @test cm.formulas == ["y~x+(1|g)"] && cm.priors[1].prior == P[1].prior && cm.priors[1].class === :b
    # by directory, or by the executable's path (walks up to model.toml)
    touch(joinpath(dir, "bin", "m1"))
    @test compiled_model(dir).dir == cm.dir
    @test compiled_model(joinpath(dir, "bin", "m1")).dir == cm.dir
    @test_throws ArgumentError compiled_model(joinpath(store, "no", "such.exe"))
    @test PG._find_by_keys(store, [k1]).name == "m1"
    @test PG._find_by_keys(store, ["nope"]) === nothing
    PG._add_keys!(cm, ["R|y~x"])
    @test "R|y~x" in compiled_model("m1"; store=store).keys
    @test occursin("m1", sprint(show, MIME"text/plain"(), s))
    @test_throws ArgumentError compiled_model("m3"; store=store)
    @test_throws ArgumentError PG._check_columns(cm, (y=[1.0], x=[1.0]))
    @test PG._parse_chain_times("fitting...\nchain times (s): 0.51, 0.49\n", 2) == [0.51, 0.49]
    remove_compiled_model("m1"; store=store)
    @test length(compiled_models(; store=store)) == 0
    rm(store; recursive=true)
end
