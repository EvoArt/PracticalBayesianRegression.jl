# Plotting and ArviZ extensions. Not part of runtests.jl (Makie is heavy):
#   julia --project=<env with CairoMakie, UnicodePlots, InferenceObjects> test/extensions.jl [OUTDIR]
using PracticalBayesianRegression, Test, Random, Statistics, LinearAlgebra
using UnicodePlots, InferenceObjects, CairoMakie
const PG = PracticalBayesianRegression
include("testdata.jl")
outdir = length(ARGS) >= 1 ? ARGS[1] : mktempdir()
mkpath(outdir)

@testset "extensions" begin
    fit = brm(@bf(yg ~ x * h + (1 | g)), TD; chains=2, iter=600, warmup=300, seed=1)
    fp = brm(@bf(yp ~ x), TD; family=poisson(), chains=2, iter=600, warmup=300, seed=2)

    @testset "UnicodePlots" begin
        p = pp_check(fit; backend=:unicode)
        @test p isa UnicodePlots.Plot
        println(p)
        @test pp_check(fit; type=:ecdf_overlay, backend=:unicode) isa UnicodePlots.Plot
        @test pp_check(fit; type=:stat, backend=:unicode) isa UnicodePlots.Plot
        @test pp_check(fp; type=:bars, backend=:unicode) isa UnicodePlots.Plot
        @test plot_effects(fit, :x; backend=:unicode) isa UnicodePlots.Plot
        @test all(p -> p isa UnicodePlots.Plot, trace_plot(fit; backend=:unicode))
        @test residual_plot(fit; backend=:unicode) isa UnicodePlots.Plot
    end

    @testset "Makie" begin
        for t in (:dens_overlay, :ecdf_overlay, :hist, :stat)
            f = pp_check(fit; type=t, backend=:makie)
            @test f isa Figure
            save(joinpath(outdir, "pp_$(t).png"), f)
        end
        f = pp_check(fp; type=:bars, backend=:makie)
        @test f isa Figure
        save(joinpath(outdir, "pp_bars.png"), f)
        f = plot_effects(fit, :x => :h; backend=:makie)
        @test f isa Figure
        save(joinpath(outdir, "effects.png"), f)
        save(joinpath(outdir, "effects_factor.png"), plot_effects(fit, :h; backend=:makie))
        f = trace_plot(fit; backend=:makie)
        @test f isa Figure
        save(joinpath(outdir, "trace.png"), f)
        save(joinpath(outdir, "resid.png"), residual_plot(fit; backend=:makie))
    end

    @testset "InferenceObjects" begin
        idata = to_inferencedata(fit)
        @test idata isa InferenceData
        @test :posterior in propertynames(idata) && :log_likelihood in propertynames(idata)
        @test size(idata.posterior[Symbol("b_x")]) == (300, 2)
        @test size(idata.log_likelihood[:yg]) == (300, 2, length(TD.yg))
        @test :diverging in keys(idata.sample_stats)
    end
end
