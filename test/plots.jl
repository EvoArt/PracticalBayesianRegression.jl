# The default plotting backend, PracticalBayesPlot (a dependency, so tested
# here; the other backends are extensions, tested in extensions.jl).
@testset "plots (PracticalBayesPlot backend)" begin
    PBP = PG.PBP
    fit = brm(@bf(yg ~ x * h + (1 | g)), TD; chains=2, iter=400, warmup=200, seed=1)
    fp = brm(@bf(yp ~ x), TD; family=poisson(), chains=2, iter=400, warmup=200, seed=2)
    @test PG._pick_backend(nothing) === :pbp
    @test_throws ArgumentError PG._pick_backend(:gr)
    svgok(fig) = fig isa PBP.Figure && startswith(PBP.svg(fig), "<svg") && !occursin("NaN", PBP.svg(fig))

    for type in (:dens_overlay, :ecdf_overlay, :hist, :stat)
        @test svgok(pp_check(fit; type))
    end
    @test svgok(pp_check(fp; type=:bars))
    @test pp_check(fit; title="checks").axes[1].title == "checks"
    @test pp_check(fit).axes[1].xlabel == "yg"
    @test svgok(plot_effects(fit, :x))
    @test svgok(plot_effects(fit, :x => :h))
    @test svgok(plot_effects(fit, :h))
    fig = trace_plot(fit)
    @test svgok(fig) && length(fig.axes) == 2 * length(PG._default_trace_vars(fit))
    @test svgok(trace_plot(fit; variables=["b_x"]))
    @test svgok(residual_plot(fit))

    # the same figures as TikZ (LaTeX) and CeTZ (Typst), by file extension
    mktempdir() do dir
        for f in (pp_check(fit), plot_effects(fit, :x => :h), trace_plot(fit))
            PBP.save(joinpath(dir, "f.tex"), f)
            tex = read(joinpath(dir, "f.tex"), String)
            @test occursin("\\begin{tikzpicture}", tex) && !occursin("NaN", tex)
            PBP.save(joinpath(dir, "f.typ"), f; standalone=true)
            typ = read(joinpath(dir, "f.typ"), String)
            @test occursin("cetz.canvas", typ) && !occursin("NaN", typ)
        end
    end

    # PracticalBayesPlot's own plots take a fit
    d = PBP.todraws(fit)
    @test first.(d) == variables(fit)
    @test last(d[1]) == draws(fit, variables(fit)[1])
    @test svgok(PBP.intervals(fit; variables=["b_Intercept", "b_x", "sigma"]))
    @test svgok(PBP.densities(fit; variables=["sigma"]))
end
