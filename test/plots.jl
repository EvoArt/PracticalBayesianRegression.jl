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
            @test save_plot(joinpath(dir, "f.tex"), f) == joinpath(dir, "f.tex")
            tex = read(joinpath(dir, "f.tex"), String)
            @test occursin("\\begin{tikzpicture}", tex) && !occursin("NaN", tex)
            save_plot(joinpath(dir, "f.typ"), f; standalone=true)
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
    @test_throws ArgumentError save_plot("f.svg", 1)

    # --- report tables (report.jl) ---------------------------------------------------
    names = vcat(["sd(Intercept)"], [PG._coef_label(v) for v in variables(fit) if startswith(v, "b_")], ["sigma"])
    tex = summary_table(fit)                                   # LaTeX by default
    @test occursin(raw"\begin{tabular}", tex) && occursin(raw"\toprule", tex) && occursin(raw"\bottomrule", tex)
    @test occursin("Group-level: g (", tex) && occursin("Regression Coefficients", tex)
    @test occursin(raw"l-95\% CI", tex) && occursin(raw"Bulk\_ESS", tex)
    @test !occursin("NaN", tex) && !occursin('|', tex)
    typ = summary_table(fit; format=:typst)
    @test occursin("table(", typ) && occursin(raw"Bulk\_ESS", typ)    # `_` escaped: no emphasis
    md = summary_table(fit; format=:markdown)
    @test all(n -> occursin(n, md), names)
    @test occursin("<table", summary_table(fit; format=:html))
    # the numbers are the summary's, with a true minus sign
    r = only(posterior_summary(fit; variables=["b_x"]))
    @test occursin(replace(@sprintf("%.2f", r.estimate), "-" => "−"), md)
    @test occursin(replace(@sprintf("%.3f", r.lower), "-" => "−"), summary_table(fit; format=:markdown, digits=3))
    @test occursin("l-90% CI", summary_table(fit; format=:markdown, prob=0.9))
    nod = summary_table(fit; format=:markdown, diagnostics=false)
    @test !occursin("Rhat", nod) && occursin("Estimate", nod)
    pt = prior_table(fit; format=:html)                        # markdown escapes the `_`
    @test occursin("student_t(3", pt) && occursin("default", pt)
    @test occursin(raw"\toprule", prior_table(fit))
    @test_throws ArgumentError summary_table(fit; format=:docx)
    mktempdir() do dir
        for (ext, fmt) in ((".tex", :latex), (".typ", :typst), (".md", :markdown), (".html", :html))
            p = joinpath(dir, "s" * ext)
            @test summary_table(p, fit) == p
            @test read(p, String) == summary_table(fit; format=fmt)
            @test read(prior_table(joinpath(dir, "p" * ext), fit), String) == prior_table(fit; format=fmt)
        end
        @test_throws ArgumentError summary_table(joinpath(dir, "s.docx"), fit)
    end
    # multivariate: one block per response, each summarised from its own fit, then rescor
    rng = Random.Xoshiro(3)
    xm = randn(rng, 80)
    dm = (; y1=1 .+ xm .+ randn(rng, 80), y2=2 .- xm .+ randn(rng, 80), x=xm)
    fm = brm(@bf(y1 ~ x) + @bf(y2 ~ x), dm; chains=2, iter=400, warmup=200, seed=4)
    mmd = summary_table(fm; format=:markdown)
    @test occursin("y1: Regression Coefficients", mmd) && occursin("y2: Regression Coefficients", mmd)
    @test occursin("Residual Correlations", mmd) && !occursin("NaN", mmd)

    # --- Quarto / Pandoc embedding: one raw block per output format --------------------
    e = embed(pp_check(fit))
    @test e isa Embedded && startswith(e.html, "<svg") && occursin(raw"\begin{tikzpicture}", e.latex) &&
          occursin("cetz.canvas", e.typst)
    md = sprint(show, MIME"text/markdown"(), e)
    @test occursin("```{=html}\n<svg", md) && occursin("```{=latex}\n", md) && occursin("```{=typst}\n#import", md)
    @test count("```{=", md) == 3
    @test_throws ArgumentError embed(1)
    st = summary_table(fit; format=:embed)
    @test st.latex == summary_table(fit; format=:latex) && st.typst == summary_table(fit; format=:typst) &&
          st.html == summary_table(fit; format=:html)
    @test prior_table(fit; format=:embed).typst == prior_table(fit; format=:typst)
    # a fence longer than any backtick run inside
    @test occursin("````{=html}\nx```y\n````", sprint(show, MIME"text/markdown"(), Embedded("x```y", "", "")))
end
