# The PracticalBayesPlot backend: the default when no other plotting package
# is loaded. PracticalBayesPlot has no dependencies and loads in
# milliseconds, so every fit can be plotted without installing a plotting
# package. Its figures are SVG; VS Code, Jupyter and Pluto show them inline
# and `PracticalBayesPlot.save` writes them to file.

# Draws of every parameter, so that PracticalBayesPlot's own plots
# (`PBP.traceplot(fit)`, `PBP.intervals(fit; variables=...)`,
# `PBP.densities(fit)`) take a fit directly.
PBP.todraws(fit::Union{BrmsFit,BrmsMVFit}) =
    Pair{String,Matrix{Float64}}[v => draws(fit, v) for v in variables(fit)]

function _pp_check(::Val{:pbp}, fit::BrmsFit; type::Symbol=:dens_overlay, ndraws::Int=50, stat=mean,
                   rng=Random.default_rng(), kwargs...)
    n = type === :stat ? size(fit.draws, 2) * nchains(fit) : ndraws   # :stat uses every draw
    y, yrep = pp_check_data(fit; ndraws=n, rng)
    xlabel = type === :stat ? "" : string(fit.info.formula.response)
    return PBP.ppcheck(y, yrep; type, stat, axis=(; xlabel, kwargs...))
end

function _plot_effects(::Val{:pbp}, ce::NamedTuple)
    names = keys(ce)
    xname = names[1]
    by = length(names) == 5 ? names[2] : nothing
    fig = PBP.Figure(; size=(600, 400))
    ax = PBP.Axis(fig[1, 1]; xlabel=string(xname), ylabel="estimate")
    groups = by === nothing ? [nothing] : unique(ce[by])
    for (k, gv) in enumerate(groups)
        sel = gv === nothing ? eachindex(ce.estimate) : findall(==(gv), ce[by])
        x = ce[xname][sel]
        col = PBP.wong_colors()[mod1(k, 7)]
        if eltype(x) <: Real
            PBP.band!(ax, x, ce.lower[sel], ce.upper[sel]; color=(col, 0.25))
            PBP.lines!(ax, x, ce.estimate[sel]; color=col, linewidth=2, label=gv === nothing ? "" : string(gv))
        else
            PBP.rangebars!(ax, string.(x), ce.lower[sel], ce.upper[sel]; color=col, whiskerwidth=8)
            PBP.scatter!(ax, string.(x), ce.estimate[sel]; color=col, markersize=10)
        end
    end
    by === nothing || PBP.axislegend(ax, string(by); position=:lt)
    return fig
end

_trace_plot(::Val{:pbp}, fit::BrmsFit, variables) =
    PBP.traceplot(Pair{String,Matrix{Float64}}[String(v) => draws(fit, v) for v in variables])

function _residual_plot(::Val{:pbp}, fit::BrmsFit)
    fig = PBP.Figure(; size=(600, 400))
    ax = PBP.Axis(fig[1, 1]; xlabel="fitted (posterior mean)", ylabel="residual")
    PBP.scatter!(ax, StatsAPI.fitted(fit), StatsAPI.residuals(fit); color=(:steelblue, 0.5), markersize=5)
    PBP.hlines!(ax, 0.0; color=:black, linestyle=:dash)
    return fig
end

"""
    save_plot(path, fig; standalone = false) -> path

Write a figure from [`pp_check`](@ref), [`plot_effects`](@ref),
[`trace_plot`](@ref) or [`residual_plot`](@ref) (the default
PracticalBayesPlot backend) to `path`. The extension picks the format:

- `.svg`: an SVG image; `.html`: a page showing it;
- `.tex`: a TikZ picture for `\\input` into a LaTeX document, its text set
  by LaTeX in the document's font (needs `\\usepackage{tikz}`);
- `.typ`: a CeTZ drawing for `#include` into a Typst document, its text set
  by Typst.

`standalone = true` makes a `.tex` or `.typ` file a document that compiles
on its own. It is PracticalBayesPlot's `save`, so this works without adding
PracticalBayesPlot to your environment. Figures from Makie are saved with
Makie's own `save`.
"""
function save_plot(path::AbstractString, fig::PBP.Figure; standalone::Bool=false)
    PBP.save(path, fig; standalone)
    return path
end
save_plot(path::AbstractString, fig; kwargs...) =
    throw(ArgumentError("save_plot writes PracticalBayesPlot figures (the default backend); " *
                        "save a $(typeof(fig)) with its own package (Makie: `save(path, fig)`)"))

"""
    Embedded

A figure or table in three forms, for a document that Pandoc renders (Quarto,
R Markdown): HTML, LaTeX and Typst. A code cell whose value is an `Embedded`
puts all three into the document as raw blocks, and each output format keeps
its own: the web page gets the SVG or HTML table, a LaTeX PDF the TikZ picture
or booktabs table, a Typst PDF the CeTZ drawing or Typst table. Made by
[`embed`](@ref) and by `summary_table(...; format = :embed)`.
"""
struct Embedded
    html::String
    latex::String
    typst::String
end

# Raw blocks, which Pandoc drops for every output format but their own. The
# fence is longer than any run of backticks in the content.
function Base.show(io::IO, ::MIME"text/markdown", e::Embedded)
    for (fmt, s) in (("html", e.html), ("latex", e.latex), ("typst", e.typst))
        n = max(3, maximum(m -> length(m.match) + 1, eachmatch(r"`+", s); init=0))
        f = repeat('`', n)
        print(io, f, "{=", fmt, "}\n", chomp(s), "\n", f, "\n\n")
    end
end
Base.show(io::IO, ::MIME"text/plain", e::Embedded) =
    print(io, "Embedded (HTML, LaTeX and Typst forms: show it in a Quarto cell)")

"""
    embed(fig) -> Embedded

A figure for a Quarto (or other Pandoc) document: the value of a code cell
that is `embed(pp_check(fit))` shows the figure as SVG in HTML output, as a
TikZ picture in a LaTeX PDF and as a CeTZ drawing in a Typst PDF. In the PDFs
the figure is typeset with the document, its text in the document's font.

A LaTeX PDF needs TikZ loaded, in the document's header:

```yaml
format:
  pdf:
    include-in-header:
      text: \\usepackage{tikz}
```

Typst output needs nothing: the figure imports CeTZ itself (Typst 0.15 or
later). See [A report with Quarto](@ref).
"""
embed(fig::PBP.Figure) = Embedded(PBP.svg(fig), PBP.tikz(fig), PBP.cetz(fig))
embed(x) = throw(ArgumentError("embed takes a PracticalBayesPlot figure (the default plotting backend); " *
                               "got a $(typeof(x)). For tables use `summary_table(fit; format = :embed)`."))
