# Plot a fit

The plotting functions draw with PracticalBayesPlot, a dependency, unless
another plotting package is loaded:

| Package | Output | Use |
|:--------|:-------|:----|
| PracticalBayesPlot (the default, always loaded) | SVG figures, shown inline in VS Code, Jupyter, Pluto and Quarto; saved as SVG, TikZ (LaTeX) or CeTZ (Typst) | everyday checks and reports, with nothing extra to install |
| CairoMakie (or GLMakie, WGLMakie) | Makie figures, saved as PNG, SVG or PDF | publication figures |
| PlotlyLight | interactive HTML | notebooks, VS Code, the browser; quick to install |
| UnicodePlots | text | the terminal |

```julia
using PracticalBayesianRegression

pp_check(fit)                         # posterior predictive density overlay
pp_check(fit; type = :bars)           # for counts and other discrete responses
pp_check(fit; type = :stat, stat = mean)
plot_effects(fit, :x)                 # conditional effect of x
plot_effects(fit, :x => :group)       # one curve per level of group
trace_plot(fit)                       # chains of the first few parameters
trace_plot(fit; variables = ["b_x", "sigma"])
residual_plot(fit)
```

`pp_check` types: `:dens_overlay` (default), `:ecdf_overlay`, `:hist`,
`:stat` and `:bars`. `ndraws` sets the number of simulated data sets drawn
(50 by default).

To choose when more than one is loaded, pass `backend = :makie`, `:plotly`,
`:unicode` or `:pbp`.

## Save a figure

By default each function returns a PracticalBayesPlot `Figure`.
[`save_plot`](@ref) writes it to a file, the extension picking the format:

```julia
save_plot("ppcheck.svg", pp_check(fit))      # an SVG image
save_plot("ppcheck.tex", pp_check(fit))      # a TikZ picture, for \input in LaTeX
save_plot("ppcheck.typ", pp_check(fit))      # a CeTZ drawing, for #include in Typst
```

A TikZ or CeTZ figure is typeset with the document: its text is in the
document's font and size. `standalone = true` makes the `.tex` or `.typ`
file a document that compiles on its own. With Makie loaded, the functions
return Makie figures instead; save those with Makie's
`save("ppcheck.png", pp_check(fit))`.

## Figures and tables in a report

In a [Quarto](https://quarto.org) document, a code cell whose value is
`embed(fig)` shows the figure as SVG in the web page, as a TikZ picture in a
PDF made with LaTeX and as a CeTZ drawing in one made with Typst, so that in
a PDF it is typeset in the document's font. [`summary_table`](@ref) and
[`prior_table`](@ref) give the summary and the priors as tables in the same
way (`format = :embed`), or as Markdown, LaTeX, Typst or HTML. See
[A report with Quarto](@ref), and for a document written directly in LaTeX
or Typst, [Put results in a LaTeX or Typst document](@ref).

## Your own plots

The data behind the plots are available without a plotting package:
[`pp_check_data`](@ref) returns the observed response and simulated
replicates, [`conditional_effects`](@ref) the grid, estimates and intervals,
and [`draws`](@ref) the draws of any parameter. PracticalBayesPlot's own
plots (once it is in your environment) also take a fit directly:

```julia
import PracticalBayesPlot as PBP
PBP.intervals(fit; variables = ["b_Intercept", "b_x", "sigma"])
PBP.densities(fit; variables = ["sigma"])
```

## ArviZ

[`to_inferencedata`](@ref) converts a fit to an `InferenceData` object for
ArviZ.jl's plots and diagnostics (it needs `using InferenceObjects` or
`using ArviZ`):

```julia
using ArviZ
idata = to_inferencedata(fit)
plot_trace(idata)
```
