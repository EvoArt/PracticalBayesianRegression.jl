# Plot a fit

The plotting functions draw with PracticalBayesPlot, a dependency, unless
another plotting package is loaded:

| Package | Output | Use |
|:--------|:-------|:----|
| PracticalBayesPlot (the default, always loaded) | SVG figures, shown inline in VS Code, Jupyter and Pluto; saved as SVG, TikZ (LaTeX) or CeTZ (Typst) | everyday checks, with nothing extra to install |
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

By default each function returns a PracticalBayesPlot `Figure`; save it
with `PracticalBayesPlot.save("ppcheck.svg", pp_check(fit))` (after
`Pkg.add("PracticalBayesPlot")` and `import PracticalBayesPlot`, since a
dependency is not loadable by name until it is in your environment). The
file extension picks the format: `.svg`, `.html`, `.tex` (a TikZ picture,
whose text LaTeX typesets in the document's fonts) or `.typ` (a CeTZ
drawing for Typst); `standalone = true` makes a `.tex` or `.typ` file that
compiles on its own. With Makie loaded, each returns a Makie `Figure`; save
it with `save("ppcheck.png", pp_check(fit))`.

`pp_check` types: `:dens_overlay` (default), `:ecdf_overlay`, `:hist`,
`:stat` and `:bars`. `ndraws` sets the number of simulated data sets drawn
(50 by default).

To choose when more than one is loaded, pass `backend = :makie`, `:plotly`,
`:unicode` or `:pbp`.

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
