# Plot a fit

The plotting functions work once a plotting package is loaded. Three are
supported:

| Package | Output | Use |
|:--------|:-------|:----|
| CairoMakie (or GLMakie, WGLMakie) | Makie figures, saved as PNG, SVG or PDF | publication figures |
| PlotlyLight | interactive HTML | notebooks, VS Code, the browser; quick to install |
| UnicodePlots | text | the terminal |

```julia
using PracticalBayesianRegression, CairoMakie

pp_check(fit)                         # posterior predictive density overlay
pp_check(fit; type = :bars)           # for counts and other discrete responses
pp_check(fit; type = :stat, stat = mean)
plot_effects(fit, :x)                 # conditional effect of x
plot_effects(fit, :x => :group)       # one curve per level of group
trace_plot(fit)                       # chains of the first few parameters
trace_plot(fit; variables = ["b_x", "sigma"])
residual_plot(fit)
```

With Makie, each function returns a `Figure`; save it with
`save("ppcheck.png", pp_check(fit))`.

`pp_check` types: `:dens_overlay` (default), `:ecdf_overlay`, `:hist`,
`:stat` and `:bars`. `ndraws` sets the number of simulated data sets drawn
(50 by default).

When more than one plotting package is loaded, choose with `backend = :makie`,
`:plotly` or `:unicode`.

## Your own plots

The data behind the plots are available without a plotting package:
[`pp_check_data`](@ref) returns the observed response and simulated
replicates, [`conditional_effects`](@ref) the grid, estimates and intervals,
and [`draws`](@ref) the draws of any parameter.

## ArviZ

[`to_inferencedata`](@ref) converts a fit to an `InferenceData` object for
ArviZ.jl's plots and diagnostics (it needs `using InferenceObjects` or
`using ArviZ`):

```julia
using ArviZ
idata = to_inferencedata(fit)
plot_trace(idata)
```
