# Plotting and export

```@meta
CurrentModule = PracticalBayesianRegression
```

The plotting functions draw with PracticalBayesPlot, a dependency, unless
CairoMakie (or another Makie backend), PlotlyLight or UnicodePlots is
loaded. Pass `backend = :makie`, `:plotly`, `:unicode` or `:pbp` to pick
one; otherwise they use the first loaded in that order. See [Plot a fit](@ref).

```@docs
pp_check
pp_check_data
plot_effects
trace_plot
residual_plot
save_plot
embed
Embedded
to_inferencedata
```

## Tables for reports

The posterior summary and the priors as LaTeX, Typst, Markdown or HTML
tables. See [A report with Quarto](@ref) and
[Put results in a LaTeX or Typst document](@ref).

```@docs
summary_table
prior_table
```
