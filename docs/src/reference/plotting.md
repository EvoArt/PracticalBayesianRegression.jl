# Plotting and export

```@meta
CurrentModule = PracticalBayesianRegression
```

The plotting functions need a plotting package to be loaded: CairoMakie (or
another Makie backend), PlotlyLight or UnicodePlots. With more than one
loaded, pass `backend = :makie`, `:plotly` or `:unicode`; otherwise they use
the first available in that order. See [Plot a fit](@ref).

```@docs
pp_check
pp_check_data
plot_effects
trace_plot
residual_plot
to_inferencedata
```
