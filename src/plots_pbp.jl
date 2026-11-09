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
