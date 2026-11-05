module PracticalBayesianRegressionMakieExt

# Makie figures for the Bayesian workflow: posterior predictive checks,
# conditional effects, traces and residuals. Activated by loading any Makie
# backend (CairoMakie, GLMakie, WGLMakie).

using PracticalBayesianRegression
using PracticalBayesianRegression: BrmsFit, pp_check_data, _kde, _density_grid, _ecdf, _pick_backend,
    _default_trace_vars, draws, nchains
import PracticalBayesianRegression: _pp_check, _plot_effects, _trace_plot, _residual_plot
using Makie
using Statistics, Random

const YCOL = :black
const REPCOL = (:steelblue, 0.25)

function _pp_check(::Val{:makie}, fit::BrmsFit; type::Symbol=:dens_overlay, ndraws::Int=50,
                   stat=mean, rng=Random.default_rng(), kwargs...)
    y, yrep = pp_check_data(fit; ndraws, rng)
    fig = Figure(; size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel=string(fit.info.formula.response), kwargs...)
    if type === :dens_overlay
        grid = _density_grid(y, yrep)
        for s in axes(yrep, 1)
            lines!(ax, grid, _kde(view(yrep, s, :), grid); color=REPCOL, linewidth=0.8)
        end
        lines!(ax, grid, _kde(y, grid); color=YCOL, linewidth=2.5, label="y")
        ax.ylabel = "density"
    elseif type === :ecdf_overlay
        grid = _density_grid(y, yrep; n=400)
        for s in axes(yrep, 1)
            stairs!(ax, grid, _ecdf(view(yrep, s, :), grid); color=REPCOL)
        end
        stairs!(ax, grid, _ecdf(y, grid); color=YCOL, linewidth=2.5)
        ax.ylabel = "ECDF"
    elseif type === :hist
        hist!(ax, vec(yrep); normalization=:pdf, color=REPCOL, bins=40)
        hist!(ax, y; normalization=:pdf, color=(:white, 0.0), strokecolor=YCOL, strokewidth=1.5, bins=40)
    elseif type === :stat
        y2, yrep2 = pp_check_data(fit; ndraws=size(fit.draws, 2) * nchains(fit), rng)
        t = [stat(view(yrep2, s, :)) for s in axes(yrep2, 1)]
        hist!(ax, t; color=REPCOL, bins=30)
        vlines!(ax, [stat(y)]; color=YCOL, linewidth=3)
        ax.xlabel = "T = $(nameof(stat))"
    elseif type === :bars
        ks = sort(unique(vcat(y, vec(yrep))))
        obs = [count(==(k), y) for k in ks]
        reps = [[count(==(k), view(yrep, s, :)) for s in axes(yrep, 1)] for k in ks]
        barplot!(ax, ks, obs; color=(:steelblue, 0.6))
        med = median.(reps); lo = quantile.(reps, 0.05); hi = quantile.(reps, 0.95)
        rangebars!(ax, ks, lo, hi; color=YCOL)
        scatter!(ax, ks, med; color=YCOL)
        ax.ylabel = "count"
    else
        throw(ArgumentError("unknown pp_check type `$type`"))
    end
    return fig
end

function _plot_effects(::Val{:makie}, ce::NamedTuple)
    names = keys(ce)
    xname = names[1]
    by = length(names) == 5 ? names[2] : nothing
    fig = Figure(; size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel=string(xname), ylabel="estimate")
    groups = by === nothing ? [nothing] : unique(ce[by])
    for (k, gv) in enumerate(groups)
        sel = gv === nothing ? eachindex(ce.estimate) : findall(==(gv), ce[by])
        x = ce[xname][sel]
        col = Makie.wong_colors()[mod1(k, 7)]
        if eltype(x) <: Real
            band!(ax, Float64.(x), ce.lower[sel], ce.upper[sel]; color=(col, 0.25))
            lines!(ax, Float64.(x), ce.estimate[sel]; color=col, linewidth=2,
                   label=gv === nothing ? "" : string(gv))
        else
            xs = 1:length(x)
            rangebars!(ax, xs, ce.lower[sel], ce.upper[sel]; color=col)
            scatter!(ax, xs, ce.estimate[sel]; color=col, markersize=10)
            ax.xticks = (collect(xs), string.(x))
        end
    end
    by === nothing || axislegend(ax, string(by); position=:lt)
    return fig
end
function _trace_plot(::Val{:makie}, fit::BrmsFit, variables)
    fig = Figure(; size=(800, 200 * length(variables)))
    for (k, v) in enumerate(variables)
        x = draws(fit, v)
        ax = Axis(fig[k, 1]; ylabel=v)
        for c in axes(x, 2)
            lines!(ax, x[:, c]; color=(Makie.wong_colors()[mod1(c, 7)], 0.7), linewidth=0.6)
        end
        ax2 = Axis(fig[k, 2]; xlabel=k == length(variables) ? "value" : "")
        for c in axes(x, 2)
            g = range(minimum(x), maximum(x); length=200)
            lines!(ax2, g, _kde(x[:, c], g); color=Makie.wong_colors()[mod1(c, 7)])
        end
    end
    colsize!(fig.layout, 1, Relative(0.7))
    return fig
end

function _residual_plot(::Val{:makie}, fit::BrmsFit)
    f = PracticalBayesianRegression.StatsAPI.fitted(fit)
    r = PracticalBayesianRegression.StatsAPI.residuals(fit)
    fig = Figure(; size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="fitted (posterior mean)", ylabel="residual")
    scatter!(ax, f, r; color=(:steelblue, 0.5), markersize=5)
    hlines!(ax, [0.0]; color=:black, linestyle=:dash)
    return fig
end

end
