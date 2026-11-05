module PracticalBayesianRegressionUnicodePlotsExt

# Quick-look plots in the terminal: posterior predictive checks, conditional
# effects, traces and residuals with UnicodePlots — no graphics stack needed.

using PracticalBayesianRegression
using PracticalBayesianRegression: BrmsFit, pp_check_data, _kde, _density_grid, _ecdf, _pick_backend,
    _default_trace_vars, draws
import PracticalBayesianRegression: _pp_check, _plot_effects, _trace_plot, _residual_plot
using UnicodePlots
using Statistics, Random

function _pp_check(::Val{:unicode}, fit::BrmsFit; type::Symbol=:dens_overlay, ndraws::Int=20,
                   stat=mean, rng=Random.default_rng(), kwargs...)
    y, yrep = pp_check_data(fit; ndraws, rng)
    name = string(fit.info.formula.response)
    if type in (:dens_overlay, :hist)
        grid = _density_grid(y, yrep)
        p = lineplot(grid, _kde(view(yrep, 1, :), grid); color=:blue, name="y_rep", xlabel=name,
                     ylabel="density", title="posterior predictive check", width=60, height=15)
        for s in 2:size(yrep, 1)
            lineplot!(p, grid, _kde(view(yrep, s, :), grid); color=:blue)
        end
        lineplot!(p, grid, _kde(y, grid); color=:red, name="y")
        return p
    elseif type === :ecdf_overlay
        grid = _density_grid(y, yrep; n=200)
        p = lineplot(grid, _ecdf(view(yrep, 1, :), grid); color=:blue, name="y_rep", xlabel=name,
                     ylabel="ECDF", width=60, height=15)
        for s in 2:size(yrep, 1)
            lineplot!(p, grid, _ecdf(view(yrep, s, :), grid); color=:blue)
        end
        lineplot!(p, grid, _ecdf(y, grid); color=:red, name="y")
        return p
    elseif type === :stat
        _, yr = pp_check_data(fit; ndraws=10_000, rng)
        t = [stat(view(yr, s, :)) for s in axes(yr, 1)]
        p = histogram(t; title="T = $(nameof(stat)); observed $(round(stat(y); sigdigits=4))", width=50)
        return p
    elseif type === :bars
        ks = sort(unique(vcat(y, vec(yrep))))
        obs = [count(==(k), y) for k in ks]
        rep = [mean(count(==(k), view(yrep, s, :)) for s in axes(yrep, 1)) for k in ks]
        return barplot(string.(Int.(ks)), obs; title="observed counts (mean replicated: $(round.(Int, rep)))")
    end
    throw(ArgumentError("unknown pp_check type `$type`"))
end

function _plot_effects(::Val{:unicode}, ce::NamedTuple)
    names = keys(ce)
    x = ce[names[1]]
    eltype(x) <: Real || return barplot(string.(x), ce.estimate; title="conditional effect of $(names[1])")
    sel = length(names) == 5 ? findall(==(first(ce[names[2]])), ce[names[2]]) : eachindex(x)
    p = lineplot(Float64.(x[sel]), ce.estimate[sel]; name="estimate", xlabel=string(names[1]), width=60, height=15)
    lineplot!(p, Float64.(x[sel]), ce.lower[sel]; color=:cyan, name="95% CI")
    lineplot!(p, Float64.(x[sel]), ce.upper[sel]; color=:cyan)
    return p
end

function _trace_plot(::Val{:unicode}, fit::BrmsFit, vars)
    return [begin
        x = draws(fit, v)
        p = lineplot(x[:, 1]; title=v, width=60, height=6)
        for c in 2:size(x, 2)
            lineplot!(p, x[:, c])
        end
        p
    end for v in vars]
end

function _residual_plot(::Val{:unicode}, fit::BrmsFit)
    f = PracticalBayesianRegression.StatsAPI.fitted(fit)
    r = PracticalBayesianRegression.StatsAPI.residuals(fit)
    return scatterplot(f, r; xlabel="fitted", ylabel="residual", width=60, height=15)
end

end
