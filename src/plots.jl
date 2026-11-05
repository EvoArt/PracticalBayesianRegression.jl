# =============================================================================
# Plotting front: generic functions whose methods live in the package
# extensions (ext/), plus the backend-independent data they draw.
#
#   Makie loaded (CairoMakie, GLMakie, ...)  ->  publication figures
#   PlotlyLight loaded                       ->  interactive HTML, light install
#   UnicodePlots loaded                      ->  quick looks in the REPL
#   InferenceObjects loaded                  ->  `to_inferencedata` for ArviZ
#
# `backend = :makie | :plotly | :unicode` picks one when several are loaded;
# otherwise the loaded one is used, Makie first (a user who paid for it wants
# its figures). PlotlyLight is the one to recommend to new users: a similar
# time to the first plot as Makie, but much smaller to install and faster to
# load.
# =============================================================================

"""
    pp_check(fit; type=:dens_overlay, ndraws=50, backend=nothing, kwargs...)

Posterior predictive check, as brms's `pp_check` (bayesplot's `ppc_*`):
`type` is `:dens_overlay` (default), `:hist`, `:ecdf_overlay`, `:stat`
(`stat=mean` by default) or `:bars` (discrete responses). Needs a plotting
extension: load PlotlyLight (lightest install, interactive HTML), Makie
(e.g. `using CairoMakie`, for static publication figures) or UnicodePlots
(terminal).
"""
pp_check(fit::BrmsFit; backend=nothing, kwargs...) = _pp_check(Val(_pick_backend(backend)), fit; kwargs...)

"""
    plot_effects(ce; backend=nothing)
    plot_effects(fit, effect; kwargs...)

Plot a [`conditional_effects`](@ref) result (line + credible band, one per
level of the `by` variable). Needs a plotting extension.
"""
plot_effects(ce::NamedTuple; backend=nothing) = _plot_effects(Val(_pick_backend(backend)), ce)
plot_effects(fit::BrmsFit, effect; backend=nothing, kwargs...) =
    plot_effects(conditional_effects(fit, effect; kwargs...); backend)

"""
    trace_plot(fit; variables=first population-level ones, backend=nothing)

Trace plots, one panel per parameter, chains overlaid. Needs a plotting extension.
"""
trace_plot(fit::BrmsFit; variables=_default_trace_vars(fit), backend=nothing) =
    _trace_plot(Val(_pick_backend(backend)), fit, variables)

"""
    residual_plot(fit; backend=nothing)

Posterior-mean residuals against posterior-mean fitted values. Needs a
plotting extension.
"""
residual_plot(fit::BrmsFit; backend=nothing) = _residual_plot(Val(_pick_backend(backend)), fit)

# implemented per backend in the extensions, on Val(:makie) / Val(:unicode)
function _pp_check end
function _plot_effects end
function _trace_plot end
function _residual_plot end

"""
    to_inferencedata(fit; posterior_predictive=true, log_likelihood=true)

Convert a fit to an `InferenceObjects.InferenceData` (the format ArviZ.jl
works with) with posterior, sample_stats, observed_data and, optionally,
posterior_predictive and log_likelihood groups. Needs `using InferenceObjects`
(or ArviZ).
"""
function to_inferencedata end

# --- backend selection -------------------------------------------------------------

_ext(name) = Base.get_extension(@__MODULE__, name)

# Preference order when several backends are loaded and `backend` is not given.
# Makie first: if a user has paid for it, they asked for publication figures.
const _BACKENDS = ((:makie, :PracticalBayesianRegressionMakieExt, "CairoMakie (or another Makie)"),
                   (:plotly, :PracticalBayesianRegressionPlotlyLightExt, "PlotlyLight"),
                   (:unicode, :PracticalBayesianRegressionUnicodePlotsExt, "UnicodePlots"))

function _pick_backend(backend)
    if backend === nothing
        for (name, ext, _) in _BACKENDS
            _ext(ext) === nothing || return name
        end
        throw(ArgumentError("plotting needs a backend: `using " *
            join([b[3] for b in _BACKENDS], "`, `using ", "` or `using ") * "`"))
    end
    i = findfirst(b -> b[1] === backend, _BACKENDS)
    i === nothing && throw(ArgumentError("backend must be one of " *
        join([repr(b[1]) for b in _BACKENDS], ", ") * "; got $(repr(backend))"))
    _ext(_BACKENDS[i][2]) === nothing &&
        throw(ArgumentError("backend $(repr(backend)) is not loaded: `using $(_BACKENDS[i][3])`"))
    return backend
end

# --- data shared by the backends -----------------------------------------------------

"""
    pp_check_data(fit; ndraws=50, rng) -> (y, yrep)

The observed response and `ndraws` posterior predictive replicates
(ndraws x N), what every `pp_check` type is drawn from.
"""
function pp_check_data(fit::BrmsFit; ndraws::Int=50, rng=Random.default_rng())
    Y = posterior_predict(fit; rng=rng)
    S = size(Y, 1)
    idx = S <= ndraws ? (1:S) : sort(randperm(rng, S)[1:ndraws])
    return Float64.(fit.density.y), Y[idx, :]
end

# Gaussian kernel density with Silverman's bandwidth on a shared grid
function _kde(x::AbstractVector, grid::AbstractVector)
    n = length(x)
    s = std(x)
    iq = quantile(x, 0.75) - quantile(x, 0.25)
    h = 0.9 * min(s, iq / 1.34) * n^(-1 / 5)
    h = h > 0 ? h : (s > 0 ? s : 1.0) * n^(-1 / 5)
    c = 1 / (n * h * sqrt(2π))
    return [c * sum(v -> exp(-((g - v) / h)^2 / 2), x) for g in grid]
end

function _density_grid(y, yrep; n=200)
    lo = min(minimum(y), quantile(vec(yrep), 0.005))
    hi = max(maximum(y), quantile(vec(yrep), 0.995))
    return collect(range(lo, hi; length=n))
end

# empirical cdf on a grid
_ecdf(x, grid) = [count(<=(g), x) / length(x) for g in grid]

_default_trace_vars(fit) = first(filter(n -> startswith(n, "b_") || n in ("sigma", "shape", "phi", "nu"), variables(fit)), 4)
