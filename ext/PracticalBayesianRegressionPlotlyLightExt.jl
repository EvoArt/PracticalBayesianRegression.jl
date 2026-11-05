module PracticalBayesianRegressionPlotlyLightExt

# Interactive HTML figures via PlotlyLight: the light-install backend.
#
# Why this exists alongside the Makie extension: PlotlyLight installs and
# loads several times faster than CairoMakie and is far smaller on disk, with
# a similar time to the first plot. The first-render cost is one-off
# compilation, not proportional to the data; after it, even a 50-curve
# dens_overlay renders in milliseconds, so no decimation is needed below.
#
# The trade: output is HTML/JS, so it renders in VS Code's plot pane, Pluto,
# Jupyter and a browser, but NOT in a bare terminal (UnicodePlots covers
# that) and it cannot write static SVG/PDF for a manuscript without Kaleido
# (Makie covers that). Three tiers, one dispatch layer.
#
# PlotlyLight's API is a thin JSON wrapper, so there is no recipe system to
# hook: `Plot` holds `data::Vector{Config}` and `layout::Config`, `p(; kw...)`
# appends one trace, and every nested value must be a `Config`, not a
# NamedTuple (JSON3 writes a NamedTuple field as a struct, which Plotly's
# schema rejects). Hence `Config(...)` everywhere below.

using PracticalBayesianRegression
using PracticalBayesianRegression: BrmsFit, pp_check_data, _kde, _density_grid, _ecdf, _pick_backend,
    _default_trace_vars, draws, nchains
import PracticalBayesianRegression: _pp_check, _plot_effects, _trace_plot, _residual_plot
using PlotlyLight
using PlotlyLight: Plot, Config
using Statistics, Random

# bayesplot's palette: observed y dark, replicates translucent blue
const YCOL = "#1c1c1c"
const REPCOL = "rgba(70,130,180,0.25)"
const REPSOLID = "#4682b4"
# Wong, as in the Makie extension, so the two backends colour groups alike
const WONG = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442")
_wong(k) = WONG[mod1(k, 7)]

_title(t) = Config(text=t)
_base_layout() = Config(margin=Config(l=60, r=20, t=40, b=50),
                        template="plotly_white", hovermode="closest")

# A legend entry per replicate would be 50 duplicate rows, so only the first
# replicate trace carries the name and the rest are grouped under it.
function _add_rep!(p, x, y, first; shape=nothing)
    line = shape === nothing ? Config(color=REPCOL, width=1) :
                               Config(color=REPCOL, width=1, shape=shape)
    p(x=x, y=y, type="scatter", mode="lines", line=line,
      name="y_rep", legendgroup="yrep", showlegend=first)
end

function _pp_check(::Val{:plotly}, fit::BrmsFit; type::Symbol=:dens_overlay, ndraws::Int=50,
                   stat=mean, rng=Random.default_rng(), kwargs...)
    y, yrep = pp_check_data(fit; ndraws, rng)
    xname = string(fit.info.formula.response)
    p = Plot()
    lay = _base_layout()
    if type === :dens_overlay
        grid = _density_grid(y, yrep)
        for s in axes(yrep, 1)
            _add_rep!(p, grid, _kde(view(yrep, s, :), grid), s == 1)
        end
        p(x=grid, y=_kde(y, grid), type="scatter", mode="lines",
          line=Config(color=YCOL, width=2.5), name="y")
        lay.xaxis.title = _title(xname)
        lay.yaxis.title = _title("density")
        lay.title = _title("posterior predictive check")
    elseif type === :ecdf_overlay
        grid = _density_grid(y, yrep; n=400)
        for s in axes(yrep, 1)
            # shape="hv" = step-after, matching Makie's stairs!
            _add_rep!(p, grid, _ecdf(view(yrep, s, :), grid), s == 1; shape="hv")
        end
        p(x=grid, y=_ecdf(y, grid), type="scatter", mode="lines",
          line=Config(color=YCOL, width=2.5, shape="hv"), name="y")
        lay.xaxis.title = _title(xname)
        lay.yaxis.title = _title("ECDF")
    elseif type === :hist
        p(x=vec(yrep), type="histogram", histnorm="probability density",
          marker=Config(color=REPCOL), nbinsx=40, name="y_rep")
        p(x=y, type="histogram", histnorm="probability density",
          marker=Config(color="rgba(0,0,0,0)", line=Config(color=YCOL, width=1.5)),
          nbinsx=40, name="y")
        lay.barmode = "overlay"
        lay.xaxis.title = _title(xname)
        lay.yaxis.title = _title("density")
    elseif type === :stat
        # the statistic needs the full posterior, not the ndraws subsample
        _, yrep2 = pp_check_data(fit; ndraws=size(fit.draws, 2) * nchains(fit), rng)
        t = [stat(view(yrep2, s, :)) for s in axes(yrep2, 1)]
        p(x=t, type="histogram", marker=Config(color=REPSOLID), opacity=0.6,
          nbinsx=30, name="T(y_rep)")
        obs = stat(y)
        lay.xaxis.title = _title("T = $(nameof(stat))")
        lay.yaxis.title = _title("count")
        lay.shapes = [Config(type="line", x0=obs, x1=obs, yref="paper", y0=0, y1=1,
                             line=Config(color=YCOL, width=3))]
        lay.annotations = [Config(x=obs, yref="paper", y=1.02, text="T(y)",
                                  showarrow=false, font=Config(color=YCOL))]
    elseif type === :bars
        ks = sort(unique(vcat(y, vec(yrep))))
        obs = [count(==(k), y) for k in ks]
        reps = [[count(==(k), view(yrep, s, :)) for s in axes(yrep, 1)] for k in ks]
        med = median.(reps)
        lo = quantile.(reps, 0.05)
        hi = quantile.(reps, 0.95)
        p(x=ks, y=obs, type="bar", marker=Config(color="rgba(70,130,180,0.6)"), name="y")
        # replicate median with a 5-95% interval, as the Makie rangebars
        p(x=ks, y=med, type="scatter", mode="markers",
          marker=Config(color=YCOL, size=8),
          error_y=Config(type="data", symmetric=false, array=hi .- med,
                         arrayminus=med .- lo, color=YCOL), name="y_rep")
        lay.xaxis.title = _title(xname)
        lay.xaxis.tickmode = "array"
        lay.xaxis.tickvals = ks
        lay.yaxis.title = _title("count")
    else
        throw(ArgumentError("unknown pp_check type `$type`"))
    end
    p.layout = lay
    return p
end

function _plot_effects(::Val{:plotly}, ce::NamedTuple)
    names = keys(ce)
    xname = names[1]
    by = length(names) == 5 ? names[2] : nothing
    p = Plot()
    groups = by === nothing ? [nothing] : unique(ce[by])
    for (k, gv) in enumerate(groups)
        sel = gv === nothing ? eachindex(ce.estimate) : findall(==(gv), ce[by])
        x = ce[xname][sel]
        col = _wong(k)
        lbl = gv === nothing ? "estimate" : string(gv)
        if eltype(x) <: Real
            xf = Float64.(x)
            # band = invisible upper trace, then lower with fill="tonexty"
            p(x=xf, y=ce.upper[sel], type="scatter", mode="lines",
              line=Config(width=0), hoverinfo="skip", showlegend=false, legendgroup=lbl)
            p(x=xf, y=ce.lower[sel], type="scatter", mode="lines",
              line=Config(width=0), fill="tonexty", fillcolor=_rgba(col, 0.25),
              hoverinfo="skip", showlegend=false, legendgroup=lbl)
            p(x=xf, y=ce.estimate[sel], type="scatter", mode="lines",
              line=Config(color=col, width=2), name=lbl, legendgroup=lbl)
        else
            # categorical x: point estimate with an asymmetric CI bar
            p(x=string.(x), y=ce.estimate[sel], type="scatter", mode="markers",
              marker=Config(color=col, size=10),
              error_y=Config(type="data", symmetric=false,
                             array=ce.upper[sel] .- ce.estimate[sel],
                             arrayminus=ce.estimate[sel] .- ce.lower[sel], color=col),
              name=lbl, legendgroup=lbl)
        end
    end
    lay = _base_layout()
    lay.xaxis.title = _title(string(xname))
    lay.yaxis.title = _title("estimate")
    lay.showlegend = by !== nothing
    by === nothing || (lay.legend.title = _title(string(by)))
    p.layout = lay
    return p
end

# "#rrggbb" -> "rgba(r,g,b,a)"; Plotly has no alpha channel on hex colours
function _rgba(hex, a)
    r = parse(Int, hex[2:3]; base=16)
    g = parse(Int, hex[4:5]; base=16)
    b = parse(Int, hex[6:7]; base=16)
    return "rgba($r,$g,$b,$a)"
end

function _trace_plot(::Val{:plotly}, fit::BrmsFit, variables)
    n = length(variables)
    p = Plot()
    lay = Config(margin=Config(l=70, r=20, t=30, b=45), template="plotly_white",
                 hovermode="closest", height=200n)
    h = 1 / n
    pad = 0.04h
    # one row per parameter: trace on the left (66%), marginal density right
    for (k, v) in enumerate(variables)
        x = draws(fit, v)
        xa, ya = "x$(2k-1)", "y$(2k-1)"
        xb, yb = "x$(2k)", "y$(2k)"
        g = collect(range(minimum(x), maximum(x); length=200))
        for c in axes(x, 2)
            col = _wong(c)
            p(y=x[:, c], type="scatter", mode="lines", line=Config(color=col, width=0.7),
              xaxis=xa, yaxis=ya, name="chain $c", legendgroup="chain$c", showlegend=(k == 1))
            p(x=g, y=_kde(x[:, c], g), type="scatter", mode="lines",
              line=Config(color=col, width=1.2), xaxis=xb, yaxis=yb,
              legendgroup="chain$c", showlegend=false)
        end
        # stacked domains, first parameter at the top as in the Makie layout
        y0, y1 = 1 - k * h + pad, 1 - (k - 1) * h - pad
        lay["xaxis$(2k-1)"] = Config(domain=[0.0, 0.66], anchor=ya,
                                     title=_title(k == n ? "iteration" : ""))
        lay["yaxis$(2k-1)"] = Config(domain=[y0, y1], anchor=xa, title=_title(string(v)))
        lay["xaxis$(2k)"] = Config(domain=[0.72, 1.0], anchor=yb,
                                   title=_title(k == n ? "value" : ""))
        lay["yaxis$(2k)"] = Config(domain=[y0, y1], anchor=xb, showticklabels=false)
    end
    p.layout = lay
    return p
end

function _residual_plot(::Val{:plotly}, fit::BrmsFit)
    f = PracticalBayesianRegression.StatsAPI.fitted(fit)
    r = PracticalBayesianRegression.StatsAPI.residuals(fit)
    p = Plot()
    p(x=f, y=r, type="scatter", mode="markers",
      marker=Config(color="rgba(70,130,180,0.5)", size=5), name="residual")
    lay = _base_layout()
    lay.xaxis.title = _title("fitted (posterior mean)")
    lay.yaxis.title = _title("residual")
    lay.showlegend = false
    lay.shapes = [Config(type="line", xref="paper", x0=0, x1=1, y0=0.0, y1=0.0,
                         line=Config(color=YCOL, width=1, dash="dash"))]
    p.layout = lay
    return p
end

end
