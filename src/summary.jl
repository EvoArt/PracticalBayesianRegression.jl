# =============================================================================
# Summaries, printed the way brms prints them.
#
# Rhat is the rank-normalised split-Rhat and ESS the bulk/tail ESS of Vehtari
# et al. (2021) — MCMCDiagnosticTools implements the same estimators as the
# R `posterior` package that brms uses, so the columns are comparable
# number-for-number.
# =============================================================================

"""
    SummaryRow

One row of a posterior summary: brms's `Estimate` (posterior mean),
`Est.Error` (posterior sd), the central interval, `Rhat`, `Bulk_ESS`,
`Tail_ESS`.
"""
struct SummaryRow
    variable::String
    estimate::Float64
    est_error::Float64
    lower::Float64
    upper::Float64
    rhat::Float64
    bulk_ess::Float64
    tail_ess::Float64
end

function _summarise(name, x::AbstractMatrix, prob)
    a = (1 - prob) / 2
    v = vec(x)
    arr = reshape(x, size(x, 1), size(x, 2), 1)   # draws x chains x params
    rh = only(MCMCDiagnosticTools.rhat(arr; kind=:rank))
    be = only(MCMCDiagnosticTools.ess(arr; kind=:bulk))
    te = only(MCMCDiagnosticTools.ess(arr; kind=:tail))
    return SummaryRow(name, mean(v), std(v), quantile(v, a), quantile(v, 1 - a), rh, be, te)
end

"""
    posterior_summary(fit; variables = variables(fit), prob = 0.95) -> Vector{SummaryRow}

Summaries of the posterior draws of `variables`, as brms's
`posterior_summary`: one row per parameter with fields `variable`,
`estimate` (posterior mean), `est_error` (posterior standard deviation),
`lower` and `upper` (the central `prob` interval), `rhat`, `bulk_ess` and
`tail_ess`.

```julia
rows = posterior_summary(fit; variables = ["b_x", "sigma"])
rows[1].estimate
```
"""
function posterior_summary(fit::Union{BrmsFit,BrmsMVFit}; variables=variables(fit), prob=0.95)
    return [_summarise(n, draws(fit, n), prob) for n in variables]
end

"""
    fixef(fit; prob=0.95) -> Vector{SummaryRow}

Population-level ("fixed") effects of the mean, named without the `b_`
prefix, as brms's `fixef()`.
"""
function fixef(fit::BrmsFit; prob=0.95)
    names = filter(n -> startswith(n, "b_") || startswith(n, "bsp_"), variables(fit))
    rows = posterior_summary(fit; variables=names, prob=prob)
    return [SummaryRow(_coef_label(r.variable), r.estimate, r.est_error, r.lower, r.upper,
                       r.rhat, r.bulk_ess, r.tail_ess) for r in rows]
end
_coef_label(n) = startswith(n, "bsp_") ? n[5:end] : n[3:end]

# --- REPL display ------------------------------------------------------------------
#
# Colour carries meaning, it is never only decoration:
#  - estimates and interval glyphs of coefficients and correlations are blue
#    when the 95% CI lies above zero, orange below, grey when it covers zero;
#    scale parameters (sd) have no zero to compare with and get one neutral hue;
#  - only problems are lit, so they stand out: Rhat amber above 1.01 (Vehtari
#    et al. 2021, the threshold behind brms's warnings), bold red above 1.1;
#    ESS red below 100 per chain (bold) or below a quarter of the median ESS
#    of the parameters shown — the one that mixes far worse than the rest.
# The interval column draws every row of a table on one shared axis with a
# zero tick — a forest plot, so effects compare at a glance, and it is where
# the x axis lives; when the rows' scales are too different for that, each row
# gets its own axis and the caption says "per-row scale". The sparkline is only the shape of each marginal, on its
# own scale (bimodality, skew, piling at a bound), with the 95% CI lit. Escape codes are
# written only when the IO has :color, so sprint() and logs stay plain text;
# the picture columns are dropped (sparkline first) when the table would be
# wider than the terminal.

# xterm-256 indices: they look the same in every modern terminal, whereas the
# 16 named colours are re-themed per terminal (plain "blue" is unreadable on
# many dark themes)
const _PAL = (pos=75, neg=209, null=246, scale=150, head=117, group=176, special=150,
              good=114, warn=221, amber=214, bad=203, key=245, rule=240, tail=239, spark=117)

_hascolor(io) = get(io, :color, false)::Bool

# `s` in an xterm-256 colour (and/or bold), or plain `s` when the IO has no colour
function _sty(io, s; color=nothing, bold=false)
    _hascolor(io) || return string(s)
    pre = (bold ? "\e[1m" : "") * (color === nothing ? "" : "\e[38;5;$(color)m")
    return isempty(pre) ? string(s) : string(pre, s, "\e[0m")
end

# a string of glyphs, each with its own colour index (0 = default), as runs
function _glyphs(io, g::Vector{Char}, c::Vector{Int})
    b = IOBuffer()
    k = 1
    while k <= length(g)
        j = k
        while j < length(g) && c[j+1] == c[k]
            j += 1
        end
        s = String(g[k:j])
        print(b, c[k] == 0 ? s : _sty(io, s; color=c[k]))
        k = j + 1
    end
    return String(take!(b))
end

# PrettyTables measures AnsiTextCell content without its escape codes
_cell(s) = occursin('\e', s) ? AnsiTextCell(s) : s
_vwidth(s) = maximum(l -> textwidth(replace(l, r"\e\[[0-9;]*m" => "")), split(s, '\n'); init=0)
_f2(x) = @sprintf("%.2f", x)
# brms prints every number with 2 decimals, so the slope of an unscaled
# predictor (income in pounds) reads 0.00 0.00 0.00 0.00; such rows switch
# to 2 significant digits
_tiny(r) = max(abs(r.estimate), abs(r.lower), abs(r.upper)) < 0.005 && r.est_error > 0
_fnum(x, tiny) = tiny ? @sprintf("%.1e", x) : _f2(x)

# what a row is, for the choice of axis and colour. Intercepts get no forest
# glyph: they sit on the response scale and would squash every slope.
function _row_kind(name)
    (startswith(name, "sd_") || startswith(name, "sds_")) && return :sd
    (startswith(name, "cor_") || startswith(name, "rescor__")) && return :cor
    (startswith(name, "b_") || startswith(name, "bsp_")) && return occursin("Intercept", name) ? :none : :coef
    return :none
end

function _tone(kind, q)
    kind === :sd && return _PAL.scale
    kind === :none && return 0
    return q[1] > 0 ? _PAL.pos : q[5] < 0 ? _PAL.neg : _PAL.null
end

# the axis shared by a table's rows; zero is always on it
function _axis(qs)
    lo = min(0.0, minimum(q -> q[1], qs))
    hi = max(0.0, maximum(q -> q[5], qs))
    return hi > lo ? (lo, hi) : (lo - 1.0, hi + 1.0)
end

# a row's own axis: its interval, plus zero when zero is near enough to show
# without squashing the interval (otherwise an edge arrow points to it)
function _row_axis(q)
    lo, hi = q[1], q[5]
    w = hi - lo
    w > 0 || return (lo - 1.0, hi + 1.0)
    lo - 2w <= 0 <= hi + 2w && return (min(lo, 0.0) - 0.05w, max(hi, 0.0) + 0.05w)
    return (lo - 0.1w, hi + 0.1w)
end

# rows on very different scales (slopes of predictors in different units)
# would collapse to a dot on a shared axis: below this many characters for
# some 95% interval, each row gets its own axis instead
const _MIN_INTERVAL_CHARS = 2

const _FOREST_W = 19
_axpos(v, ax, W) = clamp(1 + round(Int, (v - ax[1]) / (ax[2] - ax[1]) * (W - 1)), 1, W)

# ──━━━●━━── : thin 95%, thick 50%, dot at the median; ┊ marks zero
function _forest(io, q, ax, col)
    W = _FOREST_W
    p = v -> _axpos(v, ax, W)
    g = fill(' ', W)
    c = zeros(Int, W)
    for k in p(q[1]):p(q[5])
        g[k] = '─'; c[k] = col
    end
    for k in p(q[2]):p(q[4])
        g[k] = '━'; c[k] = col
    end
    if ax[1] < 0 < ax[2]
        z = p(0.0)
        g[z] = g[z] == '─' ? '┼' : g[z] == '━' ? '╋' : '┊'
        g[z] == '┊' && (c[z] = _PAL.rule)
    end
    ax[1] > 0 && g[1] == ' ' && (g[1] = '◂'; c[1] = _PAL.rule)     # zero is off to the left
    ax[2] < 0 && g[W] == ' ' && (g[W] = '▸'; c[W] = _PAL.rule)     # ... or to the right
    m = p(q[3])
    g[m] = '●'; c[m] = col
    return _glyphs(io, g, c)
end

function _short(x)
    r = round(x; sigdigits=2)
    return isinteger(r) && abs(r) < 1e6 ? string(Int(r)) : string(r)
end

# the axis under the "95% interval" label: low end, 0, high end
function _axis_caption(ax, W)
    g = fill(' ', W)
    put(s, at) = for (k, ch) in enumerate(s)
        1 <= at + k - 1 <= W && (g[at+k-1] = ch)
    end
    a, b = _short(ax[1]), _short(ax[2])
    put(a, 1)
    put(b, W - length(b) + 1)
    if ax[1] < 0 < ax[2]
        z = _axpos(0.0, ax, W)
        length(a) + 1 < z < W - length(b) && put("0", z)
    end
    return String(g)
end

const _SPARK = ('▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')

function _sparkline(io, x, lim; nb=12)
    lo, hi = quantile(x, 0.005), quantile(x, 0.995)
    hi > lo || return _sty(io, repeat('▄', nb); color=_PAL.spark)
    cnt = zeros(Int, nb)
    for v in x
        lo <= v <= hi || continue
        cnt[min(nb, 1 + floor(Int, (v - lo) / (hi - lo) * nb))] += 1
    end
    m = maximum(cnt)
    g = [_SPARK[1+round(Int, (length(_SPARK) - 1) * n / m)] for n in cnt]
    c = [lim[1] <= lo + (k - 0.5) * (hi - lo) / nb <= lim[2] ? _PAL.spark : _PAL.tail for k in 1:nb]
    return _glyphs(io, g, c)
end

const _TABLE_FORMAT = TextTableFormat(; borders=text_table_borders__unicode_rounded,
                                      vertical_lines_at_data_columns=:none)
const _TABLE_STYLE = TextTableStyle(; table_border=Crayon(; foreground=_PAL.rule))
const _COLS = ["Estimate", "Est.Error", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS", "Tail_ESS"]

_termwidth(io) = displaysize(io)[2]

# -> (rendered table, rows); the widest variant that fits the terminal
# `summ`: the rows already summarised (name => SummaryRow); `essref`: the
# median ESS a row is compared with (NaN: absolute threshold only)
function _render_table(io, fit::Union{BrmsFit,BrmsMVFit}, vars, labels; summ=nothing, essref=NaN)
    rows = summ === nothing ? posterior_summary(fit; variables=vars) : [summ[v] for v in vars]
    xs = [vec(draws(fit, v)) for v in vars]
    qs = [Tuple(quantile(x, [0.025, 0.25, 0.5, 0.75, 0.975])) for x in xs]
    kinds = map(_row_kind, vars)
    drawn = kinds .!== :none
    ax = any(drawn) ? _axis(qs[drawn]) : (0.0, 1.0)
    shared = all(k -> (qs[k][5] - qs[k][1]) / (ax[2] - ax[1]) * (_FOREST_W - 1) >= _MIN_INTERVAL_CHARS, findall(drawn))
    lowabs, lowrel = 100 * nchains(fit), essref / 4
    num = Matrix{Any}(undef, length(rows), 7)
    for (k, r) in enumerate(rows)
        t = _tone(kinds[k], qs[k])
        ess(e) = isnan(e) ? "NaN" :
                 (n = round(Int, e); _cell(_sty(io, n; color=n < lowabs || n < lowrel ? _PAL.bad : nothing, bold=n < lowabs)))
        ty = _tiny(r)
        num[k, :] = Any[_cell(_sty(io, _fnum(r.estimate, ty); color=t in (_PAL.pos, _PAL.neg) ? t : nothing, bold=true)),
                        _fnum(r.est_error, ty), _fnum(r.lower, ty), _fnum(r.upper, ty),
                        _cell(_sty(io, _f2(r.rhat); color=r.rhat > 1.1 ? _PAL.bad : r.rhat > 1.01 ? _PAL.amber : nothing, bold=r.rhat > 1.1)), ess(r.bulk_ess), ess(r.tail_ess)]
    end
    forest = [kinds[k] === :none ? "" : _cell(_forest(io, qs[k], shared ? ax : _row_axis(qs[k]), _tone(kinds[k], qs[k]))) for k in eachindex(rows)]
    spark = [_cell(_sparkline(io, xs[k], (qs[k][1], qs[k][5]))) for k in eachindex(rows)]
    variants = !any(drawn) ? ((false, true), (false, false)) : ((true, true), (true, false), (false, false))
    local s
    for (f, sp) in variants
        data = hcat(num, (f ? [forest] : [])..., (sp ? [spark] : [])...)
        lab1 = vcat(_COLS, f ? ["95% interval"] : String[], sp ? ["posterior"] : String[])
        lab2 = vcat(fill("", 7), f ? [shared ? _axis_caption(ax, _FOREST_W) : rpad("per-row scale", _FOREST_W)] : String[], sp ? ["own scale"] : String[])
        labs = f || sp ? [lab1, lab2] : [lab1]
        s = sprint(; context=IOContext(io, :displaysize => (10_000, 10_000))) do o
            pretty_table(o, data; column_labels=labs, row_labels=labels,
                         table_format=_TABLE_FORMAT, style=_TABLE_STYLE,
                         alignment=vcat(fill(:r, 7), fill(:l, size(data, 2) - 7)),
                         fit_table_in_display_horizontally=false, fit_table_in_display_vertically=false)
        end
        _vwidth(s) <= _termwidth(io) && break
    end
    return chomp(s), rows
end

# "── Title ─────────" across the width of the tables below it
function _section(io, title, width)
    t = " " * title * " "
    println(io)
    println(io, _sty(io, "──"; color=_PAL.rule), _sty(io, t; color=_PAL.head, bold=true),
            _sty(io, repeat('─', max(2, width - 2 - textwidth(t))); color=_PAL.rule))
end

# one section: title, then (subheading, table) blocks; returns the rows shown
function _print_section(io, fit, title, blocks; kw...)
    rendered = [(sub, _render_table(io, fit, vars, labels; kw...)) for (sub, vars, labels) in blocks if !isempty(vars)]
    isempty(rendered) && return SummaryRow[]
    _section(io, title, maximum(_vwidth(t) for (_, (t, _)) in rendered))
    for (sub, (t, _)) in rendered
        sub === nothing || println(io, sub)
        println(io, t)
    end
    return reduce(vcat, (r for (_, (_, r)) in rendered))
end

# --- the formula, highlighted -------------------------------------------------------
#
# The stored text has its spaces stripped (labels are built from it); spacing
# is put back here around top-level operators and inside group terms.

function _close(s, i)
    d = 0
    for j in i:lastindex(s)
        isvalid(s, j) || continue
        s[j] == '(' && (d += 1)
        s[j] == ')' && (d -= 1) == 0 && return j
    end
    return lastindex(s)
end

function _split_top(s, sep)
    parts, d, start = String[], 0, firstindex(s)
    for (i, ch) in pairs(s)
        ch == '(' && (d += 1)
        ch == ')' && (d -= 1)
        if ch == sep && d == 0
            push!(parts, strip(s[start:prevind(s, i)]))
            start = nextind(s, i)
        end
    end
    push!(parts, strip(s[start:end]))
    return parts
end

_spaced(s) = replace(s, "||" => " || ", "|" => " | ", "+" => " + ", "*" => " * ")

function _hl_terms(io, s)
    b = IOBuffer()
    i, n = firstindex(s), lastindex(s)
    while i <= n
        ch = s[i]
        if ch == '('
            j = _close(s, i)
            inner = s[i:j]
            if occursin('|', inner)          # group term; ((1+x)|g) is stored with extra parens
                inner = replace(inner, r"^\(\(([^()]*)\)(\|\|?)" => s"(\1\2")
                print(b, _sty(io, _spaced(inner); color=_PAL.group))
            else
                print(b, "(", _hl_terms(io, s[nextind(s, i):prevind(s, j)]), ")")
            end
            i = nextind(s, j)
        elseif ch == '+' || ch == '*'
            print(b, _sty(io, " $ch "; color=_PAL.key))
            i = nextind(s, i)
        elseif isletter(ch) || ch == '_' || ch == '.'
            j = i
            while nextind(s, j) <= n && (isletter(s[nextind(s, j)]) || isdigit(s[nextind(s, j)]) || s[nextind(s, j)] in ('_', '.'))
                j = nextind(s, j)
            end
            k = nextind(s, j)
            if k <= n && s[k] == '('            # special term: s(), mo(), gp(), ...
                e = _close(s, k)
                print(b, _sty(io, replace(s[i:e], "," => ", "); color=_PAL.special))
                i = nextind(s, e)
            else
                print(b, s[i:j])
                i = k
            end
        else
            print(b, ch)
            i = nextind(s, i)
        end
    end
    return String(take!(b))
end

function _hl_formula(io, f)
    f = replace(f, " " => "")
    t = findfirst('~', f)
    t === nothing && return f
    lhs, rhs = f[1:prevind(f, t)], f[nextind(f, t):end]
    return _sty(io, _spaced(lhs); bold=true) * _sty(io, " ~ "; color=_PAL.key) * _hl_terms(io, rhs)
end

# --- header and diagnostics -----------------------------------------------------------

_key(io, k) = _sty(io, lpad(k, 8) * "  "; color=_PAL.key)
_secs(t) = t < 1 ? "$(round(Int, 1000t)) ms" : t < 60 ? "$(round(t; digits=1)) s" : "$(round(t / 60; digits=1)) min"

function _print_header(io, fit, family, links, formulas, nobs)
    println(io, _key(io, "Family"), _sty(io, family; bold=true))
    isempty(links) || println(io, _key(io, "Links"), links)
    n = 0
    for f in formulas, part in _split_top(f, ',')   # dpar formulas on their own lines
        n += 1
        println(io, _key(io, n == 1 ? "Formula" : ""), _hl_formula(io, part))
    end
    println(io, _key(io, "Data"), nobs, " observations")
    println(io, _key(io, "Draws"), nchains(fit), " chains × ", fit.iter, " iterations ",
            _sty(io, "(warmup $(fit.warmup), thin $(fit.thin))"; color=_PAL.key), " → ", ndraws(fit), " draws")
    t = isempty(fit.sampling_time) ? "" : " · " * _secs(mean(fit.sampling_time)) * " per chain"
    println(io, _key(io, "Sampler"), fit.sampler, _sty(io, t; color=_PAL.key))
end

function _extra(fit, key)
    k = FlexiChains.Extra(key)
    return k in keys(fit.chain) ? parent(fit.chain[k]) : nothing
end

function _n_divergent(fit::Union{BrmsFit,BrmsMVFit})
    d = _extra(fit, "divergent__")
    return d === nothing ? 0 : round(Int, sum(d))
end

# E-BFMI of the worst chain (Betancourt 2016); below 0.3 the momentum
# resampling cannot explore the energy distribution
function _ebfmi(E)
    E = reshape(Float64.(E), size(E, 1), :)
    return minimum(c -> (e = E[:, c]; sum(abs2, diff(e)) / sum(abs2, e .- mean(e))), 1:size(E, 2))
end

# one line of checks under the tables: ✓ passed, ✗ failed (with what to try)
function _print_footer(io, fit, rows)
    isempty(rows) && return
    items, hints = Tuple{Bool,String}[], String[]
    rmax = maximum(r -> isnan(r.rhat) ? 1.0 : r.rhat, rows)
    thr = 100 * nchains(fit)
    emin = minimum(r -> min(r.bulk_ess, r.tail_ess), filter(r -> !isnan(r.bulk_ess + r.tail_ess), rows); init=Inf)
    push!(items, rmax <= 1.01 ? (true, "Rhat ≤ 1.01") : (false, "max Rhat " * _f2(rmax)))
    push!(items, emin >= thr ? (true, "ESS ≥ $thr") : (false, "min ESS $(round(Int, emin))"))
    (rmax > 1.01 || emin < thr) && push!(hints, "chains have not mixed well: run more iterations (`iter`) or simplify the model")
    nd = _n_divergent(fit)
    push!(items, nd == 0 ? (true, "no divergences") : (false, "$nd divergent"))
    nd > 0 && push!(hints, "$nd divergent transitions after warmup; consider a higher `delta`")
    td = _extra(fit, "treedepth__")
    if td !== nothing
        md = get(fit.call, :max_depth, 10)
        nt = count(>=(md), td)
        push!(items, nt == 0 ? (true, "treedepth < $md") : (false, "$nt at max treedepth"))
        nt > 0 && push!(hints, "$nt transitions hit the maximum tree depth; consider a higher `max_depth`")
    end
    en = _extra(fit, "energy__")
    if en !== nothing
        b = _ebfmi(en)
        push!(items, b >= 0.3 ? (true, "E-BFMI ≥ 0.3") : (false, "E-BFMI " * _f2(b)))
        b < 0.3 && push!(hints, "low E-BFMI: the posterior has heavy tails or a funnel; consider reparameterising")
    end
    println(io)
    sep = _sty(io, "  ·  "; color=_PAL.rule)
    println(io, " ", join((ok ? _sty(io, "✓ "; color=_PAL.good) * s : _sty(io, "✗ " * s; color=_PAL.warn, bold=true)
                           for (ok, s) in items), sep))
    for h in hints
        println(io, _sty(io, "   " * h; color=_PAL.warn))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fit::BrmsFit)
    info = fit.info
    fam = info.family
    links = join(["$(d) = $(dpar_link_name(fam, d))" for d in dpars(fam)], "; ")
    _print_header(io, fit, family_name(fam), links, [info.formula.text], info.nobs)
    _print_sections(io, fit)
    return nothing
end

# the parameter tables of a fit (shared with the multivariate display);
# returns the rows shown, for the diagnostics line
function _print_sections(io, fit::BrmsFit; footer=true)
    secs = _sections(io, fit)
    # summarise everything shown once, so each row's ESS can be compared with
    # the median of the whole display, not just of its own table
    shown = String[v for (_, blocks) in secs for (_, vars, _) in blocks for v in vars]
    summ = Dict(r.variable => r for r in posterior_summary(fit; variables=shown))
    essref = isempty(summ) ? NaN : median(min(r.bulk_ess, r.tail_ess) for r in values(summ))
    rows = SummaryRow[]
    for (title, blocks) in secs
        append!(rows, _print_section(io, fit, title, blocks; summ, essref))
    end
    footer && _print_footer(io, fit, rows)
    return rows
end

# (title, [(subheading, variables, labels)]) for every section of a fit, brms's order
function _sections(io, fit::BrmsFit)
    names = variables(fit)
    secs = Tuple{String,Vector{Any}}[]

    # group-level hyperparameters, one block per grouping factor
    groups = unique(vcat(String[], [[gb.label for gb in p.groups] for p in _linpreds(fit.density.preds)]...))
    blocks = map(groups) do g
        nlev = first(length(gb.levels) for p in _linpreds(fit.density.preds) for gb in p.groups if gb.label == g)
        sel = filter(n -> startswith(n, "sd_$(g)__") || startswith(n, "cor_$(g)__"), names)
        labels = map(sel) do n
            if startswith(n, "sd_")
                "sd(" * n[length("sd_$(g)__")+1:end] * ")"
            else
                "cor(" * replace(n[length("cor_$(g)__")+1:end], "__" => ",") * ")"
            end
        end
        sub = _sty(io, "~" * g; color=_PAL.group, bold=true) * _sty(io, " (Number of levels: $nlev)"; color=_PAL.key)
        (sub, sel, labels)
    end
    push!(secs, ("Multilevel Hyperparameters", blocks))

    # smooth-term standard deviations (brms: "Smoothing Spline Hyperparameters")
    sds = filter(n -> startswith(n, "sds_"), names)
    push!(secs, ("Smoothing Spline Hyperparameters", [(nothing, sds, ["sds(" * n[5:end] * ")" for n in sds])]))

    bs = filter(n -> startswith(n, "b_") || startswith(n, "bsp_"), names)
    push!(secs, ("Regression Coefficients", [(nothing, bs, [_coef_label(n) for n in bs])]))

    aux = vcat(String[_scalar_label(p.dpar) for p in fit.density.preds if p isa ScalarDpar],
               String[n for p in fit.density.preds if p isa ThetaSimplex for n in ("theta$k" for k in 1:_K(p))])
    push!(secs, ("Further Distributional Parameters", [(nothing, aux, aux)]))
    return secs
end

Base.show(io::IO, fit::BrmsFit) = print(io, "BrmsFit(", fit.info.formula, ", ", family_name(fit.info.family), ")")

"""
    prior_summary(fit_or_info)

The prior used for every parameter class, defaults included, in brms
syntax. Returns the [`PriorSpec`](@ref) rows.
"""
prior_summary(fit::BrmsFit) = prior_summary(fit.info)
function prior_summary(info::ModelInfo)
    rows = info.priors
    println(rpad("prior", 28), rpad("class", 11), rpad("coef", 14), rpad("group", 10), rpad("dpar", 8), "source")
    for r in rows
        println(rpad(sprint(show, r.prior), 28), rpad(string(r.class), 11), rpad(r.coef, 14),
                rpad(r.group, 10), rpad(r.dpar, 8), r.source)
    end
    return rows
end
