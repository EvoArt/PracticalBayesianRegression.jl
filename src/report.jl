# =============================================================================
# Result tables for reports: the summary and the priors as LaTeX, Typst,
# Markdown or HTML, through PrettyTables' back ends.
#
# The rows, sections and column names are the REPL summary's (brms's), so a
# report shows what the REPL showed. The terminal graphics (forest glyphs,
# sparklines, colour) are left out: in a report the plots go in as TikZ or
# CeTZ figures (PracticalBayesPlot's `save`) next to these tables.
#
# Tables are bare (a `tabular`, a Typst `table`) so the document decides
# floats, captions and numbering: `\input{...}` inside `table`, or
# `#figure(include "...")` in Typst. LaTeX tables use booktabs rules.
# =============================================================================

const _TABLE_FORMATS = (:latex, :typst, :markdown, :html)
const _TABLE_EXT = (".tex" => :latex, ".typ" => :typst, ".md" => :markdown, ".html" => :html)

function _table_format(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    i = findfirst(p -> first(p) == ext, _TABLE_EXT)
    i === nothing && throw(ArgumentError("the file extension picks the format: " *
                                         join(first.(_TABLE_EXT), ", ", " or ") * "; got \"$path\""))
    return last(_TABLE_EXT[i])
end

function _check_format(format)
    format in _TABLE_FORMATS ||
        throw(ArgumentError("format must be one of " * join(repr.(_TABLE_FORMATS), ", ") * "; got $(repr(format))"))
end

# a number as the document should typeset it: a true minus sign, not a hyphen
function _tnum(x, digits, tiny, format)
    s = isnan(x) ? "NaN" : Printf.format(Printf.Format(tiny ? "%.$(max(digits - 1, 0))e" : "%.$(digits)f"), x)
    neg = startswith(s, "-")
    format === :latex && return LatexCell(neg ? "\$-\$" * s[2:end] : s)
    return neg ? "−" * s[2:end] : s
end
_tint(x, format) = isnan(x) ? "NaN" : string(round(Int, x))

# (row-group label, fit, variables, row labels) for the tables of a fit, in the
# REPL summary's order; a multilevel block's label names its grouping factor.
# The fit is the one holding the draws: a response's own fit in a multivariate
# model, whose variables are named as in a univariate one
function _table_blocks(fit::BrmsFit; prefix="")
    out = Tuple{String,Any,Vector{String},Vector{String}}[]
    for (title, blocks) in _sections(IOBuffer(), fit), (sub, vars, labels) in blocks
        isempty(vars) && continue
        lab = sub === nothing ? title :
              replace(sub, r"^~(\S+) \(Number of levels: (\d+)\)$" => s"Group-level: \1 (\2 levels)")
        push!(out, (prefix * lab, fit, collect(String, vars), collect(String, labels)))
    end
    return out
end
function _table_blocks(fit::BrmsMVFit)
    out = Tuple{String,Any,Vector{String},Vector{String}}[]
    for (r, v) in zip(fit.resp, fit.fits)
        append!(out, _table_blocks(v; prefix="$r: "))
    end
    if fit.density.rescor
        names = _rescor_names(fit.resp)
        push!(out, ("Residual Correlations", fit, names, [replace(n[9:end], "__" => ",") for n in names]))
    end
    return out
end

"""
    summary_table(fit; format = :latex, prob = 0.95, digits = 2, diagnostics = true) -> String
    summary_table(path, fit; kwargs...)

The posterior summary of `fit` as a table for a report: the rows and sections
the REPL summary shows (group-level standard deviations and correlations,
regression coefficients, further distributional parameters), with brms's
columns `Estimate`, `Est.Error`, the central `prob` interval and, with
`diagnostics = true`, `Rhat`, `Bulk_ESS` and `Tail_ESS`.

`format` is `:latex` (a booktabs `tabular`; load `\\usepackage{booktabs}`),
`:typst` (a `table`), `:markdown`, `:html`, or `:embed` for a Quarto (or
other Pandoc) document: an [`Embedded`](@ref) that a code cell shows as the
HTML table in a web page, the booktabs table in a LaTeX PDF and the Typst
table in a Typst PDF. With a `path`, the table is written to that file and
the extension picks the format (`.tex`, `.typ`, `.md`, `.html`); the path is
returned.

The table has no caption or float, so the document places it:

```latex
\\begin{table}\\centering
  \\caption{Posterior summary}\\input{summary.tex}
\\end{table}
```

```typst
#figure(include "summary.typ", caption: [Posterior summary])
```

Numbers have `digits` decimals; a row whose estimate and interval are all
below 0.005 in magnitude switches to scientific notation, as in the REPL.
"""
function summary_table(fit::Union{BrmsFit,BrmsMVFit}; format::Symbol=:latex, prob::Real=0.95,
                       digits::Integer=2, diagnostics::Bool=true)
    format === :embed &&
        return Embedded((summary_table(fit; format=f, prob, digits, diagnostics) for f in (:html, :latex, :typst))...)
    _check_format(format)
    blocks = _table_blocks(fit)
    labels = reduce(vcat, (b[4] for b in blocks); init=String[])
    rows = reduce(vcat, (posterior_summary(b[2]; variables=b[3], prob) for b in blocks); init=SummaryRow[])
    pct = (p = round(100 * prob; digits=6); isinteger(p) ? string(Int(p)) : string(p))   # 97.5, not 98
    cols = ["Estimate", "Est.Error", "l-$pct% CI", "u-$pct% CI"]
    diagnostics && append!(cols, ["Rhat", "Bulk_ESS", "Tail_ESS"])
    data = Matrix{Any}(undef, length(rows), length(cols))
    for (k, r) in enumerate(rows)
        ty = _tiny(r)
        num = Any[_tnum(r.estimate, digits, ty, format), _tnum(r.est_error, digits, ty, format),
                  _tnum(r.lower, digits, ty, format), _tnum(r.upper, digits, ty, format)]
        diagnostics && append!(num, Any[_tnum(r.rhat, 2, false, format), _tint(r.bulk_ess, format),
                                        _tint(r.tail_ess, format)])
        data[k, :] = num
    end
    starts = cumsum(vcat(1, [length(b[3]) for b in blocks]))[1:end-1]
    groups = Pair{Int,String}[s => b[1] for (s, b) in zip(starts, blocks)]
    return _render_report_table(data, cols, labels, groups, format)
end
summary_table(path::AbstractString, fit::Union{BrmsFit,BrmsMVFit}; kwargs...) =
    _write_table(path, summary_table(fit; format=_table_format(path), kwargs...))

"""
    prior_table(fit_or_info; format = :latex) -> String
    prior_table(path, fit_or_info)

The priors of a model, defaults included, as a table for a report: one row
per prior with its brms-syntax distribution, `class`, `coef`, `group`,
`dpar` and whether it is the user's or the default (`source`), the rows
[`prior_summary`](@ref) prints. Formats and `path` as for
[`summary_table`](@ref).
"""
function prior_table(info::ModelInfo; format::Symbol=:latex)
    format === :embed && return Embedded((prior_table(info; format=f) for f in (:html, :latex, :typst))...)
    _check_format(format)
    ps = info.priors
    cols = ["Prior", "Class", "Coef", "Group", "Dpar", "Source"]
    data = Matrix{Any}(undef, length(ps), length(cols))
    for (k, p) in enumerate(ps)
        data[k, :] = Any[sprint(show, p.prior), string(p.class), p.coef, p.group, p.dpar, string(p.source)]
    end
    # all text: left-aligned
    return _render_report_table(data, cols, nothing, nothing, format; alignment=:l)
end
prior_table(fit::BrmsFit; kwargs...) = prior_table(fit.info; kwargs...)
prior_table(path::AbstractString, x::Union{BrmsFit,ModelInfo}; kwargs...) =
    _write_table(path, prior_table(x; format=_table_format(path), kwargs...))

function _write_table(path, s)
    write(path, s)
    return path
end

# booktabs-like in every back end: rules at the top, under the header, between
# row groups and at the bottom; no vertical lines
const _TYPST_REPORT_FORMAT = TypstTableFormat(; vertical_line_at_beginning=false,
    vertical_line_after_row_number_column=false, vertical_line_after_row_label_column=false,
    vertical_lines_at_data_columns=:none, vertical_line_after_data_columns=false,
    vertical_line_after_continuation_column=false)

# PrettyTables (3.5+) escapes all of Typst's markup characters in cell text,
# so `b_x` is not an emphasis: the text goes in as is.
function _render_report_table(data, cols, labels, groups, format; alignment=:r)
    kw = (; column_labels=cols, alignment)
    labels === nothing || (kw = (; kw..., row_labels=labels, row_label_column_alignment=:l))
    groups === nothing || isempty(groups) || (kw = (; kw..., row_group_labels=groups))
    return sprint() do io
        if format === :latex
            pretty_table(io, data; backend=:latex, table_format=latex_table_format__booktabs, kw...)
        elseif format === :typst
            pretty_table(io, data; backend=:typst, table_format=_TYPST_REPORT_FORMAT, kw...)
        else
            pretty_table(io, data; backend=format, kw...)
        end
    end
end
