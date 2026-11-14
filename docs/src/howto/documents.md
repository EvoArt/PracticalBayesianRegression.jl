# Put results in a LaTeX or Typst document

In a Quarto document, [`embed`](@ref) and `format = :embed` put figures and
tables into a LaTeX or Typst PDF directly ([A report with Quarto](@ref)).
When you write the document in LaTeX or Typst yourself, without Quarto, the
package writes each piece of a fit as a file the document includes:

| Piece | Function | LaTeX | Typst |
|:------|:---------|:------|:------|
| a figure | [`save_plot`](@ref) | `.tex`: a TikZ picture | `.typ`: a CeTZ drawing |
| the posterior summary | [`summary_table`](@ref) | `.tex`: a booktabs `tabular` | `.typ`: a `table` |
| the priors | [`prior_table`](@ref) | as above | as above |
| numbers for the text | your own few lines | `.tex`: macros | `.typ`: variables |

The figures are drawings, not images: their text is typeset by LaTeX or
Typst in the document's font and size, and they stay sharp at any zoom.

The examples use this fit:

```@example docs
using PracticalBayesianRegression, Random, Printf

rng = Xoshiro(7)
n = 100
light = round.(10 .* rand(rng, n); digits = 1)
treatment = rand(rng, ["control", "fertilised"], n)
height = 4 .+ 0.6 .* light .+ 1.5 .* (treatment .== "fertilised") .+ 1.2 .* randn(rng, n)
plants = (; height, light, treatment)

fit = brm(@bf(height ~ light + treatment), plants; seed = 1)
nothing # hide
```

The plots are PracticalBayesPlot figures unless another plotting package is
loaded; `backend = :pbp` makes sure of it (this page has CairoMakie loaded).

## Typst

Write the pieces, here into a folder `report/`:

```@example docs
mkpath("report")
save_plot("report/effects.typ", plot_effects(fit, :light => :treatment; backend = :pbp))
save_plot("report/ppcheck.typ", pp_check(fit; backend = :pbp))
summary_table("report/summary.typ", fit)
prior_table("report/priors.typ", fit)
nothing # hide
```

and the numbers the text quotes, as Typst variables. A dictionary per
coefficient, named as [`fixef`](@ref) names it, keeps them together:

```@example docs
num(x) = replace(@sprintf("%.2f", x), "-" => "−")    # a minus sign, not a hyphen
rows = posterior_summary(fit)
open("report/results.typ", "w") do io
    println(io, "#let nobs = ", nobs(fit))
    println(io, "#let b = (")
    for r in fixef(fit)
        println(io, "  ", r.variable, ": (est: \"", num(r.estimate), "\", lo: \"",
                num(r.lower), "\", hi: \"", num(r.upper), "\"),")
    end
    println(io, ")")
    println(io, "#let max_rhat = \"", num(maximum(r.rhat for r in rows)), "\"")
    println(io, "#let divergent = ", round(Int, sum(nuts_params(fit)["divergent__"])))
end
print(read("report/results.typ", String))
```

The document, `report/report.typ`, imports the variables and includes the
rest. `#figure` gives each its number and caption:

```typst
#import "results.typ": *
#set page(paper: "a4", margin: 2.5cm)

= Plant height, light and fertiliser

Height increased by #b.light.est cm per unit of light (95% credible
interval #b.light.lo to #b.light.hi) in #nobs plants, and fertilised plants
were #b.treatmentfertilised.est cm taller (@summary, @effects). The chains
converged (largest $hat(R)$ #max_rhat, #divergent divergent transitions).

#figure(include "summary.typ", caption: [Posterior summary.]) <summary>
#figure(include "effects.typ", caption: [Expected height against light.]) <effects>
#figure(include "ppcheck.typ", caption: [Posterior predictive check.]) <ppcheck>
#figure(include "priors.typ", caption: [Priors.]) <priors>
```

Compile it with `typst compile report/report.typ`. The figures use the
[CeTZ](https://typst.app/universe/package/cetz) package, which Typst
downloads on the first compile; CeTZ 0.5 needs Typst 0.15 or later.

## LaTeX

Write the same pieces with `.tex` extensions:

```@example docs
save_plot("report/effects.tex", plot_effects(fit, :light => :treatment; backend = :pbp))
save_plot("report/ppcheck.tex", pp_check(fit; backend = :pbp))
summary_table("report/summary.tex", fit)
prior_table("report/priors.tex", fit)
nothing # hide
```

The numbers become macros. LaTeX command names cannot contain digits or
underscores, so one command, `\coef{name}{est}`, looks the values up by
name:

```@example docs
texnum(x) = replace(@sprintf("%.2f", x), "-" => "\\ensuremath{-}")
open("report/results.tex", "w") do io
    println(io, "\\newcommand{\\nobs}{", nobs(fit), "}")
    println(io, "\\newcommand{\\coef}[2]{\\csname coef@#1@#2\\endcsname}")
    for r in fixef(fit), (k, v) in (("est", r.estimate), ("lo", r.lower), ("hi", r.upper))
        println(io, "\\expandafter\\def\\csname coef@", r.variable, "@", k, "\\endcsname{", texnum(v), "}")
    end
    println(io, "\\newcommand{\\maxrhat}{", texnum(maximum(r.rhat for r in rows)), "}")
end
print(read("report/results.tex", String))
```

The document, `report/report.tex`, needs `tikz` for the figures and
`booktabs` for the tables:

```latex
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{tikz}
\usepackage{booktabs}
\input{results.tex}

\begin{document}
\section*{Plant height, light and fertiliser}

Height increased by \coef{light}{est} cm per unit of light (95\% credible
interval \coef{light}{lo} to \coef{light}{hi}) in \nobs{} plants, and
fertilised plants were \coef{treatmentfertilised}{est} cm taller
(Table~\ref{tab:summary}, Figure~\ref{fig:effects}). The chains converged
(largest $\hat{R}$ \maxrhat).

\begin{table}[ht]\centering
  \caption{Posterior summary.}\label{tab:summary}
  \input{summary.tex}
\end{table}

\begin{figure}[ht]\centering
  \input{effects.tex}
  \caption{Expected height against light.}\label{fig:effects}
\end{figure}

\begin{figure}[ht]\centering
  \input{ppcheck.tex}
  \caption{Posterior predictive check.}\label{fig:ppcheck}
\end{figure}
\end{document}
```

Compile it with `latexmk -pdf report.tex` (or `pdflatex` twice, for the
references) in `report/`. A figure is drawn at its size in the plotting
function (600 × 450 pixels is 450 × 338 pt); to fit it to the text width,
scaling its text with it, wrap it: `\resizebox{\linewidth}{!}{\input{effects.tex}}`.

## Other formats

[`summary_table`](@ref) and [`prior_table`](@ref) also write Markdown
(`.md`, or `format = :markdown`) and HTML (`.html`, `format = :html`);
[`save_plot`](@ref) writes SVG and HTML. `standalone = true` makes a figure's
`.tex` or `.typ` file a document of its own, to compile it on its own or to
check it.
