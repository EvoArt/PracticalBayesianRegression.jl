# Build a model binary

[`build_binary`](@ref) compiles a model into a standalone executable that
refits it to any CSV file with the same columns. The executable needs no
Julia installation, so it can run on machines, pipelines or servers without
one, much as a compiled CmdStan model does.

## Requirements

- Julia 1.12 or later.
- JuliaC, installed as a Julia app: `pkg> app add JuliaC` (and
  `~/.julia/bin` on the `PATH`).
- A C compiler (`gcc` or `clang`).

## Build

```julia
using PracticalBayesianRegression, DataFrames

exe = build_binary(@bf(y ~ x + (1 + x | g), sigma ~ z), train;
                   family = gaussian(), output = "mymodel", name = "mymodel")
```

`train` is example data: it fixes which columns are numeric and which are
factors, and the build checks the compiled model against `brm` on it.
Building takes a few minutes. The `output` directory then holds the
executable in `bin/`, the libraries it needs, and a `README.txt` that lists
the columns the model reads.

## Run

```
JULIA_NUM_THREADS=4 mymodel/bin/mymodel --data new.csv --output draws.csv --chains 4 --seed 1
```

The program fits the model to `new.csv`, writes the draws to `draws.csv`
(one row per draw, with `chain` and `iteration` columns and brms parameter
names) and prints a summary table. `--help` lists the options: `--iter`,
`--warmup`, `--delta`, `--max-depth`, `--metric`, `--init`, and others.
`--columns` prints the columns the model needs.

With the same data and `--seed s`, the executable gives the same draws as
`brm(formula, data; family, prior, seed = s)`.

Distribute the whole `output` directory; the executable needs the libraries
next to it.

## Supported models

Model binaries cover a subset of models so far:

- families: the univariate families other than ordinal, categorical,
  multi-category, mixture and custom families (so gaussian, Student-t,
  lognormal, Bernoulli, binomial, Poisson, negative binomial, gamma, beta,
  the zero-inflated and hurdle families, and others);
- population-level terms with numeric and factor columns, interactions and
  transformed predictors (`log(x)`, `I(x^2)`), `offset()`, `trials()` and
  `weights()`;
- group-level terms `(1 + x | g)`, `(1 + x || g)`, `(1 | g1:g2)`;
- formulas for distributional parameters;
- priors given as distributions, and `lkj()`.

Not yet supported: censoring, truncation and other addition terms;
`poly()`, `ns()` and `bs()`; `mo()`, `cs()`, `me()`, `mi()`; smooths,
Gaussian processes, autocorrelation and spatial terms; `mm()`, `gr()`
options and group-level IDs; shrinkage and `constant()` priors; non-linear
and multivariate models.

`build_binary` raises an error that names any unsupported feature before it
starts compiling. [`model_recipe`](@ref) runs the same check without
building.
