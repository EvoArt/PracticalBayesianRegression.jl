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
(one row per draw: `chain` and `iteration` columns, then Stan's sampler
statistics `lp__` ... `energy__`, then the parameters under their brms names)
and prints a summary table. `--unconstrained FILE` also writes the draws on
the unconstrained scale. `--help` lists the options: `--iter`,
`--warmup`, `--delta`, `--max-depth`, `--metric`, `--init`, and others.
`--columns` prints the columns the model needs.

With the same data and `--seed s`, the executable gives the same draws as
`brm(formula, data; family, prior, seed = s)`.

Distribute the whole `output` directory; the executable needs the libraries
next to it.

## Keep compiled models in the model store

[`compile_model`](@ref) builds the same executable into a model store
(`~/.pbr/models`, or the directory in `PBR_MODEL_STORE`) and remembers what it
was built for. Calling it again with the same formula, family, priors and
column types returns the stored model instead of rebuilding, even in a new
session or with the formula written differently:

```julia
m = compile_model(@bf(y ~ x + (1 | g)), train; family = bernoulli())   # minutes, once
fit = brm(m, new)                     # runs the executable: no compilation
fit2 = update(fit; newdata = more)    # runs it again

compiled_models()                     # everything in the store

# by name: pass `name` when compiling to choose one
compile_model(@bf(y ~ x + (1 | g)), train; family = bernoulli(), name = "mymod")
mymod = compiled_model("mymod")       # or the path of its directory or executable
fit = mymod(new; chains = 4, seed = 1)   # a compiled model is callable: brm(mymod, new; ...)
```

`brm(m, data)` returns an ordinary `BrmsFit`: summaries, predictions and
`loo` all work, and with `seed = s` its draws equal
`brm(formula, data; family, prior, seed = s)`. `juliac` defaults to the
command line in the environment variable `PBR_JULIAC`, else the `juliac` app.

The R package PracticalBayesianRegressionR reads the same store, so a model
compiled from Julia runs from R without starting Julia, and the reverse.

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
