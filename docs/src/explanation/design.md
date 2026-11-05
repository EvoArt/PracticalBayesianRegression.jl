# Design

PracticalBayesianRegression fits the same models as brms, but it is built
differently from brms and from Julia packages such as TuringGLM.jl. This page
explains how and why.

## No probabilistic programming language in between

brms translates a formula into a Stan program, which Stan compiles and
samples. TuringGLM.jl translates a formula into a Turing.jl model, which
Turing evaluates with automatic differentiation. In both, a general-purpose
modelling language sits between the formula and the sampler.

This package has no such layer. From the formula and the data it builds the
model's log density directly, as a Julia object holding the design matrices,
the group indices and the priors, and samples it with its own implementation
of the No-U-Turn sampler (NUTS) with Stan's warm-up adaptation. The aim is to
make each evaluation of the log density and its gradient, which is where
sampling spends almost all its time, as cheap as possible.

### Closed-form gradients

A regression model has a fixed structure. Each distributional parameter has
a linear predictor,

```math
\eta = X\beta + Z u + \dots,
```

and each observation's log-likelihood depends on the parameters only through
its own linear predictors. The gradient therefore splits into two parts:

1. the derivative of each observation's log-likelihood with respect to its
   linear predictors, a few numbers per observation, and
2. the chain rule back through the linear predictors, which is a handful of
   matrix-vector products (`X' * g` for the population-level coefficients,
   scatter-adds over the groups for the group-level effects, and the
   derivatives of the transforms for standard deviations and correlation
   matrices).

The second part is the same for every model and is written out by hand. For
the first, most families have hand-derived derivatives; the rest use one
dual-number (forward-mode) evaluation per observation, which costs a small
multiple of evaluating the log-likelihood itself and needs no tape. No
reverse-mode automatic differentiation runs while sampling.

Every closed-form derivative is tested against automatic differentiation of
a separate generic implementation of the same log density.

### Why that is fast

Generic automatic differentiation records or transforms the whole program,
including the parts whose derivatives are known in advance. Writing the
gradient out avoids that overhead and lets the code:

- compute quantities that depend only on the data (such as `log(y)` for
  positive families, or factorials for counts) once, when the model is built;
- evaluate scalar distributional parameters (a single `sigma`, say) once per
  gradient rather than once per observation;
- leave out terms that are constant during sampling, as Stan does;
- reuse preallocated buffers, so a gradient evaluation allocates no memory.

Chains run in parallel on Julia threads, each with its own scratch buffers.

## Trimmable models

Julia 1.12 can compile a program ahead of time into a standalone executable
with `juliac --trim`, which keeps only the code the program can reach. That
requires every call on the path to be resolved at compile time.

The package is split accordingly. Parsing formulas, building design matrices
and choosing default priors are dynamic and run only when a model is built.
The log density, its gradient and the sampler are written with concrete
types throughout, so a built model compiles to native code without the rest
of Julia. [`build_binary`](@ref) uses this to produce an executable that
refits a fixed model to new data; see [Build a model binary](@ref).

## Turing and PracticalBayes code as an extra

Because there is no modelling language, there is no model code to read or
edit. For users who want one, [`turing_code`](@ref) and [`pb_code`](@ref)
write the model out as a Turing.jl or PracticalBayes.jl `@model`, with the
same design matrices, parameterisation and priors. That code runs through
those packages' automatic differentiation, so it is not guaranteed to be as
fast as `brm`. For features the generator cannot write out, the generated
model calls the package's own log density; `code.fallback` names the
feature.
