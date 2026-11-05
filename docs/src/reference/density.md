# Log density interface

```@meta
CurrentModule = PracticalBayesianRegression
```

[`build_model`](@ref) returns a model's posterior log density as a
[`GLMDensity`](@ref), without sampling it. It implements the
[LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl)
interface, so any sampler or optimiser that accepts that interface can use it,
and it has a closed-form gradient.

```julia
using PracticalBayesianRegression
import LogDensityProblems

ld, info = build_model(@bf(y ~ x + (1 | g)), df; family = poisson())
θ = zeros(LogDensityProblems.dimension(ld))
g = similar(θ)
lp = logdensity_and_gradient!(g, ld, θ)     # value, gradient written into g
names = variables(ld)
values = constrain(ld, θ)                   # natural-scale values of `names`
```

`θ` is on the unconstrained scale: standard deviations enter as their logs,
correlation matrices through Stan's transforms, and so on. The density drops
constant terms and includes the Jacobians of these transforms, as Stan's
does. A `GLMDensity` holds scratch memory; give each thread its own copy with
[`fresh_workspace`](@ref).

```@docs
GLMDensity
logdensity_and_gradient!
fresh_workspace
constrain
variables(::GLMDensity)
```
