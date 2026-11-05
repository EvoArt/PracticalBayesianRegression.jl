# Use approximate inference

NUTS gives exact draws from the posterior, given enough of them. For a quick
first look at a large model, or to get starting values, an approximation can
be enough. Choose one with `algorithm`, as in brms:

```julia
fit = brm(f, df; algorithm = :laplace)
fit = brm(f, df; algorithm = :pathfinder, ndraws = 2000)
```

| `algorithm` | Method |
|:------------|:-------|
| `:sampling` | NUTS (the default) |
| `:laplace` | a normal approximation at the posterior mode on the unconstrained scale, with the Hessian at the mode |
| `:meanfield` | variational inference (ADVI) with independent normal factors |
| `:fullrank` | variational inference with a full-covariance normal |
| `:pathfinder` | Pathfinder: normal approximations along an optimisation path, the best one resampled by importance weights |

The result is an ordinary fit with `ndraws` draws (1000 by default) in one
chain, so summaries, predictions and plots work as usual. R-hat and
effective sample sizes are not meaningful for these draws.

The approximations are computed on the unconstrained scale. They tend to
understate the uncertainty when the posterior is skewed or has strong
dependence between parameters, as in multilevel models with few groups.
Check important results against a NUTS fit.
