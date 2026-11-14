# Formulas

```@meta
CurrentModule = PracticalBayesianRegression
```

A model formula is written with [`@bf`](@ref) (bare names) or [`bf`](@ref)
(a string, as in R). The syntax is brms's; the table summarises it.

| Term | Meaning |
|:-----|:--------|
| `y ~ x + z` | population-level effects of `x` and `z`, with an intercept |
| `y ~ 0 + x`, `y ~ x - 1` | no intercept |
| `x * z`, `x:z`, `x & z` | main effects and interaction; interaction only |
| `log(x)`, `I(x^2)`, `poly(x, 2)`, `ns(x, 3)`, `bs(x, 4)` | transformed predictors and spline bases |
| `offset(log(t))` | a term with coefficient fixed at 1 |
| `(1 + x \| g)` | correlated group-level intercepts and slopes |
| `(1 + x \|\| g)` | uncorrelated group-level effects |
| `(1 \| g1:g2)`, `(1 \| g1/g2)` | interaction of grouping factors; nesting |
| `(1 \| p \| g)` | group-level effects correlated across formulas sharing the ID `p` |
| `(1 \| gr(g, ...))`, `(1 \| mm(g1, g2))` | grouping options; multi-membership |
| `mo(x)`, `cs(x)` | monotonic effect of an ordinal predictor; category-specific effect |
| `me(x, sdx)`, `mi(x)` | predictor measured with error; predictor with missing values |
| `s(x)`, `t2(x, z)`, `gp(x)` | smooths, tensor-product smooths, Gaussian processes |
| `ar()`, `ma()`, `arma()`, `cosy()`, `unstr()`, `fcor(M)` | temporal and other residual correlation structures |
| `sar(W)`, `car(W, gr = g)` | spatial terms |
| `y \| trials(n)`, `weights(w)`, `se(s)`, `cens(c)`, `trunc(lb = 0)`, `rate(t)`, `mi()`, `thres(3)`, `dec(d)`, `subset(s)`, `vreal(v)`, `vint(n)` | addition terms on the response |
| `sigma ~ z`, `shape ~ z`, ... | formulas for distributional parameters |

```@docs
@bf
bf
@lf
lf
nlf
set_nl
mvbf
set_rescor
set_mecor
BrmsFormula
MVFormula
```
