# A precompilation workload: fit a few small models of the common families so
# their compiled code is cached in the package image. A first fit of one of
# these families in a fresh session then takes ~2 s instead of ~30 s (Athena,
# 2026-11); families not listed still compile on first use (~9 s each).
#
# Each family is fitted with a random-intercept formula: a fixed-effects-only
# formula, correlated slopes and other formulas of the same family share the
# same compiled types, so one formula per family covers them. A distributional
# formula (sigma ~ x) is a different density type and is listed separately.
# Chains run sequentially here (one thread during precompilation).
#
# Cost (Athena, 2026-11-17): precompiling the package goes from ~18 s to
# ~126 s and its image from 9 to 62 MB; load time +0.3 s. When developing the
# package, skip the workload in your dev environment's LocalPreferences.toml:
#     [PracticalBayesianRegression]
#     precompile_workload = false
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    rng = Random.Xoshiro(1)
    n = 40
    x = randn(rng, n)
    g = [string("g", rand(rng, 1:4)) for _ in 1:n]
    d = (y=randn(rng, n), x=x, g=g, pos=exp.(randn(rng, n)), k=rand(rng, 0:5, n), b=rand(rng, 0:1, n),
         nt=fill(5, n), s=rand(rng, 0:5, n))
    fits = (("y ~ x + (1 | g)", gaussian()), ("y ~ x + (1 | g)", student()), ("k ~ x + (1 | g)", poisson()),
            ("k ~ x + (1 | g)", negbinomial()), ("b ~ x + (1 | g)", bernoulli()),
            ("s | trials(nt) ~ x + (1 | g)", binomial()), ("pos ~ x + (1 | g)", gammafamily()),
            ("pos ~ x + (1 | g)", lognormal()))
    @compile_workload begin
        # short runs diverge and warn; keep precompilation quiet
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            for (f, fam) in fits
                brm(bf(f), d; family=fam, chains=2, iter=30, warmup=15, seed=1, threads=false)
            end
            brm(bf("y ~ x", "sigma ~ x"), d; chains=2, iter=30, warmup=15, seed=1, threads=false)
        end
    end
end
