# A multilevel model

Data often come in groups: repeated measurements on the same people, pupils
in schools, plots in sites. A multilevel model gives each group its own
intercept and slopes, and learns from all groups at once how much these vary.
This tutorial fits one, reads its group-level estimates and predicts for a
new group.

## The data

We simulate reaction times of 18 people measured on 10 consecutive days of
sleep restriction. Each person has their own baseline and their own rate of
slowing down.

```@example ml
using PracticalBayesianRegression, DataFrames, Distributions, Random

rng = Xoshiro(11)
people = ["p" * lpad(i, 2, '0') for i in 1:18]
baseline = Dict(p => 250 + 25 * randn(rng) for p in people)
slowing = Dict(p => 10 + 6 * randn(rng) for p in people)
sleep = DataFrame(person = repeat(people; inner = 10), days = repeat(0:9; outer = 18))
sleep.reaction = [baseline[p] + slowing[p] * d + 25 * randn(rng)
                  for (p, d) in zip(sleep.person, sleep.days)]
first(sleep, 3)
```

The labels are zero-padded (`p01`, ..., `p18`) so that they sort the same way
in Julia and in R; see [Coming from brms](@ref).

## Fitting

`(1 + days | person)` gives every person their own intercept and their own
slope for `days`, drawn from a joint normal distribution whose standard
deviations and correlation are estimated:

```@example ml
fit = brm(@bf(reaction ~ days + (1 + days | person)), sleep; seed = 1)
```

The last lines of the summary check the sampler. Here they report
divergent transitions: in some region of the posterior the sampler's steps
were too large to follow it, so the draws may be biased. This is common in
multilevel models with few groups, where the group-level standard deviations
and the group effects depend strongly on each other. The usual remedy is a
higher target acceptance rate, which makes the steps smaller:

```@example ml
fit = brm(@bf(reaction ~ days + (1 + days | person)), sleep; seed = 1, delta = 0.95)
```

All checks now pass. The summary has a section for the grouping factor `person`: the standard
deviation of the intercepts (`sd(Intercept)`), of the slopes (`sd(days)`) and
their correlation. The population-level coefficients describe the average
person; `sigma` is the residual standard deviation.

## Group-level estimates

[`ranef`](@ref) gives each person's deviation from the average, and
[`group_coef`](@ref) (brms's `coef`) the sum of the average and the
deviation. Here are each person's estimated slopes:

```@example ml
slopes = group_coef(fit)["person"]["days"]
DataFrame(person = [r.variable for r in slopes],
          estimate = [r.estimate for r in slopes],
          lower = [r.lower for r in slopes],
          upper = [r.upper for r in slopes])
```

Each estimate is pulled towards the average slope, more so for people whose
own data are noisy. This partial pooling is what distinguishes a multilevel
model from fitting each person separately.

## Plotting the average effect

[`conditional_effects`](@ref) computes the expected response along one
predictor, with the others held fixed and the group-level effects left out;
[`plot_effects`](@ref) draws it.

```@example ml
using CairoMakie
plot_effects(fit, :days)
```

## Predicting for a new person

Predictions for people in the data use their own estimates. For a person not
in the data, allow new levels: their intercept and slope are then drawn from
the estimated population distribution, so the prediction carries the
between-person uncertainty.

```@example ml
newperson = DataFrame(person = fill("new", 10), days = 0:9)
pred = posterior_epred(fit; newdata = newperson, allow_new_levels = true)
size(pred)
```

`pred` has one row per posterior draw and one column per row of
`newperson`. Summarise it per day:

```@example ml
DataFrame(days = 0:9,
          estimate = vec(mean(pred; dims = 1)),
          lower = [quantile(c, 0.025) for c in eachcol(pred)],
          upper = [quantile(c, 0.975) for c in eachcol(pred)])
```

`re_formula = :none` drops the group-level effects instead, which predicts for
the average person.

## Next steps

- [Predict for new data](@ref) covers the prediction functions and their
  options.
- [Choose sampler settings](@ref) explains what to do about divergent
  transitions, and when a dense metric speeds up multilevel models.
