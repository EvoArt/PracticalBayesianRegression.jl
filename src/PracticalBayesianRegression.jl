"""
    PracticalBayesianRegression

Bayesian regression models in the style of R's brms: brms formula syntax,
families and default priors, and brms parameter names. Each model's log
density and its gradient are built directly from the formula and sampled
with the package's own NUTS. Start with [`brm`](@ref) and [`@bf`](@ref).
"""
module PracticalBayesianRegression

using LinearAlgebra
using Random
using Statistics
import LogDensityProblems
import AbstractMCMC
import AdvancedHMC
import ForwardDiff
import FlexiChains
import MCMCDiagnosticTools
using MCMCDiagnosticTools: rhat     # extended for fits in extras.jl, and re-exported
import DataAPI
import Tables
import Distributions
import SpecialFunctions
import PSIS
import StatsAPI
import Serialization
import TOML
using PrettyTables: pretty_table, TextHighlighter, TextTableFormat, TextTableStyle, AnsiTextCell,
                    text_table_borders__unicode_rounded, fmt__printf
using Printf: @sprintf
using Crayons: Crayon
using SpecialFunctions: loggamma, logabsgamma, digamma, logabsbinomial, logbeta, erfc, erfcx, logerfc, besselix
using LogExpFunctions: log1pexp, logistic, logit, logaddexp, log1mexp, logsumexp

include("formula.jl")
include("design.jl")
include("rsplines.jl")
include("priors.jl")
include("links.jl")
include("families.jl")
include("special.jl")
include("splines.jl")
include("gp.jl")
include("horseshoe.jl")
include("r2d2.jl")
include("timecov_types.jl")
include("density.jl")
include("nonlinear.jl")
include("ordinal.jl")
include("censoring.jl")
include("addition.jl")
include("categorical.jl")
include("dualfamilies.jl")
include("customfamily.jl")
include("wiener.jl")
include("extrafamilies.jl")
include("mixture.jl")
include("multicat.jl")
include("cox.jl")
include("gev.jl")
include("timecov.jl")
include("build.jl")
include("nuts.jl")
include("fit.jl")
include("approx.jl")
include("mimarg.jl")
include("multivariate.jl")
include("summary.jl")
include("predict.jl")
include("postprocess.jl")
include("averaging.jl")
include("bridge.jl")
include("smooths_post.jl")
include("extras.jl")       # get_prior, VarCorr, loo_predict, pp_mixture, accessors
include("effects.jl")
include("plots.jl")
include("codegen.jl")
include("csvio.jl")       # model binaries: CSV in / draws out
include("recipe.jl")      # model structure without data; trimmable assembly
include("binary.jl")      # build_binary and the executable's driver
include("store.jl")       # the model store: compile_model, compiled_models, brm(::CompiledModel, data)

export @bf, bf, @lf, lf, set_mecor, brm, build_model, sample_model, BrmsFit, BrmsFormula
export set_rescor, MVFormula, BrmsMVFit, build_mv_model
export prior, PriorSpec, prior_summary, lkj
export gaussian, student, lognormal, bernoulli, beta_binomial, poisson, negbinomial, geometric
export gammafamily, weibull, exponential, betafamily, brmsfamily
export zero_inflated_poisson, zero_inflated_negbinomial, zero_inflated_binomial
export hurdle_poisson, hurdle_negbinomial, hurdle_gamma, hurdle_lognormal, cumulative, sratio, cratio, acat, categorical
export skew_normal, exgaussian, inverse_gaussian, frechet, shifted_lognormal
export mixture
export com_poisson, hurdle_cumulative, multinomial, dirichlet, cox, gen_extreme_value, xbeta, logistic_normal
export asym_laplace, von_mises, zero_inflated_beta, zero_one_inflated_beta, discrete_weibull, negbinomial2
export custom_family, wiener, zero_inflated_beta_binomial
export brm_multiple, combine_fits, model_weights, posterior_average, pp_average
export posterior_interval, predictive_interval, predictive_error
export bridge_sampler, bayes_factor, post_prob, posterior_smooths, conditional_smooths
export GLMDensity, variables, constrain, draws, fresh_workspace, logdensity_and_gradient!
export posterior_summary, fixef
export posterior_linpred, posterior_epred, posterior_predict, log_lik, residual_draws
export ranef, group_coef, bayes_R2, loo, waic, loo_compare, LOOResult
export hypothesis, HypothesisResult, conditional_effects
export update, kfold, loo_R2, fitted_summary, predict_summary
export pp_check, pp_check_data, plot_effects, trace_plot, residual_plot, to_inferencedata
export turing_code, pb_code, model_data, turing_model, ModelCode, model_logdensity, model_dimension, FamilyDist, thresholds, log_mixing_weights, residual_correlation
export build_binary, model_recipe, assemble, read_csv, DataTable, ModelRecipe
export compile_model, compiled_model, compiled_models, find_compiled_model, remove_compiled_model, model_store
export CompiledModel, ModelStore, run_compiled, fit_from_draws
export get_prior, default_prior, validate_prior, PriorTable, VarCorr, loo_predict, loo_epred, loo_linpred
export pp_mixture, prior_draws, nuts_params, log_posterior, rhat, neff_ratio, ndraws, nchains, niterations

end
