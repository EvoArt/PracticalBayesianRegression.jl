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
using StatsAPI: vcov, nobs        # re-exported (the StatsAPI functions, so no clash with StatsBase)
import Serialization
import TOML
import PracticalBayesPlot as PBP   # the default plotting backend (plots_pbp.jl)
using PrettyTables: pretty_table, TextHighlighter, TextTableFormat, TextTableStyle, AnsiTextCell,
                    text_table_borders__unicode_rounded, fmt__printf,
                    LatexCell, latex_table_format__booktabs, TypstTableFormat   # report tables (report.jl)
using Printf: @sprintf
import Printf
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
include("smooths_extra.jl")   # mgcv cp / bs / gp / mrf smooths
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
include("contcens.jl")   # log-cdfs of the continuous families (cens / trunc)
include("timecov.jl")
include("build.jl")
include("nuts.jl")
include("fit.jl")
include("approx.jl")
include("mimarg.jl")
include("multivariate.jl")
include("summary.jl")
include("report.jl")        # summary and prior tables as LaTeX, Typst, Markdown, HTML
include("predict.jl")
include("postprocess.jl")
include("averaging.jl")
include("bridge.jl")
include("smooths_post.jl")
include("extras.jl")       # get_prior, VarCorr, loo_predict, pp_mixture, accessors
include("brmsapi.jl")      # vcov, ngrps, reloo, kfold_predict, make_conditions, ...
include("rdists.jl")       # brms's d/p/q/r distribution functions
include("effects.jl")
include("plots.jl")
include("plots_pbp.jl")    # the default plotting backend
include("codegen.jl")
include("csvio.jl")       # model binaries: CSV in / draws out
include("recipe.jl")      # model structure without data; trimmable assembly
include("binary.jl")      # build_binary and the executable's driver
include("store.jl")       # the model store: compile_model, compiled_models, brm(::CompiledModel, data)

export @bf, bf, @lf, lf, @nlf, nlf, set_nl, set_mecor, brm, build_model, sample_model, BrmsFit, BrmsFormula
export set_rescor, MVFormula, BrmsMVFit, build_mv_model
export prior, PriorSpec, prior_summary, lkj
export gaussian, student, lognormal, bernoulli, beta_binomial, poisson, negbinomial, geometric
export gammafamily, weibull, exponential, betafamily, brmsfamily
export zero_inflated_poisson, zero_inflated_negbinomial, zero_inflated_binomial
export hurdle_poisson, hurdle_negbinomial, hurdle_gamma, hurdle_lognormal, cumulative, sratio, cratio, acat, categorical
export skew_normal, exgaussian, inverse_gaussian, frechet, shifted_lognormal
export mixture
export com_poisson, hurdle_cumulative, multinomial, dirichlet, cox, gen_extreme_value, xbeta, logistic_normal
export dirichlet_multinomial, dirichlet2
export loo_subsample, LOOSubsampleResult, vcov, nobs, logLik, ngrps, nvariables, as_draws_matrix, as_draws_df, mvbf, loo_moment_match, reloo
export loo_predictive_interval, loo_model_weights, kfold_predict, make_conditions
export logit_scaled, inv_logit_scaled, logm1, expp1
export dasym_laplace, dbeta_binomial, ddirichlet, dexgaussian, dfrechet, dgen_extreme_value
export dhurdle_gamma, dhurdle_lognormal, dhurdle_negbinomial, dhurdle_poisson, dinv_gaussian
export dlogistic_normal, dmulti_normal, dmulti_student_t, dshifted_lnorm, dskew_normal, dstudent_t
export dvon_mises, dwiener, dzero_inflated_beta, dzero_inflated_beta_binomial
export dzero_inflated_binomial, dzero_inflated_negbinomial, dzero_inflated_poisson, pasym_laplace
export pbeta_binomial, pexgaussian, pfrechet, pgen_extreme_value, phurdle_gamma, phurdle_lognormal
export phurdle_negbinomial, phurdle_poisson, pinv_gaussian, pshifted_lnorm, pskew_normal
export pstudent_t, pvon_mises, pzero_inflated_beta, pzero_inflated_beta_binomial
export pzero_inflated_binomial, pzero_inflated_negbinomial, pzero_inflated_poisson, qasym_laplace
export qfrechet, qgen_extreme_value, qshifted_lnorm, qskew_normal, qstudent_t, rasym_laplace
export rbeta_binomial, rdirichlet, rexgaussian, rfrechet, rgen_extreme_value, rinv_gaussian
export rlogistic_normal, rmulti_normal, rmulti_student_t, rshifted_lnorm, rskew_normal, rstudent_t
export rvon_mises, rwiener
export asym_laplace, zero_inflated_asym_laplace, von_mises, zero_inflated_beta, zero_one_inflated_beta, discrete_weibull, negbinomial2
export custom_family, wiener, zero_inflated_beta_binomial
export brm_multiple, combine_fits, model_weights, posterior_average, pp_average
export posterior_interval, predictive_interval, predictive_error
export bridge_sampler, bayes_factor, post_prob, posterior_smooths, conditional_smooths
export GLMDensity, variables, constrain, draws, fresh_workspace, logdensity_and_gradient!
export posterior_summary, fixef, summary_table, prior_table
export posterior_linpred, posterior_epred, posterior_predict, log_lik, residual_draws
export ranef, group_coef, bayes_R2, loo, waic, loo_compare, LOOResult
export hypothesis, HypothesisResult, conditional_effects
export update, kfold, loo_R2, fitted_summary, predict_summary
export pp_check, pp_check_data, plot_effects, trace_plot, residual_plot, save_plot, embed, Embedded, to_inferencedata
export turing_code, pb_code, model_data, turing_model, ModelCode, model_logdensity, model_dimension, FamilyDist, thresholds, log_mixing_weights, residual_correlation
export build_binary, model_recipe, assemble, read_csv, DataTable, ModelRecipe
export compile_model, compiled_model, compiled_models, find_compiled_model, remove_compiled_model, model_store
export CompiledModel, ModelStore, run_compiled, fit_from_draws
export get_prior, default_prior, validate_prior, PriorTable, VarCorr, loo_predict, loo_epred, loo_linpred
export pp_mixture, prior_draws, nuts_params, log_posterior, rhat, neff_ratio, ndraws, nchains, niterations

end
