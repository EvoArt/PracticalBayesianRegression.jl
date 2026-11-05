using Documenter
using PracticalBayesianRegression
using CairoMakie
using InferenceObjects

DocMeta.setdocmeta!(PracticalBayesianRegression, :DocTestSetup, :(using PracticalBayesianRegression);
                    recursive=true)

CairoMakie.activate!(; type="png")

makedocs(
    modules=[PracticalBayesianRegression],
    sitename="PracticalBayesianRegression.jl",
    repo=Remotes.GitHub("EvoArt", "PracticalBayesianRegression.jl"),
    authors="Arthur Newbury",
    format=Documenter.HTML(;
        ansicolor=true,
        edit_link="master",
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/PracticalBayesianRegression.jl",
        size_threshold_warn=200 * 1024,
        size_threshold=400 * 1024,
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "A first model" => "tutorials/first_model.md",
            "A multilevel model" => "tutorials/multilevel.md",
            "A distributional model" => "tutorials/distributional.md",
        ],
        "How-to guides" => [
            "Coming from brms" => "howto/from_brms.md",
            "Set priors" => "howto/priors.md",
            "Choose a family and link" => "howto/families.md",
            "Ordinal and categorical responses" => "howto/ordinal.md",
            "Censored, truncated and weighted data" => "howto/censoring.md",
            "Smooths and Gaussian processes" => "howto/smooths.md",
            "Non-linear models" => "howto/nonlinear.md",
            "Autocorrelation and spatial terms" => "howto/autocorrelation.md",
            "Missing data and measurement error" => "howto/missing.md",
            "Multivariate models" => "howto/multivariate.md",
            "Predict for new data" => "howto/predictions.md",
            "Compare models" => "howto/comparison.md",
            "Use approximate inference" => "howto/approximate.md",
            "Choose sampler settings" => "howto/sampler.md",
            "Plot a fit" => "howto/plotting.md",
            "Generate Turing or PracticalBayes code" => "howto/codegen.md",
            "Build a model binary" => "howto/binaries.md",
        ],
        "Reference" => [
            "Formulas" => "reference/formulas.md",
            "Families" => "reference/families.md",
            "Priors" => "reference/priors.md",
            "Fitting" => "reference/fitting.md",
            "Posterior summaries and predictions" => "reference/postprocessing.md",
            "Model comparison" => "reference/comparison.md",
            "Plotting and export" => "reference/plotting.md",
            "Code generation" => "reference/codegen.md",
            "Model binaries" => "reference/binaries.md",
            "Log density interface" => "reference/density.md",
        ],
        "Explanation" => [
            "Design" => "explanation/design.md",
            "How brms's conventions are followed" => "explanation/conventions.md",
        ],
    ],
    checkdocs=:exports,
    warnonly=false,
)

deploydocs(
    repo="github.com/EvoArt/PracticalBayesianRegression.jl.git",
    devbranch="master",
    push_preview=false,
)
