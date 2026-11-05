using PracticalBayesianRegression
using Test, Random, LinearAlgebra, Statistics
import ForwardDiff, LogDensityProblems, Distributions
const LDP = LogDensityProblems
const PG = PracticalBayesianRegression

include("testdata.jl")

@testset "PracticalBayesianRegression" begin
    include("formula.jl")
    include("design.jl")
    include("priors.jl")
    include("families.jl")
    include("gradients.jl")
    include("sampling.jl")
    include("predict.jl")
    include("multivariate.jl")
    include("codegen.jl")
    include("recipe.jl")
end
