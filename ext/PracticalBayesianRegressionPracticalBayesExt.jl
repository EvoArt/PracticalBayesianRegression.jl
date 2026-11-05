module PracticalBayesianRegressionPracticalBayesExt

# PracticalBayes is optional: sampling uses the package's own NUTS (nuts.jl).
# PB is needed only to run generated PB model code (`pb_model`).

import PracticalBayesianRegression as PBR
import PracticalBayes
using LinearAlgebra: LinearAlgebra

# Evaluate generated code in a fresh module that sees PracticalBayes,
# Distributions and LinearAlgebra, as a user pasting the code into a script
# would (PB does not re-export Distributions).
PBR._eval_model_code(::Val{:pb}, body::String) =
    PBR._eval_in_module(body, PracticalBayes, PBR.Distributions, LinearAlgebra, PBR)

end
