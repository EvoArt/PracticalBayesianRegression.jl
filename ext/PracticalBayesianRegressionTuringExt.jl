module PracticalBayesianRegressionTuringExt

# Turing is optional: it evaluates the code `turing_code` generates
# (`turing_model`). Turing re-exports Distributions, so the generated code
# needs only Turing and LinearAlgebra.

import PracticalBayesianRegression as PBR
import Turing
using LinearAlgebra: LinearAlgebra

# (the package itself too: the density-fallback code calls model_logdensity)
PBR._eval_model_code(::Val{:turing}, body::String) = PBR._eval_in_module(body, Turing, LinearAlgebra, PBR)

end
