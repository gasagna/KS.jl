# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

module KS


# ////// ABSTRACT DEFINITION OF FORCINGS TO AVOID CIRCULARITY //////
# NOTE: subtypes obey a callable interface, where they ALWAYS add to 
#       their input!
abstract type AbstractForcing{n} end

include("wavenumbers.jl")
include("ftfield.jl")
include("field.jl")
include("ffts.jl")
include("nonlinear.jl")
include("linear.jl")
# include("gradient.jl")
include("forcing.jl")

end