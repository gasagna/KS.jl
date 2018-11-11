using Test
using KS
using Flows
import LinearAlgebra: norm, dot, Diagonal, mul!
using Statistics: mean
using Random

include("test_wavenumbers.jl")
include("test_ftfield.jl")
include("test_field.jl")
include("test_ffts.jl")
include("test_shifts.jl")
include("test_system.jl")
# include("test_gradient.jl")
include("test_linearised.jl")