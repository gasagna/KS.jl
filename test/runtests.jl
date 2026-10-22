using Test
using LinearAlgebra
using Random
using Flows
using KS

include("helpers.jl")

# the tests of 2018-19, ported to the current Flows.jl
include("test_wavenumbers.jl")
include("test_ftfield.jl")
include("test_field.jl")
include("test_ffts.jl")
include("test_shifts.jl")
include("test_system.jl")
include("test_linearised.jl")

# dealiasing, symmetries, the reference model, time stepping
include("test_dealias.jl")
include("test_symmetry.jl")
include("test_reference.jl")
include("test_flows.jl")
