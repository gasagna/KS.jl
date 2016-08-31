# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
module KS

import Base: call, norm

export KSEq, KSEqPointControl, KSEqDistributedControl,
       ndofs,
       reconstruct!,
       reconstruct,
       KineticEnergyDensity,
       inner,
       ∂ₓ, ∂ᵥ,
       issymmetric, R⁺, R⁺!

using POF
using POF.DB

# ~~~ Abstract Kuramoto-Sivashinsky system ~~~ 
abstract AbstractKSEq

# common functions
ndofs(ks::AbstractKSEq) = ks.Nₓ

# actual call
function call(ks::AbstractKSEq, ẋ::AbstractVector, x::AbstractVector)
    @assert length(x) == length(ẋ) == ks.Nₓ
    fill!(ẋ, zero(eltype(ẋ)))
    ℒ!(ks, ẋ, x)
    𝒩!(ks, ẋ, x)
    𝒞!(ks, ẋ, x) # concrete types need to implement custom 𝒞!
end

# linear term
function ℒ!(ks::AbstractKSEq, ẋ::AbstractVector, x::AbstractVector)
    ν, Nₓ = ks.ν, ks.Nₓ
    @simd for k = 1:Nₓ
        @inbounds ẋ[k] += k*k*(1-ν*k*k)*x[k]
    end
    ẋ
end

# nonlinear term
function 𝒩!(ks::AbstractKSEq, ẋ::AbstractVector, x::AbstractVector)
    Nₓ = ks.Nₓ
    for k = 1:Nₓ
        s = zero(eltype(x))
        for m = max(-Nₓ, k-Nₓ):min(Nₓ, k+Nₓ)
            if !(k-m == 0 || m == 0)
                @inbounds s += x[abs(m)]*x[abs(k-m)]*sign(m)*sign(k-m)
            end
        end
        @inbounds ẋ[k] -= k*s
    end
    ẋ
end

# ~~~ System state jacobian ~~~
immutable KSStateJacobian{T}
    ks::T # parametrises type of control
end
∂ₓ(ks::AbstractKSEq) = KSStateJacobian(ks)

function checkdims(ksJ::KSStateJacobian, J::AbstractMatrix, x::AbstractVector)
    Nₓ = ksJ.ks.Nₓ
    size(J) == (Nₓ, Nₓ) &&
    length(x) == Nₓ || throw(ArgumentError("Wrong input dimension. " * 
        "Got J->$(size(J)), x->$(length(x))"))
    nothing
end

# actual call
function call(ksJ::KSStateJacobian, J::AbstractMatrix, x::AbstractVector)
    checkdims(ksJ, J, x)
    fill!(J, zero(eltype(J)))
    ℒ!(ksJ, J, x)
    𝒩!(ksJ, J, x)
    𝒞!(ksJ, J, x) # must implement control type
end

# linear term
function ℒ!(ksJ::KSStateJacobian, J::AbstractMatrix, x::AbstractVector)
    Nₓ, ν = ksJ.ks.Nₓ, ksJ.ks.ν
    for k = 1:Nₓ
        @inbounds J[k, k] += k*k*(1 - ν*k*k)
    end
    J
end

# nonlinear term
function 𝒩!(ksJ::KSStateJacobian, J::AbstractMatrix, x::AbstractVector)
    Nₓ, ν = ksJ.ks.Nₓ, ksJ.ks.ν
    for p = 1:Nₓ, k = 1:Nₓ
        k != p    && @inbounds J[k, p] += -2*k*x[abs(k-p)]*sign(k-p) 
        k+p <= Nₓ && @inbounds J[k, p] +=  2*k*x[k+p]
    end
    J
end


# ~~~ Concrete KS system without control ~~~
immutable KSEq <: AbstractKSEq
    ν::Float64         # Hyper viscosity
    Nₓ::Int64          # Number of Fourier modes
end

# no control 
𝒞!(ks::KSEq, ẋ::AbstractVector, x::AbstractVector) = ẋ
𝒞!(ks::KSStateJacobian, J::AbstractMatrix, x::AbstractVector) = J


# ~~~ Concrete Kuramoto-Sivashinsky system with point actuation ~~~
immutable KSEqPointControl <: AbstractKSEq
    ν::Float64         # Hyper viscosity
    Nₓ::Int64          # Number of Fourier modes
    v::Vector{Float64} # feedback parameters
    xₐ::Float64        # actuator position
    function KSEqPointControl(ν::Real, Nₓ::Integer, v::AbstractVector, xₐ::Real)
        length(v) == Nₓ || 
            throw(ArgumentError("wrong size of feedback parameters vector"))
        0 <= xₐ <= π ||     
            throw(ArgumentError("actuator position must ∈ [0, π]"))
        new(ν, Nₓ, v, xₐ)
    end
end

# control description
@inline Refk(k::Integer, xₐ::Real) = -sin(k*xₐ)/2π

# Linear state feedback driving point actuator
function 𝒞!(ks::KSEqPointControl, ẋ::AbstractVector, x::AbstractVector)
    u  = x⋅ks.v # control input
    xₐ = ks.xₐ 
    @simd for k = 1:ks.Nₓ
        @inbounds ẋ[k] += Refk(k, xₐ)*u
    end
    ẋ
end    

# jacobian of system wrt parameters with point actuation
immutable KSParamJacobianPoint
    ks::KSEqPointControl
end
∂ᵥ(ks::KSEqPointControl) = KSParamJacobianPoint(ks)

function call(ksJ::KSParamJacobianPoint, J::AbstractMatrix,  x::AbstractVector)
    # hoist variables
    Nₓ, v, xₐ = ksJ.ks.Nₓ, ksJ.ks.v, ksJ.ks.xₐ
    # checks
    checkdims(ksJ, J, x)
    for k = 1:Nₓ 
        fk = Refk(k, xₐ)
        for p = 1:Nₓ
            @inbounds J[k, p] = fk*x[p]
        end
    end
    J
end

function checkdims(ksJ::KSParamJacobianPoint, J::AbstractMatrix, x::AbstractVector)
    Nₓ = ksJ.ks.Nₓ
    size(J) == (Nₓ, Nₓ) &&
    length(x) == Nₓ || throw(ArgumentError("Wrong input dimension. " * 
        "Got J->$(size(J)), x->$(length(x))"))
    nothing
end

# jacobian of system wrt state with point actuation, control term only
function 𝒞!(ksJ::KSStateJacobian{KSEqPointControl}, J::AbstractMatrix, x::AbstractVector)
    Nₓ, v, xₐ = ksJ.ks.Nₓ, ksJ.ks.v, ksJ.ks.xₐ
    for k = 1:Nₓ 
        fk = Refk(k, xₐ)
        for p = 1:Nₓ
            @inbounds J[k, p] += fk*v[p]
        end
    end
    J
end


# ~~~ Kuramoto-Sivashinsky system with distributed actuation ~~~
immutable KSEqDistributedControl <: AbstractKSEq
    ν::Float64         # Hyper viscosity
    Nₓ::Int64          # Number of Fourier modes
    V::Matrix{Float64} # feedback parameters
    g::Vector{Float64} # temporary storage for vector of control inputs
    function KSEqDistributedControl(ν::Real, Nₓ::Integer, V::AbstractMatrix)
        size(V) == (Nₓ, Nₓ) || 
            throw(ArgumentError("wrong size of feedback parameters matrix"))
        new(ν, Nₓ, V, zeros(Float64, Nₓ))
    end
end

KSEqDistributedControl(ν::Real, Nₓ::Integer) = 
    KSEqDistributedControl(ν, Nₓ, zeros(Float64, Nₓ, Nₓ))

# Linear state feedback driving distributed control
function 𝒞!(ks::KSEqDistributedControl, ẋ::AbstractVector, x::AbstractVector)
    # A_mul_B!(ks.g, ks.V, x) # pre-compute control input vector
    # g = ks.g
    g = ks.V * x
    # loop
    @simd for k = 1:ks.Nₓ
        @inbounds ẋ[k] -= 0.5*g[k]
    end
    ẋ
end  

# jacobian of system wrt parameters with distributed actuation
immutable KSParamJacobianDistributed
    ks::KSEqDistributedControl
end
∂ᵥ(ks::KSEqDistributedControl) = KSParamJacobianDistributed(ks)

function call(ksJ::KSParamJacobianDistributed, J::AbstractMatrix, x::AbstractVector)
    # hoist variables
    Nₓ = ksJ.ks.Nₓ
    # checks
    checkdims(ksJ, J, x)
    # fill with zeros
    fill!(J, zero(eltype(J)))
    for k = 1:Nₓ
        start = (k-1)*Nₓ + 1  
        stop  = start + Nₓ - 1
        for (pi, p) in enumerate(start:stop)
            @inbounds J[k, p] = -0.5*x[pi]
        end
    end
    J
end

function checkdims(ksJ::KSParamJacobianDistributed, J::AbstractMatrix, x::AbstractVector)
    Nₓ = ksJ.ks.Nₓ
    size(J) == (Nₓ, Nₓ^2) &&
    length(x) == Nₓ || throw(ArgumentError("Wrong input dimension. " * 
        "Got J->$(size(J)), x->$(length(x))"))
    nothing
end

# jacobian of system wrt state with distributed actuation, control term only
function 𝒞!(ksJ::KSStateJacobian{KSEqDistributedControl}, J::AbstractMatrix, x::AbstractVector)
    Nₓ, V = ksJ.ks.Nₓ, ksJ.ks.V
    for k = 1:Nₓ, p = 1:Nₓ
        @inbounds J[k, p] -= 0.5*V[k, p]
    end
    J
end


# ~~~ Reconstruction functions ~~~
function reconstruct!(ks::AbstractKSEq,   # the system
                      x::AbstractVector,  # state vector
                      xg::AbstractVector, # the grid
                      u::AbstractVector)  # output
    ν, Nₓ = ks.ν, ks.Nₓ
    u[:] = 0
    length(u) == length(xg) || error("xg and u must have same length")
    @inbounds for k = 1:length(x)
        xk = x[k]
        @simd for i = 1:length(xg)
            u[i] -= 2*xk*sin(k*xg[i])
        end
    end
    u
end

function reconstruct!(ks::AbstractKSEq,   # the system
                      x::AbstractMatrix,  # state vector
                      xg::AbstractVector, # the grid
                      u::AbstractMatrix)  # output
    for ti = 1:size(u, 1)
        reconstruct!(ks, slice(x, ti, :), xg, slice(u, ti, :))
    end
    u
end

reconstruct(ks::AbstractKSEq, x::AbstractVector, xg::AbstractVector) = 
    reconstruct!(ks, x, xg, Array(eltype(x), length(xg)))

reconstruct(ks::AbstractKSEq, x::AbstractMatrix, xg::AbstractVector) = 
    reconstruct!(ks, x, xg, Array(eltype(x), size(x, 1), length(xg)))

# ~~~ inner product, norm, energy and the like ~~~

function inner(ks::AbstractKSEq, x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) == ks.Nₓ
    x⋅y
end

norm(ks::AbstractKSEq, x::AbstractVector) = sqrt(inner(ks, x, x))

# Kinetic energy density
immutable KineticEnergyDensity
    ks::AbstractKSEq
end
call(k::KineticEnergyDensity, x::AbstractVector) = inner(k.ks, x, x)

# gradient of kinetic energy density wrt state variables
immutable KEDStateGrad end
∂ₓ(k::KineticEnergyDensity) = KEDStateGrad()

call(k::KEDStateGrad, out::AbstractVector, x::AbstractVector) = 
    scale!(copy!(out, x), 2.0)

# gradient of kinetic energy density wrt feedback parameters is zero
immutable KEDParamGrad end
∂ᵥ(k::KineticEnergyDensity) = KEDParamGrad()

call(k::KEDParamGrad, out::AbstractVector, x::AbstractVector) = 
    fill!(out, zero(eltype(out)))

# ~~~ Properties of orbits ~~~

# An orbit for the KS system is symmetric if the odd 
# modes have zero mean. `S` is a vector containing
# the baricenter position.
function issymmetric(S::Vector, tol::Real=1e-6)
    for k = 1:2:length(S)
        abs(S[k]) > tol && return false
    end
    return true
end 

issymmetric(orbit::PeriodicOrbitFile, tol::Real=1e-6) =
    issymmetric(stats(orbit), tol)

# ~~~ Apply symmetries to orbits ~~~
function R⁺!(y::AbstractTrajectory, x::AbstractTrajectory)
    @inbounds for j = 1:length(x) # for each time
        for i = 1:2:ndimensions(x) # invert odd states
            y[i, j] = -x[i, j]
        end
        for i = 2:2:ndimensions(x) # copy even states
            y[i, j] = x[i, j]
        end
    end
    y
end 

R⁺(x) = R⁺!(similar(x), x)

end