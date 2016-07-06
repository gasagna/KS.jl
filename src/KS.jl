# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
module KS

import Base: call, norm

export KSEq,
       ndofs,
       reconstruct!,
       reconstruct,
       KineticEnergyDensity,
       inner,
       ∂ₓ, ∂ᵥ

# Kuramoto-Sivashinski system with single output linear state feedback
immutable KSEq{T}
    ν::Float64   # Hyper viscosity
    Nₓ::Int64    # Number of Fourier modes
    v::Vector{T} # feedback parameters
    function KSEq(ν::Real, Nₓ::Integer, v::AbstractVector{T})
        length(v) == Nₓ || 
            throw(ArgumentError("wrong size of feedback parameters vector"))
        new(ν, Nₓ, v)
    end
end
KSEq{T}(ν::Real, Nₓ::Integer, v::AbstractVector{T}) = KSEq{T}(ν, Nₓ, v)
KSEq(ν::Real, Nₓ::Integer) = KSEq(ν, Nₓ, zeros(Float64, Nₓ))

ndofs(ks::KSEq) = ks.Nₓ

function 𝒩!(ks::KSEq, ẋ::AbstractVector, x::AbstractVector)
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

function ℒ!(ks::KSEq, ẋ::AbstractVector, x::AbstractVector)
    ν, Nₓ = ks.ν, ks.Nₓ
    @simd for k = 1:Nₓ
        @inbounds ẋ[k] += k*k*(1-ν*k*k)*x[k]
    end
    ẋ
end

@inline Refk(k::Integer) = -sin(k*π/4)/2π

# Linear state feedback. Note feedback parameters are defined 
# when the object is instantiated.
function 𝒞!(ks::KSEq, ẋ::AbstractVector, x::AbstractVector)
    u = x⋅ks.v # control input
    @simd for k = 1:ks.Nₓ
        @inbounds ẋ[k] += Refk(k)*u
    end
    ẋ
end

function call(ks::KSEq, ẋ::AbstractVector, x::AbstractVector)
    @assert length(x) == length(ẋ) == ks.Nₓ
    fill!(ẋ, zero(eltype(ẋ)))
    ℒ!(ks, ẋ, x)
    𝒩!(ks, ẋ, x)
    𝒞!(ks, ẋ, x)
end

# ~~~ Jacobian of the system ~~~
function checkJacdimension(J, x, Nₓ)
    size(J) == (length(x), length(x)) &&
    length(x) == Nₓ || throw(ArgumentError("Wrong input dimension. " * 
        "Got J->$(size(J)), x->$(length(x)), v->$(length(v))"))
    nothing
end

immutable KSStateJacobian
    ks::KSEq
end
∂ₓ(ks::KSEq) = KSStateJacobian(ks)

function call(ksJ::KSStateJacobian, 
              J::AbstractMatrix, 
              x::AbstractVector)
    # hoist variables out
    ν, Nₓ, v = ksJ.ks.ν, ksJ.ks.Nₓ, ksJ.ks.v
    # check
    checkJacdimension(J, x, Nₓ)
    # reset
    J[:] = zero(eltype(J))
    for k = 1:Nₓ # linear term
        @inbounds J[k, k] = k*k*(1 - ν*k*k)
    end
    for p = 1:Nₓ, k = 1:Nₓ # nonlinear term
        k != p    && @inbounds J[k, p] += -2*k*x[abs(k-p)]*sign(k-p) 
        k+p <= Nₓ && @inbounds J[k, p] +=  2*k*x[k+p]
    end
    for k = 1:Nₓ # control term
        fk = Refk(k)
        for p = 1:length(v)
            @inbounds J[k, p] += fk*v[p]
        end
    end
    J
end

immutable KSParamJacobian
    ks::KSEq
end
∂ᵥ(ks::KSEq) = KSParamJacobian(ks)

function call(ksJ::KSParamJacobian, 
              J::AbstractMatrix, 
              x::AbstractVector)
    # hoist variables
    Nₓ, v = ksJ.ks.Nₓ, ksJ.ks.v
    # checks
    checkJacdimension(J, x, Nₓ)
    for k = 1:Nₓ 
        fk = Refk(k)
        for p = 1:Nₓ
            @inbounds J[k, p] = fk*x[p]
        end
    end
    J
end


# ~~~ Reconstruction functions ~~~

function reconstruct!(ks::KSEq,           # the system
                      x::AbstractVector,  # state vector
                      xg::AbstractVector, # the grid
                      u::AbstractVector)  # output
    ν, Nₓ = ks.ν, ks.Nₓ
    u[:] = 0
    @inbounds for k = 1:length(x)
        xk = x[k]
        @simd for i = 1:length(xg)
            u[i] -= 2*xk*sin(k*xg[i])
        end
    end
    u
end

function reconstruct!(ks::KSEq,           # the system
                      x::AbstractMatrix,  # state vector
                      xg::AbstractVector, # the grid
                      u::AbstractMatrix)  # output
    for ti = 1:size(u, 1)
        reconstruct!(ks, slice(x, ti, :), xg, slice(u, ti, :))
    end
    u
end

reconstruct(ks::KSEq, x::AbstractVector, xg::AbstractVector) = 
    reconstruct!(ks, x, xg, Array(eltype(x), length(xg)))

reconstruct(ks::KSEq, x::AbstractMatrix, xg::AbstractVector) = 
    reconstruct!(ks, x, xg, Array(eltype(x), size(x, 1), length(xg)))

# ~~~ inner product, norm, energy and the like ~~~

function inner(ks::KSEq, x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) == ks.Nₓ
    x⋅y
end

norm(ks::KSEq, x::AbstractVector) = sqrt(inner(ks, x, x))

# Kinetic energy density
immutable KineticEnergyDensity
    ks::KSEq
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

end