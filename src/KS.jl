# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
module KS

import Base: call, norm

export KSEq,
       ndofs,
       reconstruct!,
       reconstruct,
       𝒦,
       inner,
       ∂ₓ, ∂ᵥ

immutable KSEq
    ν::Float64
    Nₓ::Int64
end

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

@inline Refk(k::Integer) = - sin(k*π/2)/2π

function 𝒞!(ks::KSEq, ẋ::AbstractVector, x::AbstractVector, v::AbstractVector)
    u = x⋅v # control input
    @simd for k = 1:ks.Nₓ
        @inbounds ẋ[k] += Refk(k)*u
    end
    ẋ
end

function call(ks::KSEq, ẋ::AbstractVector, x::AbstractVector, v::AbstractVector)
    @assert length(x) == length(ẋ) == length(x) == ks.Nₓ
    # use new julia function composition syntax
    fill!(ẋ, zero(eltype(ẋ)))
    ℒ!(ks, ẋ, x)
    𝒩!(ks, ẋ, x)
    𝒞!(ks, ẋ, x, v)
end

# ~~~ Jacobian of the system ~~~
macro checkJacdimension()
    :(size(J) == (length(x), length(v)) || 
        throw(ArgumentError("Wrong input dimension. Got J->$(size(J)), " * 
            "x->$(length(x)), v->$(length(v))")))
end

immutable KSStateJacobian
    ks::KSEq
end
∂ₓ(ks::KSEq) = KSStateJacobian(ks)

function call(ksJ::KSStateJacobian, 
              J::AbstractMatrix, 
              x::AbstractVector, 
              v::AbstractVector)
    @checkJacdimension
    J[:] = zero(eltype(J))
    ν, Nₓ = ksJ.ks.ν, ksJ.ks.Nₓ
    for k = 1:Nₓ # linear term
        @inbounds J[k, k] = k*k*(1 - ν*k*k)
    end
    for p = 1:Nₓ, k = 1:Nₓ # nonlinear term
        k != p   && @inbounds J[k, p] += -2*k*x[abs(k-p)]*sign(k-p) 
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
              x::AbstractVector, 
              v::AbstractVector)
    @checkJacdimension
    Nₓ = ksJ.ks.Nₓ
    for k = 1:Nₓ 
        fk = Refk(k)
        for p = 1:length(v) 
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
            u[i] += 2*xk*sin(k*xg[i])
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
    s = zero(promote_type(eltype(x), eltype(y)))
    @simd for k in 1:ks.Nₓ
        @inbounds s += x[k]*y[k]
    end
    s
end

norm(ks::KSEq, x::AbstractVector) = sqrt(inner(ks, x, x))

# kinetic energy density
𝒦(ks::KSEq, x::AbstractVector) = inner(ks, x, x)


end