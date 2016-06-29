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
    N::Int64
end

ndofs(ks!::KSEq) = ks!.N

function 𝒩!{T<:Number}(ks!::KSEq, ẋ::AbstractVector{T}, x::AbstractVector{T})
    N = ks!.N
    for k = 1:N
        s = zero(T)
        for m = max(-N, k-N):min(N, k+N)
            if !(k-m == 0 || m == 0)
                @inbounds s += x[abs(m)]*x[abs(k-m)]*sign(m)*sign(k-m)
            end
        end
        @inbounds ẋ[k] -= k*s
    end
    ẋ
end

function ℒ!{T<:Number}(ks!::KSEq, ẋ::AbstractVector{T}, x::AbstractVector{T})
    ν, N = ks!.ν, ks!.N
    @simd for k = 1:N
        @inbounds ẋ[k] += k*k*(1-ν*k*k)*x[k]
    end
    ẋ
end

@inline Refk(k::Integer) = - sin(k*π/2)/2π

function 𝒞!{T<:Number}(ks!::KSEq, ẋ::AbstractVector{T}, x::AbstractVector{T}, v::AbstractVector)
    u = x⋅v # control input
    @simd for k = 1:ks!.N
        @inbounds ẋ[k] += Refk(k)*u
    end
    ẋ
end

function call{T<:Number}(ks!::KSEq, ẋ::AbstractVector{T}, x::AbstractVector{T}, v::AbstractVector)
    @assert length(x) == length(ẋ) == length(x) == ks!.N
    # use new julia function composition syntax
    fill!(ẋ, zero(T))
    ℒ!(ks!, ẋ, x)
    𝒩!(ks!, ẋ, x)
    𝒞!(ks!, ẋ, x, v)
end

# ~~~ Jacobian of the system ~~~
immutable KSStateJacobian
    ks::KSEq
end
∂ₓ(ks!::KSEq) = KSStateJacobian(ks!)

function call(ksJ::KSStateJacobian, 
              J::AbstractMatrix, 
              x::AbstractVector, 
              v::AbstractVector)
    J[:] = zero(eltype(J))
    ν, N = ksJ.ks.ν, ksJ.ks.N
    for k = 1:N # linear term
        @inbounds J[k, k] = k*k*(1 - ν*k*k)
    end
    for p = 1:N, k = 1:N # nonlinear term
        k != p   && @inbounds J[k, p] += -2*k*x[abs(k-p)]*sign(k-p) 
        k+p <= N && @inbounds J[k, p] +=  2*k*x[k+p]
    end
    for k = 1:N # control term
        fk = Refk(k)
        for p = 1:N 
            @inbounds J[k, p] += fk*v[p]
        end
    end
    J
end

immutable KSParamJacobian
    ks::KSEq
end
∂ᵥ(ks!::KSEq) = KSParamJacobian(ks!)

function call(ksJ::KSParamJacobian, 
              J::AbstractMatrix, 
              x::AbstractVector, 
              v::AbstractVector)
    N = ksJ.ks.N
    for k = 1:N 
        fk = Refk(k)
        for p = 1:N 
            @inbounds J[k, p] = fk*x[p]
        end
    end
    J
end


# ~~~ Reconstruction functions ~~~

function reconstruct!(ks!::KSEq,           # the system
                      x::AbstractVector,  # state vector
                      xg::AbstractVector, # the grid
                      u::AbstractVector)  # output
    ν, N = ks!.ν, ks!.N
    u[:] = 0
    @inbounds for k = 1:length(x)
        xk = x[k]
        @simd for i = 1:length(xg)
            u[i] += 2*xk*sin(k*xg[i])
        end
    end
    u
end

function reconstruct!(ks!::KSEq,           # the system
                      x::AbstractMatrix,  # state vector
                      xg::AbstractVector, # the grid
                      u::AbstractMatrix)  # output
    for ti = 1:size(u, 1)
        reconstruct!(ks!, slice(x, ti, :), xg, slice(u, ti, :))
    end
    u
end

reconstruct(ks!::KSEq, x::AbstractVector, xg::AbstractVector) = 
    reconstruct!(ks!, x, xg, Array(eltype(x), length(xg)))

reconstruct(ks!::KSEq, x::AbstractMatrix, xg::AbstractVector) = 
    reconstruct!(ks!, x, xg, Array(eltype(x), size(x, 1), length(xg)))

# ~~~ inner product, norm, energy and the like ~~~

function inner{T, S}(ks!::KSEq, x::AbstractVector{T}, y::AbstractVector{S})
    @assert length(x) == length(y) == ks!.N
    s = zero(promote_type(T, S))
    @simd for k in 1:ks!.N
        @inbounds s += x[k]*y[k]
    end
    2*s
end

norm{T}(ks!::KSEq, x::AbstractVector{T}) = sqrt(inner(ks!, x, x))

# kinetic energy density
𝒦(ks!::KSEq, x::AbstractVector) = inner(ks!, x, x)


end