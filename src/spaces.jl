# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

export AbstractFTField, AbstractField, FTField, Field, WaveNumbers, fieldsize

# ~~~ abstract field types ~~~
abstract type AbstractFTField{n, L, T} <: AbstractVector{T} end
abstract type AbstractField{n, L, T}   <: AbstractVector{T} end

# ~ indexing is zero based
Base.indices(::AbstractFTField{n}) where {n} = (0:n,)
Base.linearindices(::AbstractFTField{n}) where {n} = 0:n
Base.IndexStyle(::Type{<:AbstractFTField}) = Base.IndexLinear()

Base.indices(::AbstractField{n}) where {n} = (0:2n,)
Base.linearindices(::AbstractField{n}) where {n} = 0:2n
Base.IndexStyle(::Type{<:AbstractField}) = Base.IndexLinear()

# extract parameters
fieldsize(::AbstractField{n})   where {n} = n
fieldsize(::AbstractFTField{n}) where {n} = n

# ~~~ WAVE NUMBER VECTOR ~~~
struct WaveNumbers{n, L, Q} <: AbstractFTField{n, L, Float64}
    data::Q
    function WaveNumbers{n, L}(data::Q) where {n, L, Q}
        @assert n == length(data)-1
        new{n, L, Q}(data)
    end
end

WaveNumbers(data::AbstractVector, L::Real) = WaveNumbers{length(data)-1, L}(data)
WaveNumbers(n::Int, L::Real) = WaveNumbers(2π/L*(0:n), L)

@inline Base.getindex(qk::WaveNumbers, i::Integer) =
    (@boundscheck checkbounds(qk.data, i+1); 
     @inbounds ret = qk.data[i+1]; ret)


# ~~~ SOLUTION IN FOURIER SPACE ~~~
# n is the largest wave number
# L is the domain size
struct FTField{n, L, T<:Complex, V<:AbstractVector{T}} <: AbstractFTField{n, L, T}
    data::V
    function FTField{n, L}(data::V) where {n, L, T, V<:AbstractVector{T}}
        n+1 == length(data) || throw(ArgumentError("inconsistent input data"))
        new{n, L, T, V}(data)
    end
end

# ~ outer constructors 
FTField(n::Int, L::Real, ::Type{T}=Complex128) where {T} = FTField(zeros(T, n+1), L)
FTField(data::AbstractVector, L::Real) = FTField{length(data)-1, L}(data)

# ~ array interface
@inline Base.getindex(uk::FTField, i::Integer) =
    (@boundscheck checkbounds(uk.data, i+1); 
     @inbounds ret = uk.data[i+1]; ret)

@inline Base.setindex!(uk::FTField, val, i::Integer) =
    (@boundscheck checkbounds(uk.data, i+1); 
     @inbounds uk.data[i+1] = val; val)

Base.similar(::FTField{n, L, T}) where {n, L, T} = FTField(n, L, T)
Base.copy(uk::FTField{n, L}) where {n, L} = FTField(copy(uk.data), L)

# ~ inner product and norm
function Base.dot(uk::FTField{n}, vk::FTField{n}) where {n}
    s = uk[0]*conj(vk[0])
    @simd for k = 1:n
        @inbounds s += uk[k]*conj(vk[k])
    end
    return 2*real(s)
end

Base.norm(uk::FTField, p::Real...) = sqrt(dot(uk, uk))

# shifts
function Base.shift!(uk::FTField{n, L}, s::Real) where {n, L}
    for k = 0:n
        uk[k] .*= exp(im*2π*s/L*k)
    end
    uk
end

# ~~~ SOLUTION IN PHYSICAL SPACE ~~~
struct Field{n, L, T<:Real, V<:AbstractVector{T}} <: AbstractField{n, L, T}
    data::V
    function Field{n, L}(data::V) where {n, L, T, V<:AbstractVector{T}}
        isodd(length(data))  || throw(ArgumentError("input data must be even"))
        2n+1 == length(data) || throw(ArgumentError("inconsistent input data"))
        new{n, L, T, V}(data)
    end
end

# ~ outer constructors
Field(n::Int, L::Real, ::Type{T}=Float64) where {T} = Field(zeros(T, 2n+1), L)
Field(data::AbstractVector, L::Real) = Field{(length(data)-1)>>1, L}(data)

# ~ array interface
@inline Base.getindex(u::Field, i::Integer) =
    (@boundscheck checkbounds(u.data, i+1); 
     @inbounds ret = u.data[i+1]; ret)

@inline Base.setindex!(u::Field, val, i::Integer) =
    (@boundscheck checkbounds(u.data, i+1); 
     @inbounds u.data[i+1] = val; val)

Base.similar(::Field{n, L, T}) where {n, L, T} = Field(n, L, T)