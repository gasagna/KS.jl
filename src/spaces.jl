# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

export FTField, Field

# ~~~ SOLUTION IN FOURIER SPACE ~~~
# n is the largest wave number
# L is the domain size
struct FTField{n, L, T<:Complex, V<:AbstractVector{T}}
    data::V
    FTField{n, L}(data::V) where {n, L, T, V<:AbstractVector{T}} = 
        new{n, L, T, V}(data)
end

# ~ outer constructors 
FTField(n::Int, L::Real, ::Type{T}=Complex128) where {T} = FTField(zeros(T, n+1), L)
FTField(data::AbstractVector, L::Real) = FTField{length(data)-1, L}(data)

# ~ indexing is zero based
Base.indices(::FTField{n}) where {n} = 0:n-1
Base.linearindices(::FTField{n}) where {n} = 0:n-1
Base.IndexStyle(::Type{<:FTField}) = Base.IndexLinear()

@inline Base.getindex(uk::FTField, i::Integer) =
    (@boundscheck checkbounds(uk.data, i+1); 
     @inbounds ret = uk.data[i+1]; ret)

@inline Base.setindex!(uk::FTField, val, i::Integer) =
    (@boundscheck checkbounds(uk.data, i+1); 
     @inbounds uk.data[i+1] = val; val)

Base.similar(::FTField{n, L, T}) where {n, L, T} = FTField(n, L, T)
Base.copy(uk::FTField{n, L}) where {n, L} = FTField(copy(uk.data), L)

# ~ broadcast
@generated function Base.Broadcast.broadcast!(f, uk::FTField, args::Vararg{Any, n}) where {n}
    quote 
        $(Expr(:meta, :inline))
        broadcast!(f, unsafe_get(uk), map(unsafe_get, args)...)
        return uk
    end
end
Base.unsafe_get(uk::FTField) = uk.data

# ~ inner product and norm
function Base.dot(a::FTField{n}, b::FTField{n}) where {n}
    s = a[1]*conj(b[1])
    @simd for k = 2:n
        @inbounds s += a[k]*conj(b[k])
    end
    return 2*real(s)
end

Base.norm(a::FTField) = sqrt(dot(a, a))

# ~~~ SOLUTION IN PHYSICAL SPACE ~~~
struct Field{n, L, T<:Real, V<:AbstractVector{T}} <: AbstractVector{T}
    data::V
    function Field{n, L}(data::V) where {n, L, T, V<:AbstractVector{T}}
        iseven(length(data)) || error("input data must be even")
        2n  == length(data)  || error("inconsistent input data")
        new{n, L, T, V}(data)
    end
end

# ~ outer constructors
Field(n::Int, L::Real, ::Type{T}=Float64) where {T} = Field(zeros(T, 2n), L)
Field(data::AbstractVector, L::Real) = Field{length(data)>>1, L}(data)

# ~ indexing is zero based
Base.indices(::Field{n}) where {n} = 0:2n-1
Base.linearindices(::Field{n}) where {n} = 0:2n-1
Base.IndexStyle(::Type{<:Field}) = Base.IndexLinear()

@inline Base.getindex(u::Field, i::Integer) =
    (@boundscheck checkbounds(u.data, i+1); 
     @inbounds ret = u.data[i+1]; ret)

@inline Base.setindex!(u::Field, val, i::Integer) =
    (@boundscheck checkbounds(u.data, i+1); 
     @inbounds u.data[i+1] = val; val)

Base.similar(::Field{n, L, T}) where {n, L, T} = Field(n, L, T)

# ~ broadcast
@generated function Base.Broadcast.broadcast!(f, u::Field, args::Vararg{Any, n}) where {n}
    quote 
        $(Expr(:meta, :inline))
        broadcast!(f, unsafe_get(u), map(unsafe_get, args)...)
        return u
    end
end
Base.unsafe_get(u::Field) = u.data