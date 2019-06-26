# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export AbstractField,
       Field,
       mesh

# ////// SOLUTION IN PHYSICAL SPACE //////
abstract type AbstractField{n, T} <: AbstractVector{T} end

# TODO: make this from zero to ...
Base.size(U::AbstractField{n}) where {n} = (2*(n+1),)
Base.IndexStyle(::Type{<:AbstractField}) = Base.IndexLinear()

struct Field{n, T<:Real, V<:AbstractVector{T}} <: AbstractField{n, T}
    data::V
    function Field{n}(data::V) where {n, T, V<:AbstractVector{T}}
        isodd(length(data)) && throw(ArgumentError("input data length must be even"))
        length(data) == 2*(n+1) || throw(ArgumentError("inconsistent input data"))
        new{n, T, V}(data)
    end
end


# ////// outer constructors /////
Field(n::Int, ::Type{T}=Float64) where{T} = Field{n}(zeros(T, 2*(n+1)))


# ////// array interface //////
@inline Base.getindex(u::Field, i::Integer) =
    (@boundscheck checkbounds(u, i);
        @inbounds ret = u.data[i]; ret)

@inline Base.setindex!(u::Field, val, i::Integer) =
    (@boundscheck checkbounds(u, i);
        @inbounds u.data[i] = val; val)

Base.similar(u::Field{n}) where {n} = Field(n)
Base.copy(u::Field) = (v = similar(u); v .= u; v)
Base.objectid(U::Field) = objectid(U.data)
Base.mightalias(A::Field, B::Field) = Base.mightalias(A.data, B.data)

# ////// MESH //////
mesh(n::Int) = range(0, stop=2π, length=2*(n+1)+1)[1:2*(n+1)]
mesh(u::Field{n}) where {n} = mesh(n)
mesh(U::FTField{n}) where {n} = mesh(n)