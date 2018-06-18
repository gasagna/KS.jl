# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export AbstractField,
       Field,
       mesh

# ////// SOLUTION IN PHYSICAL SPACE //////
abstract type AbstractField{n,   ISODD, T} <: AbstractVector{T} end
# TODO: make this from zero to ...
Base.size(U::AbstractField{n}) where {n} = (2*(n+1),)
Base.IndexStyle(::Type{<:AbstractField}) = Base.IndexLinear()

struct Field{n, T<:Real, V<:AbstractVector{T}} <: AbstractField{n, T}
    data::V
    L::Float64
    function Field{n}(data::V, L::Real) where {n, T, V<:AbstractVector{T}}
        L > 0 || throw(ArgumentError("domain size must be positive"))
        isodd(length(data)) && throw(ArgumentError("input data length must be even"))
        length(data) == 2*(n+1) || throw(ArgumentError("inconsistent input data"))
        new{n, T, V}(data, L)
    end
end


# ////// outer constructors /////
Field(n::Int, L::Real) = Field{n}(zeros(2*(n+1)), L)


# ////// array interface //////
@inline Base.getindex(u::Field, i::Integer) =
    (@boundscheck checkbounds(u, i);
        @inbounds ret = u.data[i]; ret)

@inline Base.setindex!(u::Field, val, i::Integer) =
    (@boundscheck checkbounds(u, i);
        @inbounds u.data[i] = val; val)

Base.similar(u::Field{n}) where {n} = Field(n, u.L)
Base.copy(u::Field) = (v = similar(u); v .= u; v)


# ////// MESH //////
mesh(n::Int, L::Real) = linspace(0, L, 2*(n+1)+1)[1:2*(n+1)]
mesh(u::Field{n}) where {n} = mesh(n, u.L)
mesh(U::FTField{n}) where {n} = mesh(n, U.L)