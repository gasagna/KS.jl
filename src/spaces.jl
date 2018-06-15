# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export AbstractFTField,
       AbstractField,
       FTField,
       Field,
       mesh,
       ddx!

# ////// ABSTRACT FIELD TYPES //////
abstract type AbstractFTField{n, T} <: AbstractVector{T} end
abstract type AbstractField{n, T}   <: AbstractVector{T} end


# ////// ABSTRACTARRAY INTERFACE //////
Base.size(uk::AbstractFTField{n}) where {n} = (n,)
# TODO: make this from zero to ...
Base.size(uk::AbstractField{n}) where {n} = (2*(n+1),)

Base.IndexStyle(::Type{<:AbstractFTField}) = Base.IndexLinear()
Base.IndexStyle(::Type{<:AbstractField}) = Base.IndexLinear()


# ////// FULL SOLUTION IN FOURIER SPACE //////
struct FTField{n, T<:Complex, V<:AbstractVector{T}} <: AbstractFTField{n, T}
    data::V
    L::Float64
    function FTField{n}(data::V, L::Real) where {n, T<:Complex, V<:AbstractVector{T}}
        L > 0 || throw(ArgumentError("domain size must be positive"))
        n == length(data) || throw(ArgumentError("inconsistent input data"))
        new{n, T, V}(vcat(zero(T), data, zero(T)), L)
    end
end


# ////// outer constructors //////
FTField(n::Int, L::Real) = FTField(zeros(Complex{Float64}, n), L)
FTField(data::Vector{<:Complex}, L::Real) = FTField{length(data)}(data, L)

# ////// array interface //////
@inline Base.getindex(uk::FTField, i::Integer) =
    (@boundscheck checkbounds(uk, i); getindex(uk.data, i+1))

@inline Base.setindex!(uk::FTField, val, i::Integer) =
    (@boundscheck checkbounds(uk, i); setindex!(uk.data, val, i+1))

Base.similar(uk::FTField{n}) where {n} = FTField(n, uk.L)
Base.copy(uk::FTField) = (vk = similar(uk); vk .= uk; vk)


# ////// inner product and norm //////
Base.dot(uk::FTField{n}, vk::FTField{n}) where {n} =
    2*real(sum(uki*conj(vki) for (uki, vki) in zip(uk, vk)))

Base.norm(uk::FTField) = sqrt(dot(uk, uk))


# ////// squared norm of the difference //////
dotdiff(uk::FTField{n}, vk::FTField{n}) where {n} =
    2*real(sum(abs2(uki-vki) for (uki, vki) in zip(uk, vk)))


# ////// shifts and differentiation //////
Base.shift!(uk::AbstractFTField{n}, s::Real) where {n} =
    (uk .*= exp.(im.*2π.*s./uk.L.*(1:n)); uk)

ddx!(uk::FT) where {n, FT<:AbstractFTField{n}} = 
    (uk .*= im.*2π./uk.L.*(1:n); uk)


# ////// SOLUTION IN PHYSICAL SPACE //////
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
mesh(uk::FTField{n}) where {n} = mesh(n, uk.L)