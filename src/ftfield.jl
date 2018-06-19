# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export AbstractFTField,
       wavenumber,
       FTField,
       ddx!

# ////// ABSTRACT TYPE FOR SOLUTION IN FOURIER SPACE //////
# n     : is the largest wave number that can be represented
# ISODD : whether we consider odd solutions
# T     : underlying data type, e.g. Float64
abstract type AbstractFTField{n, ISODD, T} <: AbstractVector{T} end

# ////// ABSTRACTARRAY INTERFACE //////
# number of degrees of freedom
Base.size(U::AbstractFTField{n, true})  where {n} = (  n,)
Base.size(U::AbstractFTField{n, false}) where {n} = (2*n,)

Base.IndexStyle(::Type{<:AbstractFTField}) = Base.IndexLinear()


# ////// FULL SOLUTION IN FOURIER SPACE //////
struct FTField{n,
               ISODD,
               T,
               V<:AbstractVector{Complex{T}},
               VL<:AbstractVector{T}} <: AbstractFTField{n, ISODD, T}
    data::V    # the data as a complex array
    dofs::VL   # linearised data for fast look-up of the degrees of freedom
               # this is a essentially a view over data
    L::Float64 # the length of the domain

    # with complex input
    function FTField{n, ISODD}(input::V,
                                   L::Real) where {n,
                                                   ISODD,
                                                   T<:Real,
                                                   V<:AbstractVector{Complex{T}}}
        # checks
        L > 0 || throw(ArgumentError("domain size must be positive"))
        length(input) == n || throw(ArgumentError("inconsistent input"))

        # create data and view
        data = vcat(zero(Complex{T}), input, zero(Complex{T}))
        dofs = reinterpret(T, data)
        new{n, ISODD, T, typeof(data), typeof(dofs)}(data, dofs, L)
    end

    # with real input
    function FTField{n, ISODD}(input::V,
                                   L::Real) where {n,
                                                   ISODD,
                                                   T<:Real,
                                                   V<:AbstractVector{T}}
        # checks
        L > 0 || throw(ArgumentError("domain size must be positive"))
        ISODD == true  && (length(input) ==  n ||
            throw(ArgumentError("inconsistent input")))
        ISODD == false && (length(input) == 2n ||
            throw(ArgumentError("inconsistent input")))

        # create data and view
        dofs = vcat(zero(T), zero(T), _weave(input, Val{ISODD}()), zero(T), zero(T))
        data = reinterpret(Complex{T}, dofs)
        new{n, ISODD, T, typeof(data), typeof(dofs)}(data, dofs, L)
    end
end

# helper function to construct the array of degrees of freedom
function _weave(x::AbstractVector, ::Val{true})
    out = zeros(eltype(x), 2*length(x))
    out[2:2:end] .= x
    return out
end

_weave(x::AbstractVector, ::Val{false}) = x


# ////// outer constructors //////
FTField(n::Int, L::Real, isodd::Bool, fun=k->0) =
    FTField(Complex{Float64}[fun(k) for k in 1:n], L, isodd)

FTField(input::Vector{<:Complex}, L::Real, isodd::Bool) =
    FTField{length(input), isodd}(input, L)

function FTField(input::Vector{<:Real}, L::Real, isodd::Bool)
    N = length(input)
    return isodd == true ? FTField{N,    isodd}(input, L) :
                           FTField{N>>1, isodd}(input, L)
end


# ////// Enforce symmetries, if needed //////
_set_symmetry!(U::AbstractFTField{n,  true}) where {n} =
    (@inbounds @simd for k in wavenumbers(1:n)
        U[k] = im.*imag(U[k])
     end; U)

_set_symmetry!(U::AbstractFTField{n, false}) where {n} =  U



# ////// array interface //////
# custom check bounds
Base.checkbounds(U::AbstractFTField{n}, k::WaveNumber) where {n} =
    (0 < k ≤ n || throw(BoundsError(U, k)); nothing)

Base.checkbounds(U::AbstractFTField{n}, i::Int) where {n} =
    (0 < i ≤ length(U) || throw(BoundsError(U.dofs, i)); nothing)


# indexing over the degrees of freedom
@inline Base.getindex(U::FTField{n, false}, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); getindex(U.dofs, i+2))

@inline Base.setindex!(U::FTField{n, false}, val, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); setindex!(U.dofs, val, i+2))

@inline Base.getindex(U::FTField{n, true}, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); getindex(U.dofs, 2i+2))

@inline Base.setindex!(U::FTField{n, true}, val, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); setindex!(U.dofs, val, 2i+2))

# indexing over the wave numbers
@inline Base.getindex(U::FTField, k::WaveNumber) =
    (@boundscheck checkbounds(U, k); getindex(U.data, k+1))

# no guarantee we do not break the invariance!!
@inline Base.setindex!(U::FTField, val, k::WaveNumber) =
    (@boundscheck checkbounds(U, k); setindex!(U.data, val, k+1))


Base.similar(U::FTField{n, ISODD}) where {n, ISODD} = FTField(n, U.L, ISODD)
Base.copy(U::FTField) = (V = similar(U); V .= U; V)


# ////// inner product and norm //////
Base.dot(U::FTField{n}, V::FTField{n}) where {n} =
    2*real(sum(U[k]*conj(V[k]) for k in wavenumbers(1:n)))

Base.norm(U::FTField) = sqrt(dot(U, U))

# ////// squared norm of the difference //////
dotdiff(U::FTField{n}, V::FTField{n}) where {n} =
    2*real(sum(abs2(U[k] - V[k]) for k in wavenumbers(1:n)))


# ////// shifts and differentiation //////
Base.shift!(U::AbstractFTField{n}, s::Real) where {n} =
    (@inbounds @simd for k in wavenumbers(1:n)
         U[k] *= exp(im*2π*s/U.L*k)
     end; U)

ddx!(U::AbstractFTField{n}) where {n} =
    (@inbounds @simd for k in wavenumbers(1:n)
         U[k] *= im*2π/U.L*k
     end; U)