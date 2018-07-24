# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export AbstractFTField,
       wavenumber,
       FTField,
       ddx!,
       dotdiff

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
    # with complex input
    function FTField{n, ISODD}(input::V) where {n,
                                                ISODD,
                                                T<:Real,
                                                V<:AbstractVector{Complex{T}}}
        # checks
        length(input) == n || throw(ArgumentError("inconsistent input"))

        # create data and view
        data = vcat(zero(Complex{T}), input, zero(Complex{T}))
        dofs = reinterpret(T, data)
        new{n, ISODD, T, typeof(data), typeof(dofs)}(data, dofs)
    end

    # with real input
    function FTField{n, ISODD}(input::V) where {n,
                                                ISODD,
                                                T<:Real,
                                                V<:AbstractVector{T}}
        # checks
        ISODD == true  && (length(input) ==  n ||
            throw(ArgumentError("inconsistent input")))
        ISODD == false && (length(input) == 2n ||
            throw(ArgumentError("inconsistent input")))

        # create data and view
        dofs = vcat(zero(T), zero(T), _weave(input, Val{ISODD}()), zero(T), zero(T))
        data = reinterpret(Complex{T}, dofs)
        new{n, ISODD, T, typeof(data), typeof(dofs)}(data, dofs)
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
FTField(n::Int, isodd::Bool, fun=k->0) =
    FTField(Complex{Float64}[fun(k) for k in 1:n], isodd)

FTField(input::Vector{<:Complex}, isodd::Bool) =
    FTField{length(input), isodd}(input)

function FTField(input::Vector{<:Real}, isodd::Bool)
    N = length(input)
    return isodd == true ? FTField{N,    isodd}(input) :
                           FTField{N>>1, isodd}(input)
end

# allow constructing object from type specification
function Base.convert(::Type{<:KS.FTField{n, ISODD}}, dofs::Array{<:Real, 1}) where {n, ISODD} 
    flag = ISODD == false ? (length(dofs) == 2n) : length(dofs) == n
    flag || throw(ArgumentError("inconsistent input"))
    return KS.FTField(dofs, ISODD)
end


# ////// Enforce symmetries, if needed //////
_set_symmetry!(U::AbstractFTField{n,  true}) where {n} =
    (@inbounds @simd for k in wavenumbers(n)
        U[k] = im*imag(U[k])
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


Base.similar(U::FTField{n, ISODD}) where {n, ISODD} = FTField(n, ISODD)
Base.copy(U::FTField) = (V = similar(U); V .= U; V)
Base.deepcopy(U::FTField) = copy(U)


# ////// inner product and norm //////
Base.dot(U::FTField{n}, V::FTField{n}) where {n} =
    real(sum(U[k]*conj(V[k]) for k in wavenumbers(n)))

Base.norm(U::FTField) = sqrt(dot(U, U))

# ////// squared norm of the difference //////
dotdiff(U::FTField{n}, V::FTField{n}) where {n} =
    real(sum(abs2(U[k] - V[k]) for k in wavenumbers(n)))

# ////// shifts and differentiation //////
function Base.shift!(U::AbstractFTField{n}, s::Real) where {n}
    if s != 0
        @inbounds @simd for k in wavenumbers(n)
            U[k] *= exp(im*s*k)
        end
    end
    return U
end

# modify first argument
ddx!(iKU::AbstractFTField{n}, U::AbstractFTField{n}) where {n} =
    (@inbounds for k in wavenumbers(n)
         iKU[k] = im*(k*U[k])
     end; iKU)

# in-place function
ddx!(U::AbstractFTField) = ddx!(U, U)