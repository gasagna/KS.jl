# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export VecFTField,
	   field

# ///// WRAPPER TO MAKE FTFIELDS BEHAVE LIKE ABSTRACTVECTORS //////
struct VecFTField{T, ISODD, n, P, FT} <: AbstractVector{T}
    p::P
    uk::FT
    function VecFTField{ISODD}(uk::FTField{n, Complex{T}}) where {ISODD, n, T<:Real}
    	p = convert(Ptr{T}, pointer(uk.data))
    	new{T, ISODD, n, typeof(p), typeof(uk)}(p, uk)
    end
end

# from a FTField
VecFTField(uk::FTField, ISODD::Bool) = VecFTField{ISODD}(uk)

# from a Julia Vector
function VecFTField(x::AbstractVector{T}, L::Real, ISODD::Bool) where {T}
	if ISODD 
		return VecFTField{ISODD}(FTField(im*x, L))
	else
		N = length(x)
		N == 2*(N>>1) || throw(ArgumentError("input size must be even"))
		return VecFTField{ISODD}(FTField(reinterpret(Complex{T}, copy(x)), L))
	end
end


# ////// ACCESSOR //////
field(x::VecFTField) = x.uk


# ////// ARRAY INTERFACE //////
Base.IndexStyle(::Type{<:VecFTField}) = Base.IndexLinear()
Base.similar(x::VecFTField{T, ISODD}) where {T, ISODD} = VecFTField(similar(x.uk), ISODD)
Base.copy(x::VecFTField{T, ISODD}) where {T, ISODD} = VecFTField(copy(x.uk), ISODD)

Base.size(x::VecFTField{T, true, n})  where {T, n} = (n,)
Base.size(x::VecFTField{T, false, n}) where {T, n} = (2n,)


# /// with odd symmetry ///
@inline Base.getindex(x::VecFTField{T, true}, i::Int) where {T} =
    (@boundscheck checkbounds(x, i); unsafe_load(x.p, 2i + 2))

@inline Base.setindex!(x::VecFTField{T, true}, val::Real, i::Int) where {T} =
    (@boundscheck checkbounds(x, i); unsafe_store!(x.p, val, 2i + 2))

# /// with no symmetry ///
@inline Base.getindex(x::VecFTField{T, false}, i::Int) where {T} =
	(@boundscheck checkbounds(x, i); unsafe_load(x.p, i + 2))

@inline Base.setindex!(x::VecFTField{T, false}, val::Real, i::Int) where {T} =
	(@boundscheck checkbounds(x, i); unsafe_store!(x.p, val, i + 2))



#= 

struct Aop{S, n, OP, FT<:FTField{n, Complex{S}}} <: AbstractMatrix{S}
    ψ::OP
    uk::FT
    T::Float64
    L::Float64
    isodd::Bool
    function Aop(psi::OP, uk::FTField{n, Complex{S}}, T::Real, isodd::Bool) where {OP, n, S}
        new{S, n, typeof(ψ), typeof(uk)}(ψ, uk, T, uk.L, isodd)
    end
end

Base.issymmetric(::Aop) = false
Base.size(A::Aop{S, n}) where {S, n} = (n, n)

Base.A_mul_B!(y::AbstractVector, A::Aop, x::AbstractVector) = 
    (y .= A.ψ(VecFTField(x, A.L, A.isodd), VecFTField(copy(A.uk), A.isodd), (0, A.T)); y)

=#