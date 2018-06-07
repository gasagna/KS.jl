# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export VecFTField

# ///// WRAPPER TO MAKE FTFIELDS BEHAVE LIKE ABSTRACTVECTORS //////
struct VecFTField{T, ISODD, n, FT<:AbstractFTField{n, Complex{T}}} <: AbstractVector{T}
    uk::FT
end

VecFTField(uk::FTField{n, T}, ISODD::Bool) where {n, T} = VecFTField{T, ISODD, n, typeof(uk)}(uk)

Base.IndexStyle(::Type{<:VecFTField}) = Base.IndexLinear()
Base.similar(x::VecFTField{T, ISODD}) where {T, ISODD} = VecFTField(similar(x.uk), ISODD)
Base.copy(x::VecFTField{T, ISODD}) where {T, ISODD} = VecFTField(copy(x.uk), ISODD)


# ////// WITH ODD SYMMETRY //////
Base.size(x::VecFTField{T, true, n}) where {T, n} = (n,)

@inline function Base.getindex(x::VecFTField{T, true, n}, i::Int) where {T, n}
    @boundscheck checkbounds(x, i)
    @inbounds val = imag(uk[i])
    val
end

@inline function Base.setindex!(x::VecFTField{T, true, n}, val::Real, i::Int) where {T, n}
    @boundscheck checkbounds(x, i)
    @inbounds uk[i] .= im*val
    val
end