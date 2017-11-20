# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

# ~~~ SOLUTION IN FOURIER SPACE ~~~
struct FTField{n, T<:Complex, M<:AbstractVector{T}} <: AbstractVector{T}
    data::M
    FTField{n}(data::M) where {n, T, M<:AbstractVector{T}} = new{n, T, M}(data)
end

# n is the largest wave number
FTField(n::Int, ::Type{T}=Complex128) where {T} = FTField(zeros(T, n+1))
FTField(data::AbstractVector) = FTField{length(data)}(data)

# indexing is zero based
Base.indices(f::FTField{n}) where {n} = (0:n-1)
Base.linearindices(f::FTField{n}) where {n} = 1:n
Base.IndexStyle(::Type{<:FTField}) = Base.IndexLinear()

@inline Base.getindex(U::FTField, i::Integer) =
    (@boundcheck checkbounds(u.data, i+1); 
     @inbounds ret = u.data[i+1]; ret)

@inline Base.setindex!(U::FTField, val, i::Integer) =
    (@boundcheck checkbounds(u.data, i+1); 
     @inbounds u.data[i+1] = val); val)

Base.similar(U::FTField{n, T}) where {n, T} = FTField(n, T)

# inner product between two fields # CHECK
function Base.dot(a::FTField{n, T}, b::FTField{n, T})
    s = real(a[1]*conj(b[1]))
    @simd for k = 2:n
        @inbounds s += real(a[k]*conj(b[k]))
    end
    return s
end

# norm
Base.norm(a::FTField) = sqrt(dot(a, a))


# ~~~ SOLUTION IN PHYSICAL SPACE ~~~
struct Field{n, T<:Real, M<:AbstractMatrix{T}} <: AbstractVector{T}
    data::M
    function Field{n}(data::M) where {n, T, M<:AbstractVector{T}}
        2n+1 == length(data) || error("wrong input data")
        new{n, T, M}(data)
    end
end

Field(n::Int, ::Type{T}=Float64) where {T} = Field(zeros(T, 2n+1))
Field(data::AbstractVector) = Field{length(data)>>1+1}(data)
