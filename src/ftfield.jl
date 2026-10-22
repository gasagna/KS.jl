# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export AbstractFTField,
       wavenumber,
       FTField,
       ddx!,
       normdiff,
       shift!,
       reflect!,
       diffmat,
       minnormdiff,
       grow,
       shrink,
       toresol

import LinearAlgebra: dot, norm, reflect!
import SparseArrays: spdiagm

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
               P<:Ptr{T}} <: AbstractFTField{n, ISODD, T}
    data::V # the data as a complex array
    dptr::P # linearised data for fast look-up of the degrees of freedom
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
        dptr = convert(Ptr{T}, pointer(data))
        new{n, ISODD, T, typeof(data), typeof(dptr)}(data, dptr)
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
        data = _write(zeros(Complex{T}, n+2), input, Val{ISODD}())
        dptr = convert(Ptr{T}, pointer(data))
        new{n, ISODD, T, typeof(data), typeof(dptr)}(data, dptr)
    end
end

# helper function to construct the array of degrees of freedom
function _write(data::AbstractVector, input::AbstractVector, ::Val{true})
    for i = 1:length(input)
        data[i+1] = im*input[i]
    end
    return data
end

function _write(data::AbstractVector, input::AbstractVector, ::Val{false})
    for i = 1:length(data)-2
        data[1 + i] = input[2i - 1] + im*input[2i]
    end
    return data
end

# ////// outer constructors //////
"""
    FTField(n, ISODD, fun=k->0)

The Fourier coefficients U_k = fun(k), k = 1, …, n, of the real field
u(x) = Σ_{0<|k|≤n} U_k e^{ikx} on [0, 2π), with U_{-k} = conj(U_k) and zero mean.
`U[WaveNumber(k)]` is U_k. The linear index runs over the degrees of freedom,
(Re U_1, Im U_1, …, Re U_n, Im U_n) in the full space (`ISODD = false`) and
(Im U_1, …, Im U_n) in the odd space, where u = -2 Σ_k Im U_k sin(kx).
"""
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
    (0 < i ≤ length(U) || throw(BoundsError(U.dptr, i)); nothing)


# indexing over the degrees of freedom
@inline Base.getindex(U::FTField{n, false}, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); unsafe_load(U.dptr, i+2))

@inline Base.setindex!(U::FTField{n, false}, val, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); unsafe_store!(U.dptr, val, i+2))

@inline Base.getindex(U::FTField{n, true}, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); unsafe_load(U.dptr, 2i+2))

@inline Base.setindex!(U::FTField{n, true}, val, i::Int) where {n} =
    (@boundscheck checkbounds(U, i); unsafe_store!(U.dptr, val, 2i+2))

# indexing over the wave numbers
@inline Base.getindex(U::FTField, k::WaveNumber) =
    (@boundscheck checkbounds(U, k); @inbounds ret = U.data[k+1]; ret)

# no guarantee we do not break the invariance!!
@inline Base.setindex!(U::FTField, val, k::WaveNumber) =
    (@boundscheck checkbounds(U, k); @inbounds U.data[k+1] = val; val)


Base.similar(U::FTField{n, ISODD}) where {n, ISODD} = FTField(n, ISODD)
Base.copy(U::FTField) = (V = similar(U); V .= U; V)
Base.parent(U::FTField) = U.data

# The default deepcopy keeps dptr as it is, a pointer into the data of U, so that
# writes to the copy would go to U. This one builds a new pointer, into the new data
function Base.deepcopy_internal(U::FTField{n, ISODD}, dict::IdDict) where {n, ISODD}
    haskey(dict, U) && return dict[U]
    V = FTField{n, ISODD}(U.data[2:n+1])
    dict[U] = V
    return V
end

function grow(U::FTField{n, ISODD}, m::Int) where {n, ISODD}
    if m < n
        throw(ArgumentError("cannot grow to smaller resolution"))
    end
    return FTField(vcat(U.data[2:end-1], im*zeros(m - n)), ISODD)
end

function shrink(U::FTField{n, ISODD}, m::Int) where {n, ISODD}
    if m > n
        throw(ArgumentError("cannot shrink to larger resolution"))
    end
    return FTField(U.data[2:(m+1)] + im*zeros(m), ISODD)
end

function toresol(U::FTField{n, ISODD}, m::Int) where {n, ISODD}
    n == m && return U
    m  > n && return grow(U, m)
    m  < n && return shrink(U, m)
end

# see julia issue #28178
Base.objectid(U::FTField) = objectid(U.data)
Base.mightalias(A::FTField, B::FTField) = Base.mightalias(A.data, B.data)

# ////// inner product and norm //////
"""
    dot(U, V)

The inner product Σ_k Re(U_k conj(V_k)) = (1/4π) ∫ u v dx over [0, 2π). In the
full space it is the Euclidean product of the degrees of freedom. `norm(U)` is
sqrt(dot(U, U)).
"""
function dot(U::FTField{n, ISODD, T},
             V::FTField{n, ISODD, T}) where {n, ISODD, T}
    out = zero(T)
    @inbounds @simd for k in wavenumbers(n)
        out += real(sum(U[k]*conj(V[k])))
    end
    return out
end

norm(U::FTField) = sqrt(dot(U, U))

# ////// norm of the difference //////
normdiff(U::FTField{n}, V::FTField{n}) where {n} =
    sqrt(real(sum(abs2(U[k] - V[k]) for k in wavenumbers(n))))

"""
    minnormdiff(U, V, N=20; method=:grid) -> (d, (s,))

The shift s that brings `U` closest to `V`, with d = |shift!(U, s) - V| the
norm of the difference (`normdiff`). Both methods start from the N shifts
s = 2πj/N.

With `method = :grid`, the default, s is the best of the N shifts. The search
shifts `U` in place N times, which leaves `U` as it was up to round-off.

With `method = :newton`, Newton's method refines the best of the N shifts
toward the local minimum next to it. That shift is kept when the refinement
does not lower d, so d never exceeds its value there. Neither argument
changes. The global minimum can sit next to another of the N shifts, which
happens less often for larger N. Take N ≥ 8n.
"""
function minnormdiff(     U::FTField{n},
                          V::FTField{n},
                          N::Int=20;
                     method::Symbol=:grid) where {n}
    # ---- input checks ----
    method in (:grid, :newton) ||
        throw(ArgumentError("method must be :grid or :newton"))

    # ---- grid search, shifting U in place ----
    if method === :grid
        dmin = normdiff(U, V)
        imin = 0
        for i = 1:N
            # by shifting N times by 2π/N we leave U unchanged at the exit
            shift!(U, 2π/N)
            d = normdiff(U, V)
            if d < dmin
                dmin = d
                imin = i
            end
        end
        return dmin, (imin * 2π/N, )
    end

    # ---- C(s), its derivatives, and the distance without cancellation ----
    # |shift!(U, s) - V|² = |U|² + |V|² - 2C(s), C(s) = Σ_k Re(U_k conj(V_k) e^{iks}).
    # They take the shift as an argument, so that the shift is not boxed below
    C(σ)    = sum(real(U[k]*conj(V[k])*cis(k*σ))      for k in wavenumbers(n))
    C′(σ)   = sum(real(im*k*U[k]*conj(V[k])*cis(k*σ)) for k in wavenumbers(n))
    C″(σ)   = sum(real(-k^2*U[k]*conj(V[k])*cis(k*σ)) for k in wavenumbers(n))
    dist(σ) = sqrt(sum(abs2(U[k]*cis(k*σ) - V[k])     for k in wavenumbers(n)))

    # ---- best of the N shifts ----
    j  = argmax(i -> C(2π*i/N), 0:N-1)
    s₀ = 2π*j/N

    # ---- Newton's method on C'(s) = 0, from the best of the N shifts ----
    s = s₀
    for _ in 1:30
        d2 = C″(s)
        d2 < 0 || break # not near a maximum, as when V = 0
        δ  = -C′(s)/d2
        s += δ
        abs(δ) < 1e-14 && break
    end

    # ---- keep the best of the N shifts when Newton's method ended above it ----
    # a step from a point where C is barely concave can jump to a lower maximum
    d  = dist(s)
    d₀ = dist(s₀)
    if d₀ < d
        s = s₀
        d = d₀
    end

    return d, (mod(s, 2π), )
end

# ////// shifts and differentiation //////
"""
    shift!(U, s) -> U

Translate `U` in place, u(x) → u(x + s), that is U_k → U_k e^{iks}. It preserves
`dot`, and its generator is `ddx!`. In the odd space only s = 0 and s = π keep
a field odd.
"""
function shift!(U::AbstractFTField{n}, s::Real) where {n}
    if s != 0
        @inbounds @simd for k in wavenumbers(n)
            U[k] *= exp(im*s*k)
        end
    end
    return U
end

"""
    reflect!(U) -> U

Reflect `U` in place, u(x) → -u(-x), that is U_k → -conj(U_k). The equation
is equivariant under the reflection, and odd fields do not change.
"""
function reflect!(U::AbstractFTField{n}) where {n}
    # ---- U_k → -conj(U_k), the real parts change sign ----
    @inbounds for k in wavenumbers(n)
        U[k] = -conj(U[k])
    end

    return U
end

"""
    ddx!(V, U) -> V
    ddx!(U) -> U

The derivative ∂x u, V_k = ik U_k. It is the generator of `shift!`, the
derivative of shift!(U, s) at s = 0.
"""
ddx!(iKU::AbstractFTField{n}, U::AbstractFTField{n}) where {n} =
    (@inbounds for k in wavenumbers(n)
         iKU[k] = im*(k*U[k])
     end; iKU)

# in-place function
ddx!(U::AbstractFTField) = ddx!(U, U)


# CONSTRUCT DIFFERENTIATION MATRIX

# little helper function to insert zeros into a vector
function _add_zeros(x::AbstractVector)
    out = zeros(eltype(x), 2*length(x) - 1)
    for i = 1:length(x)
        out[2i-1] = x[i]
    end
    return out
end

function diffmat(n::Int, ISODD::Bool, D::AbstractMatrix)
    # check D has the right size
    ISODD == false ||
        throw(ArgumentError("only implemented for non odd fields"))
    return spdiagm(1=>-_add_zeros(1:n), -1=>_add_zeros(1:n))
end