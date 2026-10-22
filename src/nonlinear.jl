# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Flows

export ForwardEquation,
       splitexim,
       TangentMode,
       AdjointMode

import LinearAlgebra: mul!

# This triggers a different evaluation of the linearised 
# operator, but the required software infrastructure is 
# the same in both cases.
abstract type AbstractMode end
abstract type AbstractLinearMode <: AbstractMode end

struct ForwardMode <: AbstractMode end
struct TangentMode <: AbstractLinearMode end
struct AdjointMode <: AbstractLinearMode end

# ////// LINEAR TERM //////
struct LinearTerm{n, FT<:AbstractFTField{n}}
    A::FT
    function LinearTerm{n}(ν::Real, ISODD::Bool) where {n}
        ν > 0 || throw(ArgumentError("viscosity must be positive"))
        A = FTField(n, ISODD)
        for k in wavenumbers(n)
            A[k] = k^2 - ν*k^4
        end
        new{n, typeof(A)}(A)
    end
end

LinearTerm(n::Int, ν::Real, ISODD::Bool) = LinearTerm{n}(ν, ISODD)

# obey Flows interface. The operator is self-adjoint!
mul!(dUdt::AbstractFTField{n},
   imTerm::LinearTerm{n},
        U::AbstractFTField{n}) where {n} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
         dUdt[k] = imTerm.A[k] * U[k]
     end; dUdt)

Flows.ImcA!(imTerm::LinearTerm{n},
                 c::Real,
                 U::AbstractFTField{n, ISODD, T},
              dUdt::AbstractFTField{n, ISODD, T}) where {n, ISODD, T} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
          dUdt[k] = U[k]/(one(T) - c*real(imTerm.A[k]))
     end; dUdt)


# ////// NONLINEAR TERM //////
# number of modes m of the grid of the products. With :pad the grid has
# 2(m + 1) ≥ 3n + 1 points, and the modes 1, …, n of the products are exact
_gridmodes(n::Int, dealias::Symbol) = dealias === :pad ? cld(3n - 1, 2) : n

# the option a term was built with, from its inverse transform, for deepcopy
_dealias(::InverseFFT{n, m}) where {n, m} = m == n ? :none : :pad

struct NonLinearExTerm{n, ISODD, FT<:AbstractFTField{n}, F<:AbstractField, IP, FP}
       V::Tuple{FT, FT} # temporaries in Fourier space
       u::Tuple{F, F}   # temporaries in physical space, on the grid of the products
    ifft::IP            # inverse transform, n modes to the grid
     fft::FP            # forward transform, grid to n modes
    function NonLinearExTerm{n}(ISODD::Bool; dealias::Symbol=:none) where {n}
        # ---- input checks ----
        dealias in (:none, :pad) ||
            throw(ArgumentError("dealias must be :none or :pad"))

        # ---- temporaries and transforms ----
        m    = _gridmodes(n, dealias)
        VV   = (FTField(n, ISODD), FTField(n, ISODD))
        uu   = (Field(m), Field(m))
        fft  = ForwardFFT(uu[1], n)
        ifft = InverseFFT(VV[1], m)

        # ---- assemble ----
        return new{n,
                   ISODD,
                   typeof(VV[1]),
                   typeof(uu[1]),
                   typeof(ifft),
                   typeof(fft)}(VV,
                                uu,
                                ifft,
                                fft)
    end
end

NonLinearExTerm(n::Int, ISODD::Bool; dealias::Symbol=:none) =
    NonLinearExTerm{n}(ISODD; dealias=dealias)

# @inline function
#     (exTerm::NonLinearExTerm{n, ISODD, FT})(t::Real,
#                                             U::FT,
#                                          dUdt::FT,
#                                           add::Bool=false) where {n, ISODD, FT}
#     _set_symmetry!(U)
#     exTerm.ifft(U, exTerm.u)        # copy and inverse transform
#     exTerm.u .= exTerm.u.^2         # square
#     exTerm.fft(exTerm.u, exTerm.V)  # forward transform
#     ddx!(exTerm.V)                  # differentiate

#     # store and enforce symmetries
#     add == true ? (dUdt .-= 0.5.*exTerm.V) : (dUdt .= .-0.5.*exTerm.V)

#     return dUdt
# end

@inline function
    (exTerm::NonLinearExTerm{n, ISODD, FT})(t::Real,
                                            U::FT,
                                         dUdt::FT,
                                          add::Bool=false) where {n, ISODD, FT}
    # enforce symmetry
    _set_symmetry!(U)

    # differentiate
    ddx!(exTerm.V[1], U)

    # transform to physical space
    exTerm.ifft(U, exTerm.u[1])
    exTerm.ifft(exTerm.V[1], exTerm.u[2])

    # multiply in physical space
    exTerm.u[1] .= exTerm.u[1] .* exTerm.u[2]

    # go to spectral space again
    exTerm.fft(exTerm.u[1], exTerm.V[1])

    # store and enforce symmetries
    add == true ? (dUdt .-= exTerm.V[1]) : (dUdt .= .- exTerm.V[1])

    return dUdt
end

function Base.deepcopy_internal(x::NonLinearExTerm{n, ISODD}, 
                             dict::IdDict) where {n, ISODD}
    if !( haskey(dict, x) )
        dict[x] = NonLinearExTerm(n, ISODD; dealias=_dealias(x.ifft))
    end
    return dict[x]
end

# ////// COMPLETE EQUATION //////
struct ForwardEquation{n,
                       G<:AbstractForcing{n},
                       LIN<:LinearTerm{n},
                       NLIN<:NonLinearExTerm{n}}
                   imTerm::LIN
                  exTerm::NLIN
               forcing::G
    function ForwardEquation{n}(      ν::Real,
                                  ISODD::Bool,
                                forcing::G;
                                dealias::Symbol=:none) where {n, G}
        exTerm = NonLinearExTerm(n, ISODD; dealias=dealias)
        imTerm = LinearTerm(n, ν, ISODD)
        new{n, typeof(forcing), 
            typeof(imTerm), typeof(exTerm)}(imTerm, exTerm, forcing)
    end
end

"""
    ForwardEquation(n, ν, ISODD, forcing=DummyForcing(n); dealias=:none)

The right-hand side of u_t = -u u_x - u_xx - ν u_xxxx + forcing on [0, 2π), with
n Fourier modes, in the odd subspace (`ISODD = true`) or in the full space. Call
it as `eq(t, U, dUdt)`, the right-hand side of `flow(eq, RK8(U), ...)`, or split
it with `splitexim(eq)` for the IMEX schemes.

With `dealias = :none`, the default, the product u u_x is taken on the 2n + 2
points of the grid, with aliasing. With `dealias = :pad` it is taken on
2(m + 1) ≥ 3n + 1 points. The model is then the exact Galerkin truncation, and it
is equivariant under all translations. The standard form on [0, L) has
ν = (2π/L)², see the README.
"""
ForwardEquation(      n::Int,
                      ν::Real,
                  ISODD::Bool,
                forcing::AbstractForcing=DummyForcing(n);
                dealias::Symbol=:none) =
    ForwardEquation{n}(ν, ISODD, forcing; dealias=dealias)

# split into implicit and explicit terms
function splitexim(eq::ForwardEquation{n}) where {n}
    wrapper(t::Real, U::AbstractFTField{n}, dUdt::AbstractFTField{n}) =
        (eq.exTerm(t, U, dUdt, false); eq.forcing(t, U, dUdt); dUdt)
    return wrapper, eq.imTerm
end

# evaluate right hand side of equation
(eq::ForwardEquation{n})(t::Real,
                         U::FT,
                         dUdt::FT) where {n, FT} =
    (mul!(dUdt, eq.imTerm, U);
     eq.exTerm(t, U, dUdt, true);
     eq.forcing(t, U, dUdt); dUdt)