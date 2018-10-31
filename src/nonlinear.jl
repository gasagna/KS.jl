# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Flows
import VectorPairs

export ForwardEquation,
       splitexim,
       TangentMode,
       AdjointMode

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
Base.A_mul_B!(dUdt::AbstractFTField{n},
            imTerm::LinearTerm{n},
                 U::AbstractFTField{n}) where {n} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
         dUdt[k] = imTerm.A[k] * U[k]
     end; dUdt)

Flows.ImcA!(imTerm::LinearTerm{n},
                 c::Real,
                 U::AbstractFTField{n},
              dUdt::AbstractFTField{n}) where {n} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
          dUdt[k] = U[k]/(1 - c*imTerm.A[k])
     end; dUdt)


# ////// NONLINEAR TERM //////
struct NonLinearExTerm{n, FT<:AbstractFTField{n}, F<:AbstractField{n}}
     V::FT      # temporary in Fourier space
     u::F       # solution in physical space
    ifft        # plans
    fft         #
    function NonLinearExTerm{n}(ISODD::Bool) where {n}
        V = FTField(n, ISODD); u = Field(n)
        fft, ifft = ForwardFFT(u), InverseFFT(V)
        new{n, typeof(V), typeof(u)}(V, u, ifft, fft)
    end
end

NonLinearExTerm(n::Int, ISODD::Bool) = NonLinearExTerm{n}(ISODD)

@inline function
    (exTerm::NonLinearExTerm{n, FT})(t::Real,
                                     U::FT,
                                     dUdt::FT,
                                     add::Bool=false) where {n, FT}
    _set_symmetry!(U)
    exTerm.ifft(U, exTerm.u)        # copy and inverse transform
    exTerm.u .= exTerm.u.^2         # square
    exTerm.fft(exTerm.u, exTerm.V)  # forward transform
    ddx!(exTerm.V)                  # differentiate

    # store and enforce symmetries
    add == true ? (dUdt .-= 0.5.*exTerm.V) : (dUdt .= .-0.5.*exTerm.V)

    return dUdt
end


# ////// COMPLETE EQUATION //////
struct ForwardEquation{n,
                       G<:AbstractForcing{n},
                       LIN<:LinearTerm{n},
                       NLIN<:NonLinearExTerm{n}}
                   imTerm::LIN
                  exTerm::NLIN
               forcing::G
    function ForwardEquation{n}(ν::Real, ISODD::Bool, forcing::G) where {n, G}
        exTerm = NonLinearExTerm(n, ISODD)
        imTerm = LinearTerm(n, ν, ISODD)
        new{n, typeof(forcing), 
            typeof(imTerm), typeof(exTerm)}(imTerm, exTerm, forcing)
    end
end

ForwardEquation(n::Int,
                ν::Real,
                ISODD::Bool,
                forcing::AbstractForcing=DummyForcing(n)) =
                                           ForwardEquation{n}(ν, ISODD, forcing)

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
    (A_mul_B!(dUdt, eq.imTerm, U);
     eq.exTerm(t, U, dUdt, true);
     eq.forcing(t, U, dUdt); dUdt)