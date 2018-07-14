# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Flows

export ForwardEquation,
       splitexim

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
    function LinearTerm{n}(ν::Real, ISODD::Bool, mode::M) where {n, M<:AbstractMode}
        ν > 0 || throw(ArgumentError("viscosity must be positive"))
        A = FTField(n, ISODD)
        # for the adjoint equation reverse the sign of the linear term
        sgn = M <: AdjointMode ? -1 : 1
        for k in wavenumbers(n)
            A[k] = sgn*(k^2 - ν*k^4)
        end
        new{n, typeof(A)}(A)
    end
end
LinearTerm(n::Int, ν::Real, ISODD::Bool, mode::AbstractMode) = 
    LinearTerm{n}(ν, ISODD, mode)

# obey Flows interface
@inline Base.A_mul_B!(dUdt::AbstractFTField{n},
                      lks::LinearTerm{n},
                      U::AbstractFTField{n}) where {n} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
         dUdt[k] = lks.A[k] * U[k]
     end; dUdt)

@inline Flows.ImcA!(lks::LinearTerm{n},
                    c::Real,
                    U::AbstractFTField{n},
                    dUdt::AbstractFTField{n}) where {n} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
          dUdt[k] = U[k]/(1 - c*lks.A[k])
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
    (nlks::NonLinearExTerm{n, FT})(t::Real,
                                   U::FT,
                                   dUdt::FT,
                                   add::Bool=false) where {n, FT}
    _set_symmetry!(U)
    nlks.ifft(U, nlks.u)        # copy and inverse transform
    nlks.u .= nlks.u.^2         # square
    nlks.fft(nlks.u, nlks.V)    # forward transform
    ddx!(nlks.V)                # differentiate

    # store and enforce symmetries
    add == true ? (dUdt .+= 0.5.*nlks.V) : (dUdt .= 0.5.*nlks.V)

    return dUdt
end


# ////// COMPLETE EQUATION //////
struct ForwardEquation{n,
                       G<:AbstractForcing{n},
                       LIN<:LinearTerm{n},
                       NLIN<:NonLinearExTerm{n}}
                   lks::LIN
                  nlks::NLIN
               forcing::G
    function ForwardEquation{n}(ν::Real, ISODD::Bool, forcing::G) where {n, G}
        nlks = NonLinearExTerm(n, ISODD)
        lks  = LinearTerm(n, ν, ISODD, ForwardMode())
        new{n, typeof(forcing), typeof(lks), typeof(nlks)}(lks, nlks, forcing)
    end
end

ForwardEquation(n::Int,
                ν::Real,
                ISODD::Bool,
                forcing::AbstractForcing=DummyForcing(n)) =
                                           ForwardEquation{n}(ν, ISODD, forcing)

# split into implicit and explicit terms
function splitexim(ks::ForwardEquation{n}) where {n}
    wrapper(t::Real, U::AbstractFTField{n}, dUdt::AbstractFTField{n}) =
        (ks.nlks(t, U, dUdt, false); ks.forcing(t, U, dUdt); dUdt)
    return wrapper, ks.lks
end

# evaluate right hand side of equation
(ks::ForwardEquation{n})(t::Real,
                         U::FT,
                         dUdt::FT) where {n, FT<:AbstractFTField{n}} =
    (A_mul_B!(dUdt, ks.lks, U);
     ks.nlks(t, U, dUdt, true);
     ks.forcing(t, U, dUdt); dUdt)