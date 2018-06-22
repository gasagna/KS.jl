# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import Flows

export KSEq,
       splitexim,
       LinearisedKSEq,
       set_χ!


# ////// LINEAR TERM //////
struct LinearKSEqTerm{n, ISODD, FT<:AbstractFTField{n}}
    A::FT
    function LinearKSEqTerm{n, ISODD}(ν::Real) where {n, ISODD}
        ν > 0 || throw(ArgumentError("viscosity must be positive"))
        A = FTField(n, ISODD)
        for k in wavenumbers(n)
            A[k] = k^2 - ν*k^4
        end
        new{n, ISODD, typeof(A)}(A)
    end
end
LinearKSEqTerm(n::Int, ν::Real, ISODD::Bool) = LinearKSEqTerm{n, ISODD}(ν)

# obey Flows interface
@inline Base.A_mul_B!(dUdt::AbstractFTField{n},
                      lks::LinearKSEqTerm{n, ISODD},
                      U::AbstractFTField{n}) where {n, ISODD} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
         dUdt[k] = lks.A[k] * U[k]
     end; dUdt)

@inline Flows.ImcA!(lks::LinearKSEqTerm{n, ISODD},
                    c::Real,
                    U::AbstractFTField{n},
                    dUdt::AbstractFTField{n}) where {n, ISODD} =
    (_set_symmetry!(U);
     @inbounds for k in wavenumbers(n)
          dUdt[k] = U[k]/(1 - c*lks.A[k])
     end; dUdt)


# ////// NONLINEAR TERM //////
struct NonLinearKSEqTerm{n, ISODD, FT<:AbstractFTField{n}, F<:AbstractField{n}}
     V::FT      # temporary in Fourier space
     u::F       # solution in physical space
    ifft        # plans
    fft         #
    function NonLinearKSEqTerm{n, ISODD}() where {n, ISODD}
        V = FTField(n, ISODD); u = Field(n)
        fft, ifft = ForwardFFT(u), InverseFFT(V)
        new{n, ISODD, typeof(V), typeof(u)}(V, u, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, ISODD::Bool) = NonLinearKSEqTerm{n, ISODD}()

@inline function
    (nlks::NonLinearKSEqTerm{n, ISODD, FT})(t::Real,
                                            U::FT,
                                            dUdt::FT,
                                            add::Bool=false) where {n,
                                                  ISODD, FT<:AbstractFTField{n}}
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
struct KSEq{n,
            ISODD,
            G<:AbstractForcing{n},
            LIN<:LinearKSEqTerm{n},
            NLIN<:NonLinearKSEqTerm{n, ISODD}}
        lks::LIN
       nlks::NLIN
    forcing::G
    function KSEq{n, ISODD}(ν::Real, forcing::G) where {n, ISODD, G}
        nlks = NonLinearKSEqTerm(n, ISODD)
        lks  = LinearKSEqTerm(n, ν, ISODD)
        new{n, ISODD, typeof(forcing), typeof(lks), typeof(nlks)}(lks,
                                                                  nlks,
                                                                  forcing)
    end
end

KSEq(n::Int,
     ν::Real,
     ISODD::Bool,
     forcing::AbstractForcing=DummyForcing(n)) =
    KSEq{n, ISODD}(ν, forcing)

# split into implicit and explicit terms
function splitexim(ks::KSEq{n}) where {n}
    wrapper(t::Real, U::AbstractFTField{n}, dUdt::AbstractFTField{n}) =
        (ks.nlks(t, U, dUdt, false); ks.forcing(t, U, dUdt); dUdt)
    return wrapper, ks.lks
end

# evaluate right hand side of equation
(ks::KSEq{n, ISODD})(t::Real,
                     U::FT,
                     dUdt::FT) where {n, ISODD, FT<:AbstractFTField{n}} =
    (A_mul_B!(dUdt, ks.lks, U);
     ks.nlks(t, U, dUdt, true);
     ks.forcing(t, U, dUdt); dUdt)


# ////// LINEAR EQUATION //////
struct LinearisedKSEqExTerm{n, ISODD, FT<:FTField{n}, F<:Field{n}}
    TMP1::FT # temporary in Fourier space
    TMP2::FT # temporary in Fourier space
    tmp1::F  # temporary in physical space
    tmp2::F  # temporary in physical space
    tmp3::F  # temporary in physical space
    ifft     # plans
    fft      #
    function LinearisedKSEqExTerm{n, ISODD}() where {n, ISODD}
        TMP1 = FTField(n, ISODD); TMP2 = FTField(n, ISODD)
        tmp1 = Field(n); tmp2 = Field(n); tmp3 = Field(n)
        ifft = InverseFFT(TMP1); fft  = ForwardFFT(tmp1)
        new{n, ISODD, typeof(TMP1), typeof(tmp1)}(TMP1, 
                TMP2, tmp1, tmp2, tmp3, ifft, fft)
    end
end

# constructor
LinearisedKSEqExTerm(n::Int, ISODD::Bool) = 
    LinearisedKSEqExTerm{n, ISODD}()


# evaluate linear operator around U
function (lks::LinearisedKSEqExTerm{n})(t::Real,
                                        U::FTField{n},
                                        V::FTField{n},
                                        dVdt::FTField{n},
                                        add::Bool=false) where {n}
    # /// calculate u * vₓ ///
    lks.ifft(U, lks.tmp1)            # transform U to physical space
    ddx!(lks.TMP2, V)                # differentiate V
    lks.ifft(lks.TMP2, lks.tmp2)     # transform to physical space
    lks.tmp3 .= lks.tmp1 .* lks.tmp2 # multiply in physical space

    # /// calculate v * uₓ ///
    lks.ifft(V, lks.tmp1)             # transform V to physical space
    ddx!(lks.TMP2, U)                 # differentiate U
    lks.ifft(lks.TMP2, lks.tmp2)      # transform to physical space
    lks.tmp3 .+= lks.tmp1 .* lks.tmp2 # multiply in physical space

    # /// transform back bilinear term to wave number space ///
    lks.fft(lks.tmp3, lks.TMP2)

    # /// change sign, store and set symmetry ///
    add == true ? (dVdt .-= lks.TMP2) : (dVdt .= .-lks.TMP2)
    _set_symmetry!(dVdt)

    return dVdt
end

# ~~~ SOLVER OBJECT FOR THE LINEAR EQUATIONS ~~~
mutable struct LinearisedKSEq{n,
                      IT<:LinearKSEqTerm{n},
                      ET<:LinearisedKSEqExTerm{n},
                      G<:AbstractForcing{n},
                      M<:Flows.AbstractMonitor,
                      FT<:AbstractFTField}
     imTerm::IT
     exTerm::ET
    forcing::G
        mon::M
        TMP::FT
          χ::Float64
end

# set χ constant
set_χ!(eq::LinearisedKSEq, χ::Real) = (eq.χ = χ; nothing)

# outer constructor: main entry point
function LinearisedKSEq(n::Int,
                        ν::Real,
                        ISODD::Bool,
                        mon::Flows.AbstractMonitor{T, X},
                        χ::Real=0,
                        forcing::AbstractForcing=DummyForcing(n)) where {T, X}
    X <: FTField{n, ISODD} || error("invalid monitor object")
    imTerm = LinearKSEqTerm(n, ν, ISODD)
    exTerm = LinearisedKSEqExTerm(n, ISODD)
    TMP = FTField(n, ISODD)
    LinearisedKSEq{n,
                   typeof(imTerm),
                   typeof(exTerm),
                   typeof(forcing),
                   typeof(mon),
                   typeof(TMP)}(imTerm, exTerm, forcing, mon, TMP, χ)
end

# obtain two components
function splitexim(eq::LinearisedKSEq{n}) where {n}
    function wrapper(t::Real, V::AbstractFTField{n}, dVdt::AbstractFTField{n})
        # interpolate U and evaluate nonlinear interaction term and forcing
        eq.mon(eq.TMP, t, Val{0}())
        eq.exTerm(t, eq.TMP, V, dVdt, false)
        eq.forcing(t, eq.TMP, V, dVdt)

        # interpolate dUdt if needed
        if eq.χ != 0
            eq.mon(eq.TMP, t, Val{1}())
            dVdt .+= eq.χ .* eq.TMP
        end

        return dVdt
    end
    return wrapper, eq.imTerm
end