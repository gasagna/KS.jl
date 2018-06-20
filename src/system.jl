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
    function LinearKSEqTerm{n, ISODD}(L::Real) where {n, ISODD}
        qk = FTField(n, L, ISODD)
        for k in wavenumbers(n)
            qk[k] = (2π/L*k)^2 - (2π/L*k)^4
        end
        new{n, ISODD, typeof(qk)}(qk)
    end
end
LinearKSEqTerm(n::Int, L::Real, ISODD::Bool) = LinearKSEqTerm{n, ISODD}(L)

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
     c::Float64 # mean flow velocity
     V::FT      # temporary in Fourier space
     u::F       # solution in physical space
    ifft        # plans
    fft         #
    function NonLinearKSEqTerm{n, ISODD}(L::Real, c::Real) where {n, ISODD}
        V = FTField(n, L, ISODD); u = Field(n, L)
        fft, ifft = ForwardFFT(u), InverseFFT(V)
        new{n, ISODD, typeof(V), typeof(u)}(c, V, u, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, L::Real, c::Real, ISODD::Bool) =
    NonLinearKSEqTerm{n, ISODD}(L, c)

@inline function
    (nlks::NonLinearKSEqTerm{n, ISODD, FT})(t::Real,
                                            U::FT,
                                            dUdt::FT,
                                            add::Bool=false) where {n,
                                                  ISODD, FT<:AbstractFTField{n}}
    nlks.ifft(U, nlks.u)                  # copy and inverse transform
    nlks.u .= .- 0.5.*(nlks.c.+nlks.u).^2 # sum c, square and divide by 2
    nlks.fft(nlks.u, nlks.V)              # forward transform
    ddx!(nlks.V)                          # differentiate

    # store and enforce symmetries
    add == true ? (dUdt .+= nlks.V) : (dUdt .= nlks.V)
    _set_symmetry!(dUdt)

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
    function KSEq{n, ISODD}(L::Real, c::Real, forcing::G) where {n, ISODD, G}
        nlks = NonLinearKSEqTerm(n, L, c, ISODD)
        lks  = LinearKSEqTerm(n, L, ISODD)
        new{n, ISODD, typeof(forcing), typeof(lks), typeof(nlks)}(lks,
                                                                  nlks,
                                                                  forcing)
    end
end

KSEq(n::Int,
     L::Real,
     c::Real,
     ISODD::Bool,
     forcing::AbstractForcing=DummyForcing(n)) =
    KSEq{n, ISODD}(L, c, forcing)

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
       c::Float64  # mean flow velocity
    TMP1::FT       # temporary in Fourier space
    TMP2::FT       # temporary in Fourier space
    tmp1::F        # temporary in physical space
    tmp2::F        # temporary in physical space
    tmp3::F        # temporary in physical space
    ifft           # plans
    fft            #
    function LinearisedKSEqExTerm{n, ISODD}(L::Real, c::Real) where {n, ISODD}
        TMP1 = FTField(n, L, ISODD); TMP2 = FTField(n, L, ISODD)
        tmp1 = Field(n, L); tmp2 = Field(n, L); tmp3 = Field(n, L)
        ifft = InverseFFT(TMP1); fft  = ForwardFFT(tmp1)
        new{n, ISODD, typeof(TMP1), typeof(tmp1)}(c,
                TMP1, TMP2, tmp1, tmp2, tmp3, ifft, fft)
    end
end

# constructor
LinearisedKSEqExTerm(n::Int,
                     L::Real,
                     c::Real,
                     ISODD::Bool) = LinearisedKSEqExTerm{n, ISODD}(L, c)


# evaluate linear operator around U
function (lks::LinearisedKSEqExTerm{n})(t::Real,
                                        U::FTField{n},
                                        V::FTField{n},
                                        dVdt::FTField{n},
                                        add::Bool=false) where {n}
    # /// calculate (c + u) * vₓ ///
    lks.ifft(U, lks.tmp1)                      # transform U to physical space
    ddx!(lks.TMP2, V)                          # differentiate V
    lks.ifft(lks.TMP2, lks.tmp2)               # transform to physical space
    lks.tmp3 .= (lks.c .+ lks.tmp1).* lks.tmp2 # multiply in physical space

    # /// calculate v * uₓ ///
    lks.ifft(V, lks.tmp1)                       # transform V to physical space
    ddx!(lks.TMP2, U)                           # differentiate U
    lks.ifft(lks.TMP2, lks.tmp2)                # transform to physical space
    lks.tmp3 .+= lks.tmp1 .* lks.tmp2           # multiply in physical space

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
set_χ!(l::LinearisedKSEq, χ::Real) = (l.χ = χ; nothing)

# outer constructor: main entry point
function LinearisedKSEq(n::Int,
                        L::Real,
                        c::Real,
                        ISODD::Bool,
                        mon::Flows.AbstractMonitor{T, X},
                        χ::Real=0,
                        forcing::AbstractForcing=DummyForcing(n)) where {T, X}
    X <: FTField{n, ISODD} || error("invalid monitor object")
    imTerm = LinearKSEqTerm(n, L, ISODD)
    exTerm = LinearisedKSEqExTerm(n, L, c, ISODD)
    TMP = FTField(n, L, ISODD)
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