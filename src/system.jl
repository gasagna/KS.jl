# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import Flows

export KSEq,
       imex

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
    function NonLinearKSEqTerm{n, ISODD}(L::Real, c::Real, mode::Symbol) where {n, ISODD}
        if mode == :forward
            V = FTField(n, L, ISODD); u = Field(n, L)
        elseif mode == :tangent
            V = VarFTField(n, L, ISODD); u = VarField(n, L)
        else
            throw(ArgumentError("mode :$mode not understood." *
                                " Must be :forward or :tangent"))
        end
        fft, ifft = ForwardFFT(u), InverseFFT(V)
        new{n, ISODD, typeof(V), typeof(u)}(c, V, u, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, L::Real, c::Real, ISODD::Bool, mode::Symbol=:forward) =
    NonLinearKSEqTerm{n, ISODD}(L, c, mode)

@inline function (nlks::NonLinearKSEqTerm{n, ISODD, FT})(t::Real,
                                                         U::FT,
                                                         dUdt::FT,
                                                         add::Bool=false) where {n, ISODD, FT<:AbstractFTField{n}}
    _set_symmetry!(U)         # enforce symmetries
    nlks.ifft(U, nlks.u)                  # copy and inverse transform
    nlks.u .= .- 0.5.*(nlks.c.+nlks.u).^2  # sum c, square and divide by 2
    nlks.fft(nlks.u, nlks.V)              # forward transform
    ddx!(nlks.V)                          # differentiate
    _set_symmetry!(nlks.V)    # enforce symmetries

    add == true ? (dUdt .+= nlks.V) : (dUdt .= nlks.V)
    dUdt
end


# ////// COMPLETE EQUATION //////
struct KSEq{n, ISODD, LIN<:LinearKSEqTerm{n}, NLIN<:NonLinearKSEqTerm{n, ISODD}, G<:Union{AbstractForcing{n}, Void}}
        lks::LIN
       nlks::NLIN
    forcing::G
    function KSEq{n, ISODD}(L::Real, c::Real, mode::Symbol, forcing::G) where {n, ISODD, G}
        nlks = NonLinearKSEqTerm(n, L, c, ISODD, mode)
        lks  = LinearKSEqTerm(n, L, ISODD)
        new{n, ISODD, typeof(lks), typeof(nlks), typeof(forcing)}(lks, nlks, forcing)
    end
end

KSEq(n::Int,
     L::Real,
     c::Real,
     ISODD::Bool,
     mode::Symbol=:forward,
     forcing::Union{AbstractForcing, Void}=nothing) = KSEq{n, ISODD}(L, c, mode, forcing)

# split into implicit and explicit terms
function imex(ks::KSEq{n, ISODD, LIN, NLIN, G}) where {n, ISODD, LIN, NLIN, G<:Union{AbstractForcing{n}, Void}}
    @inline function wrapper(t::Real, U::AbstractFTField{n}, dUdt::AbstractFTField{n})
        ks.nlks(t, U, dUdt, false)                     # eval nonlinear term
        G <: AbstractForcing && ks.forcing(t, U, dUdt) # only eval if there is a forcing
        return dUdt
    end
    return ks.lks, wrapper
end

# evaluate right hand side of equation
(ks::KSEq{n, ISODD, LIN, NLIN, G})(t::Real, U::FT, dUdt::FT) where {n, ISODD, LIN, NLIN, G, FT<:AbstractFTField{n}} =
    (A_mul_B!(dUdt, ks.lks, U);                      # linear term
     ks.nlks(t, U, dUdt, true);                      # nonlinear term (add value)
     G <: AbstractForcing && ks.forcing(t, U, dUdt); # add forcing
     return dUdt)


# # ////// LINEARISED EQUATION //////
# struct LinearisedKSEq{n, FT<:FTField{n}, F<:Field{n}}
#      c::Float64  # mean flow velocity
#     nk::FT       # temporary in Fourier space
#      u::F        # temporary in physical space
#      w::F        # temporary in physical space
#     ifft         # plans
#     fft          #
#     function LinearisedKSEq{n}(L::Real, c::Real) where {n}
#         u = Field(n, L); w = Field(n, L); nk = FTField(n, L)
#         ifft = InverseFFT(nk); fft  = ForwardFFT(u)
#         new{n, L, typeof(nk), typeof(u)}(c, nk, u, w, ifft, fft)
#     end
# end

# LinearisedKSEq(n::Int, L::Real, c::Real) = LinearisedKSEq{n}(L, c)

# # evaluate linear operator around u
# function (lks::LinearisedKSEq{n})(t::Real, U::FTField{n}, wk::FTField{n}, dwkdt::FTField{n}) where {n}
#     # ffts
#     lks.ifft(U, lks.u)    # transform u to physical space
#     lks.ifft(wk, lks.w)    # transform w to physical space

#     # compute term  -(u + c)*w, in place over u
#     lks.u .= .- (lks.u .+ c).*lks.w  # compute product in physical space
#     lks.fft(lks.u, dwkdt)            # transform to Fourier space

#     for k = 1:n
#         qk = 2π*k/L
#         dwkdt[k] *= im*qk*dwdt[k]       # differentiate
#         dwkdt[k] += wk[k]*(qk^2 - qk^4) # add terms w₂ₓ - w₄ₓ
#     end

#     dwkdt
# end