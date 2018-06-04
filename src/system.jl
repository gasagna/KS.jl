# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import IMEXRKCB

export KSEq, 
       imex, 
       LinearisedKSEq, 
       SteadyForcing

# ////// LINEAR TERM //////
struct LinearKSEqTerm{n, FT<:AbstractFTField{n}}
    A::FT
    function LinearKSEqTerm{n}(L::Real) where {n}
        qk = FTField(n, L)
        for k = 1:n
            qk[k] = (2π/L*k)^2 - (2π/L*k)^4
        end
        new{n, typeof(qk)}(qk)
    end
end
LinearKSEqTerm(n::Int, L::Real) = LinearKSEqTerm{n}(L)

# obey IMEXRKCB interface
@inline Base.A_mul_B!(dukdt::AbstractFTField{n}, lks::LinearKSEqTerm{n}, uk::AbstractFTField{n}) where {n} =
    (dukdt .= lks.A .* uk; dukdt)

@inline IMEXRKCB.ImcA!(lks::LinearKSEqTerm{n}, c::Real, uk::AbstractFTField{n}, dukdt::AbstractFTField{n}) where {n} =
    dukdt .= uk./(1 .- c.*lks.A)


# ////// NONLINEAR TERM //////
struct NonLinearKSEqTerm{n, ISODD, FT<:AbstractFTField{n}, F<:AbstractField{n}}
     U::Float64 # mean flow velocity
    vk::FT      # temporary in Fourier space
     u::F       # solution in physical space
    ifft        # plans
    fft         #
    function NonLinearKSEqTerm{n, ISODD}(L::Real, U::Real, mode::Symbol) where {n, ISODD}
        if mode == :forward
            vk = FTField(n, L); u = Field(n, L)
        elseif mode == :tangent
            vk = VarFTField(n, L); u = VarField(n, L)
        else
            throw(ArgumentError("mode :$mode not understood." * 
                                " Must be :forward or :tangent"))
        end
        fft, ifft = ForwardFFT(u), InverseFFT(vk)
        new{n, ISODD, typeof(vk), typeof(u)}(U, vk, u, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, L::Real, U::Real, ISODD::Bool, mode::Symbol=:forward) = 
    NonLinearKSEqTerm{n, ISODD}(L, U, mode)

@inline function (nlks::NonLinearKSEqTerm{n, ISODD, FT})(t::Real,
                                                         uk::FT,
                                                         dukdt::FT,
                                                         add::Bool=false) where {n, ISODD, FT<:AbstractFTField{n}}
    _set_symmetry!(Val{ISODD}, uk)         # enforce symmetries
    nlks.ifft(uk, nlks.u)                  # copy and inverse transform
    nlks.u .= .- 0.5.*(nlks.U.+nlks.u).^2  # sum U, square and divide by 2
    nlks.fft(nlks.u, nlks.vk)              # forward transform
    ddx!(nlks.vk)                          # differentiate
    _set_symmetry!(Val{ISODD}, nlks.vk)    # enforce symmetries

    add == true ? (dukdt .+= nlks.vk) : (dukdt .= nlks.vk)
    dukdt
end

# enforce symmetries, if needed
_set_symmetry!(::Val{true},  uk::FTField{n}) where {n} = (uk .= im.*imag.(uk); uk)
_set_symmetry!(::Val{false}, uk::FTField{n}) where {n} = uk


# ////// FORCING //////
abstract type AbstractForcing{n} end

struct SteadyForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n}
    hk::FT
end
SteadyForcing(hk::AbstractFTField{n}) where {n} = SteadyForcing{n, typeof(hk)}(hk)

# allow indexing this 
Base.getindex(sf::SteadyForcing, i::Int) = sf.hk[i]
Base.setindex!(sf::SteadyForcing, val, i::Int) = (sf.hk[i] = val)

# add to dukdt by default
@inline (sf::SteadyForcing{n, FT})(t::Real, uk::FT, dukdt::FT) where {n, FT<:AbstractFTField{n}} =
    (dukdt .+= sf.hk; return dukdt)


# ////// COMPLETE EQUATION //////
struct KSEq{n, ISODD, LIN<:LinearKSEqTerm{n}, NLIN<:NonLinearKSEqTerm{n, ISODD}, G<:Union{AbstractForcing{n}, Void}}
        lks::LIN
       nlks::NLIN
    forcing::G
    function KSEq{n, ISODD}(L::Real, U::Real, mode::Symbol, forcing::G) where {n, ISODD, G}
        nlks = NonLinearKSEqTerm(n, L, U, ISODD, mode)
        lks  = LinearKSEqTerm(n, L)
        new{n, ISODD, typeof(lks), typeof(nlks), typeof(forcing)}(lks, nlks, forcing)
    end
end

KSEq(n::Int, 
     L::Real, 
     U::Real, 
     ISODD::Bool,
     mode::Symbol=:forward, 
     forcing::Union{AbstractForcing, Void}=nothing) = KSEq{n, ISODD}(L, U, mode, forcing)

# split into implicit and explicit terms
function imex(ks::KSEq{n, ISODD, LIN, NLIN, G}) where {n, ISODD, LIN, NLIN, G<:Union{AbstractForcing{n}, Void}}
    @inline function wrapper(t::Real, uk::AbstractFTField{n}, dukdt::AbstractFTField{n})
        ks.nlks(t, uk, dukdt, false)                     # eval nonlinear term
        G <: AbstractForcing && ks.forcing(t, uk, dukdt) # only eval if there is a forcing
        return dukdt
    end
    return ks.lks, wrapper
end

# evaluate right hand side of equation
@inline (ks::KSEq{n, ISODD, LIN, NLIN, G})(t::Real, uk::FT, dukdt::FT) where {n, ISODD, LIN, NLIN, G, FT<:AbstractFTField{n}} =
    (A_mul_B!(dukdt, ks.lks, uk);                      # linear term
     ks.nlks(t, uk, dukdt, true);                      # nonlinear term (add value)
     G <: AbstractForcing && ks.forcing(t, uk, dukdt); # add forcing
     return dukdt) 


# # ////// LINEARISED EQUATION //////
# struct LinearisedKSEq{n, FT<:FTField{n}, F<:Field{n}}
#      U::Float64  # mean flow velocity
#     nk::FT       # temporary in Fourier space
#      u::F        # temporary in physical space
#      w::F        # temporary in physical space
#     ifft         # plans
#     fft          #
#     function LinearisedKSEq{n}(L::Real, U::Real) where {n}
#         u = Field(n, L); w = Field(n, L); nk = FTField(n, L)
#         ifft = InverseFFT(nk); fft  = ForwardFFT(u)
#         new{n, L, typeof(nk), typeof(u)}(U, nk, u, w, ifft, fft)
#     end
# end

# LinearisedKSEq(n::Int, L::Real, U::Real) = LinearisedKSEq{n}(L, U)

# # evaluate linear operator around u
# function (lks::LinearisedKSEq{n})(t::Real, uk::FTField{n}, wk::FTField{n}, dwkdt::FTField{n}) where {n}
#     # ffts
#     lks.ifft(uk, lks.u)    # transform u to physical space
#     lks.ifft(wk, lks.w)    # transform w to physical space

#     # compute term  -(u + U)*w, in place over u
#     lks.u .= .- (lks.u .+ U).*lks.w  # compute product in physical space
#     lks.fft(lks.u, dwkdt)            # transform to Fourier space
    
#     for k = 1:n
#         qk = 2π*k/L
#         dwkdt[k] *= im*qk*dwdt[k]       # differentiate 
#         dwkdt[k] += wk[k]*(qk^2 - qk^4) # add terms w₂ₓ - w₄ₓ
#     end   
   
#     dwkdt
# end