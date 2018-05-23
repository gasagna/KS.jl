# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

import IMEXRKCB

export KSEq, imex, LinearisedKSEq, SteadyForcing

# ~~~ LINEAR TERM ~~~
struct LinearKSEqTerm{n, L, W}
    A::W
    function LinearKSEqTerm{n, L}() where {n, L}
        qk = WaveNumbers([qk^2 - qk^4 for qk = 2π/L*(0:n)], L)
        new{n, L, typeof(qk)}(qk)
    end
end
LinearKSEqTerm(n::Int, L::Real) = LinearKSEqTerm{n, L}()

# obey IMEXRKCB interface
@inline Base.A_mul_B!(dukdt::AbstractFTField{n, L}, lks::LinearKSEqTerm{n, L}, uk::AbstractFTField{n, L}) where {n, L} =
    (dukdt .= lks.A .* uk; dukdt)

@inline IMEXRKCB.ImcA!(lks::LinearKSEqTerm{n, L}, c::Real, uk::AbstractFTField{n, L}, dukdt::AbstractFTField{n, L}) where {n, L} =
    dukdt .= uk./(1 .- c.*lks.A)


# ~~~ NONLINEAR TERM ~~~
struct NonLinearKSEqTerm{n, L, FT<:AbstractFTField{n, L}, F<:AbstractField{n, L}}
     U::Float64 # mean flow velocity
    vk::FT      # temporary in Fourier space
     u::F       # solution in physical space
    ifft        # plans
    fft         #
    function NonLinearKSEqTerm{n, L}(U::Real, mode::Symbol) where {n, L}
        if mode == :forward
            vk = FTField(n, L); u = Field(n, L)
        elseif mode == :tangent
            vk = VarFTField(n, L); u = VarField(n, L)
        else
            throw(ArgumentError("mode :$mode not understood." * 
                                " Must be :forward or :tangent"))
        end
        fft, ifft = ForwardFFT(u), InverseFFT(vk)
        new{n, L, typeof(vk), typeof(u)}(U, vk, u, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, U::Real, L::Real, mode::Symbol=:forward) = 
    NonLinearKSEqTerm{n, L}(U, mode)

@inline function (nlks::NonLinearKSEqTerm{n, L, FT})(t::Real,
                                             uk::FT,
                                             dukdt::FT,
                                             add::Bool=false) where {n, L, FT<:AbstractFTField{n, L}}
    # setup
    vk, u, U = nlks.vk, nlks.u, nlks.U # aliases
    qk = WaveNumbers(n, L)             # wave numbers

    # compute nonlinear term using FFTs
    uk[0] = 0               # make sure mean is zero
    nlks.ifft(vk .= uk, u)  # copy and inverse transform
    u .= .- 0.5.*(U.+u).^2  # sum U, square and divide by 2
    nlks.fft(u, vk)         # forward transform
    vk .*= im.*qk           # differentiate

    # add terms and return
    add == true ? (dukdt .+= vk) : (dukdt .= vk) # note minus sign on rhs
    dukdt[0] = 0; return dukdt                   # make sure mean does not change
end

# ~~~ FORCING ~~~
abstract type AbstractForcing{n, L} end

struct SteadyForcing{n, L, FT<:AbstractFTField{n, L}} <: AbstractForcing{n, L}
	hk::FT
end
SteadyForcing(hk::AbstractFTField{n, L}) where {n, L} = 
	SteadyForcing{n, L, typeof(hk)}(hk)

# allow indexing this 
Base.getindex(sf::SteadyForcing, i::Int) = sf.hk[i]
Base.setindex!(sf::SteadyForcing, val, i::Int) = (sf.hk[i] = val)

# add to dukdt by default
@inline (sf::SteadyForcing{n, L, FT})(t::Real, uk::FT, dukdt::FT) where {n, L, FT<:AbstractFTField{n, L}} =
	(dukdt .+= sf.hk; return dukdt)

# ~~~ COMPLETE EQUATION ~~~
struct KSEq{n, L, LIN<:LinearKSEqTerm{n, L}, NLIN<:NonLinearKSEqTerm{n, L}, G<:Union{AbstractForcing{n, L}, Void}}
        lks::LIN
       nlks::NLIN
    forcing::G
    function KSEq{n, L}(U::Real, mode::Symbol, forcing::G) where {n, L, G}
        nlks = NonLinearKSEqTerm(n, U, L, mode)
        lks  = LinearKSEqTerm(n, L)
        new{n, L, typeof(lks), typeof(nlks), typeof(forcing)}(lks, nlks, forcing)
    end
end
KSEq(n::Int, 
	 U::Real, 
	 L::Real, 
	 mode::Symbol=:forward, 
	 forcing::Union{AbstractForcing, Void}=nothing) = KSEq{n, L}(U, mode, forcing)

# split into implicit and explicit terms
function imex(ks::KSEq{n, L, LIN, NLIN, G}) where {n, L, LIN, NLIN, G<:Union{AbstractForcing{n, L}, Void}}
	@inline function wrapper(t::Real, uk::AbstractFTField{n, L}, dukdt::AbstractFTField{n, L})
		ks.nlks(t, uk, dukdt, false)                     # eval nonlinear term
		G <: AbstractForcing && ks.forcing(t, uk, dukdt) # only eval if there is a forcing
		return dukdt
	end
	return ks.lks, wrapper
end

# evaluate right hand side of equation
@inline (ks::KSEq{n, L, LIN, NLIN, G})(t::Real, uk::FT, dukdt::FT) where {n, L, LIN, NLIN, G, FT<:AbstractFTField{n, L}} =
    (A_mul_B!(dukdt, ks.lks, uk);                      # linear term
     ks.nlks(t, uk, dukdt, true);                      # nonlinear term (add value)
     G <: AbstractForcing && ks.forcing(t, uk, dukdt); # add forcing
     return dukdt) 

# ~~~ LINEARISED EQUATION ~~~
struct LinearisedKSEq{n, L, FT<:FTField{n, L}, F<:Field{n, L}}
     U::Float64  # mean flow velocity
    nk::FT       # temporary in Fourier space
     u::F        # temporary in physical space
     w::F        # temporary in physical space
    ifft         # plans
    fft          #
    function LinearisedKSEq{n, L}(U::Real) where {n, L}
        u = Field(n, L); w = Field(n, L); nk = FTField(n, L)
        ifft = InverseFFT(nk); fft  = ForwardFFT(u)
        new{n, L, typeof(nk), typeof(u)}(U, nk, u, w, ifft, fft)
    end
end

LinearisedKSEq(n::Int, U::Real, L::Real) = LinearisedKSEq{n, L}(U)

# ~~~ EVALUATE LINEAR OPERATOR AROUND U ~~~
function (lks::LinearisedKSEq{n, L})(t::Real, uk::FTField{n, L}, wk::FTField{n, L}, dwkdt::FTField{n, L}) where {n, L}
    # setup
    w, u, nk, U = lks.w, lks.u, lks.nk, lks.U # aliases
    qk = WaveNumbers(n, L)                    # wave numbers
    uk[0] = 0; wk[0] = 0                      # make sure zero mean

    # ffts
    lks.ifft(uk, u)    # transform u to physical space
    lks.ifft(wk, w)    # transform w to physical space

    # compute term  -(u + U)*w, in place over u
    u .= .- (u .+ U).*w # compute product in physical space
    lks.fft(u, dwkdt)   # transform to Fourier space
    dwkdt .*= im.*qk    # differentiate
   
    # compute term  -w₂ₓ - w₄ₓ and return
    dwkdt .+= wk.*(qk.^2 .- qk.^4); return dwkdt # differentiate wk, add other term
end
