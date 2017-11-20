# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

module KS

import FFTW
import IMEXRKCB

export KSEq, imex, LinearisedKSEq

# ~~~ LINEAR TERM ~~~
struct LinearKSEqTerm
    A::Vector{Float64}
    L::Float64
    LinearKSEqTerm(N::Int, L::Real) = 
        new(Float64[(2π*k/L)^2 - (2π*k/L)^4 for k = 0:N], L)
end

# obey IMEXRKCB interface
Base.A_mul_B!(dukdt::Vector, lks::LinearKSEqTerm, uk::Vector) =
    (dukdt .= lks.A .* uk; dukdt)

IMEXRKCB.ImcA!(lks::LinearKSEqTerm, c::Real, uk::Vector, dukdt::Vector) =
    dukdt .= uk./(1 .- c.*lks.A)


# ~~~ NONLINEAR TERM ~~~
struct NonLinearKSEqTerm
     U::Float64                   # mean flow velocity
     L::Float64                   # domain size
     u::Vector{Float64}           # solution in physical space
    vk::Vector{Complex{Float64}}  # temporary in Fourier space
    iplan                         # plans
    fplan                         #
    # N is the maximum wavenumber
    function NonLinearKSEqTerm(N::Int, U::Real, L::Real)
        u  = zeros(Float64,          2N+1)
        vk = zeros(Complex{Float64},  N+1)
        fplan = plan_rfft(  u,       flags=FFTW.PATIENT)
        iplan = plan_brfft(vk, 2N+1, flags=FFTW.PATIENT)
        new(U, L, u, vk, iplan, fplan)
    end
end

function (nlks::NonLinearKSEqTerm)(t::Real, uk::Vector, dukdt::Vector, add::Bool=false)
    # setup
    vk, u = nlks.vk, nlks.u                         # aliases
    N, L, U = length(uk)-1, nlks.L, nlks.U          # parameters
    qk = 2π/L*(0:N)                                 # wave numbers

    # compute nonlinear term using FFTs
    uk[1] = 0                                       # make sure mean is zero
    FFTW.unsafe_execute!(nlks.iplan, vk .= uk, u)   # copy and inverse transform
    u .= 0.5.*(U.+u).^2                             # sum U, square and divide by 2
    FFTW.unsafe_execute!(nlks.fplan, u, vk)         # forward transform
    vk .*= im.*qk./N                                # differentiate and normalise

    # add terms and return
    add == true ? (dukdt .-= vk) : (dukdt .= .- vk) # note minus sign on rhs
    dukdt[1] = 0; return dukdt                      # make sure mean does not change
end


# ~~~ COMPLETE EQUATION ~~~
struct KSEq
     lks::LinearKSEqTerm
    nlks::NonLinearKSEqTerm
    KSEq(N::Int, U::Real, L::Real) =
        new(LinearKSEqTerm(N, L), NonLinearKSEqTerm(N, U, L))
end

# split linear and nonlinear term
imex(KSEq) = KSEq.lks, KSEq.nlks

# evaluate right hand side of equation
(ks::KSEq)(t::Real, uk::Vector, dukdt::Vector) =
    (A_mul_B!(dukdt, ks.lks, uk);        # linear term
     ks.nlks(t, uk, dukdt, true); dukdt) # nonlinear term (add value)


# ~~~ LINEARISED EQUATION ~~~
struct LinearisedKSEq
     U::Float64                  # mean flow velocity
     L::Float64                  # domain size
    nk::Vector{Complex{Float64}} # temporary in Fourier space
     u::Vector{Float64}          # temporary in physical space
     w::Vector{Float64}          # temporary in physical space
    iplan                        # plans
    fplan                        #
    function LinearisedKSEq(N::Int, U::Real, L::Real)
        u  = zeros(Float64,          2N+1)
        w  = zeros(Float64,          2N+1)
        nk = zeros(Complex{Float64},  N+1)
        fplan = plan_rfft(  u,       flags=FFTW.PATIENT)
        iplan = plan_brfft(nk, 2N+1, flags=FFTW.PATIENT)
        new(U, L, nk, u, w, iplan, fplan)
    end
end

# ~~~ EVALUATE LINEAR OPERATOR AROUND U ~~~
function (lks::LinearisedKSEq)(t::Real, uk::Vector, wk::Vector, dwkdt::Vector)
    # setup
    w, u, nk = lks.w, lks.u, lks.nk      # aliases
    N, L, U = length(wk)-1, lks.L, lks.U # parameters
    qk = 2π/L*(0:N)                      # wave numbers
    uk[1] = 0; wk[1] = 0                 # make sure zero mean

    # ffts
    FFTW.unsafe_execute!(lks.iplan, uk, u)    # transform u to physical space
    FFTW.unsafe_execute!(lks.iplan, wk, w)    # transform w to physical space

    # compute term  -(u + U)*w, in place over u
    u .= .- (u .+ U).*w                       # compute product in physical space
    FFTW.unsafe_execute!(lks.fplan, u, dwkdt) # transform to Fourier space
    dwkdt .*= im.*qk./N                       # differentiate and normalise
   
    # compute term  -w₂ₓ - w₄ₓ and return
    dwkdt .+= wk.*(qk.^2 .- qk.^4); return dwkdt # differentiate wk, add other term
end

end