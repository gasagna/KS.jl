module KS

import FFTW
import IMEXRKCB

export KSEq, imex

# ~~~ LINEAR TERM ~~~
struct LinearKSEqTerm
    A::Vector{Float64}
    L::Float64
    LinearKSEqTerm(N::Int, L::Real) = new(Float64[(k/L)^2 - (k/L)^4 for k = 0:N], L)
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
    (A_mul_B!(dukdt, ks.lks, uk); # linear term
     ks.nlks(t, uk, dukdt, true)) # nonlinear term (add value)


# ~~~ LINEARISED EQUATION ~~~
struct LinearisedKSEq
     U::Float64                   # mean flow velocity
     L::Float64                   # domain size
     p::Vector{Float64}           # temporary in physical space
    pk::Vector{Complex{Float64}}  # temporary in Fourier space
    wk::Vector{Complex{Float64}}  # temporary in Fourier space
    iplan                         # plans
    fplan                         #
end

# ~~~ EVALUATE LINEAR OPERATOR AROUND U ~~~
function (lks::LinearisedKSEq)(t::Real, u::Vector, w::Vector, dwdt::Vector)
    # setup
    wk, pk, p = lks.wk, lks.pk, lks.p     # aliases
    N, L, U = length(wk)-1, lks.L, lks.U  # parameters
    qk = 2π/L*(0:N)                       # wave numbers

    # compute term  -(u + U)*w
    p .= .- (u .+ U).*w                   # compute product in physical space
    FFTW.unsafe_execute(lks.fplan, p, pk) # transform to Fourier space
    pk .*= im.*qk./N                      # differentiate and normalise
   
    # compute term  -u₂ₓ - u₄ₓ
    FFTW.unsafe_execute(lks.fplan, w, wk) # transform to Fourier space
    wk .*= (qk.^2 .- qk.^4)./N            # differentiate wk and normalise

    # add, transform to physical space and return
    FFTW.unsafe_execute(lks.iplan, pk .+= wk, dwdt); return dwdt
end

end