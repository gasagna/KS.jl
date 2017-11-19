module KS

import FFTW: unsafe_execute!
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
    iplan
    fplan
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
    vk, u = nlks.vk, nlks.u                         # aliases
    N, L, U = length(uk)-1, nlks.L, nlks.U          # parameters
    uk[1] = 0                                       # make sure mean is zero
    unsafe_execute!(nlks.iplan, vk .= uk, u)        # copy and inverse transform
    u .= 0.5.*(U.+u).^2                             # sum U, square and divide by 2
    unsafe_execute!(nlks.fplan, u, vk)              # forward transform
    vk .*= (2π/L/N).*im.*(0:N)                      # differentiate 1/2(u+U)^2 and normalise
    add == true ? (dukdt .-= vk) : (dukdt .= .- vk) # note minus sign on rhs
    dukdt[1] = 0                                    # make sure mean does not change
    return dukdt                                    # return
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
(ks::KSEq)(t::Real, uk::Vector{Float64}, dukdt::Vector{Float64}) = 
    (A_mul_B!(dukdt, ks.lks, uk); # linear term
     ks.nlks(t, uk, dukdt, true)) # nonlinear term (add value)

end