# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

import IMEXRKCB

export KSEq, imex, LinearisedKSEq

# ~~~ LINEAR TERM ~~~
struct LinearKSEqTerm{n}
    A::Vector{Float64}
    L::Float64
    LinearKSEqTerm{n}(L::Real) where {n} =
        new{n}(Float64[(2π*k/L)^2 - (2π*k/L)^4 for k = 0:n], L)
end
LinearKSEqTerm(n::Int, L::Real) = LinearKSEqTerm{n}(L)

# obey IMEXRKCB interface
Base.A_mul_B!(dukdt::FTField{n}, lks::LinearKSEqTerm{n}, uk::FTField{n}) where {n} =
    (dukdt .= lks.A .* uk; dukdt)

IMEXRKCB.ImcA!(lks::LinearKSEqTerm{n}, c::Real, uk::FTField{n}, dukdt::FTField{n}) where {n} =
    dukdt .= uk./(1 .- c.*lks.A)


# ~~~ NONLINEAR TERM ~~~
struct NonLinearKSEqTerm{n, FT<:FTField{n}, F<:Field{n}}
     U::Float64 # mean flow velocity
     L::Float64 # domain size
     u::F       # solution in physical space
    vk::FT      # temporary in Fourier space
    ifft        # plans
    fft         #
    function NonLinearKSEqTerm{n}(U::Real, L::Real) where {n}
        vk = FTField(n); ifft = InverseFFT(vk)
        u  = Field(n);   fft  = ForwardFFT(u)
        new{n, typeof(vk), typeof(u)}(U, L, u, vk, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, U::Real, L::Real) = NonLinearKSEqTerm{n}(U, L)

function (nlks::NonLinearKSEqTerm{n})(t::Real, uk::FTField{n}, dukdt::FTField{n}, add::Bool=false) where {n}
    # setup
    vk, u = nlks.vk, nlks.u  # aliases
    L, U = nlks.L, nlks.U    # parameters
    qk = 2π/L*(0:n)          # wave numbers

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


# ~~~ COMPLETE EQUATION ~~~
struct KSEq{n, NL<:NonLinearKSEqTerm{n}}
     lks::LinearKSEqTerm{n}
    nlks::NL
    function KSEq{n}(U::Real, L::Real) where {n} 
        exTerm = NonLinearKSEqTerm(n, U, L)
        imTerm = LinearKSEqTerm(n, L)
        new{n, typeof(exTerm)}(imTerm, exTerm)
    end
end
KSEq(n::Int, U::Real, L::Real) = KSEq{n}( U, L)

# split linear and nonlinear term
imex(KSEq) = KSEq.lks, KSEq.nlks

# evaluate right hand side of equation
(ks::KSEq{n})(t::Real, uk::FTField{n}, dukdt::FTField{n}) where {n}=
    (A_mul_B!(dukdt, ks.lks, uk);        # linear term
     ks.nlks(t, uk, dukdt, true); dukdt) # nonlinear term (add value)


# ~~~ LINEARISED EQUATION ~~~
struct LinearisedKSEq{n, FT<:FTField{n}, F<:Field{n}}
     U::Float64  # mean flow velocity
     L::Float64  # domain size
    nk::FT       # temporary in Fourier space
     u::F        # temporary in physical space
     w::F        # temporary in physical space
    ifft         # plans
    fft          #
    function LinearisedKSEq{n}(U::Real, L::Real) where {n}
        u = Field(n); w = Field(n); nk = FTField(n)
        ifft = InverseFFT(nk); fft  = ForwardFFT(u)
        new{n, typeof(nk), typeof(u)}(U, L, nk, u, w, ifft, fft)
    end
end

LinearisedKSEq(n::Int, U::Real, L::Real) = LinearisedKSEq{n}(U, L)

# ~~~ EVALUATE LINEAR OPERATOR AROUND U ~~~
function (lks::LinearisedKSEq{n})(t::Real, uk::FTField{n}, wk::FTField{n}, dwkdt::FTField{n}) where {n}
    # setup
    w, u, nk = lks.w, lks.u, lks.nk # aliases
    L, U = lks.L, lks.U             # parameters
    qk = 2π/L*(0:n)                 # wave numbers
    uk[1] = 0; wk[1] = 0            # make sure zero mean

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
