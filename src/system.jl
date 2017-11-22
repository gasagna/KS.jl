# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

import IMEXRKCB

export KSEq, imex, LinearisedKSEq

# ~~~ LINEAR TERM ~~~
struct LinearKSEqTerm{n, L}
    A::Vector{Float64}
    LinearKSEqTerm{n, L}() where {n, L} =
        new{n, L}(Float64[(2π*k/L)^2 - (2π*k/L)^4 for k = 0:n])
end
LinearKSEqTerm(n::Int, L::Real) = LinearKSEqTerm{n, L}()

# obey IMEXRKCB interface
Base.A_mul_B!(dukdt::FTField{n, L}, lks::LinearKSEqTerm{n, L}, uk::FTField{n, L}) where {n, L} =
    (dukdt .= lks.A .* uk; dukdt)

IMEXRKCB.ImcA!(lks::LinearKSEqTerm{n, L}, c::Real, uk::FTField{n, L}, dukdt::FTField{n, L}) where {n, L} =
    dukdt .= uk./(1 .- c.*lks.A)


# ~~~ NONLINEAR TERM ~~~
struct NonLinearKSEqTerm{n, L, FT<:FTField{n, L}, F<:Field{n, L}}
     U::Float64 # mean flow velocity
     u::F       # solution in physical space
    vk::FT      # temporary in Fourier space
    ifft        # plans
    fft         #
    function NonLinearKSEqTerm{n, L}(U::Real) where {n, L}
        vk = FTField(n, L); ifft = InverseFFT(vk)
        u  = Field(n, L);   fft  = ForwardFFT(u)
        new{n, L, typeof(vk), typeof(u)}(U, u, vk, ifft, fft)
    end
end

NonLinearKSEqTerm(n::Int, U::Real, L::Real) = NonLinearKSEqTerm{n, L}(U)

function (nlks::NonLinearKSEqTerm{n, L})(t::Real, uk::FTField{n, L}, dukdt::FTField{n, L}, add::Bool=false) where {n, L}
    # setup
    vk, u, U = nlks.vk, nlks.u, nlks.U # aliases
    qk = 2π/L*(0:n)                    # wave numbers

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
struct KSEq{n, L, NL<:NonLinearKSEqTerm{n, L}}
     lks::LinearKSEqTerm{n, L}
    nlks::NL
    function KSEq{n, L}(U::Real) where {n, L}
        exTerm = NonLinearKSEqTerm(n, U, L)
        imTerm = LinearKSEqTerm(n, L)
        new{n, L, typeof(exTerm)}(imTerm, exTerm)
    end
end
KSEq(n::Int, U::Real, L::Real) = KSEq{n, L}(U)

# split linear and nonlinear term
imex(KSEq) = KSEq.lks, KSEq.nlks

# evaluate right hand side of equation
(ks::KSEq{n, L})(t::Real, uk::FTField{n, L}, dukdt::FTField{n, L}) where {n, L} =
    (A_mul_B!(dukdt, ks.lks, uk);        # linear term
     ks.nlks(t, uk, dukdt, true); dukdt) # nonlinear term (add value)


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
    qk = 2π/L*(0:n)                           # wave numbers
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
