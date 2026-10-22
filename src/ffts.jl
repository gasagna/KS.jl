# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import FFTW

export ForwardFFT, 
       InverseFFT,
       IFFT


# //// UTILS //////
_fix_FFT!(U::FTField{n}) where {n} =
    (@inbounds U.data[1]   = 0;
     @inbounds U.data[n+2] = 0; U)

# scale complex Fourier coefficients
_normalise!(U::FTField{n}) where {n} = (U.data .*= 1/(2n+2); U)

# ////// FORWARD TRANSFORM //////
"""
    ForwardFFT(u::Field{m}, n::Int=m)

Forward transform from the 2(m + 1) points x_j = 2πj/(2m + 2) of `u` to the
modes 1, …, n of an `FTField{n}`, U_k = (1/(2m + 2)) Σ_j u(x_j) e^{-ikx_j}. The
mean and the modes above n are dropped. With 2(m + 1) ≥ 3n + 1, the modes 1, …, n
of the product of two fields with n modes are exact. Planning uses
`FFTW.EXHAUSTIVE` and overwrites the data of `u`.
"""
struct ForwardFFT{m, n, P}
    plan::P                  # real-to-complex plan on 2(m + 1) points
     buf::Vector{ComplexF64} # the m + 2 coefficients of the grid, used when n < m

    function ForwardFFT(u::AbstractField{m}, n::Int=m) where {m}
        # ---- input checks ----
        n ≤ m ||
            throw(ArgumentError("the grid must have m ≥ n modes"))

        # ---- plan ----
        buf  = zeros(ComplexF64, m + 2)
        plan = FFTW.plan_rfft(u.data, flags=FFTW.EXHAUSTIVE)

        return new{m, n, typeof(plan)}(plan, buf)
    end
end

# ////// callable interface //////
# same grid, as before
@inline (f::ForwardFFT{n, n})(u::Field{n}, U::FTField{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, u.data, U.data); _fix_FFT!(_normalise!(U)))

# finer grid, transform into the buffer and keep the modes 1, …, n
@inline function (f::ForwardFFT{m, n})(u::Field{m}, U::FTField{n}) where {m, n}
    # ---- transform on the grid ----
    FFTW.unsafe_execute!(f.plan, u.data, f.buf)

    # ---- the first n modes, normalised ----
    @inbounds for k in 1:n
        U.data[k+1] = f.buf[k+1] * (1/(2m+2))
    end

    return _fix_FFT!(U)
end


# ////// INVERSE TRANSFORM //////
"""
    InverseFFT(U::FTField{n}, m::Int=n)

Inverse transform from the modes of `U` to the 2(m + 1) points of a `Field{m}`,
u(x_j) = 2 Re Σ_{k=1}^n U_k e^{ikx_j}, unnormalised. With m > n the modes
n + 1, …, m are zero, so `u` is the field of `U` on a finer grid, and `U` does not
change. Planning uses `FFTW.EXHAUSTIVE`. With m = n the plan is made on the data
of `U`, which planning overwrites. With m > n it is made on an internal buffer,
and `U` keeps its data.
"""
struct InverseFFT{n, m, P}
    plan::P                  # complex-to-real plan on 2(m + 1) points
     buf::Vector{ComplexF64} # the m + 2 coefficients, padded, used when m > n

    function InverseFFT(U::AbstractFTField{n}, m::Int=n) where {n}
        # ---- input checks ----
        m ≥ n ||
            throw(ArgumentError("the grid must have m ≥ n modes"))

        # ---- plan, on the data of U as before, or on the buffer ----
        buf   = zeros(ComplexF64, m + 2)
        input = m == n ? U.data : buf
        plan  = FFTW.plan_brfft(input, 2*(m+1), flags=FFTW.EXHAUSTIVE)

        return new{n, m, typeof(plan)}(plan, buf)
    end
end

# ////// callable interface //////
# same grid, as before
@inline (f::InverseFFT{n, n})(U::FTField{n}, u::Field{n}) where {n} =
    (_fix_FFT!(U); FFTW.unsafe_execute!(f.plan, U.data, u.data); u)

# finer grid, pad the modes of U with zeros in the buffer, U does not change
@inline function (f::InverseFFT{n, m})(U::FTField{n}, u::Field{m}) where {n, m}
    # ---- padded coefficients, mean and modes above n set to zero ----
    buf = f.buf
    @inbounds begin
        buf[1] = 0
        for k in 1:n
            buf[k+1] = U.data[k+1]
        end
        for k in n+2:m+2
            buf[k] = 0
        end
    end

    # ---- transform ----
    FFTW.unsafe_execute!(f.plan, buf, u.data)

    return u
end


# ~~~ ALLOCATING VERSIONS ~~~

function IFFT(U::FTField{n, ISODD, T}) where {n, ISODD, T}
    V = copy(U)
    fun = InverseFFT(V); V .= U
    return fun(V, Field(n, T))
end