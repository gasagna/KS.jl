# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import FFTW

export ForwardFFT, 
       InverseFFT


# //// UTILS //////
_fix_FFT!(U::FTField{n}) where {n} =
    (@inbounds U.data[1]   = 0; 
     @inbounds U.data[n+2] = 0; U)

# scale complex Fourier coefficients
_normalise!(U::FTField{n}) where {n} = (U.data .*= 1/(2n+2); U)

# ////// FORWARD TRANSFORM //////
struct ForwardFFT{n, P}
    plan::P
    function ForwardFFT(u::AbstractField{n}) where {n}
        plan = FFTW.plan_rfft(u.data, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

# ////// callable interface //////
@inline (f::ForwardFFT{n})(u::Field{n}, U::FTField{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, u.data, U.data); _fix_FFT!(_normalise!(U)))


# ////// INVERSE TRANSFORM //////
struct InverseFFT{n, P}
    plan::P
    function InverseFFT(U::AbstractFTField{n}) where {n}
        plan = FFTW.plan_brfft(U.data, 2*(n+1), flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

# ////// callable interface //////
@inline (f::InverseFFT{n})(U::FTField{n}, u::Field{n}) where {n} =
    (_fix_FFT!(U); FFTW.unsafe_execute!(f.plan, U.data, u.data); u)