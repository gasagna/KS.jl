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

_normalise!(U::FTField{n}) where {n} = (U .*= 1/(2*(n+1)); U)

# ////// FORWARD TRANSFORM //////
struct ForwardFFT{n, P}
    plan::P
    function ForwardFFT(u::AbstractField{n}) where {n}
        plan = FFTW.plan_rfft(state(u).data, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

# ////// callable interface //////
@inline (f::ForwardFFT{n})(u::Field{n}, U::FTField{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, u.data, U.data); _fix_FFT!(_normalise!(U)))

@inline (f::ForwardFFT{n})(u::VarField{n}, U::VarFTField{n}) where {n} =
    (f(state(u), state(U)); f(prime(u), prime(U)); U)


# ////// INVERSE TRANSFORM //////
struct InverseFFT{n, P}
    plan::P
    function InverseFFT(U::AbstractFTField{n}) where {n}
        plan = FFTW.plan_brfft(state(U).data, 2*(n+1), flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

# ////// callable interface //////
@inline (f::InverseFFT{n})(U::FTField{n}, u::Field{n}) where {n} =
    (_fix_FFT!(U); FFTW.unsafe_execute!(f.plan, U.data, u.data); u)

@inline (f::InverseFFT{n})(U::VarFTField{n}, u::VarField{n}) where {n} =
    (f(state(U), state(u)); f(prime(U), prime(u)); u)