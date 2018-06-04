# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import FFTW

export ForwardFFT, 
	   InverseFFT


# //// UTILS //////
_fix_FFT!(uk::FTField{n}) where {n} = 
	(@inbounds uk.data[1] = 0; @inbounds uk.data[n+1] = 0; uk)

_normalise!(uk::FTField{n}) where {n} = (uk .*= 1/(2*(n+1)); uk)

# ////// FORWARD TRANSFORM //////
struct ForwardFFT{n, P}
    plan::P
	function ForwardFFT(u::AbstractField{n}) where {n}
		plan = FFTW.plan_rfft(state(u).data, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

# ////// callable interface //////
@inline (f::ForwardFFT{n})(u::Field{n}, uk::FTField{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, u.data, uk.data); _fix_FFT!(_normalise!(uk)))

@inline (f::ForwardFFT{n})(u::VarField{n}, uk::VarFTField{n}) where {n} =
    (f(state(u), state(uk)); f(prime(u), prime(uk)); uk)


# ////// INVERSE TRANSFORM //////
struct InverseFFT{n, P}
    plan::P
	function InverseFFT(uk::AbstractFTField{n}) where {n}
		plan = FFTW.plan_brfft(state(uk).data, 2*(n+1), flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

# ////// callable interface //////
@inline (f::InverseFFT{n})(uk::FTField{n}, u::Field{n}) where {n} =
    (_fix_FFT!(uk); FFTW.unsafe_execute!(f.plan, uk.data, u.data); u)

@inline (f::InverseFFT{n})(uk::VarFTField{n}, u::VarField{n}) where {n} =
    (f(state(uk), state(u)); f(prime(uk), prime(u)); u)
