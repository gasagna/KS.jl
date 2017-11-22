# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

import FFTW
export ForwardFFT, InverseFFT

# ~~~ FORWARD TRANSFORM ~~~
struct ForwardFFT{n, P}
    plan::P
    function ForwardFFT{n}(u::Field{n}) where {n}
        plan = FFTW.plan_rfft(u.data, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

ForwardFFT(u::Field{n}) where {n} = ForwardFFT{n}(u)
ForwardFFT(n::Int) = ForwardFFT(Field(n, 1)) # domain size not needed

# ~ callable interface
@inline (f::ForwardFFT{n})(u::Field{n}, uk::FTField{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, u.data, uk.data); uk .*= 1/2n; uk)


# ~~~ INVERSE TRANSFORM ~~~
struct InverseFFT{n, P}
    plan::P
    function InverseFFT{n}(uk::FTField{n}) where {n}
        plan = FFTW.plan_brfft(uk.data, 2n, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

InverseFFT(uk::FTField{n}) where {n} = InverseFFT{n}(uk)
InverseFFT(n::Int) = InverseFFT(FTField(n))

# ~ callable interface
@inline (f::InverseFFT{n})(uk::FTField{n}, u::Field{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, uk.data, u.data); u)