# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

import FFTW
export ForwardFFT, InverseFFT

# ~~~ FORWARD TRANSFORM ~~~
struct ForwardFFT{n, P}
    plan::P
    function ForwardFFT{n}(u::AbstractField{n}) where {n}
        plan = FFTW.plan_rfft(state(u).data, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

ForwardFFT(u::AbstractField{n}) where {n} = ForwardFFT{n}(u)
ForwardFFT(n::Int) = ForwardFFT(Field(n, 1))

# ~ callable interface
@inline (f::ForwardFFT{n})(u::Field{n}, uk::FTField{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, u.data, uk.data); uk .*= 1/2n; uk)

@inline (f::ForwardFFT{n})(u::VarField{n}, uk::VarFTField{n}) where {n} =
    (f(state(u), state(uk)); f(prime(u), prime(uk)); uk)


# ~~~ INVERSE TRANSFORM ~~~
struct InverseFFT{n, P}
    plan::P
    function InverseFFT{n}(uk::AbstractFTField{n}) where {n}
        plan = FFTW.plan_brfft(state(uk).data, 2n, flags=FFTW.PATIENT)
        new{n, typeof(plan)}(plan)
    end
end

InverseFFT(uk::AbstractFTField{n}) where {n} = InverseFFT{n}(uk)
InverseFFT(n::Int) = InverseFFT(FTField(n))

# ~ callable interface
@inline (f::InverseFFT{n})(uk::FTField{n}, u::Field{n}) where {n} =
    (FFTW.unsafe_execute!(f.plan, uk.data, u.data); u)

@inline (f::InverseFFT{n})(uk::VarFTField{n}, u::VarField{n}) where {n} =
    (f(state(uk), state(u)); f(prime(uk), prime(u)); u)