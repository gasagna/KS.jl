# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export WaveNumber,
       wavenumbers

# trick to index FTFields over the wave numbers rather
# than over the degrees of freedom
primitive type WaveNumber <: Signed 64 end

# constructor
WaveNumber(k::Int64) = reinterpret(WaveNumber, k)
Int64(k::WaveNumber) = reinterpret(Int64, k)

# This allows the REPL to show values of type WaveNumber
Base.show(io::IO, k::WaveNumber) = print(io, Int64(k))

# conversion and promotion rules
Base.convert(::Type{WaveNumber}, k::WaveNumber) = k
Base.convert(::Type{WaveNumber}, k::Number)     = WaveNumber(Int64(k))

Base.convert(::Type{S},          k::WaveNumber)  where {S <: Number} = S(Int64(k))
Base.promote_rule(::Type{WaveNumber}, ::Type{S}) where {S <: Number} = promote_type(S, Int64)

# arithmetic
for op in [:+, :-, :*, :/, :<, :<=]
    @eval begin
        Base.$(op)(a::WaveNumber, b::WaveNumber) = $(op)(Int64(a), Int64(b))
    end
end

# obtain a vector of wave numbers for iteration
wavenumbers(rng::Range) =
    range(WaveNumber.((first(rng), step(rng), last(rng)))...)