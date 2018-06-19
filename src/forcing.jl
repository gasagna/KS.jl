# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export SensitivityForcing, SteadyForcing

# ////// FORCING //////
abstract type AbstractForcing{n} end


# ////// STEADY FORCING //////
struct SteadyForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n}
    H::FT
end
SteadyForcing(H::AbstractFTField{n}) where {n} = SteadyForcing{n, typeof(H)}(H)

# allow indexing this object
Base.getindex(sf::SteadyForcing, i::Int) = sf.H[i]
Base.setindex!(sf::SteadyForcing, val, i::Int) = (sf.H[i] = val)

# add to dUdt by default
@inline (sf::SteadyForcing{n, FT})(t::Real, U::FT, dUdt::FT) where {n, FT<:AbstractFTField{n}} =
    (dUdt .+= sf.H; return dUdt)



# ////// FORCING FOR THE SENSITIVITY EQUATIONS //////
struct SensitivityForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n}
    tmp::FT    # temporary: set to full space 
    χ::Float64
end

# constructors
SensitivityForcing(n::Int, L::Real, χ::Real) =
    SensitivityForcing(FTField(n, L, false), χ)

SensitivityForcing(U::FTField{n}, χ::Real) where {n} =
    SensitivityForcing{n, typeof(U)}(U, χ)

# obey callable interface
@inline function (sf::SensitivityForcing{n})(t::Real, 
                                             UV::FT, 
                                             dUVdt::FT) where {n, 
                                                ISODD, FT<:VarFTField{n, ISODD}}
    # aliases
    dUVdt_state, dUVdt_prime = state(dUVdt), prime(dUVdt)

    # this is fₚ(u(x,t))
    sf.tmp .= state(UV)
    ddx!(sf.tmp)
    dUVdt_prime .-= sf.tmp

    # this is χ⋅f(u(x,t))
    sf.χ != 0 && (dUVdt_prime .+= sf.χ .* state(dUVdt))

    return dUVdt
end