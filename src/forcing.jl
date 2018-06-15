# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export SensitivityForcing, SteadyForcing

# ////// FORCING //////
abstract type AbstractForcing{n} end


# ////// STEADY FORCING //////
struct SteadyForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n}
    hk::FT
end
SteadyForcing(hk::AbstractFTField{n}) where {n} = SteadyForcing{n, typeof(hk)}(hk)

# allow indexing this object
Base.getindex(sf::SteadyForcing, i::Int) = sf.hk[i]
Base.setindex!(sf::SteadyForcing, val, i::Int) = (sf.hk[i] = val)

# add to dukdt by default
@inline (sf::SteadyForcing{n, FT})(t::Real, uk::FT, dukdt::FT) where {n, FT<:AbstractFTField{n}} =
    (dukdt .+= sf.hk; return dukdt)



# ////// FORCING FOR THE SENSITIVITY EQUATIONS //////
struct SensitivityForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n}
	tmp::FT
	χ::Float64
end

# constructors
SensitivityForcing(n::Int, L::Real, χ::Real) = 
	SensitivityForcing(FTField(n, L), χ)

SensitivityForcing(uk::FTField{n}, χ::Real) where {n} = 
	SensitivityForcing{n, typeof(uk)}(uk, χ)

# obey callable interface
@inline function (sf::SensitivityForcing{n})(t::Real, uvk::FT, duvkdt::FT) where {n, FT<:VarFTField{n}}
	# aliases
	duvkdt_state, duvkdt_prime = state(duvkdt), prime(duvkdt)

	# this is fₚ(u(x,t))
	sf.tmp .= state(uvk)
	ddx!(sf.tmp)
	duvkdt_prime .-= sf.tmp

	# this is χ⋅f(u(x,t))
	sf.χ != 0 && (duvkdt_prime .+= sf.χ * state(duvkdt))

	return duvkdt
end