# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export SensitivityForcing, SteadyForcing

# ////// FORCING //////
abstract type AbstractForcing{n} end


# ////// DUMMY FORCING - DOES NOTHING //////
struct DummyForcing{n} <: AbstractForcing{n} end

DummyForcing(n::Int) = DummyForcing{n}()

# call for nonlinear equation
(::DummyForcing{n})(t, U::FT, dUdt::FT) where {n, FT<:AbstractFTField{n}} = dUdt

# and for the linear equation
(::DummyForcing{n})(t, U::FT,
                       V::FT, dVdt::FT) where {n, FT<:AbstractFTField{n}} = dVdt


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



# ////// FORCING FOR THE SENSITIVITY EQUATIONS - THIS IS -uₓ //////
struct SensitivityForcing{n} <: AbstractForcing{n} end

# constructors
SensitivityForcing(n::Int) = SensitivityForcing{n}()

# obey callable interface
(::SensitivityForcing{n})(t::Real,
                          U::FT,
                          V::FT,
                          dVdt::FT) where {n, FT<:FTField{n}} =
    (@inbounds @simd for k in wavenumbers(n);
          dVdt[k] -= im*2π/U.L*k*U[k]
     end; dVdt)
