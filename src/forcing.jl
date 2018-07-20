# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export SensitivityWRTViscosity,
       SteadyForcing,
       DummyForcing,
       FlowForcing

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


# ////// FORCING ALONG f(x(T)) //////
mutable struct FlowForcing{n, FEQ<:ForwardEquation{n}, FT<:FTField{n}} <: AbstractForcing{n} 
      f::FEQ
    TMP::FT
      χ::Float64
    FlowForcing(f::FEQ, TMP::FT) where {n, 
                                        FEQ<:ForwardEquation{n}, 
                                        FT<:FTField} = 
        new{n, FEQ, FT}(f, TMP)

end

# constructors
FlowForcing(f::ForwardEquation{n}, ISODD::Bool, χ::Real=1.0) where {n} = 
    FlowForcing(f, FTField(n, ISODD))

# obey callable interface
(ff::FlowForcing{n})(t::Real, U::FT, V::FT, dVdt::FT) where {n, FT<:FTField{n}} =
    (ff.f(t, U, ff.TMP); dVdt .+= ff.χ.*ff.TMP; dVdt)


# ////// Sensitivity with respect to ν //////
struct SensitivityWRTViscosity{n} <: AbstractForcing{n} end

# constructors
SensitivityWRTViscosity(n::Int) = SensitivityWRTViscosity{n}()

# obey callable interface
(::SensitivityWRTViscosity{n})(t::Real,
                               U::FT,
                               V::FT,
                               dVdt::FT) where {n, FT<:FTField{n}} =
    (@inbounds @simd for k in wavenumbers(n);
          dVdt[k] -= k^4*U[k]
     end; dVdt)