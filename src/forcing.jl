# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export SensitivityWRTViscosity,
       SteadyForcing,
       DummyForcing,
       FlowForcing,
       EnergyGradient,
       BlowingForcing


struct BlowingForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n} 
    TMP::FT
    c::Float64
end

BlowingForcing(U::AbstractFTField{n}, c::Real) where {n} = 
    BlowingForcing{n, typeof(U)}(copy(U), c)

# call for nonlinear equation
(b::BlowingForcing{n})(t, U::FT, dUdt::FT) where {n, FT<:AbstractFTField{n}} = 
    (ddx!(b.TMP, U); dUdt .+= b.c.*b.TMP; dUdt)

# ////// DUMMY FORCING - DOES NOTHING //////
struct DummyForcing{n} <: AbstractForcing{n} end

DummyForcing(n::Int) = DummyForcing{n}()

# call for nonlinear equation
(::DummyForcing{n})(t, U::FT, dUdt::FT) where {n, FT<:AbstractFTField{n}} = dUdt

# and for the linear equations
(::DummyForcing{n})(t, U::FT, dUdt::FT,
                       V::FT, dVdt::FT) where {n, FT<:AbstractFTField{n}} = dVdt

(::DummyForcing{n})(t, U::FT,
                       V::FT, dVdt::FT) where {n, FT<:AbstractFTField{n}} = dVdt


# ////// STEADY FORCING //////
struct SteadyForcing{n, FT<:AbstractFTField{n}} <: AbstractForcing{n}
    H::FT
end

# allow indexing this object
Base.getindex(sf::SteadyForcing, i::Int) = sf.H[i]
Base.setindex!(sf::SteadyForcing, val, i::Int) = (sf.H[i] = val)

# add to dUdt by default
@inline (sf::SteadyForcing{n, FT})(t::Real, U::FT, dUdt::FT) where {n, FT<:AbstractFTField{n}} =
    (dUdt .+= sf.H; return dUdt)

@inline (sf::SteadyForcing{n, FT})(t::Real, U::FT, V::FT, dVdt::FT) where {n, FT<:AbstractFTField{n}} =
    (dVdt .+= sf.H; return dVdt)

@inline (sf::SteadyForcing{n, FT})(t::Real, U::FT, dUdt::FT, V::FT, dVdt::FT) where {n, FT<:AbstractFTField{n}} =
    (dVdt .+= sf.H; return dVdt)


# ////// FORCING ALONG f(x(T)) //////
mutable struct FlowForcing{n} <: AbstractForcing{n}
    χ::Float64
end

# constructors
FlowForcing(n::Int, χ::Real=1.0) = FlowForcing{n}(χ)

# obey callable interface
(ff::FlowForcing{n})(t::Real, U::FT, dUdt::FT,
                              V::FT, dVdt::FT) where {n, FT<:FTField{n}} =
    (dVdt .+= ff.χ.*dUdt; dVdt)


# ////// Sensitivity with respect to ν //////
struct SensitivityWRTViscosity{n} <: AbstractForcing{n} end

# constructors
SensitivityWRTViscosity(n::Int) = SensitivityWRTViscosity{n}()

# obey callable interface
(f::SensitivityWRTViscosity{n})(t::Real,
                                U::FT,
                                V::FT,
                             dVdt::FT) where {n, FT<:FTField{n}} =
    f(t, U, U, V, dVdt)

function (::SensitivityWRTViscosity{n})(t::Real,
                                        U::FT,
                                     dUdt::FT,
                                        V::FT,
                                     dVdt::FT) where {n, FT<:FTField{n}}
    @inbounds @simd for k in wavenumbers(n)
          dVdt[k] -= k^4*U[k]
    end
    return dVdt
end

# used in VaPOrE
function (f::SensitivityWRTViscosity{n})(dVdt::FT,
                                            U::FT) where {n, FT<:FTField{n}}
    @inbounds @simd for k in wavenumbers(n)
          dVdt[k] = -k^4*U[k]
    end
    return dVdt
end

# forcing for adjoint equation based on energy density gradient
struct EnergyGradient{n} <: AbstractForcing{n} end

EnergyGradient(n::Int) = EnergyGradient{n}()

(::EnergyGradient{n})(t::Real,
                      U::FT,
                      V::FT,
                      dVdt::FT) where {n, FT<:FTField{n}} = 
    (dVdt .-= U; dVdt)