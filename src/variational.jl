# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import DualNumbers: Dual, value, epsilon

export VarFTField, 
       VarField, 
       state, 
       prime

# ////// VERSIONS OF FTFIELD AND FIELD WITH PERTURBATION //////
for (FT, numtype) in [(:FTField, Complex), (:Field, Real)]
    typename = Symbol(:Var, FT)
    abstractname = Symbol(:Abstract, FT)
    @eval begin
        struct $typename{n, T<:Dual{<:$numtype}, F<:$FT{n, <:$numtype}} <: $abstractname{n, T}
            uk::F # state
            vk::F # perturbation
            L::Float64
        end

        # ////// constructors //////
        $typename(n::Int, L::Real) = 
            $typename($FT(n, L), $FT(n, L))

        function $typename(uk::$FT{n, T}, vk::$FT{n, T}) where {n, T}
        	uk.L != vk.L && throw(ArgumentError("fields must have same domain length"))
            $typename{n, Dual{T}, typeof(uk)}(uk, vk, uk.L)
        end

        # ////// accessors //////
        state(ukvk::$typename) = ukvk.uk
        prime(ukvk::$typename) = ukvk.vk
        state(ukvk::$FT) = ukvk # no op for FTField and Field

        # ////// array interface //////
        Base.@propagate_inbounds @inline Base.getindex(ukvk::$typename, i...) =
            Dual(ukvk.uk[i...], ukvk.vk[i...])
        Base.@propagate_inbounds @inline Base.setindex!(ukvk::$typename, v::Dual, i...) = 
            (ukvk.uk[i...] = value(v); ukvk.vk[i...] = epsilon(v); v)
        Base.@propagate_inbounds @inline Base.setindex!(ukvk::$typename, v::Real, i...) = 
            (ukvk.uk[i...] = v; v)

        Base.similar(ukvk::$typename) = $typename(similar(ukvk.uk), similar(ukvk.vk))
        Base.copy(ukvk::$typename) = $typename(copy(ukvk.uk), copy(ukvk.vk))
    end
end