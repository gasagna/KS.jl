# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import DualNumbers: Dual, value, epsilon

export VarFTField,
       VarField,
       state,
       prime


# ////// Embed the perturbation and the state together //////
struct VarFTField{n,
                  ISODD,
                  T<:Dual{<:Real},
                  F<:FTField{n, ISODD}} <: AbstractFTField{n, ISODD, T}
    s::F       # state
    p::F       # perturbation
    L::Float64 # domain length
end

# ////// constructors //////
VarFTField(n::Int, L::Real, isodd::Bool) =
    VarFTField(FTField(n, L, isodd), FTField(n, L, isodd))

VarFTField(U::FT, V::FT) where {n, ISODD, T, FT<:FTField{n, ISODD, T}} =
    (U.L != V.L && throw(ArgumentError("fields must have same domain length"));
     VarFTField{n, ISODD, Dual{T}, FT}(U, V, U.L))



# ////// Same for field objects //////
struct VarField{n,
                T<:Dual{<:Real},
                F<:Field{n}} <: AbstractField{n, T}
    s::F       # state
    p::F       # perturbation
    L::Float64 # domain length
end

# ////// constructors //////
VarField(n::Int, L::Real) = VarField(Field(n, L), Field(n, L))

VarField(u::F, v::F) where {n, T, F<:Field{n, T}} =
    (u.L != v.L && throw(ArgumentError("fields must have same domain length"));
     VarField{n, Dual{T}, F}(u, v, u.L))



# ////// Define remaining methods //////
for FT in (:FTField, :Field)
    VARFT = Symbol(:Var, FT)
    @eval begin
        # ////// accessors //////
        state(x::$VARFT) = x.s
        prime(x::$VARFT) = x.p
        state(x::$FT) = x # no op for FTField and Field

        # ////// array interface //////
        Base.@propagate_inbounds @inline Base.getindex(x::$VARFT, i...) =
            Dual(x.s[i...], x.p[i...])
        Base.@propagate_inbounds @inline Base.setindex!(x::$VARFT, v::Dual, i...) =
            (x.s[i...] = value(v); x.p[i...] = epsilon(v); v)
        Base.@propagate_inbounds @inline Base.setindex!(x::$VARFT, v::Number, i...) =
            (x.s[i...] = v; v)

        Base.similar(x::$VARFT) = $VARFT(similar(x.s), similar(x.p))
        Base.copy(x::$VARFT)    = $VARFT(copy(x.s),    copy(x.p))
    end
end