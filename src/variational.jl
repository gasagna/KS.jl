import DualNumbers: Dual, value, epsilon

export VarFTField, VarField, state, prime

# ~~ VERSIONS OF FTFIELD AND FIELD WITH PERTURBATION ~~~
for (FT, numtype) in [(:FTField, Complex), (:Field, Real)]
    typename = Symbol(:Var, FT)
    abstractname = Symbol(:Abstract, FT)
    @eval begin
        struct $typename{n, L, T<:Dual{<:$numtype}, F<:$FT{n, L, <:$numtype}} <: $abstractname{n, L, T}
            uk::F # state
            vk::F # perturbation
        end

        # Constructors
        $typename(n::Int, L::Real) = $typename(n, L, Dual{$(numtype == Complex ? Complex128 : Float64)})
        $typename(n::Int, L::Real, ::Type{Dual{T}}) where {T} = $typename($FT(n, L, T), $FT(n, L, T))
        $typename(uk::$FT{n, L, T}, vk::$FT{n, L, T}) where {n, L, T} = $typename{n, L, Dual{T}, typeof(uk)}(uk, vk)

        # accessors
        state(U::$typename) = U.uk
        prime(U::$typename) = U.vk
        state(U::$FT) = U # no op for FTField and Field

        # ~~ Array interface ~~
        Base.@propagate_inbounds @inline Base.getindex(U::$typename, i...) =
            Dual(U.uk[i...], U.vk[i...])
        Base.@propagate_inbounds @inline Base.setindex!(U::$typename, v::Dual, i...) = 
            (U.uk[i...] = value(v); U.vk[i...] = epsilon(v); v)
        Base.@propagate_inbounds @inline Base.setindex!(U::$typename, v::Real, i...) = 
            (U.uk[i...] = v; v)

        Base.similar(U::$typename{n, L, T}) where {n, L, T} = $typename(similar(U.uk), similar(U.vk))
    end
end