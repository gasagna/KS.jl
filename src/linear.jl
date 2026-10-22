export LinearisedEquation

# ////// LINEARISED EQUATION //////
struct LinearisedExTerm{n, ISODD, M<:AbstractLinearMode, FT<:FTField, F<:Field, IP, FP}
    TMP1::FT # temporary in Fourier space
    TMP2::FT # temporary in Fourier space
    tmp1::F  # temporary in physical space, on the grid of the products
    tmp2::F  # temporary in physical space
    tmp3::F  # temporary in physical space
    ifft::IP # inverse transform, n modes to the grid
     fft::FP # forward transform, grid to n modes
    function LinearisedExTerm{n, ISODD, M}(; dealias::Symbol=:none) where {n, ISODD, M}
        # ---- input checks ----
        dealias in (:none, :pad) ||
            throw(ArgumentError("dealias must be :none or :pad"))

        # ---- temporaries and transforms ----
        m    = _gridmodes(n, dealias)
        TMP1 = FTField(n, ISODD)
        TMP2 = FTField(n, ISODD)
        tmp1 = Field(m)
        tmp2 = Field(m)
        tmp3 = Field(m)
        ifft = InverseFFT(TMP1, m)
        fft  = ForwardFFT(tmp1, n)

        # ---- assemble ----
        return new{n,
                   ISODD,
                   M,
                   typeof(TMP1),
                   typeof(tmp1),
                   typeof(ifft),
                   typeof(fft)}(TMP1,
                                TMP2,
                                tmp1,
                                tmp2,
                                tmp3,
                                ifft,
                                fft)
    end
end

# constructor
LinearisedExTerm(      n::Int,
                   ISODD::Bool,
                    mode::M;
                 dealias::Symbol=:none) where {M <: AbstractLinearMode} =
    LinearisedExTerm{n, ISODD, M}(; dealias=dealias)

function Base.deepcopy_internal(x::LinearisedExTerm{n, ISODD, M}, 
                             dict::IdDict) where {n, ISODD, M}
    if !( haskey(dict, x) )
        dict[x] = LinearisedExTerm{n, ISODD, M}(; dealias=_dealias(x.ifft))
    end
    return dict[x]
end

# evaluate linear operator around U
function (lks::LinearisedExTerm{n, ISODD, TangentMode})(t::Real,
                                                        U::FTField{n},
                                                        V::FTField{n},
                                                     dVdt::FTField{n},
                                                      add::Bool=false) where {n, ISODD}

    # /// calculate - v * uₓ term ///
    lks.ifft(V, lks.tmp1)               # transform V to physical space
    ddx!(lks.TMP1, U)                   # differentiate U
    lks.ifft(lks.TMP1, lks.tmp2)        # transform Uₓ to physical space
    lks.tmp3 .= .- lks.tmp1 .* lks.tmp2 # multiply in physical space

    # /// calculate - u * vₓ term ///
    lks.ifft(U, lks.tmp1)               # transform U to physical space
    ddx!(lks.TMP2, V)                   # differentiate V
    lks.ifft(lks.TMP2, lks.tmp2)        # transform V to physical space
    lks.tmp3 .-= lks.tmp1 .* lks.tmp2   # multiply

    # transform sum to fourier space
    lks.fft(lks.tmp3, lks.TMP2)

    # /// store and set symmetry ///
    add == true ? (dVdt .+= lks.TMP2) : (dVdt .= lks.TMP2)
    _set_symmetry!(dVdt)

    return dVdt
end

# evaluate adjoint operator around U
function (lks::LinearisedExTerm{n, ISODD, AdjointMode})(t::Real,
                                                        U::FTField{n},
                                                        V::FTField{n},
                                                     dVdt::FTField{n},
                                                      add::Bool=false) where {n, ISODD}

    # /// calculate - v * uₓ term ///
    lks.ifft(V, lks.tmp1)               # transform V to physical space
    ddx!(lks.TMP1, U)                   # differentiate U
    lks.ifft(lks.TMP1, lks.tmp2)        # transform Uₓ to physical space
    lks.tmp3 .= lks.tmp1 .* lks.tmp2    # multiply in physical space
    lks.fft(lks.tmp3, lks.TMP1)         # transform product to fourier space

    # /// calculate + (u*v)ₓ term ///
    lks.ifft(U, lks.tmp2)               # transform U to physical space
    lks.tmp2 .= lks.tmp1 .* lks.tmp2    # multiply in physical space
    lks.fft(lks.tmp2, lks.TMP2)         # transform product to fourier space
    ddx!(lks.TMP2)                      # differentiate in place

    # sum the two terms
    lks.TMP2 .-= lks.TMP1

    # /// store and set symmetry ///
    add == true ? (dVdt .+= lks.TMP2) : (dVdt .= lks.TMP2)
    _set_symmetry!(dVdt)

    return dVdt
end

# ~~~ SOLVER OBJECT FOR THE LINEARISED EQUATIONS ~~~
mutable struct LinearisedEquation{n,
                                  ISODD,
                                  M<:AbstractLinearMode,
                                  IT<:LinearTerm{n},
                                  ET<:LinearisedExTerm{n, ISODD, M},
                                  G}
     imTerm::IT
     exTerm::ET
    forcing::G
end

"""
    LinearisedEquation(n, ν, ISODD, mode, forcing=DummyForcing(n); dealias=:none)

The tangent (`mode = TangentMode()`) or the adjoint (`AdjointMode()`) of the
right-hand side of `ForwardEquation(n, ν, ISODD; dealias)`, about the state U.
Call it as `L(t, U, V, dVdt)`, the right-hand side of the linearised modes of
Flows.jl, or as `L(t, U, dUdt, V, dVdt)` in a coupled system. The tangent is the
exact Jacobian of the right-hand side, and the adjoint is its exact transpose in
`dot`, with or without dealiasing. Use the same `dealias` as the forward equation.
"""
function LinearisedEquation(      n::Int,
                                  ν::Real,
                              ISODD::Bool,
                               mode::AbstractLinearMode,
                            forcing=DummyForcing(n);
                            dealias::Symbol=:none)
    imTerm = LinearTerm(n, ν, ISODD)
    exTerm = LinearisedExTerm(n, ISODD, mode; dealias=dealias)
    LinearisedEquation{n,
                       ISODD,
                       typeof(mode),
                       typeof(imTerm),
                       typeof(exTerm),
                       typeof(forcing)}(imTerm, exTerm, forcing)
end


# /// SPLIT EXPLICIT AND IMPLICIT PARTS /// 
function splitexim(eq::LinearisedEquation{n}) where {n}
    function wrapper(t::Real, 
                     U::FTField{n}, 
                  dUdt::FTField{n},
                     V::FTField{n}, 
                  dVdt::FTField{n}, 
                   add::Bool=false)
        eq.exTerm(t, U, V, dVdt, add)
        eq.forcing(t, U, dUdt, V, dVdt) 
        return dVdt
    end

    function wrapper(t::Real,
                     U::FTField{n},
                     V::FTField{n},
                     dVdt::FTField{n},
                     add::Bool=false)
        eq.exTerm( t, U, V, dVdt, add)
        eq.forcing(t, U, V, dVdt)
        return dVdt
    end

    return wrapper, eq.imTerm
end

# /// EVALUATE RIGHT HAND SIDE OF LINEARISED EQUATION ///
(eq::LinearisedEquation{n})(t::Real, 
                            U::FTField{n}, 
                         dUdt::FTField{n},
                            V::FTField{n}, 
                         dVdt::FTField{n}) where {n} =
    (mul!(dVdt, eq.imTerm, V);
        eq.exTerm(t, U, V, dVdt, true);
            eq.forcing(t, U, dUdt, V, dVdt); dVdt)


(eq::LinearisedEquation{n})(t::Real,
                            U::FTField{n},
                            V::FTField{n},
                         dVdt::FTField{n}) where {n} =
    (mul!(dVdt, eq.imTerm, V);
        eq.exTerm(t, U, V, dVdt, true);
            eq.forcing(t, U, V, dVdt); dVdt)