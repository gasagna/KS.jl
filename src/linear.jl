export LinearisedEquation

# ////// LINEARISED EQUATION //////
struct LinearisedExTerm{n, M<:AbstractLinearMode, FT<:FTField{n}, F<:Field{n}}
    TMP1::FT # temporary in Fourier space
    TMP2::FT # temporary in Fourier space
    tmp1::F  # temporary in physical space
    tmp2::F  # temporary in physical space
    tmp3::F  # temporary in physical space
    ifft     # plans
    fft      #
    function LinearisedExTerm{n, M}(ISODD::Bool) where {n, M}
        TMP1 = FTField(n, ISODD); TMP2 = FTField(n, ISODD)
        tmp1 = Field(n); tmp2 = Field(n); tmp3 = Field(n)
        ifft = InverseFFT(TMP1); fft  = ForwardFFT(tmp1)
        new{n, M, typeof(TMP1), typeof(tmp1)}(TMP1,
                TMP2, tmp1, tmp2, tmp3, ifft, fft)
    end
end

# constructor
LinearisedExTerm(n::Int, ISODD::Bool, mode::M) where {M <: AbstractLinearMode} = 
    LinearisedExTerm{n, M}(ISODD)


# evaluate linear operator around U
function (lks::LinearisedExTerm{n, TangentMode})(t::Real,
                                                 U::FTField{n},
                                              dUdt::FTField{n},
                                                 V::FTField{n},
                                              dVdt::FTField{n},
                                               add::Bool=false) where {n}

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
function (lks::LinearisedExTerm{n, AdjointMode})(t::Real,
                                                 U::FTField{n},
                                                 V::FTField{n},
                                              dVdt::FTField{n},
                                               add::Bool=false) where {n}

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
                                  IT<:LinearTerm{n},
                                  ET<:LinearisedExTerm{n},
                                  G}
     imTerm::IT
     exTerm::ET
    forcing::G
end

# outer constructor: main entry point
function LinearisedEquation(n::Int,
                            ν::Real,
                            ISODD::Bool,
                            mode::AbstractLinearMode,
                            forcing=nothing)
    imTerm = LinearTerm(n, ν, ISODD)
    exTerm = LinearisedExTerm(n, ISODD, mode)
    LinearisedEquation{n,
                       typeof(imTerm),
                       typeof(exTerm),
                       typeof(forcing)}(imTerm, exTerm, forcing)
end


# /// SPLIT EXPLICIT AND IMPLICIT PARTS /// 

# If we have a forcing we have to integrate it explicitly, and we require passing the
# time derivative of the main state too. This is because the linearised equations do 
# depend on dUdt, but the forcing might.
function splitexim(eq::LinearisedEquation{n, IT, ET, F}) where {n, IT, ET, F<:AbstractForcing}
    
    # integrate explicit part and forcing explicitly
    function wrapper(t::Real,
                     U::FTField{n},
                     dUdt::FTField{n},
                     V::AbstractFTField{n},
                     dVdt::AbstractFTField{n},
                     add::Bool=false)
        eq.exTerm( t, U, dUdt, V, dVdt, add)
        eq.forcing(t, U, dUdt, V, dVdt) # note forcing always adds to dVdt
        return dVdt
    end

    return wrapper, eq.imTerm
end

# If there is no forcing, things are much easier, and we just return the two fields. 
# In this case, the explicit term does not depend on the time derivative of the main state.
splitexim(eq::LinearisedEquation{n, IT, ET, Nothing}) where {n, IT, ET} =
    (eq.exTerm, eq.imTerm)

# /// EVALUATE RIGHT HAND SIDE OF LINEARISED EQUATION ///
# Same here, if we have forcing we evaluate it, and we require passing dUdt too.
(eq::LinearisedEquation{n, IT, ET, F})(t::Real, U, dUdt, V, dVdt) where {n, IT, ET, F<:AbstractForcing} =
    (mul!(dVdt, eq.imTerm, V);
        eq.exTerm(t, U, dUdt, V, dVdt, true);
            eq.forcing(t, U, dUdt, V, dVdt); dVdt)

(eq::LinearisedEquation{n, IT, ET, Nothing})(t::Real, U, dUdt, V, dVdt) where {n, IT, ET} =
    (mul!(dVdt, eq.imTerm, V);
        eq.exTerm(t, U, dUdt, V, dVdt, true);
            return dVdt)

# Obtain jacobian matrix (only for systems with no forcing!)
function (eq::LinearisedEquation{n, IT, ET, Nothing})(J::AbstractMatrix,
                                                      U::FT,
                                                   tmp1::FT,
                                                   tmp2::FT) where {n, FT, IT,
                                                                    ET, Nothing}
    # set to zero initially, for safety
    tmp1 .= 0
    for i = 1:length(tmp1)
        # perturb one component
        tmp1[i] = 1
        # apply linearised operator
        eq(0, U, tmp1, tmp2)
        # write to matrix
        J[:, i] .= tmp2
        # reset component to zero
        tmp1[i] = 0
    end
    return J
end