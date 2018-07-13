export LinearisedEquation,
       set_χ!

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
function (lks::LinearisedExTerm{n, M})(t::Real,
                                       U::FTField{n},
                                       V::FTField{n},
                                       dVdt::FTField{n},
                                       add::Bool=false) where {n, M}

    # /// calculate - v * uₓ term ///
    lks.ifft(V, lks.tmp1)               # transform V to physical space
    ddx!(lks.TMP1, U)                   # differentiate U
    lks.ifft(lks.TMP1, lks.tmp2)        # transform to physical space
    lks.tmp3 .= .- lks.tmp1 .* lks.tmp2 # multiply in physical space
    lks.fft(lks.tmp3, lks.TMP1)

    # /// calculate - u * vₓ or + (u*v)ₓ term ///
    if M <: TangentMode
        lks.ifft(U, lks.tmp1)               # transform U to physical space
        ddx!(lks.TMP2, V)                   # differentiate V
        lks.ifft(lks.TMP2, lks.tmp2)        # transform Vto physical space
        lks.tmp3 .= .- lks.tmp1 .* lks.tmp2 # multiply
    end
    if M <: AdjointMode
        lks.ifft(U, lks.tmp1)              # transform U to physical space
        lks.ifft(V, lks.tmp2)              # transform V to physical space
        lks.tmp1 .= lks.tmp1 .* lks.tmp2   # multiply, overwriting v
        lks.fft(lks.tmp1, lks.TMP2)        # transform term to fourier space
        ddx!(lks.TMP2)                     # differentiate term
    end

    # /// store and set symmetry ///
    add == true ? (dVdt .+= lks.TMP2) : (dVdt .= lks.TMP2)
    _set_symmetry!(dVdt)

    return dVdt
end

# ~~~ SOLVER OBJECT FOR THE LINEARISED EQUATIONS ~~~
mutable struct LinearisedEquation{n,
                                  IT<:LinearTerm{n},
                                  ET<:LinearisedExTerm{n},
                                  G<:AbstractForcing{n},
                                  M<:Flows.AbstractMonitor,
                                  FT<:AbstractFTField}
     imTerm::IT
     exTerm::ET
    forcing::G
        mon::M
        TMP::FT
          χ::Float64
end

# set χ constant
set_χ!(eq::LinearisedEquation, χ::Real) = (eq.χ = χ; nothing)

# outer constructor: main entry point
function LinearisedEquation(n::Int,
                            ν::Real,
                            ISODD::Bool,
                            mode::AbstractLinearMode,
                            mon::Flows.AbstractMonitor{T, X},
                            χ::Real=0,
                            forcing::AbstractForcing=DummyForcing(n)) where {T, X}
    X <: FTField{n, ISODD} || error("invalid monitor object")
    imTerm = LinearTerm(n, ν, ISODD, mode)
    exTerm = LinearisedExTerm(n, ISODD, mode)
    TMP = FTField(n, ISODD)
    LinearisedEquation{n,
                       typeof(imTerm),
                       typeof(exTerm),
                       typeof(forcing),
                       typeof(mon),
                       typeof(TMP)}(imTerm, exTerm, forcing, mon, TMP, χ)
end

# obtain two components
function splitexim(eq::LinearisedEquation{n}) where {n}
    function wrapper(t::Real, V::AbstractFTField{n}, dVdt::AbstractFTField{n})
        # interpolate U and evaluate nonlinear interaction term and forcing
        eq.mon(eq.TMP, t, Val{0}())
        eq.exTerm(t, eq.TMP, V, dVdt, false)
        eq.forcing(t, eq.TMP, V, dVdt)

        # interpolate dUdt if needed
        if eq.χ != 0
            eq.mon(eq.TMP, t, Val{1}())
            dVdt .+= eq.χ .* eq.TMP
        end

        return dVdt
    end
    return wrapper, eq.imTerm
end