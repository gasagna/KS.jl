# KS.jl against the exact Galerkin model of the standard form at L = 22 with 16 modes,
# through the change of variables x = qX, t = q²T, u = U/q, ν = q² of the README.
# The reference is the transcription in ks_model.jl. Setting ENV["KS_REFERENCE_MODEL"]
# to a file that defines ks_model(L, n), such as the model file of the bridge study,
# runs the same tests against that file too. Without it the tests use no file outside
# the package.

module Transcription
    include(joinpath(@__DIR__, "ks_model.jl"))
end

module External
    const PATH = get(ENV, "KS_REFERENCE_MODEL", "")
    isempty(PATH) || include(PATH)
end

function test_reference(M)
    # ---- the study ----
    L  = 22.0
    n  = 16
    q  = 2π / L
    ν  = q^2
    Δt = 0.005

    # ---- the map of the states, U_k = c_k/q with c_k = (a_k - i b_k)/2 ----
    S(x)  = [isodd(i) ? x[(i + 1) ÷ 2] / (2q) : -x[n + i ÷ 2] / (2q) for i in 1:2n]
    Si(y) = vcat([2q * y[2k - 1] for k in 1:n], [-2q * y[2k] for k in 1:n])
    ft(x) = FTField(S(x), false)

    # ---- KS.jl, padded and aliased ----
    eq  = ForwardEquation(n, ν, false; dealias=:pad)
    eqa = ForwardEquation(n, ν, false)
    tan = LinearisedEquation(n, ν, false, TangentMode(); dealias=:pad)
    adj = LinearisedEquation(n, ν, false, AdjointMode(); dealias=:pad)

    # ---- a state on the attractor of the reference, 100 time units ----
    x = vcat([0.1 * exp(-0.3k) * cos(1.7k) for k in 1:n],
             [0.1 * exp(-0.3k) * sin(1.7k) for k in 1:n])
    g = (t, x, dxdt) -> M.f!(dxdt, x)
    rk8!(g, x, Δt, 20000)
    v = Si(dofs(smooth(n, false; a=0.2, φ=0.4)))
    w = Si(dofs(smooth(n, false; a=0.1, φ=2.3)))

    # ---- the operators, in the units of the reference, f_ref = q² S⁻¹ f(Sx) ----
    f_ks = Si(ν .* dofs(eq(0.0, ft(x), FTField(n, false))))
    fa   = Si(ν .* dofs(eqa(0.0, ft(x), FTField(n, false))))
    t_ks = Si(ν .* dofs(tan(0.0, ft(x), ft(v), FTField(n, false))))
    a_ks = Si(ν .* dofs(adj(0.0, ft(x), ft(w), FTField(n, false))))
    @test rel(f_ks, M.f!(zeros(2n), x))          < 1e-13
    @test rel(t_ks, M.tangent!(zeros(2n), x, v)) < 1e-13
    @test rel(a_ks, M.adjoint!(zeros(2n), x, w)) < 1e-13
    @test rel(fa,   M.f!(zeros(2n), x))          > 1e-5   # aliasing, without padding

    # ---- RK8 over 20 time units, Δt_KS = q² Δt ----
    xr = rk8!(g, copy(x), Δt, 4000)
    y  = rk8!(eq, ft(x), ν * Δt, 4000)
    @test rel(Si(dofs(y)), xr) < 1e-11

    # ---- inner product, (2/L)∫UV dX = 4q² dot(U, V) ----
    @test abs(dot(x, v) - 4q^2 * dot(ft(x), ft(v))) < 1e-13 * abs(dot(x, v))

    # ---- shift σ_ℓ = shift!(⋅, -qℓ), generator -q ddx!, reflection ----
    ℓ = 3.7
    @test rel(dofs(shift!(ft(x), -q * ℓ)), S(M.shift(x, ℓ))) < 1e-14
    @test rel(-q .* dofs(ddx!(FTField(n, false), ft(x))), S(M.generator(x))) < 1e-14
    @test dofs(reflect!(ft(x))) == S(M.reflect(x))

    # ---- best shift, d_ref = 2q d and ℓ = mod(-s/q, L) ----
    x2 = rk8!(g, copy(x), Δt, 7460)
    dref, ℓref = M.best_shift(x, x2)
    d, (s,) = minnormdiff(ft(x), ft(x2), 8n; method=:newton)
    @test abs(2q * d - dref) < 1e-10 * dref
    @test abs(mod(-s / q - ℓref + L/2, L) - L/2) < 1e-9

    return nothing
end

@testset "reference model, L = 22, n = 16         " begin
    @testset "transcription, test/ks_model.jl       " begin
        test_reference(Transcription.ks_model(22.0, 16))
    end

    if isdefined(External, :ks_model)
        @testset "file in KS_REFERENCE_MODEL            " begin
            test_reference(External.ks_model(22.0, 16))
        end
    end
end
