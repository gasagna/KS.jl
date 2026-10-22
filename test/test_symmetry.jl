# Translations, their generator, the reflection and the best shift, in the full space.

@testset "translations and reflection             " begin
    n = 16
    ν = (2π / 22)^2
    U = smooth(n, false; a=0.1)
    V = smooth(n, false; a=0.2, φ=0.4)

    # ---- shift!, u(x) → u(x + s), a group action, orthogonal in dot ----
    Us = shift!(copy(U), 0.9)
    Vs = shift!(copy(V), 0.9)
    @test rel(dofs(shift!(copy(U), 2π)), dofs(U)) < 1e-13
    @test rel(dofs(shift!(shift!(copy(U), 1.3), -1.3)), dofs(U)) < 1e-14
    @test abs(dot(Us, Vs) - dot(U, V)) < 1e-14 * abs(dot(U, V))

    # ---- ddx! is the generator of shift! ----
    h  = 1e-5
    gd = (dofs(shift!(copy(U), h)) .- dofs(shift!(copy(U), -h))) ./ (2h)
    @test rel(dofs(ddx!(FTField(n, false), U)), gd) < 1e-7

    # ---- reflect!, u(x) → -u(-x), an involution that keeps odd fields ----
    @test dofs(reflect!(reflect!(copy(U)))) == dofs(U)
    @test dofs(reflect!(copy(smooth(n, true)))) == dofs(smooth(n, true))
    @test reflect! === LinearAlgebra.reflect!

    # ---- equivariance, exact when padded, only for grid shifts when aliased ----
    for dealias in (:pad, :none)
        eq = ForwardEquation(n, ν, false; dealias=dealias)
        f(W) = eq(0.0, copy(W), FTField(n, false))
        @test rel(dofs(f(reflect!(copy(U)))), dofs(reflect!(f(U)))) < 1e-13
        @test rel(dofs(f(shift!(copy(U), 2π/34))), dofs(shift!(f(U), 2π/34))) < 1e-13
        e = rel(dofs(f(shift!(copy(U), 1.234))), dofs(shift!(f(U), 1.234)))
        dealias === :pad ? (@test e < 1e-13) : (@test e > 1e-4)
    end
end

@testset "best shift                              " begin
    n = 16
    U = smooth(n, false; a=0.1)

    # ---- a known shift is recovered, and neither argument changes ----
    for s0 in (0.123, 1.234, 4.5, 2π - 1e-3)
        V  = shift!(copy(U), s0)
        U0 = dofs(U)
        V0 = dofs(V)
        d, (s,) = minnormdiff(U, V, 8n; method=:newton)
        @test abs(mod(s - s0 + π, 2π) - π) < 1e-12
        @test d < 1e-13
        @test dofs(U) == U0
        @test dofs(V) == V0
    end

    # ---- two unrelated states, the minimum of a fine scan and not below it ----
    U1 = smooth(n, false; a=0.15)
    U2 = shift!(smooth(n, false; a=0.25, φ=0.9), 2.0)
    d, (s,) = minnormdiff(U1, U2, 8n; method=:newton)
    scan = [normdiff(shift!(copy(U1), σ), U2) for σ in range(0, 2π; length=10001)]
    @test d ≤ minimum(scan) + 1e-14
    @test minimum(scan) - d < 1e-6
    @test abs(normdiff(shift!(copy(U1), s), U2) - d) < 1e-14

    # ---- the default grid search is unchanged, N = 20 shifts of 2π/20 ----
    dg, (sg,) = minnormdiff(copy(U1), U2)
    @test any(abs(sg - 2π * i / 20) < 1e-14 for i in 0:19)
    @test dg ≥ d

    # ---- never above the best of its own N shifts, random pairs ----
    # Newton's method alone, from the best of the N shifts, ends above it in
    # some of these pairs for N = 20
    rng = Xoshiro(3)
    for N in (20, 8n)
        above = 0
        for _ in 1:500
            A     = FTField(randn(rng, 2n) .* repeat(exp.(-0.05 .* (1:n)), inner=2), false)
            B     = FTField(randn(rng, 2n) .* repeat(exp.(-0.05 .* (1:n)), inner=2), false)
            dnewt = minnormdiff(A, B, N; method=:newton)[1]
            dgrid = minimum(normdiff(shift!(copy(A), 2π * j / N), B) for j in 0:N-1)
            above += dnewt > dgrid * (1 + 1e-12)
        end
        @test above == 0
    end

    # ---- the option ----
    @test_throws ArgumentError minnormdiff(U1, U2; method=:foo)
end
