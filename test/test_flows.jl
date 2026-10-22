# Time stepping with Flows.jl. The discrete tangent and adjoint of RK8 and of an IMEX
# scheme, the coupled tangent, the other schemes against RK8, allocations, and deep
# copies.

@testset "Flows.jl, padded, L = 22, n = 16        " begin
    n  = 16
    ν  = (2π / 22)^2
    Δt = 0.005 * ν
    K  = 200
    eq  = ForwardEquation(n, ν, false; dealias=:pad)
    tan = LinearisedEquation(n, ν, false, TangentMode(); dealias=:pad)
    adj = LinearisedEquation(n, ν, false, AdjointMode(); dealias=:pad)

    # ---- a state on the attractor, 100 time units ----
    x0 = rk8!(eq, smooth(n, false; amp=0.1), Δt, 20000)
    δ  = smooth(n, false; a=0.2, φ=0.4)
    a  = smooth(n, false; a=0.1, φ=2.3)

    # ---- RK8, the discrete tangent against differences of K steps ----
    cache = RAMStageCache(11, x0)
    xK    = rk8!(eq, copy(x0), Δt, K; cache=cache)
    v     = replay!(RK8(similar(x0), DiscreteMode(false)), tan, copy(δ), cache, false)
    ε     = 1e-7
    xp    = rk8!(eq, FTField(dofs(x0) .+ ε .* dofs(δ), false), Δt, K)
    xm    = rk8!(eq, FTField(dofs(x0) .- ε .* dofs(δ), false), Δt, K)
    @test rel(dofs(v), (dofs(xp) .- dofs(xm)) ./ (2ε)) < 1e-6

    # ---- RK8, the dot-product test of the discrete adjoint ----
    w = replay!(RK8(similar(x0), DiscreteMode(true)), adj, copy(a), cache, true)
    @test abs(dot(a, v) - dot(w, δ)) < 1e-12 * abs(dot(a, v))

    # ---- the coupled tangent is the discrete tangent ----
    Lc = flow(couple(eq, tan), RK8(couple(similar(x0), similar(x0))), TimeStepConstant(Δt))
    z  = Lc(couple(copy(x0), copy(δ)), (0, K * Δt))
    @test rel(dofs(z[2]), dofs(v)) < 1e-14
    @test rel(dofs(z[1]), dofs(xK)) < 1e-14

    # ---- CB3R2R3e with splitexim, close to RK8, and its discrete adjoint ----
    ϕ  = flow(splitexim(eq)..., CB3R2R3e(similar(x0)), TimeStepConstant(Δt))
    @test rel(dofs(ϕ(copy(x0), (0, 10Δt))), dofs(rk8!(eq, copy(x0), Δt, 10))) < 1e-7
    c4 = RAMStageCache(4, x0)
    ϕ(copy(x0), (0, K * Δt), c4)
    mt = CB3R2R3e(similar(x0), DiscreteMode(false))
    ma = CB3R2R3e(similar(x0), DiscreteMode(true))
    ψ  = flow(splitexim(tan)..., mt, TimeStepFromCache())
    ψa = flow(splitexim(adj)..., ma, TimeStepFromCache())
    v4 = ψ(copy(δ), c4)
    w4 = ψa(copy(a), c4)
    @test abs(dot(a, v4) - dot(w4, δ)) < 1e-12 * abs(dot(a, v4))

    # ---- RK4 with the full right-hand side, CB4R3R4 and CNRK2 with splitexim ----
    x10 = rk8!(eq, copy(x0), Δt, 10)
    ϕr4 = flow(eq, RK4(similar(x0)), TimeStepConstant(Δt))
    ϕb4 = flow(splitexim(eq)..., CB4R3R4(similar(x0)), TimeStepConstant(Δt))
    ϕc2 = flow(splitexim(eq)..., CNRK2(similar(x0)), TimeStepConstant(Δt))
    @test rel(dofs(ϕr4(copy(x0), (0, 10Δt))), dofs(x10)) < 1e-10
    @test rel(dofs(ϕb4(copy(x0), (0, 10Δt))), dofs(x10)) < 1e-8
    @test rel(dofs(ϕc2(copy(x0), (0, 10Δt))), dofs(x10)) < 1e-6
end

@testset "allocations                             " begin
    n  = 16
    ν  = (2π / 22)^2
    Δt = 0.005 * ν
    for ISODD in (false, true), dealias in (:none, :pad)
        eq  = ForwardEquation(n, ν, ISODD; dealias=dealias)
        tan = LinearisedEquation(n, ν, ISODD, TangentMode(); dealias=dealias)
        adj = LinearisedEquation(n, ν, ISODD, AdjointMode(); dealias=dealias)
        U   = smooth(n, ISODD)
        V   = smooth(n, ISODD; a=0.2)

        # ---- the right-hand sides ----
        @test allocs_rhs(eq, U, similar(U)) == 0
        @test allocs_lin(tan, U, V, similar(U)) == 0
        @test allocs_lin(adj, U, V, similar(U)) == 0

        # ---- a step of RK8, of CB3R2R3e, and of RK8 with a coupled tangent ----
        rk8  = RK8(similar(U))
        cb3  = CB3R2R3e(similar(U))
        rk8c = RK8(couple(similar(U), similar(U)))
        sysc = Flows.System(couple(eq, tan), couple(nothing, nothing))
        @test allocs_step(rk8,  Flows.System(eq, nothing),       copy(U), Δt) == 0
        @test allocs_step(cb3,  Flows.System(splitexim(eq)...),  copy(U), Δt) == 0
        @test allocs_step(rk8c, sysc, couple(copy(U), copy(V)), Δt) == 0

        # ---- a step of RK4, of CB4R3R4 and of CNRK2 ----
        rk4 = RK4(similar(U))
        cb4 = CB4R3R4(similar(U))
        cn2 = CNRK2(similar(U))
        @test allocs_step(rk4, Flows.System(eq, nothing),      copy(U), Δt) == 0
        @test allocs_step(cb4, Flows.System(splitexim(eq)...), copy(U), Δt) == 0
        @test allocs_step(cn2, Flows.System(splitexim(eq)...), copy(U), Δt) == 0
    end
end

@testset "deepcopy                                " begin
    n = 16
    ν = (2π / 22)^2
    U = smooth(n, false)

    # ---- an FTField copy owns its data ----
    D = deepcopy(U)
    @test dofs(D) == dofs(U)
    D[1] = 100.0
    @test U[1] != 100.0
    Uo = smooth(n, true)
    Uo[WaveNumber(2)] = 0.3 + 0.4im
    @test deepcopy(Uo).data == Uo.data
    Z = couplecopy(2, U)
    Z[1][1] = 7.0
    @test U[1] != 7.0 && Z[2][1] != 7.0

    # ---- equations and flows, same option, same numbers, own buffers ----
    eq = ForwardEquation(n, ν, false; dealias=:pad)
    ec = deepcopy(eq)
    @test typeof(ec) == typeof(eq)
    @test dofs(ec(0.0, copy(U), similar(U))) == dofs(eq(0.0, copy(U), similar(U)))
    tan = LinearisedEquation(n, ν, false, TangentMode(); dealias=:pad)
    @test typeof(deepcopy(tan)) == typeof(tan)
    G  = flow(eq, RK8(similar(U)), TimeStepConstant(0.005ν))
    G2 = deepcopy(G)
    @test dofs(G2(copy(U), (0, 0.5ν))) == dofs(G(copy(U), (0, 0.5ν)))
end
