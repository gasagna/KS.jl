using Base.Test
using KS
using Flows

@testset "test adjoint identities                " begin
    # parameters
    ν        = (2π/100)^2
    Δt       = 0.1*ν
    T        = 100*ν
    ISODD    = false
    n        = 100

    # seed rng
    srand(0)

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # flow
    ϕ = flow(splitexim(F)..., 
             CB3R2R3e(FTField(n, ISODD), :NORMAL), TimeStepConstant(Δt))

    # define the stage cache
    cache = RAMStageCache(4, FTField(n, ISODD))

    # random initial condition
    U = FTField(n, ISODD, k->exp(2π*im*rand())/k)

    # proceed forward, then store forward solution for a small bit
    ϕ(U, (0, T)); ϕ(U, (0, 100*ν), reset!(cache))

    # rhs
    L  = LinearisedEquation(n, ν, ISODD, KS.TangentMode())
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode())

    # will use these random field for testing
    V  = FTField(n, ISODD, k->exp(2π*im*rand())/k)
    V⁺ = FTField(n, ISODD, k->exp(2π*im*rand())/k);

    # test that ⟨V, ℒ⁺[V⁺]⟩ = ⟨V⁺, ℒ[V]⟩ to machine accuracy
    v1 = dot(V,  L⁺(0, U, V⁺, FTField(n, ISODD)))
    v2 = dot(V⁺, L( 0, U, V,  FTField(n, ISODD)))

    @test abs(v1-v2)/abs(v1) < 1e-14

    # test that the difference ⟨V, ψ⁺[V⁺]⟩ - ⟨V⁺, ψ[V]⟩
    # goes to zero within machine accuracy
    ψ  = flow(splitexim(L)..., 
                   CB3R2R3e(FTField(n, ISODD), :TAN), TimeStepFromCache())
    ψ⁺ = flow(splitexim(L⁺)..., 
                   CB3R2R3e(FTField(n, ISODD), :ADJ), TimeStepFromCache())
    v1 = dot(V, ψ⁺(copy(V⁺),cache))
    v2 = dot(V⁺, ψ(copy(V), cache))
    
    @test abs(v1-v2)/abs(v1) < 5e-14
end

@testset "perturbations to the trivial base      " begin

    # parameters
    ν        = (2π/10)^2
    Δt       = 0.01*ν
    T        = 1*ν
    ISODD    = false
    n        = 15

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # flow
    ϕ = flow(splitexim(F)..., 
                   CB3R2R3e(FTField(n, ISODD), :NORMAL), TimeStepConstant(Δt))

    # define the stage cache
    cache = RAMStageCache(4, FTField(n, ISODD))

    # zero initial condition
    U = FTField(n, ISODD, k->0)

    # now store forward solution
    ϕ(U, (0, T), reset!(cache))

    # define linearised operator
    L  = LinearisedEquation(n, ν, ISODD, KS.TangentMode())
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode())

    # and the flows
    ψ  = flow(splitexim(L)...,
                    CB3R2R3e(FTField(n, ISODD), :TAN), TimeStepFromCache())
    ψ⁺ = flow(splitexim(L⁺)...,
                    CB3R2R3e(FTField(n, ISODD), :ADJ), TimeStepFromCache())

    # will use this field
    V = FTField(n, ISODD)

    for k = wavenumbers(n)
        # TANGENT
        # set initial condition to be proportional to cos(2π/L⋅k⋅x)
        V .= 0; V[k] = 1
    
        # propagate tangent problem
        ψ(V, cache)

        # check match with exact solution
        @test abs(V[k] - exp( T*(k^2 - ν*k^4) )) < 1e-7
        
        # ADJOINT
        # set terminal condition to be proportional to cos(2π/L⋅k⋅x)
        V .= 0; V[k] = 1
    
        # propagate adjoint problem
        ψ⁺(V, cache)

        # check match with exact solution
        @test abs(V[k] - exp(T*(k^2 - ν*k^4) )) < 1e-7
    end
end

@testset "perturbations to nonlinear trajectory  " begin
    # parameters
    ν        = (2π/22)^2
    Δt       = 1e-1*ν
    T        = 1*ν
    ISODD    = false
    n        = 22

    # seed rng
    srand(0)

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # flow
    ϕ = flow(splitexim(F)..., 
                   CB3R2R3e(FTField(n, ISODD), :NORMAL), TimeStepConstant(Δt))

    # define the stage cache
    cache = RAMStageCache(4, FTField(n, ISODD))

    # land on attractor
    U₀ = ϕ(FTField(n, ISODD, k->exp(2π*im*rand())/k), (0, 100*ν))

    # rhs
    L  = LinearisedEquation(n, ν, ISODD, KS.TangentMode())

    # flows
    ψ  = flow(splitexim(L)...,
                    CB3R2R3e(FTField(n, ISODD), :TAN), TimeStepFromCache())

    # copy initial condition
    _U₀ = copy(U₀)

    # and get final point
    Uₜ0 = ϕ(copy(_U₀), (0, T), reset!(cache))

    # copy initial condition and perturb it
    _U₀ = copy(U₀); _U₀[1] += 1e-6

    # and get final point
    Uₜ1 = ϕ(copy(_U₀), (0, T))

    # Now use the direct method, i.e. calculate the direction derivative along a particular perturbation
    V  = FTField(n, ISODD, k->0); V[1] = 1

    # integrate direct equations, then apply shift
    ψ(V, cache)

    # cannot demand too much from finite differences
    @test abs(V[1] - 1e6*(Uₜ1 - Uₜ0)[1]) < 3e-6
    @test abs(V[2] - 1e6*(Uₜ1 - Uₜ0)[2]) < 3e-6
end
