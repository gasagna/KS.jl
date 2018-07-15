using Base.Test
using KS
using Flows

@testset "test adjoint identities                " begin
    # parameters
    ν        = (2π/100)^2
    Δt       = 0.01*ν
    T        = 100*ν
    rkmethod = :CB4_4R3R
    ISODD    = false
    n        = 100

    # seed rng
    srand(0)

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # integrator
    ϕ = integrator(splitexim(F)..., 
                   Scheme(rkmethod, FTField(n, ISODD)), TimeStepConstant(Δt))

    # monitor to store the forward solution
    sol = Monitor(FTField(n, ISODD), copy)

    # random initial condition
    U = FTField(n, ISODD, k->exp(2π*im*rand())/k);

    # now store forward solution
    ϕ(U, (0, T), reset!(sol));

    # rhs
    L  = LinearisedEquation(n, ν, ISODD, KS.TangentMode(), sol)
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode(), sol)

    # will use these random field for testing
    V  = FTField(n, ISODD, k->exp(2π*im*rand())/k)
    V⁺ = FTField(n, ISODD, k->exp(2π*im*rand())/k);

    # test that ⟨V, ℒ⁺[V⁺]⟩ + ⟨V⁺, ℒ[V]⟩ = 0 to machine accuracy
    v1 = dot(V,  L⁺(0, U, U, V⁺, FTField(n, ISODD)))
    v2 = dot(V⁺, L( 0, U, U, V,  FTField(n, ISODD)))

    @test abs(v1+v2) < 2*eps(v1)*n

    # test that the difference ⟨V, ψ⁺[V⁺]⟩ - ⟨V⁺, ψ[V]⟩
    # goes to zero as we refine the time step, since
    # the tangent and adjoint integrators are not
    # discretely consistent with each other.
    for Δt in ν.*[1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
        # integrators
        ψ  = integrator(splitexim(L)..., 
                       Scheme(rkmethod, FTField(n, ISODD)), TimeStepConstant(Δt))
        ψ⁺ = integrator(splitexim(L⁺)..., 
                       Scheme(rkmethod, FTField(n, ISODD)), TimeStepConstant(Δt))
        v1 = dot(V, ψ⁺(copy(V⁺), (100*ν, 99*ν)))
        v2 = dot(V⁺, ψ(copy(V),  (99*ν, 100*ν)))
        @test abs(v1-v2)/Δt^4 < 10^9
    end
end

@testset "perturbations to the trivial base      " begin

    # parameters
    ν        = (2π/10)^2
    Δt       = 0.01*ν
    T        = 1*ν
    rkmethod = :CB4_4R3R
    ISODD    = false
    n        = 15

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # integrator
    ϕ = integrator(splitexim(F)..., 
                   Scheme(rkmethod, FTField(n, ISODD)), TimeStepConstant(Δt))

    # monitor to store the forward solution
    sol = Monitor(FTField(n, ISODD), copy)

    # random initial condition
    U = FTField(n, ISODD, k->0)

    # now store forward solution
    ϕ(U, (0, T), reset!(sol))

    # define linearised operator
    L  = LinearisedEquation(n, ν, ISODD, KS.TangentMode(), sol)
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode(), sol)

    # and the integrators
    ψ  = integrator(splitexim(L)...,
                    Scheme(rkmethod, FTField(n, ISODD)), TimeStepConstant(Δt))
    ψ⁺ = integrator(splitexim(L⁺)...,
                    Scheme(rkmethod, FTField(n, ISODD)), TimeStepConstant(Δt))

    # will use this field
    V = FTField(n, ISODD)

    for k = wavenumbers(n)
        # TANGENT
        # set initial condition to be proportional to cos(2π/L⋅k⋅x)
        V .= 0; V[k] = 1
    
        # propagate tangent problem
        ψ(V, (0, T))

        # check match with exact solution
        @test abs(V[k] - exp( T*(k^2 - ν*k^4) )) < 1e-10
        
        # ADJOINT
        # set terminal condition to be proportional to cos(2π/L⋅k⋅x)
        V .= 0; V[k] = 1
    
        # propagate adjoint problem
        ψ⁺(V, (T, 0))

        # check match with exact solution
        @test abs(V[k] - exp(T*(k^2 - ν*k^4) )) < 1e-10
    end
end