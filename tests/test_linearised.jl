using Base.Test
using KS
using DualNumbers
using Flows

@testset "perturbations to u(x) = 0 decays exponentially" begin

    # parameters
    Δt       = 1e-2
    Δt       = 1e-2
    T        = 1
    rkmethod = :CB4_4R3R
    ISODD    = false
    n        = 15
    L        = 10
    c        = 0.0

    # system right hand side
    f = KSEq(n, L, c, ISODD)

    # integrator
    ϕ = integrator(splitexim(f)..., Scheme(rkmethod, FTField(n, L, ISODD)), Δt)

    # monitor to store the forward solution
    sol = Monitor(FTField(n, L, ISODD), copy)

    # random initial condition
    U = FTField(n, L, ISODD);

    # store forward solution
    ϕ(U, (0, T), reset!(sol));

    # rhs
    lf = LinearisedKSEq(n, L, c, ISODD, sol)

    # integrator
    ψ = integrator(splitexim(lf)..., Scheme(rkmethod, FTField(n, L, ISODD)), Δt)

    # will use this field
    V = FTField(n, L, ISODD)

    for k = wavenumbers(n)
        # set initial condition to be proportional to cos(2π/L⋅k⋅x)
        V .= 0; V[k] = 1
    
        # propagate tangent problem
        ψ(V, (0, T))

        # check match with exact solution
        @test abs(V[k] - exp( T*((2π/L*k)^2 - (2π/L*k)^4) )) < 1e-10
    end
end