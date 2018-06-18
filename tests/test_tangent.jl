@testset "tangent code                           " begin
    @testset "compare with nonlinear simulations " begin
        # setup
        n, c, L, dt, ISODD = 31, 0, 50, 1, false

        # nonlinear system
        F = KSEq(n, L, c, false, :forward)
        scheme = Scheme(:CB4_4R3R, FTField(n, L, ISODD))
        _a, _b = imex(F)
        ϕ = integrator(_b, _a, scheme, dt)

        # augmented system
        F = KSEq(n, L, c, false, :tangent)
        scheme = Scheme(:CB4_4R3R, VarFTField(n, L, ISODD))
        _a, _b = imex(F)
        ϕψ = integrator(_b, _a, scheme, dt)

        # random initial condition
        U = FTField(n, L, ISODD); U .= 1e-2

        # propagate to attractor
        ϕ(U, (0, 1000))

        # relative perturbation of the real parts by ϵ when integrating by T
        ϵ = 1e-6
        T = 1

        for k = 1:n
            # do two nonlinear simulations
            Utmp = copy(U)
            ϕ(Utmp, (0, T))
            a = real(Utmp[WaveNumber(k)])

            Utmp = copy(U); Utmp[WaveNumber(k)] += ϵ*abs(real(U[WaveNumber(k)]))
            ϕ(Utmp, (0, T))
            b = real(Utmp[WaveNumber(k)])

            # do one linearised simulation
            Utmp = FTField(n, L, ISODD); Utmp[WaveNumber(k)] += 1
            ϕψ(VarFTField(copy(U), Utmp), (0, T))
            c = real(Utmp[WaveNumber(k)])

            val = abs((b-a)/(ϵ*abs(real(U[WaveNumber(k)]))) - c)/abs(c)
            # @printf "%0.4d - %.3e\n" k val
        end
    end

    @testset "compare with analytical solution" begin
        # setup
        n, c, L, dt, ISODD = 21, 0, 50, 0.1, false

        # augmented system
        F = KSEq(n, L, c, false, :tangent)
        scheme = Scheme(:CB4_4R3R, VarFTField(n, L, ISODD))
        _a, _b = imex(F)
        ϕψ = integrator(_b, _a, scheme, dt)

        # zero initial condition
        U = FTField(n, L, ISODD)

        # integration horizon
        T = 1

        # perturb real part
        for k = 1:n
            V = FTField(n, L, ISODD); V[WaveNumber(k)] = 1
            ϕψ(VarFTField(copy(U), V), (0, T))
            actual = real(V[WaveNumber(k)])
            qk = 2π*k/L
            exact = exp((qk^2 - qk^4)*T)

            @test abs(actual - exact) < 4e-7
       end
    end
end