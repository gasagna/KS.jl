using Base.Test
using ForwardDiff
using KS
using POF
using POF.DB

# used below
const ν = 1/(39/2π)^2 # 1/L̃^2

# Compute jacobian of function f(out, x) using finite differences.
function fdjac(f, Nout, x::AbstractVector, h::Float64=1e-7)
    # Note that f : R^Nin → R^Nout
    Nin = length(x)
    # preallocate output and temporary arrays
    out_p = zeros(Nout)
    out_m = zeros(Nout)
    J = zeros(Nout, Nin)
    for j = 1:Nin # for each parameter
        x[j] += h
        f(out_p, x)
        x[j] -= 2h
        f(out_m, x)
        x[j] += h
        for i = 1:Nout # for each output
            J[i, j] = (out_p[i] - out_m[i])/(2h)
        end
    end
    J
end

# test finite difference jacobian
let 
    # example function
    function foo(out, x) 
        @assert length(x) == length(out) == 3
        out[1] = x[1]*x[2]
        out[2] = x[2]*x[3]
        out[3] = x[3]*x[1]
        out
    end

    srand(0)
    x = rand(3)
    @test fdjac(foo, 3, x) ≈ [x[2]  x[1]  0   ;
                              0     x[3]  x[2];
                              x[3]  0     x[1]]
end

# test basic call works for no control 
let 
    ks = KSEq(ν, 10)
    x = zeros(ndofs(ks))
    ẋ = similar(x)
    # zero is an equilibrium
    @test ks(ẋ, x) == zeros(ndofs(ks))
    @test ndofs(ks) == 10
end

# constructors with point actuation
let 
    ks = KSEqPointControl(ν, 10, randn(10), π/2)
    x = zeros(ndofs(ks))
    ẋ = similar(x)
    # zero is an equilibrium
    @test ks(ẋ, x) == zeros(ndofs(ks))
    @test ndofs(ks) == 10
end

# constructors with distributed actuation
let 
    ks = KSEqDistributedControl(ν, 10, randn(10, 10))
    x = zeros(ndofs(ks))
    ẋ = similar(x)
    # zero is an equilibrium
    @test ks(ẋ, x) == zeros(ndofs(ks))
    @test ndofs(ks) == 10
end


# test reconstruct
let
    # single vector
    ks = KSEq(ν, 3)
    x = [5.0, -2.0, 4.0] 
    grid = linspace(0, 2π, 10)
    u = reconstruct!(ks, x, grid, similar(grid))
    # note the minus
    @test u ≈ -2*(5*sin(1*grid) - 
                 2*sin(2*grid) + 
                 4*sin(3*grid))

    # full matrix
    x = [5.0 -2.0 4.0;
         1.0 -1.0 2.0] 
    grid = linspace(0, 2π, 10)
    u = reconstruct(ks, x, grid)
    # note the minus
    @test vec(u[1, :]) ≈ -2*(5*sin(1*grid) -
                            2*sin(2*grid) +
                            4*sin(3*grid))
    @test vec(u[2, :]) ≈ -2*(1*sin(1*grid) -
                            1*sin(2*grid) +
                            2*sin(3*grid))
end

# test state jacobian
let 
    srand(0)
    for N = 1:32
        for ks in [KSEq(ν, N), 
                   KSEqPointControl(ν, N, randn(N), π*rand()),
                   KSEqDistributedControl(ν, N, randn(N, N))]
        
            # fix a random point
            x = randn(N)
            
            # analytic jacobian
            J_ex = KS.∂ₓ(ks)(zeros(N, N), x)

            # fd jacobian
            J_fd = fdjac(ks, N, x, 1e-7)
            @test J_fd ≈ J_ex
         end
    end
end

# test parameter jacobian for point actuation
let 
    srand(0)
    for N = 1:32
        # fix a point
        x = randn(N)
        v = randn(N)
        
        # define function that we want to differentiate
        fun(out, v) = KSEqPointControl(ν, N, v, π/2)(out, x)

        # define system
        ks = KSEqPointControl(ν, N, v, π/2)

        # analytic jacobian
        J_ex = ∂ᵥ(ks)(zeros(N, N), x)

        # fd jacobian, evaluated at v
        J_fd = fdjac(fun, N, v, 1e-6)
        @test maxabs(J_fd - J_ex) < 2e-6
    end
end

# test parameter jacobian for distributed actuation
let 
    srand(0)
    for N = 1:32
        # fix a point
        x = randn(N)
        V = randn(N, N)
        
        # define function that we want to differentiate
        fun(out, v) = KSEqDistributedControl(ν, N, reshape(v, N, N))(out, x)

        # define system
        ks = KSEqDistributedControl(ν, N, V)

        # analytic jacobian
        J_ex = ∂ᵥ(ks)(zeros(N, N*N), x)

        # fd jacobian, evaluated at v
        J_fd = fdjac(fun, N, V[:], 2e-6)
        @test maxabs(J_fd - J_ex) < 2e-6
    end
end

# test kinetic energy density
let 
    # 
    ks = KSEq(ν, 3)
    x = [1, 2, 3] 
    grid = linspace(0, 2π, 11)
    u = reconstruct!(ks, x, grid, similar(grid))
    # note the minus
    @test u ≈ -2*(1*sin(grid) + 2*sin(2*grid) +  3*sin(3*grid))
    # use composite trapezoidal rule
    𝒦 = KineticEnergyDensity(ks)
    @test 𝒦(x) ≈ 1/2*sum(u[2:end-1].^2)*grid[2]/2π
end

# test inner product, norm
let 
    ks = KSEq(ν, 3)
    𝒦 = KineticEnergyDensity(ks)
    x = [1, 2, 3] 
    y = [2, 3, 4]
    @test inner(ks, x, y) == inner(ks, y, x) 
    @test norm(ks, x)^2 ≈ 𝒦(x)

    grid = linspace(0, 2π, 11)
    u = reconstruct!(ks, x, grid, similar(grid))
    v = reconstruct!(ks, y, grid, similar(grid))
    # note the minus
    @test u ≈ -2*(1*sin(grid) + 2*sin(2*grid) +  3*sin(3*grid))
    @test v ≈ -2*(2*sin(grid) + 3*sin(2*grid) +  4*sin(3*grid))
    # use composite trapezoidal rule
    @test inner(ks, x, y) ≈ 1/2*sum( (u.*v)[2:end-1] )*grid[2]/2π
end    

# test dissipation and production work
let
    ks = KSEq((2π/39)^2, 32)
    𝒫 = ProductionDensity()
    𝒟 = DissipationDensity(ks)

    # on a periodic orbit the average production is the same
    # as the average dissipation
    orb = PeriodicOrbitFile("tmphyHYMD.orb")
    @test ptrapz(map(𝒫, trajectory(orb))) == ptrapz(map(𝒟, trajectory(orb)))
end

let
    x = Float64[1, 2, 3]
    ks = KSEq(1, length(x))
    ϕ = KineticEnergyDensity(ks)

    @test     ϕ(x) == 1^2 + 2^2 + 3^2
    @test ∂ₓ(ϕ)(similar(x), x) == [2, 4, 6]
    @test ∂ᵥ(ϕ)(similar(x), x) == [0, 0, 0]
end

# test symmetry
let
    x = [1, 1, 1, 1]
    @test issymmetric(x) == false

    x = [1e-7, 1, 1e-7, 1]
    @test issymmetric(x) == true

    x = [1e-7, 1, 1e-7, 1]
    @test issymmetric(x, 1e-8) == false

    x = [1e-7, 1, 1e-6, 1]
    @test issymmetric(x, 2e-7) == false
end

# test application of symmetries
let
    x = [1 1 1 1;
         2 2 2 2;
         3 3 3 3;
         4 4 4 4] 
    out = [-1 -1 -1 -1;
            2  2  2  2;
           -3 -3 -3 -3;
            4  4  4  4] 
    @test R⁺(PeriodicTrajectory(x)) == PeriodicTrajectory(out)
end

# make sure symmetric orbit is also a solution
let
    orb = PeriodicOrbitFile("tmphyHYMD.orb")
    @test issymmetric(orb) == false

    # number of Fourier modes
    Nₓ = 32

    # system
    f = KS.KSEq(ν, Nₓ)
    fₓ = KS.∂ₓ(f)

    # options
    opts = POFOptions(verbose=false, maxiter=3)

    # check this is an orbit
    @test isconverged(pof!(f, fₓ, 
            data(trajectory(orb)),     2π/period(orb); options=opts)) == true

    # symmetrize orbit
    @test isconverged(pof!(f, fₓ, 
            data(R⁺(trajectory(orb))), 2π/period(orb); options=opts)) == true
end

# make sure gradient works
let
    orb = PeriodicOrbitFile("tmphyHYMD.orb")

    # number of Fourier modes
    Nₓ = 32

    # system with no control
    f = KS.KSEqDistributedControl(ν, Nₓ, zeros(Nₓ, Nₓ))
    fₓ = KS.∂ₓ(f)
    fᵥ = KS.∂ᵥ(f)

    # cost
    ϕ = KS.KineticEnergyDensity(f)
    ϕₓ = KS.∂ₓ(ϕ)
    ϕᵥ = KS.∂ᵥ(ϕ)

    # options
    opts = POFOptions(verbose=false, maxiter=10)

    # check this is an orbit
    @test isconverged(pof!(f, fₓ, 
        data(trajectory(orb)), 2π/period(orb); options=opts)) == true

    # compute gradient
    ωᵥ, ϕ̄ᵥ = grad(f, fₓ, fᵥ, ϕ, ϕₓ, ϕᵥ, data(trajectory(orb)), 2π/period(orb), Nₓ^2)

    # perturb system in the gradient direction
    λ = 1e-7
    V = -λ*reshape(ϕ̄ᵥ, Nₓ, Nₓ) # note the transpose here
    fp = KS.KSEqDistributedControl(ν, Nₓ, V)
    fpₓ = KS.∂ₓ(fp)
    fpᵥ = KS.∂ᵥ(fp)

    # find perturbed orbit
    rp = pof!(fp, fpₓ, data(trajectory(orb)), 2π/period(orb); options=opts)
    @test isconverged(rp)

    # compute difference in cost
    ϕ̄p = POF.average(ϕ, PeriodicTrajectory(rp.X))
    ϕ̄  = POF.average(ϕ, trajectory(orb))

    # show, for the sake of checking
    # @printf "%.10f\n" (ϕ̄p - ϕ̄)/λ
    # @printf "%.10f\n" -norm(ϕ̄ᵥ)^2

    # test
    @test abs( ((ϕ̄p - ϕ̄)/λ) - (-norm(ϕ̄ᵥ)^2) ) < 1e-6
end