using Base.Test
using ForwardDiff
using KS

const ν = 1/(39/2π)^2 # 1/L̃^2

# test basic call works
let 
    ks = KSEq(ν, 10)
    x = zeros(ndofs(ks))
    ẋ = similar(x)
    # zero is an equilibrium
    @test ks(ẋ, x) == zeros(ndofs(ks))
end

# provide gains
let 
    ks = KSEq(ν, 10, randn(10))
    x = zeros(ndofs(ks))
    ẋ = similar(x)
    # zero is an equilibrium
    @test ks(ẋ, x) == zeros(ndofs(ks))
end

# test ndofs
let 
    ks = KSEq(ν, 3)
    @test ndofs(ks) == 3
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
    for N = 1:5
        ks = KSEq(ν, N, randn(N))
        
        # fix a random point
        x = randn(N)

        # define function
        ksfun(x) = ks(similar(x), x)
        
        # analytic jacobian
        J_ex = KS.∂ₓ(ks)(zeros(N, N), x)

        # ad jacobian
        J_ad = ForwardDiff.jacobian!(zeros(N, N), ksfun, x)
        @test J_ad ≈ J_ex
    end
end

# test parameter jacobian using 
let 
    srand(0)
    for N = 3:3
        # fix a point
        x = randn(N)
        v = randn(N)
        
        # define function
        ksfun(v) = KSEq(ν, N, v)(similar(v), x)

        # analytic jacobian
        ks = KSEq(ν, N, v)
        J_ex = ∂ᵥ(ks)(zeros(N, N), x)

        # ad jacobian
        J_ad = ForwardDiff.jacobian!(zeros(N, N), ksfun, v)
        @test J_ad ≈ J_ex
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


let
    x = Float64[1, 2, 3]
    ks = KSEq(1, length(x))
    ϕ = KineticEnergyDensity(ks)

    @test     ϕ(x) == 1^2 + 2^2 + 3^2
    @test ∂ₓ(ϕ)(similar(x), x) == [2, 4, 6]
    @test ∂ᵥ(ϕ)(similar(x), x) == [0, 0, 0]
end