using Base.Test
using ForwardDiff
using KS

const ν = 1/(38.5/2π)^2 # 1/L̃^2

# test basic call works
let 
    ks! = KSEq(ν, 3)
    x = rand(ndofs(ks!))
    ẋ = similar(x)
    ks!(ẋ, x)
end

# test ndofs
let 
    ks! = KSEq(ν, 3)
    @test ndofs(ks!) == 3
end

# test reconstruct
let
    # single vector
    ks! = KSEq(ν, 3)
    x = [5.0, -2.0, 4.0] 
    grid = linspace(0, 2π, 10)
    u = reconstruct!(ks!, x, grid, similar(grid))
    @test u ≈ 2*(5*sin(1*grid) - 
                 2*sin(2*grid) + 
                 4*sin(3*grid))

    # full matrix
    x = [5.0 -2.0 4.0;
         1.0 -1.0 2.0] 
    grid = linspace(0, 2π, 10)
    u = reconstruct(ks!, x, grid)
    @test vec(u[1, :]) ≈ 2*(5*sin(1*grid) -
                            2*sin(2*grid) +
                            4*sin(3*grid))
    @test vec(u[2, :]) ≈ 2*(1*sin(1*grid) -
                            1*sin(2*grid) +
                            2*sin(3*grid))
end

# test jacobian
let 
    srand(0)
    for N = 1:16
        x = randn(N)
        ks! = KSEq(ν, N)
        ks(x) = ks!(similar(x), x)
        J_ex = KS.jacobian(ks!)(zeros(N, N), x)
        J_ad = ForwardDiff.jacobian!(zeros(N, N), ks, x)
        @test J_ad ≈ J_ex
    end
end

# test kinetic energy density
let 
    # 
    ks! = KSEq(ν, 3)
    x = [1, 2, 3] 
    grid = linspace(0, 2π, 11)
    u = reconstruct!(ks!, x, grid, similar(grid))
    @test u ≈ 2*(1*sin(grid) + 2*sin(2*grid) +  3*sin(3*grid))
    # use composite trapezoidal rule
    @test 𝒦(ks!, x) ≈ sum(u[2:end-1].^2)*grid[2]/2π
end

# test inner product, norm
let 
    ks! = KSEq(ν, 3)
    x = [1, 2, 3] 
    y = [2, 3, 4]
    @test inner(ks!, x, y) == inner(ks!, y, x) 
    @test norm(ks!, x)^2 ≈ 𝒦(ks!, x)

    grid = linspace(0, 2π, 11)
    u = reconstruct!(ks!, x, grid, similar(grid))
    v = reconstruct!(ks!, y, grid, similar(grid))
    @test u ≈ 2*(1*sin(grid) + 2*sin(2*grid) +  3*sin(3*grid))
    @test v ≈ 2*(2*sin(grid) + 3*sin(2*grid) +  4*sin(3*grid))
    # use composite trapezoidal rule
    @test inner(ks!, x, y) ≈ sum( (u.*v)[2:end-1] )*grid[2]/2π
end    
