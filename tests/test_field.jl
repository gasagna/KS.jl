@testset "Field                                  " begin
    # indexing
    u = Field(2, 1)
    @test eltype(u) == Float64
    @test length(u) == 6
    @test size(u) == (6, )

    @test_throws BoundsError u[0]
          u[1] =  2
    @test u[1] == 2
          u[2] =  3
    @test u[2] == 3
    @test_throws BoundsError u[7]

    # broadcast
    u   = Field(2, 1)
    u .+= 1
    @test u[1] == u[2] == u[3] == u[4] == u[5] == u[6] == 1

    u .= u .+ 1
    @test u[1] == u[2] == u[3] == u[4] == u[5] == u[6] == 2

    u .*= u
    @test u[1] == u[2] == u[3] == u[4] == u[5] == u[6] == 4

    u .= [1, 1, 1, 1, 1, 1]
    @test u[1] == u[2] == u[3] == u[4] == u[5] == u[6] == 1

    # mesh
    x = mesh(Field(1, 2π))
    @test abs(x[1] - 0)    < 1e-15
    @test abs(x[2] - π/2)  < 1e-15
    @test abs(x[3] - π)    < 1e-15
    @test abs(x[4] - 3π/2) < 1e-15
end