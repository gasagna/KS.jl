@testset "variational                            " begin
    n, L = 16, 22
    @testset "Field" begin
        # constructors
        U = VarField(n, L)
        V = similar(U)
        W = VarField(Field(n, L), Field(n, L))

        # accessors
        state(V) .= 1
        prime(V) .= 2
        state(W) .= 3
        prime(W) .= 4

        # broadcast
        U .= V .+ W

        # FIXME data
        @test all(state(U) .== 4) == true
        @test all(prime(U) .== 6) == true

        # indexing
        @test U[1]       == Dual(4, 6)
        @test U[2*(n+1)] == Dual(4, 6)

        @test_throws BoundsError U[0]
        @test_throws BoundsError U[2*(n+1) + 1]
    end
    @testset "FTField" begin
        # constructors
        U = VarFTField(n, L, false)
        V = similar(U)
        W = VarFTField(FTField(n, L, false), FTField(n, L, false))

        # broadcast on accessors
        state(V) .= 1
        prime(V) .= 2
        state(W) .= 3
        prime(W) .= 4

        @test all(state(V) .== 1)
        @test all(prime(V) .== 2)
        @test all(state(W) .== 3)
        @test all(prime(W) .== 4)

        # broadcast over degrees of freedom
        U .= V .+ W
        @test all(state(U) .== 4)
        @test all(prime(U) .== 6)

        # indexing over wave numbers
        U[WaveNumber(1)] = 1.0 + 2.0*im
        @test U[WaveNumber(1)] == Dual(1.0 + 2.0*im, 0.0 +0.0*im)

        U[WaveNumber(1)] = Dual(1.0 + 2.0*im, 3.0 + 4.0*im)
        @test U[WaveNumber(1)] == Dual(1.0 + 2.0*im, 3.0 + 4.0*im)


        # indexing over degrees of freedom
        U[1] = 1.0
        @test U[1] == Dual(1.0, 0.0)

        U[1] = Dual(1.0, 2.0)
        U[2] = Dual(3.0, 4.0)
        @test U[1] == Dual(1.0, 2.0)
        @test U[2] == Dual(3.0, 4.0)
        @test U[WaveNumber(1)] == Dual(1.0 + 3.0*im, 2.0 + 4.0*im)
    end

    @testset "symmetry                           " begin
        U = VarFTField(n, L, true)
        state(U)[WaveNumber(1)] = 1.0 + 1.0*im
        prime(U)[WaveNumber(1)] = 2.0 + 2.0*im
        KS._set_symmetry!(U)
        @test state(U)[WaveNumber(1)] == 0.0 + 1.0*im
        @test prime(U)[WaveNumber(1)] == 0.0 + 2.0*im
    end
end
