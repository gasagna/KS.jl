@testset "constructors and indexing              " begin
    @testset "full space                         " begin
        for U in [FTField([1.0, 2.0, 3.0, 4.0], 1, false),
                   FTField([1.0+2.0*im, 3.0+4.0*im], 1, false)]

            @test eltype(U) == Float64
            @test length(U) == 4
            @test size(U) == (4, )

            # indexing over degrees of freedom
            for i = 1:4
                @test U[i] ==  i
                      U[i] =  2i
                @test U[i] == 2i
            end
            @test_throws BoundsError U[0]
            @test_throws BoundsError U[5]
        end

        # other constructor and wave numbers
        U = FTField(2, 1, false)
        @test eltype(U) == Float64
        @test length(U) == 4
        @test size(U) == (4, )

        U[WaveNumber(1)]        = 1.0+2.0*im
        U[WaveNumber(2)]        = 3.0+4.0*im
        @test U[WaveNumber(1)] == 1.0+2.0*im
        @test U[WaveNumber(2)] == 3.0+4.0*im
        @test_throws BoundsError U[WaveNumber(0)]
        @test_throws BoundsError U[WaveNumber(3)]
    end
    @testset "odd space                           " begin
        for U in [FTField([1.0, 2.0, 3.0, 4.0], 1, true),
                   FTField([1.0*im, 2.0*im, 3.0*im, 4.0*im], 1, true)]

            @test eltype(U) == Float64
            @test length(U) == 4
            @test size(U) == (4, )

            # indexing over degrees of freedom
            for i = 1:4
                @test U[i] ==  i
                      U[i] =  2i
                @test U[i] == 2i
            end
            @test_throws BoundsError U[0]
            @test_throws BoundsError U[5]
        end

        # other constructor and wave numbers
        U = FTField(4, 1, true)
        @test eltype(U) == Float64
        @test length(U) == 4
        @test size(U) == (4, )

        # note we can break the odd invariance!
        U[WaveNumber(1)]        = 1.0+2.0*im
        @test U[WaveNumber(1)] == 1.0+2.0*im
        @test_throws BoundsError U[WaveNumber(0)]
        @test_throws BoundsError U[WaveNumber(5)]
    end
end

@testset "similar and copy                       " begin
    @testset "odd space                           " begin
        U = FTField(4, 1, true)
        V = similar(U)
        @test length(V) == 4
        @test size(V) == (4, )
        @test V[WaveNumber(1)] == 0.0 + 0.0*im
        @test V[WaveNumber(2)] == 0.0 + 0.0*im
        @test V[WaveNumber(3)] == 0.0 + 0.0*im
        @test V[WaveNumber(4)] == 0.0 + 0.0*im

        V = copy(FTField([1, 2], 1, true))
        @test length(V) == 2
        @test size(V) == (2, )
        @test V[WaveNumber(1)] == 0.0+1.0*im
        @test V[WaveNumber(2)] == 0.0+2.0*im
        @test V[1] == 1.0
        @test V[2] == 2.0
    end
    @testset "full space                           " begin
        U = FTField(2, 1, false)
        V = similar(U)
        @test length(V) == 4
        @test size(V) == (4, )
        @test V[WaveNumber(1)] == 0.0 + 0.0*im
        @test V[WaveNumber(2)] == 0.0 + 0.0*im

        V = copy(FTField([1.0+2.0*im, 3.0+4.0*im], 1, false))
        @test length(V) == 4
        @test size(V) == (4, )
        @test V[WaveNumber(1)] == 1.0+2.0*im
        @test V[WaveNumber(2)] == 3.0+4.0*im
        @test V[1] == 1.0
        @test V[2] == 2.0
        @test V[3] == 3.0
        @test V[4] == 4.0
    end
end

@testset "dot and norm                           " begin
    # cos(x)*cos(x)
    U = FTField(2, 1, false); U[WaveNumber(1)] = 0.5
    @test dot(U, U) == 0.5
    @test norm(U) == sqrt(0.5)

    # cos(x)*sin(x)
    U = FTField(2, 1, false); U[WaveNumber(1)] =  0.5
    V = FTField(2, 1, false); V[WaveNumber(1)] = -0.5*im
    @test dot(U, V) == 0.0

    # sin(2x)*sin(2x)
    U = FTField(2, 1, true); U[WaveNumber(1)] = -0.5*im
    V = FTField(2, 1, true); V[WaveNumber(1)] = -0.5*im
    @test dot(U, U) == 0.5
    @test norm(U) == sqrt(0.5)
end

@testset "broadcast                              " begin
    @testset "full space                         " begin
        U = FTField(2, 1, false)
        U .= [1, 2, 3, 4]
        @test U[1] == 1
        @test U[2] == 2
        @test U[3] == 3
        @test U[4] == 4
        U .= U.*U .+ 1
        @test U[1] == 2
        @test U[2] == 5
        @test U[3] == 10
        @test U[4] == 17
    end
    @testset "odd space                          " begin
        U = FTField(4, 1, true)
        U .= [1, 2, 3, 4]
        @test U[1] == 1
        @test U[2] == 2
        @test U[3] == 3
        @test U[4] == 4
        U .= U.*U .+ 1
        @test U[1] == 2
        @test U[2] == 5
        @test U[3] == 10
        @test U[4] == 17
    end
end

@testset "symmetry                               " begin
    @testset "full space                         " begin
        U = FTField([1, 2, 3, 4], 1, false)

        KS._set_symmetry!(U)
        @test U[WaveNumber(1)] == 1+2*im
        @test U[WaveNumber(2)] == 3+4*im
    end

    @testset "odd space                         " begin
        U  = FTField(2, 1, true)

        # break the invariance
        U[WaveNumber(1)] = 1+2*im
        U[WaveNumber(2)] = 3+4*im

        KS._set_symmetry!(U)
        @test U[WaveNumber(1)] == 0+2*im
        @test U[WaveNumber(2)] == 0+4*im
    end
end