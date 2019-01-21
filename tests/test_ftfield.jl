@testset "constructors and indexing              " begin
    @testset "pass function to FTField           " begin
        U = FTField(2, true)
        @test U[WaveNumber(1)] == 0
        @test U[WaveNumber(2)] == 0

        V = FTField(2, true, k->k+im*k)
        @test V[WaveNumber(1)] == 1+im*1
        @test V[WaveNumber(2)] == 2+im*2
    end
    @testset "full space                         " begin
        for U in [FTField([1.0, 2.0, 3.0, 4.0], false),
                  FTField([1.0+2.0*im, 3.0+4.0*im], false)]

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
        U = FTField(2, false)
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
        for U in [FTField([1.0, 2.0, 3.0, 4.0], true),
                  FTField([1.0*im, 2.0*im, 3.0*im, 4.0*im], true)]

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
        U = FTField(4, true)
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
        U = FTField(4, true)
        V = similar(U)
        @test length(V) == 4
        @test size(V) == (4, )
        @test V[WaveNumber(1)] == 0.0 + 0.0*im
        @test V[WaveNumber(2)] == 0.0 + 0.0*im
        @test V[WaveNumber(3)] == 0.0 + 0.0*im
        @test V[WaveNumber(4)] == 0.0 + 0.0*im

        V = copy(FTField([1, 2], true))
        @test length(V) == 2
        @test size(V) == (2, )
        @test V[WaveNumber(1)] == 0.0+1.0*im
        @test V[WaveNumber(2)] == 0.0+2.0*im
        @test V[1] == 1.0
        @test V[2] == 2.0
    end
    @testset "full space                           " begin
        U = FTField(2, false)
        V = similar(U)
        @test length(V) == 4
        @test size(V) == (4, )
        @test V[WaveNumber(1)] == 0.0 + 0.0*im
        @test V[WaveNumber(2)] == 0.0 + 0.0*im

        V = copy(FTField([1.0+2.0*im, 3.0+4.0*im], false))
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

@testset "grow                                   " begin
    U = FTField(3, false)
    U[WaveNumber(1)] = 2.0 + im/1
    U[WaveNumber(2)] = 3.0 + im/2
    U[WaveNumber(3)] = 4.0 + im/3
    V = grow(U, 5)
    @test typeof(V) == FTField{5, false, Float64, Array{Complex{Float64}, 1}, Ptr{Float64}}
    @test V[WaveNumber(1)] == 2.0 + im/1
    @test V[WaveNumber(2)] == 3.0 + im/2
    @test V[WaveNumber(3)] == 4.0 + im/3
    @test V[WaveNumber(4)] == 0
    @test V[WaveNumber(5)] == 0
end

@testset "derivative                             " begin
    U = FTField(3, false); 
    U[WaveNumber(1)] = 2.0 + im/1
    U[WaveNumber(2)] = 3.0 + im/2
    U[WaveNumber(3)] = 4.0 + im/3

    V = ddx!(similar(U), U)
    @test V == [-1, 2, -1, 6, -1, 12]

    # check differentiation matrix
    D = diffmat(3, false, zeros(6, 6))
    
    # copy U to a vector
    U_vec = zeros(length(U))
    U_vec .= U
    # then check with using ddx!
    @test V == D*U_vec
end

@testset "shifts identities                      " begin
    Random.seed!(0)
    n, ISODD = 30, false
    U = FTField(n, ISODD, k->exp(2π*im*rand())/k)
    V = FTField(n, ISODD, k->exp(2π*im*rand())/k)

    shouldbeU = shift!(shift!(copy(U), 1), -1)
    @test sqrt(dotdiff(U, shouldbeU)) < 3e-16

    @test abs(dot(U, shift!(copy(V), 1)) - dot(shift!(copy(U), -1), V)) < 3e-16
end

@testset "mindiff                                " begin
    U = FTField([1.0, 2.0, 3.0, 4.0], false)
    V = shift!(copy(U), 4*2π/20)
    dmin, (smin, ) = mindotdiff(U, V)
    @test dmin < 1e-16
    @test smin == 4*2π/20
    @test isapprox(U[WaveNumber(1)], 1.0 + 2.0*im)
    @test isapprox(U[WaveNumber(2)], 3.0 + 4.0*im)
end

@testset "dot and norm                           " begin
    # we divide by two to be consistent with the fact that 
    # our degrees of freedom do not include the `negative`
    # wavenumbers. This becomes consistent with the dot
    # product of the `linearised` array of degrees of freedom.
    # cos(x)*cos(x)
    U = FTField(2, false); U[WaveNumber(1)] = 0.5
    @test dot(U, U) == 0.5/2
    @test norm(U) == sqrt(0.5/2)

    # cos(x)*sin(x)
    U = FTField(2, false); U[WaveNumber(1)] =  0.5
    V = FTField(2, false); V[WaveNumber(1)] = -0.5*im
    @test dot(U, V) == 0.0

    # sin(2x)*sin(2x)
    U = FTField(2, true); U[WaveNumber(1)] = -0.5*im
    V = FTField(2, true); V[WaveNumber(1)] = -0.5*im
    @test dot(U, U) == 0.5/2
    @test norm(U) == sqrt(0.5/2)
end

@testset "broadcast                              " begin
    @testset "full space                         " begin
        U = FTField(2, false)
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
        U = FTField(4, true)
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
        U = FTField([1, 2, 3, 4], false)

        KS._set_symmetry!(U)
        @test U[WaveNumber(1)] == 1+2*im
        @test U[WaveNumber(2)] == 3+4*im
    end

    @testset "odd space                         " begin
        U  = FTField(2, true)

        # break the invariance
        U[WaveNumber(1)] = 1+2*im
        U[WaveNumber(2)] = 3+4*im

        KS._set_symmetry!(U)
        @test U[WaveNumber(1)] == 0+2*im
        @test U[WaveNumber(2)] == 0+4*im
    end
end

@testset "deepcopy                               " begin
    U = FTField([1, 2, 3, 4], false)
    V = deepcopy(U)
    V[1] = 5
    @test real(V.data[2]) == 5
end
