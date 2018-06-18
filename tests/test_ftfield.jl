using Base.Test
using KS

@testset "constructors and indexing              " begin
    @testset "full space                         " begin
        for uk in [FTField([1.0, 2.0, 3.0, 4.0], 1, false),
                   FTField([1.0+2.0*im, 3.0+4.0*im], 1, false)]

            @test eltype(uk) == Float64
            @test length(uk) == 4
            @test size(uk) == (4, )

            # indexing over degrees of freedom
            for i = 1:4
                @test uk[i] ==  i
                      uk[i] =  2i
                @test uk[i] == 2i
            end
            @test_throws BoundsError uk[0]
            @test_throws BoundsError uk[5]
        end

        # other constructor and wave numbers
        uk = FTField(2, 1, false)
        @test eltype(uk) == Float64
        @test length(uk) == 4
        @test size(uk) == (4, )

        uk[WaveNumber(1)]        = 1.0+2.0*im
        uk[WaveNumber(2)]        = 3.0+4.0*im
        @test uk[WaveNumber(1)] == 1.0+2.0*im
        @test uk[WaveNumber(2)] == 3.0+4.0*im
        @test_throws BoundsError uk[WaveNumber(0)]
        @test_throws BoundsError uk[WaveNumber(3)]
    end
    @testset "odd space                           " begin
        for uk in [FTField([1.0, 2.0, 3.0, 4.0], 1, true),
                   FTField([1.0*im, 2.0*im, 3.0*im, 4.0*im], 1, true)]

            @test eltype(uk) == Float64
            @test length(uk) == 4
            @test size(uk) == (4, )

            # indexing over degrees of freedom
            for i = 1:4
                @test uk[i] ==  i
                      uk[i] =  2i
                @test uk[i] == 2i
            end
            @test_throws BoundsError uk[0]
            @test_throws BoundsError uk[5]
        end

        # other constructor and wave numbers
        uk = FTField(4, 1, true)
        @test eltype(uk) == Float64
        @test length(uk) == 4
        @test size(uk) == (4, )

        # note we can break the odd invariance!
        uk[WaveNumber(1)]        = 1.0+2.0*im
        @test uk[WaveNumber(1)] == 1.0+2.0*im
        @test_throws BoundsError uk[WaveNumber(0)]
        @test_throws BoundsError uk[WaveNumber(5)]
    end
end

@testset "similar and copy                       " begin
    @testset "odd space                           " begin
        uk = FTField(4, 1, true)
        vk = similar(uk)
        @test length(vk) == 4
        @test size(vk) == (4, )
        @test vk[WaveNumber(1)] == 0.0 + 0.0*im
        @test vk[WaveNumber(2)] == 0.0 + 0.0*im
        @test vk[WaveNumber(3)] == 0.0 + 0.0*im
        @test vk[WaveNumber(4)] == 0.0 + 0.0*im

        vk = copy(FTField([1, 2], 1, true))
        @test length(vk) == 2
        @test size(vk) == (2, )
        @test vk[WaveNumber(1)] == 0.0+1.0*im
        @test vk[WaveNumber(2)] == 0.0+2.0*im
        @test vk[1] == 1.0
        @test vk[2] == 2.0
    end
    @testset "full space                           " begin
        uk = FTField(2, 1, false)
        vk = similar(uk)
        @test length(vk) == 4
        @test size(vk) == (4, )
        @test vk[WaveNumber(1)] == 0.0 + 0.0*im
        @test vk[WaveNumber(2)] == 0.0 + 0.0*im

        vk = copy(FTField([1.0+2.0*im, 3.0+4.0*im], 1, false))
        @test length(vk) == 4
        @test size(vk) == (4, )
        @test vk[WaveNumber(1)] == 1.0+2.0*im
        @test vk[WaveNumber(2)] == 3.0+4.0*im
        @test vk[1] == 1.0
        @test vk[2] == 2.0
        @test vk[3] == 3.0
        @test vk[4] == 4.0
    end
end

@testset "dot and norm                           " begin
    # cos(x)*cos(x)
    uk = FTField(2, 1, false); uk[WaveNumber(1)] = 0.5
    @test dot(uk, uk) == 0.5
    @test norm(uk) == sqrt(0.5)

    # cos(x)*sin(x)
    uk = FTField(2, 1, false); uk[WaveNumber(1)] =  0.5
    vk = FTField(2, 1, false); vk[WaveNumber(1)] = -0.5*im
    @test dot(uk, vk) == 0.0

    # sin(2x)*sin(2x)
    uk = FTField(2, 1, true); uk[WaveNumber(1)] = -0.5*im
    vk = FTField(2, 1, true); vk[WaveNumber(1)] = -0.5*im
    @test dot(uk, uk) == 0.5
    @test norm(uk) == sqrt(0.5)
end

@testset "broadcast                              " begin
    @testset "full space                         " begin
        uk = FTField(2, 1, false)
        uk .= [1, 2, 3, 4]
        @test uk[1] == 1
        @test uk[2] == 2
        @test uk[3] == 3
        @test uk[4] == 4
        uk .= uk.*uk .+ 1
        @test uk[1] == 2
        @test uk[2] == 5
        @test uk[3] == 10
        @test uk[4] == 17
    end
    @testset "odd space                          " begin
        uk = FTField(4, 1, true)
        uk .= [1, 2, 3, 4]
        @test uk[1] == 1
        @test uk[2] == 2
        @test uk[3] == 3
        @test uk[4] == 4
        uk .= uk.*uk .+ 1
        @test uk[1] == 2
        @test uk[2] == 5
        @test uk[3] == 10
        @test uk[4] == 17
    end
end

@testset "symmetry                               " begin
    @testset "full space                         " begin
        uk = FTField([1, 2, 3, 4], 1, false)

        KS._set_symmetry!(uk)
        @test uk[WaveNumber(1)] == 1+2*im
        @test uk[WaveNumber(2)] == 3+4*im
    end

    @testset "odd space                         " begin
        uk  = FTField(2, 1, true)

        # break the invariance
        uk[WaveNumber(1)] = 1+2*im
        uk[WaveNumber(2)] = 3+4*im

        KS._set_symmetry!(uk)
        @test uk[WaveNumber(1)] == 0+2*im
        @test uk[WaveNumber(2)] == 0+4*im
    end
end