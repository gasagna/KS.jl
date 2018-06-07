using Base.Test
using KS
using DualNumbers

@testset "spaces                                 " begin
    @testset "FTField                                " begin
        # indexing
        uk = FTField(2, 1)

    	@test eltype(uk) == Complex{Float64}
		@test length(uk) == 2
		@test size(uk) == (2, )

              uk[1] =  1.0+2.0*im
        @test uk[1] == 1.0+2.0*im
              uk[2] =  3.0+4.0*im
        @test uk[2] == 3.0+4.0*im
        @test_throws BoundsError uk[0]
        @test_throws BoundsError uk[3]

        # similar
        vk = similar(uk)
    	@test eltype(vk) == Complex{Float64}
		@test length(vk) == 2
		@test size(vk) == (2, )
        @test vk[1] == 0.0+0.0*im
        @test vk[2] == 0.0+0.0*im

        # dot and norm
        # cos(x)*cos(x)
        uk = FTField(2, 1); uk[1] = 0.5
        @test dot(uk, uk) == 0.5
        @test norm(uk) == sqrt(0.5)

        # cos(x)*sin(x)
        uk = FTField(2, 1); uk[1] =  0.5
        vk = FTField(2, 1); vk[1] = -0.5*im
        @test dot(uk, vk) == 0.0
        
        #broadcast
        uk = FTField(3, 1)
        uk .= 1 + 2*im
        @test uk[1] == 1 + 2*im
        @test uk[2] == 1 + 2*im
        @test uk[3] == 1 + 2*im

        uk .= [1, 2, 3]
        @test uk[1] == 1
        @test uk[2] == 2
        @test uk[3] == 3

        uk .= uk.*uk .+ 1
        @test uk[1] == 2
        @test uk[2] == 5
        @test uk[3] == 10
    end
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

    @testset "symmetry                           " begin
        uk = FTField(2, 1)
        uk .= [1, 2] .+ im*[3, 4]

        KS._set_symmetry!(Val(false), uk) 
        @test uk[1] == 1+3*im
        @test uk[2] == 2+4*im

        KS._set_symmetry!(Val(true), uk) 
        @test uk[1] == 0+3*im
        @test uk[2] == 0+4*im
    end
end


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
        U = VarFTField(n, L)
        V = similar(U)
        W = VarFTField(FTField(n, L), FTField(n, L))

        # accessors
        state(V) .= 1 + 3im
        prime(V) .= 2 + 6im
        state(W) .= 3 + 2im
        prime(W) .= 4 + 1im

        # broadcast
        U .= V .+ W
        @test all(state(U) .== 4 + 5*im)
        @test all(prime(U) .== 6 + 7*im)

        # indexing
        @test U[1] == Dual(4 + 5*im, 6 + 7*im)
        @test U[n] == Dual(4 + 5*im, 6 + 7*im)
        @test_throws BoundsError U[0]
        @test_throws BoundsError U[n+1]

        # broadcast
        U .= 1
        @test U[1] == 1
        @test U[n] == 1
    end
end


@testset "ffts                                   " begin
	n = 2
	u = Field(n, 2π)

	# non zero mean field gets filtered
	fun(x) = 1 + cos(x) + sin(2*x)
	u .= fun.(mesh(u))

	uk = ForwardFFT(similar(u))(u, FTField(n, 2π))
	@test abs(uk[1] -  0.5 )    < 1e-15
	@test abs(uk[2] +  0.5*im ) < 1e-15
	@test uk.data[1] == 0
	@test uk.data[4] == 0

	# inverse transform
	n = 2
	uk = FTField(n, 2π)
	uk[1] = 0.5
	uk[2] = 0.5

	fun(x) = cos(x) + cos(2*x)
	v = InverseFFT(similar(uk))(uk, Field(n, 2π))
	@test norm(v - fun.(mesh(u))) < 1e-15
end


@testset "shift                                  " begin
    # cos(x)
    # uk = FTField(Complex{Float64}[0.0+im*0.0, 0.5+im*0.0, 0.0+im*0.0], 2π)
    uk = FTField(1, 2π); uk[1] = 0.5
    shift!(uk, π/2) # a left shift by π/2 become - sin x
    @test abs(uk[1] - (0.0 + im*0.5)) < 1e-16

    # shift a wave left and check it's ok
    fun(x) = cos(x) + sin(2*x)
    u = Field(10, 2π)
    x = mesh(u)
    u .= fun.(x)
    uk = ForwardFFT(similar(u))(u, FTField(10, 2π))
    shift!(uk, 2π/(2*10 + 2 )) # shift left by one grid point 
    v = InverseFFT(similar(uk))(uk, Field(10, 2π))
    @test abs(v[1] - fun(x[2])) < 9e-16
end


# @testset "derivative                             " begin
#     # cos(2π/4*x) - sin(2*2π/4*x)
#     uk = FTField(Complex{Float64}[0.0+im*0.0, 0.5+im*0.0, 0.0+im*0.5], 4)
#     qk = WaveNumbers(2, 4)
    
#     # derivative, in place
#     uk .= uk.*im.*qk

#     @test uk[0] ==   0.0
#     @test uk[1] == ( 0.0 + im*0.5)*2π/4
#     @test uk[2] == (-0.5 + im*0.0)*2π/4*2

#     # also test the in place version
#     vk = FTField(Complex{Float64}[0.0+im*0.0, 0.5+im*0.0, 0.0+im*0.5], 4)
#     ddx!(vk)
#     @test all(vk.data .== uk.data) == true # fixme
# end

@testset "system                                 " begin
    F = KSEq(2, 4, 0, false)
    L, N = imex(F)
    uk  = FTField(2, 4)
    uk[1] = 0.5+im*0.1
    uk[2] = 0.2+im*0.1
    out1, out2, out3 = similar(uk), similar(uk), similar(uk)
    # output of F
    F(0.0, uk, out1) 
    # output of L and N
    A_mul_B!(out2, L, uk)
    N(0.0, uk, out3)
    @test out1[1] == out2[1] + out3[1]
    @test out1[2] == out2[2] + out3[2]
end


# @testset "linearisation                          " begin
#     # setup
#     n, U, L = 5, 0, 5
#     F = KSEq(n, U, L)
#     ℒ = LinearisedKSEq(n, U, L)
    
#     # allocate
#     srand(0)
#     uk    = ForwardFFT(n)(Field(randn(2n+1), L), FTField(n, L)); uk[0] = 0 
#     wk    = similar(uk)
#     dwkdt = similar(uk)

#     # perturb k-th real direction
#     for k in 2:n

#         # get value from linearisation
#         wk[k] += 1
#         ℒ(0.0, uk, wk, dwkdt)
#         wk[k] -= 1

#         # approximate using finite differences
#         Δ = 1e-7
#         uk[k] -= Δ
#         out1 = F(0.0, uk, similar(uk))
#         uk[k] += 2Δ
#         out2 = F(0.0, uk, similar(uk))
#         uk[k] -= Δ

#         # compare all elements
#         for j = 2:n
#             a = dwkdt[j]
#             b = (out2[j]-out1[j])/2Δ
#             @test abs(a - b) < 2e-6
#         end
#     end
# end