using Base.Test
using KS
using DualNumbers

@testset "spaces                                 " begin
    @testset "FTField                                " begin
        # indexing
        uk = FTField(2, 1)
        @test length(uk.data) == 3
              uk[0] =  0.0+0.0*im
        @test uk[0] == 0.0+0.0*im
              uk[1] =  1.0+2.0*im
        @test uk[1] == 1.0+2.0*im
              uk[2] =  3.0+4.0*im
        @test uk[2] == 3.0+4.0*im
        @test_throws BoundsError uk[3]
        @test_throws BoundsError uk[-1]
        # similar
        vk = similar(uk)
        @test vk[0] == 0.0+0.0*im
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
        uk  = FTField(3, 1)
        uk .= 1 + 2*im
        @test uk[1] == 1 + 2*im
        # broadcast with wave numbers
        uk .= WaveNumbers(im.*(0:3), 1)
        @test uk[0] == 0*im
        @test uk[1] == 1*im
        @test uk[2] == 2*im
        @test uk[3] == 3*im
    end
    @testset "Field                                  " begin
        # indexing
        u = Field(2, 1)
              u[0] =  1
        @test u[0] == 1
              u[1] =  2
        @test u[1] == 2

        # broadcast
        u   = Field(2, 1)
        u .+= 1
        @test u[0] == u[1] == u[2] == u[3] == u[4] == 1

        v = Field([0, 1, 2, 3, 4], 1)
        @test fieldsize(v) == 2
    end
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

# @testset "variational                            " begin
#     n, L = 16, 22
#     @testset "Field" begin
#         # constructors
#         U = VarField(n, L)
#         V = similar(U)
#         W = VarField(Field(n, L), Field(n, L))

#         # accessors
#         state(V) .= 1
#         prime(V) .= 2
#         state(W) .= 3
#         prime(W) .= 4

#         # broadcast
#         U .= V .+ W

#         # FIXME data
#         @test all(state(U).data .== 4) == true
#         @test all(prime(U).data .== 6) == true

#         # indexing
#         @test U[0]    == Dual(4, 6)
#         @test U[2n-1] == Dual(4, 6)
#     end
#     @testset "FTField" begin
#         # constructors
#         U = VarFTField(n, L)
#         V = similar(U)
#         W = VarFTField(FTField(n, L), FTField(n, L))

#         # accessors
#         state(V) .= 1 + 3im
#         prime(V) .= 2 + 6im
#         state(W) .= 3 + 2im
#         prime(W) .= 4 + 1im

#         # broadcast
#         U .= V .+ W
#         @test all(state(U).data .== 4 + 5*im) == true
#         @test all(prime(U).data .== 6 + 7*im) == true

#         # indexing
#         @test U[0] == Dual(4 + 5*im, 6 + 7*im)
#         @test U[n] == Dual(4 + 5*im, 6 + 7*im)

#         # broadcast with wavenumbers
#         qk = WaveNumbers(n, L)
#         U .= qk

#         @test U[0] == 0
#         @test U[1] == 2π/L*1
#         @test U[n] == 2π/L*n

#     end
# end

# @testset "wavenumbers                            " begin
#     w = WaveNumbers(2, 3)
#     @test w[0] == 0*2π/3
#     @test w[1] == 1*2π/3
#     @test w[2] == 2*2π/3
# end

# @testset "shift                                  " begin
#     # cos(x)
#     uk = FTField(Complex{Float64}[0.0+im*0.0, 0.5+im*0.0, 0.0+im*0.0], 2π)
#     shift!(uk, π/2) # a left shift by π/2 become - sin x
#     @test abs(uk[1] - (0.0 + im*0.5)) < 1e-16

#     # shift a wave left and check it's ok
#     x = linspace(0, 2, 8)[1:7]
#     fun(x) = cos(2π/2*x) + sin(2*2π/2*x)
#     u = Field(fun.(x), 2)
#     uk = ForwardFFT(similar(u))(u, FTField(2, 2))
#     shift!(uk, 2/5) # shift left by one grid point 
#     v = InverseFFT(similar(uk))(uk, Field(2, 2))
#     @test abs(v[0] - fun(x[2])) < 2e-16
# end

# @testset "derivative                             " begin
#     # cos(2π/4*x) - sin(2*2π/4*x)
#     uk = FTField(Complex{Float64}[0.0+im*0.0, 0.5+im*0.0, 0.0+im*0.5], 4)
#     qk = WaveNumbers(2, 4)
    
#     # derivative, in place
#     uk .= uk.*im.*qk

#     @test uk[0] ==   0.0
#     @test uk[1] == ( 0.0 + im*0.5)*2π/4
#     @test uk[2] == (-0.5 + im*0.0)*2π/4*2
# end

# @testset "system                                 " begin
#     F = KSEq(2, 0, 4)
#     L, N = imex(F)
#     uk  = FTField(Complex{Float64}[0.0+im*0.0, 0.5+im*0.1, 0.2+im*0.1], 4)
#     out1, out2, out3 = similar(uk), similar(uk), similar(uk)
#     # output of F
#     F(0.0, uk, out1) 
#     # output of L and N
#     A_mul_B!(out2, L, uk)
#     N(0.0, uk, out3)
#     @test out1[0] == out2[0] + out3[0]
#     @test out1[1] == out2[1] + out3[1]
#     @test out1[2] == out2[2] + out3[2]
# end