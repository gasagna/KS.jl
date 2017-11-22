using Base.Test
using KS

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
        uk .= im.*(0:3)
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
        @test u[0] == u[1] == 1
    end
end

@testset "linearisation                          " begin
    # setup
    n, U, L = 5, 0, 5
    F = KSEq(n, U, L)
    ℒ = LinearisedKSEq(n, U, L)
    
    # allocate
    srand(0)
    uk    = ForwardFFT(n)(Field(randn(2n), L), FTField(n, L)); uk[0] = 0 
    wk    = similar(uk)
    dwkdt = similar(uk)

    # perturb k-th real direction
    for k in 2:n

        # get value from linearisation
        wk[k] += 1
        ℒ(0.0, uk, wk, dwkdt)
        wk[k] -= 1

        # approximate using finite differences
        Δ = 1e-7
        uk[k] -= Δ
        out1 = F(0.0, uk, similar(uk))
        uk[k] += 2Δ
        out2 = F(0.0, uk, similar(uk))
        uk[k] -= Δ

        # compare all elements
        for j = 2:n
            a = dwkdt[j]
            b = (out2[j]-out1[j])/2Δ
            @test abs(a - b) < 2e-6
        end
    end
end