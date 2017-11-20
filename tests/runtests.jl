using Base.Test
using KS

@testset "linearisation                          " begin
    # setup
    N, U, L = 5, 0, 5
    F = KSEq(N, U, L)
    L = LinearisedKSEq(N, U, L)
    
    # allocate
    srand(0)
    uk    = randn(N+1) + im*randn(N+1)
    uk[1] = 0 
    wk    = zeros(uk)
    dwkdt = zeros(uk)

    # perturb k-th real direction
    for k in 2:N

        # get value from linearisation
        wk[k] += 1
        L(0.0, uk, wk, dwkdt)
        wk[k] -= 1

        # approximate using finite differences
        Δ = 1e-7
        uk[k] -= Δ
        out1 = F(0.0, uk, similar(uk))
        uk[k] += 2Δ
        out2 = F(0.0, uk, similar(uk))
        uk[k] -= Δ

        # compare all elements
        for j = 2:N
            a = dwkdt[j]
            b = (out2[j]-out1[j])/2Δ
            @test abs(a - b) < 2e-6
        end
    end
end