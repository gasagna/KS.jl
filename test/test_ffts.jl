@testset "ffts                                   " begin
    n = 2
    u = Field(n)

    # non zero mean field gets filtered
    fun(x) = 1 + cos(x) + sin(2*x)
    u .= fun.(mesh(u))

    U = ForwardFFT(similar(u))(u, FTField(n, false))
    @test abs(U[WaveNumber(1)] -  0.5 )    < 1e-15
    @test abs(U[WaveNumber(2)] +  0.5*im ) < 1e-15
    @test U.data[1] == 0
    @test U.data[4] == 0

    # inverse transform
    n = 2
    U = FTField(n, false)
    U[WaveNumber(1)] = 0.5
    U[WaveNumber(2)] = 0.5

    gun(x) = cos(x) + cos(2*x)
    v = InverseFFT(similar(U))(U, Field(n))
    @test norm(v - gun.(mesh(u))) < 1e-15
end

# The transforms between n modes and a finer grid of 2(m + 1) points, used by
# dealias = :pad, and their size checks.
@testset "padded transforms                       " begin
    for ISODD in (false, true), (n, m) in ((5, 8), (16, 24))
        U  = smooth(n, ISODD)
        U0 = dofs(U)

        # ---- the field of U on the finer grid, U unchanged ----
        u  = InverseFFT(U, m)(U, Field(m))
        ex = [2real(sum(U[WaveNumber(k)] * cis(k * x) for k in 1:n)) for x in mesh(m)]
        @test maximum(abs, u .- ex) < 1e-13
        @test dofs(U) == U0

        # ---- back to the n modes ----
        V = ForwardFFT(similar(u), n)(u, FTField(n, ISODD))
        @test rel(dofs(V), U0) < 1e-14
    end

    # ---- the grid cannot have fewer modes than the field ----
    @test_throws ArgumentError ForwardFFT(Field(4), 5)
    @test_throws ArgumentError InverseFFT(FTField(5, false), 4)
end
