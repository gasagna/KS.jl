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

    fun(x) = cos(x) + cos(2*x)
    v = InverseFFT(similar(U))(U, Field(n))
    @test norm(v - fun.(mesh(u))) < 1e-15
end