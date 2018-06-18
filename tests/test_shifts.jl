using Base.Test
using KS

@testset "shift                                  " begin
    # cos(x)
    U = FTField(1, 2π, false); U[WaveNumber(1)] = 0.5
    shift!(U, π/2) # a left shift by π/2 becomes - sin x
    @test abs(U[WaveNumber(1)] - (0.0 + im*0.5)) < 1e-16

    # shift a wave left and check it's ok
    fun(x) = cos(x) + sin(2*x)
    u = Field(10, 2π)
    x = mesh(u)
    u .= fun.(x)
    U = ForwardFFT(similar(u))(u, FTField(10, 2π, false))
    shift!(U, 2π/(2*10 + 2 )) # shift left by one grid point
    v = InverseFFT(similar(U))(U, Field(10, 2π))
    @test abs(v[1] - fun(x[2])) < 9e-16
end


@testset "ddx                                    " begin
    # d/dx[cos(x) - sin(2*x)] = -sin(x) - 2*cos(2x)
    U = FTField(3, 2π, false)
    U[WaveNumber(1)] = 0.5
    U[WaveNumber(2)] = 0.5*im
    U[WaveNumber(3)] = 0.0

    ddx!(U)
    @test U[WaveNumber(1)] ==  0.5*im
    @test U[WaveNumber(2)] == -1
    @test U[WaveNumber(3)] == 0
end