@testset "system                                 " begin
    F = KSEq(2, 4, 0, false)
    N, L = splitexim(F)
    U  = FTField(2, 4, false)
    U[WaveNumber(1)] = 0.5+im*0.1
    U[WaveNumber(2)] = 0.2+im*0.1
    out1, out2, out3 = similar(U), similar(U), similar(U)
    # output of F
    F(0.0, U, out1) 
    # output of L and N
    A_mul_B!(out2, L, U)
    N(0.0, U, out3)
    @test out1[WaveNumber(1)] == out2[WaveNumber(1)] + out3[WaveNumber(1)]
    @test out1[WaveNumber(2)] == out2[WaveNumber(2)] + out3[WaveNumber(2)]
end
