@testset "arithmetic                             " begin
    a = WaveNumber(1)
    @test (a+1) == 2
    @test (a+a) == 2
    @test (a*a) == 1
    @test (a<a) == false
    @test (a<2) == true
    @test typeof(a+1)      == Int64
    @test typeof(a*2)      == Int64
    @test typeof(a*2.0)    == Float64
    @test typeof(a*2.0*im) == Complex{Float64}
end

@testset "wave numbers vector                    " begin
    ks = wavenumbers(1:3:5)
    for (i, k) in zip(1:3:5, ks)
        @test typeof(k) == WaveNumber
        @test k         == i
    end
end