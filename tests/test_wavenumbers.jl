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
    ks = wavenumbers(5)
    for (i, k) in zip(1:5, ks)
        @test typeof(k) == WaveNumber
        @test k         == i
    end
end

@testset "allocations                            " begin
    foo(o, f) = (for i = 1:1000; ddx!(o, f) end)
    f = FTField(100, 1.0, true)
    ddx!(f)
    @test (@allocated ddx!(f)) == 0
end