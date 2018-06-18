using KS

L = 1.0

@printf " n  -  t [μs]\n"
for n = 500:1000
    F = KSEq(n, L, 0, false)
    uk, vk = FTField(n, L), FTField(n, L)
    F(0.0, uk, vk) 
    t = minimum([@elapsed F(0.0, uk, vk) for i = 1:100])

    @printf "%03d - %5.4f\n" n t*10^6
end