using KS

L = 1.0

@printf " n  -  t_forw [μs]\n"
for n = (1000-30):(1000+30)
	# define right hand side
    F  = KSEq(n, L, 0, false)

    # benchmark forward code
    U, V = FTField(n, L, false), FTField(n, L, false)
    F(0.0, U, V) 
    t = minimum([@elapsed F(0.0, U, V) for i = 1:5000])

    @printf "%03d - %7.4f\n" n t*10^6
end