using KS

L = 1.0

@printf " n  -  t_forw [μs] - t_coupled [μs] - ratio\n"
for n = 10:100
	# define right hand side
    F  = KSEq(n, L, 0, false, :forward)
    FL = KSEq(n, L, 0, false, :tangent)

    # benchmark forward code
    U, V = FTField(n, L, false), FTField(n, L, false)
    F(0.0, U, V) 
    tF = minimum([@elapsed F(0.0, U, V) for i = 1:5000])

    # benchmark forward+tangent code
    U, V = VarFTField(n, L, false), VarFTField(n, L, false)
    FL(0.0, U, V) 
    tFL = minimum([@elapsed FL(0.0, U, V) for i = 1:5000])

    @printf "%03d - %7.4f      - %9.4f      - %5.3f\n" n tF*10^6 tFL*10^6 tFL/tF
end