using Base.Test
using KS
using asis


@testset "tovector/fromvector business           " begin

	# odd fields
	a = FTField([1, 2, 3], true)
	b = FTField([4, 5, 6], true)
	x = tovector((a, b), 7, 8)
	@test x == [1, 2, 3, 4, 5, 6, 7, 8]

	c = FTField(3, true)
	d = FTField(3, true)
	out, T, s = fromvector!((c, d), x)

	@test all(c .== [1, 2, 3])
	@test all(d .== [4, 5, 6])
	@test T == 7
	@test s == 8

	# full-space fields
	a = FTField([1, 2, 3, 4], false)
	b = FTField([5, 6, 7, 8], false)
	x = tovector((a, b), 9, 10)
	@test x == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	c = FTField(2, false)
	d = FTField(2, false)
	out, T, s = fromvector!((c, d), x)

	@test all(c .== [1, 2, 3, 4])
	@test all(d .== [5, 6, 7, 8])
	@test T == 9
	@test s == 10
end