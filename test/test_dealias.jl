# The nonlinear term against direct sums. The default is the advective product on
# 2n + 2 points, with aliasing, and dealias=:pad is the exact Galerkin truncation.

# the default, -P[u u_x] with u and u_x on the 2n + 2 points, by direct sums
function aliased_oracle(U::FTField{n}) where {n}
    M  = 2n + 2
    xs = 2π .* (0:M-1) ./ M
    u  = [2real(sum(U[WaveNumber(k)] * cis(k * x) for k in 1:n)) for x in xs]
    ux = [2real(sum(im * k * U[WaveNumber(k)] * cis(k * x) for k in 1:n)) for x in xs]
    return [-sum(u[j] * ux[j] * cis(-k * xs[j]) for j in 1:M) / M for k in 1:n]
end

# the exact Galerkin right-hand side (kind = :rhs), tangent (:tan) and adjoint (:adj)
function galerkin_oracle(U::FTField{n}, V::FTField{n}, ν::Real, kind::Symbol) where {n}
    c(W, k) = k > 0 ? W[WaveNumber(k)]        :
              k < 0 ? conj(W[WaveNumber(-k)]) : zero(ComplexF64)
    out = zeros(ComplexF64, n)
    for k in 1:n
        λ  = k^2 - ν * k^4
        ms = k-n:n
        if kind === :rhs
            out[k] = λ * c(U, k) - (im * k / 2) * sum(c(U, m) * c(U, k - m) for m in ms)
        elseif kind === :tan
            out[k] = λ * c(V, k) - im * k * sum(c(U, m) * c(V, k - m) for m in ms)
        else
            out[k] = λ * c(V, k) + sum(c(U, m) * im * (k - m) * c(V, k - m) for m in ms)
        end
    end
    return out
end

coeffs(U::FTField{n}) where {n} = [U[WaveNumber(k)] for k in 1:n]

@testset "dealiasing                              " begin
    ν = (2π / 22)^2

    for ISODD in (false, true)
        n = 16
        U = smooth(n, ISODD)
        V = smooth(n, ISODD; a=0.2, amp=0.3, φ=0.4)

        # ---- the default is unchanged, the aliased advective product ----
        N = FTField(n, ISODD)
        KS.NonLinearExTerm(n, ISODD)(0.0, copy(U), N, false)
        @test rel(coeffs(N), aliased_oracle(U)) < 1e-13
        f1 = ForwardEquation(n, ν, ISODD)(0.0, copy(U), FTField(n, ISODD))
        f2 = ForwardEquation(n, ν, ISODD; dealias=:none)(0.0, copy(U), FTField(n, ISODD))
        @test dofs(f1) == dofs(f2)
        @test rel(coeffs(f1), galerkin_oracle(U, V, ν, :rhs)) > 1e-5

        # ---- padded, the exact Galerkin truncation ----
        eq  = ForwardEquation(n, ν, ISODD; dealias=:pad)
        tan = LinearisedEquation(n, ν, ISODD, TangentMode(); dealias=:pad)
        adj = LinearisedEquation(n, ν, ISODD, AdjointMode(); dealias=:pad)
        f   = eq(0.0, copy(U), FTField(n, ISODD))
        t   = tan(0.0, copy(U), copy(V), FTField(n, ISODD))
        a   = adj(0.0, copy(U), copy(V), FTField(n, ISODD))
        @test rel(coeffs(f), galerkin_oracle(U, V, ν, :rhs)) < 1e-13
        @test rel(coeffs(t), galerkin_oracle(U, V, ν, :tan)) < 1e-13
        @test rel(coeffs(a), galerkin_oracle(U, V, ν, :adj)) < 1e-13
    end

    # ---- the padded grid, 2(m + 1) ≥ 3n + 1 points, for small and large n ----
    for n in (1, 2, 3, 8, 15, 16, 17, 33)
        U = smooth(n, false; a=0.05)
        f = ForwardEquation(n, ν, false; dealias=:pad)(0.0, copy(U), FTField(n, false))
        @test rel(coeffs(f), galerkin_oracle(U, U, ν, :rhs)) < 1e-13
    end

    # ---- the option ----
    @test_throws ArgumentError ForwardEquation(4, ν, false; dealias=:foo)
    @test_throws ArgumentError LinearisedEquation(4, ν, false, TangentMode(); dealias=:foo)
end

# The tangent against centred differences, which are exact for the quadratic
# right-hand side up to round-off, and the adjoint by the dot-product test.
@testset "tangent and adjoint operators           " begin
    n = 16
    ν = (2π / 22)^2
    ε = 1e-3

    for ISODD in (false, true), dealias in (:none, :pad)
        eq  = ForwardEquation(n, ν, ISODD; dealias=dealias)
        tan = LinearisedEquation(n, ν, ISODD, TangentMode(); dealias=dealias)
        adj = LinearisedEquation(n, ν, ISODD, AdjointMode(); dealias=dealias)
        U   = smooth(n, ISODD)
        V   = smooth(n, ISODD; a=0.2, amp=0.3, φ=0.4)
        W   = smooth(n, ISODD; a=0.1, φ=2.3)

        # ---- tangent ----
        fp = eq(0.0, FTField(dofs(U) .+ ε .* dofs(V), ISODD), FTField(n, ISODD))
        fm = eq(0.0, FTField(dofs(U) .- ε .* dofs(V), ISODD), FTField(n, ISODD))
        JV = tan(0.0, copy(U), copy(V), FTField(n, ISODD))
        @test rel(dofs(JV), (dofs(fp) .- dofs(fm)) ./ (2ε)) < 1e-10

        # ---- adjoint ----
        JW = adj(0.0, copy(U), copy(W), FTField(n, ISODD))
        @test abs(dot(W, JV) - dot(JW, V)) < 1e-13 * abs(dot(W, JV))
    end
end

# Values of the working tree of 21 Oct 2026, with the advective product of Dec 2023,
# for n = 16, ν = (2π/22)², Δt = 0.005ν and smooth(16, ISODD). In order, the norm and
# the first and last entries of the right-hand side, the norm and the first entry
# after 100 steps of CB3R2R3e, and the same after 100 steps of RK8.
@testset "default behaviour unchanged             " begin
    golden = Dict(false => ( 5.94741927088240416e+01,
                            -1.37328167615733326e-01,
                             6.71865656073323070e+00,
                             5.55735599135834812e-01,
                            -1.59010420312914319e-01,
                             5.55735595808535709e-01,
                            -1.59010421451718315e-01),
                  true  => ( 4.44423508315933091e+01,
                             4.71820441603905238e-01,
                             6.72511972037439598e+00,
                             5.24499126540699967e-01,
                             3.57123742140649181e-01,
                             5.24499125374461195e-01,
                             3.57123741867021893e-01))
    n  = 16
    ν  = (2π / 22)^2
    Δt = 0.005 * ν

    for ISODD in (false, true)
        eq = ForwardEquation(n, ν, ISODD)
        U  = smooth(n, ISODD)
        ϕ  = flow(splitexim(eq)..., CB3R2R3e(FTField(n, ISODD)), TimeStepConstant(Δt))
        f  = eq(0.0, copy(U), FTField(n, ISODD))
        y  = ϕ(copy(U), (0, 100Δt))
        z  = rk8!(eq, copy(U), Δt, 100)
        G  = golden[ISODD]
        @test isapprox(norm(f), G[1]; rtol=1e-10)
        @test isapprox(f[1],    G[2]; rtol=1e-10)
        @test isapprox(f[end],  G[3]; rtol=1e-10)
        @test isapprox(norm(y), G[4]; rtol=1e-10)
        @test isapprox(y[1],    G[5]; rtol=1e-10)
        @test isapprox(norm(z), G[6]; rtol=1e-10)
        @test isapprox(z[1],    G[7]; rtol=1e-10)
    end
end
