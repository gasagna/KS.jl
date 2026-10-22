# The exact Galerkin model of the Kuramoto–Sivashinsky equation in the standard form
#
#     U_T = -U U_X - U_XX - U_XXXX,     X in [0, L), periodic, zero mean,
#
# with n complex modes and no aliasing. It transcribes the model of the bridge study,
# bridging-paper/code/src/ks_long_model.jl (21 Oct 2026), so that the tests do not
# depend on that folder. Set KS_REFERENCE_MODEL to that file to test against it too.
# The state is x = (a_1, …, a_n, b_1, …, b_n), with
#
#     U(X) = Σ_k a_k cos(qkX) + b_k sin(qkX),     q = 2π/L,     c_k = (a_k - i b_k)/2.
#
# The right-hand side, its tangent -∂X(U V) and its adjoint U ∂X W are written as
# direct sums over the triads, the shift is U(X) → U(X - ℓ), the generator -∂X U and
# the reflection U(X) → -U(-X).

using LinearAlgebra

function ks_model(L::Real, n::Int)
    q  = 2π / L
    qk = q .* (1:n)
    Λ  = vcat(qk .^ 2 .- qk .^ 4, qk .^ 2 .- qk .^ 4)

    # ---- the complex coefficients c_{-n}, …, c_n, with c_k at k + n + 1 ----
    function coefficients(x)
        c = zeros(ComplexF64, 2n + 1)
        for k in 1:n
            c[n + 1 + k] = complex(x[k], -x[n + k]) / 2
            c[n + 1 - k] = conj(c[n + 1 + k])
        end
        return c
    end

    # ---- the real state of the coefficients F_k, k = 1, …, n ----
    states(F) = vcat(2 .* real.(F), -2 .* imag.(F))

    # ---- Σ_m w_{k,m} c_m(u) c_{k-m}(v), over |m|, |k - m| ≤ n ----
    triads(cu, cv, w) = [sum(w(k, m) * cu[n + 1 + m] * cv[n + 1 + k - m] for m in k - n:n)
                         for k in 1:n]

    # ---- the right-hand side, the tangent -∂X(U V) and the adjoint U ∂X W ----
    function f!(dxdt, x)
        c = coefficients(x)
        F = triads(c, c, (k, m) -> -im * qk[k] / 2)
        dxdt .= states(F) .+ Λ .* x
        return dxdt
    end

    function tangent!(dvdt, u, v)
        F = triads(coefficients(u), coefficients(v), (k, m) -> -im * qk[k])
        dvdt .= states(F) .+ Λ .* v
        return dvdt
    end

    function adjoint!(dwdt, u, w)
        F = triads(coefficients(u), coefficients(w), (k, m) -> im * q * (k - m))
        dwdt .= states(F) .+ Λ .* w
        return dwdt
    end

    # ---- the shift by ℓ, c_k → c_k e^{-iqkℓ}, its generator and the reflection ----
    shift(x, ℓ)  = vcat(x[1:n] .* cos.(qk .* ℓ) .- x[n+1:2n] .* sin.(qk .* ℓ),
                        x[1:n] .* sin.(qk .* ℓ) .+ x[n+1:2n] .* cos.(qk .* ℓ))
    generator(x) = vcat(-qk .* x[n+1:2n], qk .* x[1:n])
    reflect(x)   = vcat(-x[1:n], x[n+1:2n])

    # ---- the best shift, on 128 shifts and then by Newton's method ----
    function best_shift(x, y)
        p = (complex.(x[1:n], -x[n+1:2n]) ./ 2) .* conj.(complex.(y[1:n], -y[n+1:2n]) ./ 2)
        C(ℓ) = real(sum(p .* cis.(-qk .* ℓ)))
        ℓ = (argmax(C.(range(0, L; length=129)[1:end-1])) - 1) * L / 128
        for _ in 1:30
            e  = p .* cis.(-qk .* ℓ)
            δ  = -real(sum(-im .* qk .* e)) / real(sum(-(qk .^ 2) .* e))
            ℓ += δ
            abs(δ) < 1e-14 && break
        end
        return norm(shift(x, ℓ) .- y), mod(ℓ, L)
    end

    return (q          = q,
            f!         = f!,
            tangent!   = tangent!,
            adjoint!   = adjoint!,
            shift      = shift,
            generator  = generator,
            reflect    = reflect,
            best_shift = best_shift)
end
