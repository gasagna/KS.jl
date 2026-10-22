# KS.jl

The Kuramoto–Sivashinsky equation in Fourier space, for time stepping with
[Flows.jl](https://github.com/gasagna/Flows.jl).

## The equation

KS.jl solves

    u_t = -u u_x - u_xx - ν u_xxxx,     x in [0, 2π), periodic, zero mean.

`ISODD = true` restricts it to odd fields, u(-x) = -u(x). `ISODD = false` is the
full space.

## States

`FTField(n, ISODD)` holds the coefficients U_k, k = 1, …, n, of

    u(x) = Σ_{0<|k|≤n} U_k e^{ikx},     U_{-k} = conj(U_k).

`U[WaveNumber(k)]` is the complex U_k. The linear index `U[i]` runs over the
degrees of freedom. They are (Re U_1, Im U_1, …, Re U_n, Im U_n) in the full space
and (Im U_1, …, Im U_n) in the odd space. So u = Σ a_k cos(kx) + b_k sin(kx) with
a_k = 2 Re U_k and b_k = -2 Im U_k.

`Field(n)` holds u at the 2n + 2 points x_j = 2πj/(2n + 2). The forward transform
is U_k = (1/(2n + 2)) Σ_j u(x_j) e^{-ikx_j}. The inverse transform is not
normalised, u(x_j) = 2 Re Σ_k U_k e^{ikx_j}. `ForwardFFT(u, n)` and
`InverseFFT(U, m)` also go between n modes and a finer grid of 2m + 2 points.

`dot(U, V)` is Σ_k Re(U_k conj(V_k)), which is (1/4π) ∫ u v dx. In the full space
it is the Euclidean product of the degrees of freedom. `norm(U)` is its square
root.

## Equations

    eq = ForwardEquation(n, ν, ISODD; dealias=:none)
    L  = LinearisedEquation(n, ν, ISODD, TangentMode(); dealias=:none)
    L⁺ = LinearisedEquation(n, ν, ISODD, AdjointMode(); dealias=:none)

`eq(t, U, dUdt)` is the right-hand side, for the explicit schemes RK4 and RK8 of
Flows.jl. `splitexim(eq)` gives the explicit and implicit parts for its IMEX
schemes. `L(t, U, V, dVdt)` is the tangent J(U)V and `L⁺(t, U, W, dWdt)` the
adjoint J(U)ᵀW in `dot`. Both are exact, so the discrete modes of Flows.jl give
the exact derivative of a run and its adjoint. `couple(eq, L)` integrates the
state and a tangent vector together.

## Dealiasing

By default the product u u_x is taken on the 2n + 2 points of the grid. It has
modes up to 2n, and those above n + 1 fold back onto the modes kept. This aliasing
breaks the continuous translation symmetry. With `dealias = :pad` the product is
taken on 2(m + 1) ≥ 3n + 1 points, with m = ⌈(3n - 1)/2⌉, and only the modes
1, …, n are kept. The model is then the exact Galerkin truncation with n modes. At
n = 16 it costs no more than the default. Use the same option for the forward and
the linearised equations.

## Symmetries

| Operation | Effect | Coefficients |
|---|---|---|
| `shift!(U, s)` | u(x) → u(x + s) | U_k → U_k e^{iks} |
| `ddx!(V, U)` | ∂x u, the generator of `shift!` | V_k = ik U_k |
| `reflect!(U)` | u(x) → -u(-x) | U_k → -conj(U_k) |
| `minnormdiff(U, V, 8n; method=:newton)` | (d, (s,)) with shift!(U, s) closest to V | |

`shift!` preserves `dot`. Odd fields do not change under `reflect!`, and only the
shifts 0 and π keep them odd. `reflect!` is a method of `LinearAlgebra.reflect!`.

`minnormdiff(U, V, N)` starts from the N shifts 2πj/N. The default
`method = :grid` keeps the grid search of 2018. It returns the best of the N
shifts, and it shifts `U` in place and back. `method = :newton` changes neither
field. It refines the best of the N shifts by Newton's method, toward the local
minimum next to it, and keeps the shift of the grid when the refinement does not
lower d. The global minimum can sit next to another of the N shifts, less often
for larger N. Take N ≥ 8n.

## The standard form on [0, L)

The equation U_T = -U U_X - U_XX - U_XXXX on X in [0, L) is the same as KS.jl
with

    q = 2π/L,     x = qX,     t = q²T,     u = U/q,     ν = q².

| Quantity | Standard form | KS.jl |
|---|---|---|
| state | U(X) = Σ a_k cos(qkX) + b_k sin(qkX) | U_k = (a_k - i b_k)/(2q) |
| time, period | T | q²T |
| Lyapunov or Floquet exponent | λ | λ/q² |
| right-hand side | dx/dT | q² times that of KS.jl, mapped |
| inner product | (2/L) ∫ U V dX | 4q² dot(U, V) |
| distance | d | d/(2q) |
| shift σ_ℓ U(X) = U(X - ℓ) | ℓ | `shift!(U, -qℓ)` |
| generator of σ_ℓ | -∂X U | -q ∂x u, `ddx!` times -q |
| reflection | U(X) → -U(-X) | `reflect!` |
| best shift | (d, ℓ) | ℓ = mod(-s/q, L), d = 2q d_KS |

Tangent and adjoint vectors map like states. Gradients map like states times 4q².
Floquet multipliers do not change.

At L = 22, q = 0.285599332 and ν = 0.0815669785. A step of 0.005 is 4.07835e-4 in
KS.jl, and periods from 600 to 6000 are 48.94 to 489.4.

## Tests

    julia --project=. -e 'using Pkg; Pkg.test()'

The environment of the package takes Flows.jl from `../Flows.jl`, and so does
`test/Project.toml`. The tests compare KS.jl with the exact Galerkin model in
`test/ks_model.jl`, and use no file outside the package. To run the same tests
against another file that defines `ks_model(L, n)`, such as the model file of the
bridge study, set `KS_REFERENCE_MODEL` to its path.

    KS_REFERENCE_MODEL=/path/to/ks_long_model.jl julia --project=. -e 'using Pkg; Pkg.test()'
