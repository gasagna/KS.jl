# Shared by the tests.

dofs(U)   = collect(Float64, U)
rel(a, b) = norm(a .- b) / max(norm(b), 1e-300)

# a smooth field without random numbers, odd when ISODD is true
smooth(n, ISODD; a=0.3, amp=0.5, φ=1.7) =
    KS._set_symmetry!(FTField(n, ISODD, k -> amp * exp(-a * k) * cis(φ * k + 0.3k^2)))

# N steps of RK8 of Flows.jl, one at a time, optionally into a stage cache
function rk8!(g, x, h, N; cache=nothing)
    method = RK8(similar(x))
    system = Flows.System(g, nothing)
    for s in 0:N-1
        Flows.step!(method, system, s * h, h, x, cache)
    end
    return x
end

# a stage cache replayed through a linearised equation, forward or backward
function replay!(method, L, x, cache, backward::Bool)
    system = Flows.System(L, nothing)
    steps  = backward ? reverse(eachindex(cache.ts)) : eachindex(cache.ts)
    for s in steps
        Flows.step!(method, system, cache.ts[s], cache.Δts[s], x, cache.xs[s])
    end
    return x
end

# bytes allocated by one call, after a first call that compiles it
allocs_rhs(eq, U, dU)          = (eq(0.0, U, dU); @allocated eq(0.0, U, dU))
allocs_lin(L, U, V, dV)        = (L(0.0, U, V, dV); @allocated L(0.0, U, V, dV))
allocs_step(method, sys, x, h) = (Flows.step!(method, sys, 0.0, h, x, nothing);
                                  @allocated Flows.step!(method, sys, 0.0, h, x, nothing))
