import asis: tovector!,
			 tovector,
			 fromvector!, 
			 _checksize

# IO functions for gradient calculations
function tovector!(out::Vector, ∇ᵤJ::NTuple{N, FTField{n}}, ∇ₜJ::Real, ∇sJ::Real) where {N, n}
    _checksize(out, ∇ᵤJ)
    idx = 1
    for i = 1:N
        @inbounds @simd for el in ∇ᵤJ[i]
            out[idx] = el
            idx += 1
        end
    end
    out[end-1] = ∇ₜJ
    out[end]   = ∇sJ
    return out
end

tovector(∇ᵤJ::NTuple{N, FTField{n}}, ∇ₜJ::Real, ∇sJ::Real) where {N, n} = 
	tovector!(zeros(sum(length.(∇ᵤJ)) + 2), ∇ᵤJ, ∇ₜJ, ∇sJ)

function fromvector!(∇ᵤJ::NTuple{N, FTField{n}}, out::Vector) where {N, n}
    _checksize(out, ∇ᵤJ)
    idx = 1
    for i = 1:N
        ∇ᵤJᵢ = ∇ᵤJ[i] # alias
        @inbounds @simd for j in 1:length(∇ᵤJᵢ)
            ∇ᵤJᵢ[j] = out[idx]
            idx += 1
        end
    end
    return ∇ᵤJ, out[end-1], out[end]
end

# ---------------------------------------------------------------------------- #
# IO functions for gradient calculations with new API   
function tovector!(out::Vector, dJdU0_L::FTField{n}, dJdU0_R::FTField{n}, 
                   ∇TLJ::Real, ∇TRJ::Real, ∇sJ::Real) where {n}
    _checksize(out, dJdU0_L, dJdU0_R)
    idx = 1
    @inbounds @simd for el in dJdU0_L
        out[idx] = el
        idx += 1
    end
    @inbounds @simd for el in dJdU0_R
        out[idx] = el
        idx += 1
    end
    out[end-2] = ∇TLJ
    out[end-1] = ∇TRJ
    out[end]   = ∇sJ
    return out
end

tovector(dJdU0_L::FTField{n}, dJdU0_R::FTField{n}, 
         ∇TLJ::Real, ∇TRJ::Real, ∇sJ::Real) where {n} =
    tovector!(zeros(2*length(dJdU0_L) + 3), dJdU0_L, dJdU0_R, ∇TLJ, ∇TRJ, ∇sJ)

function fromvector!(dJdU0_L::FTField{n}, dJdU0_R::FTField{n}, out::Vector) where {n}
    _checksize(out, dJdU0_L, dJdU0_R)
    idx = 1
    @inbounds @simd for j in 1:length(dJdU0_L)
        dJdU0_L[j] = out[idx]
        idx += 1
    end
    @inbounds @simd for j in 1:length(dJdU0_R)
        dJdU0_R[j] = out[idx]
        idx += 1
    end
    return dJdU0_L, dJdU0_R, out[end-2], out[end-1], out[end]
end
