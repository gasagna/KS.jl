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
