import asis: tovector!,
			 tovector,
			 fromvector!, 
			 _checksize

# IO functions for gradient calculations
function tovector!(out::Vector, ∇ᵤJ::FTField{n}, ∇ₜJ::Real, ∇ₛJ::Real) where {n}
    _checksize(out, ∇ᵤJ)
    @inbounds @simd for i in eachindex(out)
        out[i] = ∇ᵤJ[i]
    end
    out[end-1] = ∇ₜJ
    out[end]   = ∇ₛJ
    return out
end

function tovector(∇ᵤJ::FTField{n}, ∇ₜJ::Real, ∇ₛJ::Real) where {n}
	out = zeros(length(∇ᵤJ) + 2)
    @inbounds @simd for i in eachindex(out)
        out[i] = ∇ᵤJ[i]
    end
    out[end-1] = ∇ₜJ
    out[end]   = ∇ₛJ
    return out
end

function fromvector!(∇ᵤJ::FTField{n}, out::Vector) where {n}
    _checksize(out, ∇ᵤJ)
    @inbounds @simd for i in eachindex(out)
        ∇ᵤJ[i] = out[i]
    end
    return ∇ᵤJ, out[end-1], out[end]
end
