### atempting to plement chiral pots models
# https://arxiv.org/pdf/2403.00852 (2024)

using MPSKitModels, TensorKit

"""
    potts_exchange([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q=3)

The Potts exchange operator ``Z ⊗ Z' + Z' ⊗ Z``, where ``Z^q = 1``.
"""
function potts_exchange end
potts_exchange(; kwargs...) = potts_exchange(ComplexF64, Trivial; kwargs...)
potts_exchange(elt::Type{<:Number}; kwargs...) = potts_exchange(elt, Trivial; kwargs...)
function potts_exchange(symmetry::Type{<:Sector}; kwargs...)
    return potts_exchange(ComplexF64, symmetry; kwargs...)
end

function potts_exchange(elt::Type{<:Number}, ::Type{Trivial}; q=3)
    pspace = ComplexSpace(q)
    Z = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        Z[i, i] = cis(2π * (i - 1) / q)
    end
    return Z ⊗ Z' + Z' ⊗ Z
end
"""
    potts_field([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q=3) 

The Potts field operator ``X + X'``, where ``X^q = 1``.
"""

function potts_field end
potts_field(; kwargs...) = potts_field(ComplexF64, Trivial; kwargs...)
potts_field(elt::Type{<:Number}; kwargs...) = potts_field(elt, Trivial; kwargs...)
function potts_field(symmetry::Type{<:Sector}; kwargs...)
    return potts_field(ComplexF64, symmetry; kwargs...)
end

function potts_field(elt::Type{<:Number}, ::Type{Trivial}; q=3)
    pspace = ComplexSpace(q)
    X = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        X[mod1(i - 1, q), i] = one(elt)
    end
    return X + X'
end




lambda = 5
J0 = 4
J1 = 2
J3 = 4
h=2













H_3_chiral_Potts = @mpoham sum((-J0 * potts_exchange(; q=5)){i, i+1} + (h * potts_field(; q = 5)){i} for i in vertices(InfiniteChain(2)))
