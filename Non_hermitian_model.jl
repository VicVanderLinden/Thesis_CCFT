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
J = 4
h = 2
### The potts_field has not been adapeted to the symmetry type, so it might not go as fast. If needed implement this in MPSKitModels
##thats why both are used without symmetry imposed
H_Potts = @mpoham sum((J * potts_exchange(; q=5)){i, i+1} + (h * potts_field(; q = 5)){i} for i in vertices(InfiniteChain(2)))

#H1 =  @mpoham lambda * sum((potts_field(; q = 5){i} + potts_field(; q = 5){i+1})   for i in vertices(InfiniteChain(2)))



### i need to ask about how {i,j} works, since in exchange it works, but in field in doesn't and i don't quite know chy
## this for manual implementation
### aditionally how can i do {i+1} in some parts??





"""
    potts_phase([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q=3)

The Potts phase operator sigma |n> = e^{2pin/Q} |n>.
"""
function potts_phase end
potts_phase(; kwargs...) = potts_exchange(ComplexF64, Trivial; kwargs...)
potts_phase(elt::Type{<:Number}; kwargs...) = potts_phase(elt, Trivial; kwargs...)
function potts_phase(symmetry::Type{<:Sector}; kwargs...)
    return potts_phase(ComplexF64, symmetry; kwargs...)
end


function potts_phase(elt::Type{<:Number}, ::Type{Trivial}; q=3)
    pspace = ComplexSpace(q)
    sigma = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        sigma[i, i] = cis(2*pi*i/q)
    end
    return sigma
end



function potts_spin_shift_dag end
potts_spin_shift_dag(; kwargs...) = spin_shift_dag(ComplexF64, Trivial; kwargs...)
potts_spin_shift_dag(elt::Type{<:Number}; kwargs...) = spin_shift_dag(elt, Trivial; kwargs...)
function potts_spin_shift_dag(symmetry::Type{<:Sector}; kwargs...)
    return potts_spin_shift_dag(ComplexF64, symmetry; kwargs...)
end

function potts_spin_shift_dag(elt::Type{<:Number}, ::Type{Trivial}; q=3)
    pspace = ComplexSpace(q)
    tau = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        tau[mod1(i + 1, q), i] = one(elt)
    end
    return tau
end

function potts_spin_shift end
potts_spin_shift(; kwargs...) = spin_shift(ComplexF64, Trivial; kwargs...)
potts_spin_shift(elt::Type{<:Number}; kwargs...) = spin_shift(elt, Trivial; kwargs...)
function potts_spin_shift(symmetry::Type{<:Sector}; kwargs...)
    return potts_spin_shift(ComplexF64, symmetry; kwargs...)
end

function potts_spin_shift(elt::Type{<:Number}, ::Type{Trivial}; q=3)
    pspace = ComplexSpace(q)
    tau = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        tau[mod1(i - 1, q), i] = one(elt)
    end
    return tau
end


#H_Potts_alt = @mpoham sum(J * (potts_phase(; q=5)* potts_phase(; q=5)){i} for i in vertices(InfiniteChain(2)))
