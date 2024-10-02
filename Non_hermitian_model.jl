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
Q = 2
### The potts_field has not been adapeted to the symmetry type, so it might not go as fast. If needed implement this in MPSKitModels
##thats why both are used without symmetry imposed
H_Potts = @mpoham sum((J * potts_exchange(; q=Q)){i, i+1} + (h * potts_field(; q = Q)){i} for i in vertices(InfiniteChain()))




### i need to ask about how {i,j} works, since in exchange it works, but in field in doesn't and i don't quite know chy
## this for manual implementation
### aditionally how can i do {i+1} in some parts??


"""
    potts_phase([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q=3)

The Potts phase operator sigma |n> = e^{2pin/Q} |n>.
"""
function potts_spin_shift end
potts_spin_shift(; kwargs...) = potts_spin_shift(ComplexF64, Trivial; kwargs...)
potts_spin_shift(elt::Type{<:Number}; kwargs...) = potts_spin_shift(elt, Trivial; kwargs...)
function potts_spin_shift(symmetry::Type{<:Sector}; kwargs...)
    return potts_spin_shift(ComplexF64, symmetry; kwargs...)
end


function potts_spin_shift(elt::Type{<:Number}, ::Type{Trivial}; q=3, k=1)
    pspace = ComplexSpace(q)
    tau = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        tau[mod1(i + 1, q), i] = one(elt)
    end
    return tau^k
end
function potts_phase end
potts_phase(; kwargs...) = potts_phase(ComplexF64, Trivial; kwargs...)
potts_phase(elt::Type{<:Number}; kwargs...) = potts_phase(elt, Trivial; kwargs...)
function potts_phase(symmetry::Type{<:Sector}; kwargs...)
    return potts_phase(ComplexF64, symmetry; kwargs...)
end


function potts_phase(elt::Type{<:Number}, ::Type{Trivial}; q=3,k=1)
    pspace = ComplexSpace(q)
    sigma = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        sigma[i, i] = cis(2*pi*(i-1)/q)
    end
    return (sigma^k'⊗ sigma^k)
end
function potts_phase_shift_combined end
potts_phase_shift_combined(; kwargs...) = potts_phase_shift_combined(ComplexF64, Trivial; kwargs...)
potts_phase_shift_combined(elt::Type{<:Number}; kwargs...) = potts_phase_combined(elt, Trivial; kwargs...)
function potts_phase_shift_combined(symmetry::Type{<:Sector}; kwargs...)
    return potts_phase_combined(ComplexF64, symmetry; kwargs...)
end


function potts_phase_shift_combined(elt::Type{<:Number}, ::Type{Trivial}; q=3,k=1,p=1)
    pspace = ComplexSpace(q)
    sigma = TensorMap(zeros, elt, pspace ← pspace)
    tau = TensorMap(zeros, elt, pspace ← pspace)
    identity_e = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        sigma[i, i] = cis(2*pi*(i-1)/q)
        tau[mod1(i + 1, q), i] = one(elt)
        identity_e[i,i]= 1
    end
    return (tau^k'⊗ identity_e) * (sigma^p'⊗ sigma^p) + (identity_e ⊗ tau^k') * (sigma^k'⊗ sigma^k) + (sigma^k'⊗ sigma^k) * (tau^p'⊗ identity_e) +  (sigma^k'⊗ sigma^k) * (identity_e ⊗ tau^p')
end

lambda = 5
J = 4
h = 2
Q = 2
L = 10
H_Potts_alt = @mpoham sum(sum((J * potts_phase(; q=Q,k=j)){i,i+1} + (h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)])
H1 =  @mpoham lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)])

###Ying Tang hamiltonian
H = H1+H_Potts_alt




### MPS quantum
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials
ψ₀ = FiniteMPS(ℂ^Q, ℂ^25)
ψ, envs = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-7, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
  

