# https://arxiv.org/pdf/2403.00852 (2024)
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials
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
        tau[i,mod1(i - 1, q)] = one(elt)
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
    return (sigma'⊗ sigma)^k
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
        tau[i,mod1(i - 1, q)] = one(elt)
        identity_e[i,i]= 1
    end
    return (tau^k'⊗ identity_e) * (sigma'⊗ sigma)^p + (identity_e ⊗ tau^k') * (sigma'⊗ sigma)^k + (sigma'⊗ sigma)^k * (tau^p'⊗ identity_e) +  (sigma'⊗ sigma)^k * (identity_e ⊗ tau^p')
end




lambda = 0.079 + 0.060im
J = 1
h = 1
Q = 5
D= 25

                                     ##### Simulating energies
L_list = 8:1:13
N_sizes = length(L_list)
N_levels = 10 ## Gets until the N'th energie level
Energie_levels = zeros(ComplexF64,(N_sizes,N_levels+1))
println(size(Energie_levels))
for (i,L) in enumerate(L_list)

    ###Yin Tang hamiltonian
    H = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
    +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
    + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
    + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
    ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    println("start")
    (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-5, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
    Energie_levels[i,1] = expectation_value(ψ,H,envir)
    states = (ψ, )

      
    En , other  = excitations(H,FiniteExcited(gsalg =DMRG(maxiter = 200,tol=1e-3, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))), states,num=N_levels);
    println(En)
    println(other)
    Energie_levels[i,2:end] = En
    println(other)
    
end


using JLD2
save_object("MPSNonHermitian_pottsq5[8,9,10,11,12,13]-$N_levels,_energies.jld2", Energie_levels)




