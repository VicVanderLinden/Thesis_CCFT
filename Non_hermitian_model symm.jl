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
    return TensorMap(((P_in * tau* P)^k).data,Vp←Vp)                         ### nonzero elements on the parts that intermingle between sectors
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
    return TensorMap((((P_inv*sigma*P)'⊗ (P_inv*sigma*P))^k).data,Vp⊗Vp←Vp⊗Vp)  ### nonzero elements on the parts that intermingle between sectors
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
    tau_n = P_in * tau* P
    sigma_n = P_in * sigma* P
    return TensorMap(((tau_n^k'⊗ identity_e) * (sigma_n'⊗ sigma_n)^p + (identity_e ⊗ tau_n^k') * (sigma_n'⊗ sigma_n)^k + (sigma_n'⊗ sigma_n)^k * (tau_n^p'⊗ identity_e) +  (sigma_n'⊗ sigma_n)^k * (identity_e ⊗ tau_n^p')).data,Vp⊗Vp←Vp⊗Vp)
end




### parameters
lambda = 0.079 + 0.060im
J = 1
h = 1
Q = 5
D= 50


### symmetry
using LinearAlgebra     
Vp = Vect[ZNIrrep{5}](0=>1,1=>1,2=>1,3=>1,4 =>1)
tau = zeros(ComplexF64,Q,Q) 
for i in 1:Q
    tau[i,mod1(i - 1, Q)] = 1
end
eigenv = eigvecs(tau)       
pspace = ComplexSpace(Q)    
P = TensorMap(eigenv, pspace ← pspace)      
P_inv = TensorMap(inv(eigenv), pspace ← pspace) 


L = 10

H = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]))





















                                     ##### Simulating energies
L_list = 8:1:30
N_sizes = length(L_list)
N_levels = 1 ## Gets until the N'th energie level
Energie_levels = zeros(ComplexF64,(N_sizes,N_levels+1))

run = false
if run
for (i,L) in enumerate(L_list)

    ###Yin Tang hamiltonian
    H = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
    +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
    + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
    + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
    ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    println("start")
    (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-7, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
    Energie_levels[i,1] = expectation_value(ψ,H,envir)
    states = (ψ, )
    if N_levels !=1
      
        En , other  = excitations(H,FiniteExcited(gsalg =DMRG(maxiter = 200,tol=1e-3, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))), states,num=N_levels);
        Energie_levels[i,2:end] = En
    end
end
using JLD2
save_object("MPSNonHermitian_pottsq5VicVanderLinden-$N_levels,_energies.jld2", Energie_levels)
end