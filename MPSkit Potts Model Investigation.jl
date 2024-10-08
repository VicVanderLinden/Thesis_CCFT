### MPSkitModel Investigation
# https://arxiv.org/pdf/2403.00852 (2024)

using MPSKitModels, TensorKit
"""
    potts_spin_shift([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q,k)

The Potts spin shift operator tau |n> = |(n+1) mod Q>.
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

"""
    potts_phase([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q,k)

The Potts phase operator sigma |n> = e^{2pin/Q} |n>.
"""
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







### MPS quantum
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using JLD2

J = 1
h = 1
Q = 5### Run it for Q=3 and Q = 5 and you see that they will repectively agree and not agree
D=10
L_list = 6:4:22



energies = similar(L_list,ComplexF64)
global i=0
for L in L_list
    ###Potts Hamiltonian
    H = quantum_potts(FiniteChain(L),q=Q)

    global i+=1
    ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    println("start")
    ψ, envs , delta   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-5, eigalg =MPSKit.Defaults.alg_eigsolve()));
    energies[i] = expectation_value(ψ,H,envs)
end
x_values = 1 ./L_list.^2
divided_energies = similar(L_list,ComplexF64)
for j in 1:1:length(L_list)
    divided_energies[j] = energies[j]/L_list[j]
end

p = plot(; xlabel="1/L²", ylabel="Re(E0/L)")
plot!(x_values,real(divided_energies) ; seriestype=:scatter,label="Potts_no_boundary")
energies = similar(L_list,ComplexF64)
global i=0
for L in L_list
    ###Yin Tang hamiltonian
    H = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model w no open bc
    +sum(-h * potts_spin_shift(; q = Q,k=j){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1))
    ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    println("start")
    global i+=1
    ψ, envs , delta   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-5, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
    energies[i] = expectation_value(ψ,H,envs)
end


x_values = 1 ./L_list.^2
divided_energies = similar(L_list,ComplexF64)
for j in 1:1:length(L_list) 
    divided_energies[j] = energies[j]/L_list[j]
end
println("here")
plot!(x_values,real(divided_energies) ; seriestype=:scatter,label="Yin_tang_no_boundary")
savefig(p,"Energy scaling_PottsMPSvsHomemade D = $D, Q = $Q.png")








### conculsion this is the true potts hamiltonian with open boundary conditions
#H_potts = @mpoham sum( (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1)+ sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} - (h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]); 
   