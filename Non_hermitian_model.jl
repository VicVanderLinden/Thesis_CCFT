# https://arxiv.org/pdf/2403.00852 (2024)
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials
using JLD2
include("Potts-Operators & Hamiltonian.jl")


lambda = 0.079 + 0.060im
J = 1
h = 1
Q = 5
D= 100

                                     ##### Simulating energies
L_list = [8,9,10,11,12]
N_sizes = length(L_list)
N_levels = 0 ## Gets until the N'th energie level
Energie_levels = Vector{ComplexF64}[]

run = false
if run
for (i,L) in enumerate(L_list)

    ###Yin Tang hamiltonian, no sym
    H = Potts_Hamiltonian(L; sym=false)
    ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    println("start")
    (ψ, envir , delta) = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=8e-5, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    Ground_energy = expectation_value(ψ,H)
    states = (ψ, )
    if N_levels !=0
    #     En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(1),num=N_levels)
        # En , other  = excitations(H,FiniteExcited(gsalg =QuasiparticleAnsatz(maxiter = 200,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))), states,num=N_levels);
        En, st = excitations(H, ChepigaAnsatz2(;), ψ, envir; sector=ZNIrrep{5}(0),num=N_levels)
        energy_diff = zeros(ComplexF64,(length(En)+1,))
        energy_diff[1] = Ground_energy
        energy_diff[2:end] = En .-Ground_energy
        push!(Energie_levels,energy_diff)
        println(Energie_levels)
    end
    if N_levels == 0
        push!(Energie_levels,[Ground_energy])
    end
end
save_object("non_sym_Ground_state_MPSNonHermitian_pottsq$Q excited-N$N_levels,D$D,energies-L$L_list.jld2", Energie_levels)
end
Q=7
println( sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1))