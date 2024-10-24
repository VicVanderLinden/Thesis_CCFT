# https://arxiv.org/pdf/2403.00852 (2024)
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials
using LinearAlgebra 
using JLD2
include("Potts-Operators & Hamiltonian.jl")
                                    ##### Simulating energies
Q = 5
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)  
L_list = [8,9,10,11,12]
N_sizes = length(L_list)
N_levels = 0 ## Gets until the N'th energie level
Energie_levels = Vector{ComplexF64}[]

function run_sum()
    D = 100
    for (i,L) in enumerate(L_list)
        ###Yin Tang hamiltonian
        H = Potts_Hamiltonian(L)
        ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)) #;left=Vleft, right=Vright)
        println("start")
        (ψ, envir , delta) = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
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
#     Vleft = Vect[ZNIrrep{5}](1=>1)
#     Vright = Vect[ZNIrrep{5}](3=>1)
#     ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D);left=Vleft, right=Vright)
#     (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-5, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
#     Energie_levels[i,2] = expectation_value(ψ,H)
    end
    save_object("Ground_state_MPSNonHermitian_pottsq$Q excited-N$N_levels,D$D,energies-L$L_list.jld2", Energie_levels)
end
run_sum()