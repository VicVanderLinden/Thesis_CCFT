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



J=1
h=1
Q=5


                    ##### Simulating energies
Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)  
L_list = [8,9,10,11,12]
N_sizes = length(L_list)
N_levels = 5 ## Gets until the N'th energie level
Energie_levels = Vector{ComplexF64}[]
function run_sum(L_list)
    D = 50
    param = 0.083332 + 0.072775im
    #param = 0.08333+ 0.07217im
    for (i,L) in enumerate(L_list)
        # ###Yin Tang hamiltonian
        H = Potts_Hamiltonian(L,Q=Q,lambda=param)
        # H0 = @mpoham (sum(TensorMap((P_inv*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data,Vp←Vp){i} for i in vertices(FiniteChain(L))[1:(end)])) ### Potts
        # H1 = @mpoham (sum(TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(FiniteChain(L))[1:(end-1)]) + TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}) ##¨Potts with BC
        # H2 =  @mpoham lambda * sum( TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1)*sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data,Vp⊗Vp←Vp⊗Vp){i,i+1}   for i in vertices(FiniteChain(L))[1:(end - 1)]) + lambda * TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data, Vp⊗Vp←Vp⊗Vp ){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} ###Extra term
        # H = H0+H1+H2
        ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](sector=>D for sector in 0:Q-1)) #;left=Vleft, right=Vright)
        println("start")
        (ψ, envir , delta) = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))

        Ground_energy = expectation_value(ψ,H)
        if N_levels !=0

            ### We use QuasiparticleAnsatz
            En0, st0 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{Q}(0),num=2)
            En1, st1 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{Q}(1),num=2)
            En = vcat(Ground_energy,En1,En0)
            push!(Energie_levels,En)
            println(Energie_levels)
        end
        if N_levels == 0
            push!(Energie_levels,[Ground_energy])
        end
    end
    save_object("MPS_altlambda_pottsq$Q excited-N$N_levels,D$D,energies-L$L_list.jld2", Energie_levels)
end
run_sum(L_list)   

