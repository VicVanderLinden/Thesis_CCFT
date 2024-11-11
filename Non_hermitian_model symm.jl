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


### alternative term

J=1
h=1
Q=5
lambda=0.079 + 0.060im
T = zeros(ComplexF64,Q,Q)           
ω = exp(2*π*im/Q)   
Vp = Vect[ZNIrrep{5}](0=>1,1=>1,2=>1,3=>1,4=>1)
for i in 0:Q-1
    for j in 0:Q-1
        T[i+1,j+1] = (ω^i)^j
    end
end
P = TensorMap(T/sqrt(5),ℂ^Q←ℂ^Q)
pspace = ComplexSpace(Q)                     
P_inv = TensorMap(inv(P.data), pspace ← pspace)  







                                    ##### Simulating energies
Q = 5
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)  
L_list = [8,9,10,11,12]
N_sizes = length(L_list)
N_levels = 1 ## Gets until the N'th energie level
Energie_levels = Vector{ComplexF64}[]

function run_sum()
    D = 100
    for (i,L) in enumerate(L_list)
        # ###Yin Tang hamiltonian
        # H = Potts_Hamiltonian(L)
        ## alt term
        H0 = @mpoham (sum(TensorMap((P_inv*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data,Vp←Vp){i} for i in vertices(FiniteChain(L))[1:(end)])) ### Potts
        H1 = @mpoham (sum(TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(FiniteChain(L))[1:(end-1)]) + TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}) ##¨Potts with BC
        H2 =  @mpoham lambda * sum( TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1)*sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data,Vp⊗Vp←Vp⊗Vp){i,i+1}   for i in vertices(FiniteChain(L))[1:(end - 1)]) + lambda * TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data, Vp⊗Vp←Vp⊗Vp ){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} ###Extra term
        H = H0+H1+H2



        ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)) #;left=Vleft, right=Vright)
        println("start")
        (ψ, envir , delta) = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        Ground_energy = expectation_value(ψ,H)
        states = (ψ, )
        if N_levels !=0
        #     En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(1),num=N_levels)
            # En , other  = excitations(H,FiniteExcited(gsalg =QuasiparticleAnsatz(maxiter = 200,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))), states,num=N_levels);
            En, st = excitations(H, QuasiparticleAnsatz(maxiter = 200,tol=1e-6, ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=N_levels)
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
    save_object("MPSNonHermitian_pottsq$Q excited-N$N_levels,D$D,energies-L$L_list.jld2", Energie_levels)
end
run_sum()   