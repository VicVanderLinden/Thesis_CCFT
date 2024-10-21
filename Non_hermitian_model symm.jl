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
    return (tau^k)                    ### nonzero elements on the parts that intermingle between sectors
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
    return ((sigma'⊗ sigma)^k)
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




### parameters
lambda = 0.068 + 0.058im
J = 1
h = 1
Q = 5
D= 100


### symmetry
using LinearAlgebra     
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)         
T = zeros(ComplexF64,Q,Q)           
ω = exp(2*π*im/Q)   
for i in 0:Q-1
    for j in 0:Q-1
        T[i+1,j+1] = (ω^i)^j
    end
end 
P = TensorMap(T/sqrt(5),ℂ^Q←ℂ^Q)       
pspace = ComplexSpace(Q)                     
P_inv = TensorMap(inv(P.data), pspace ← pspace)        


                                    ##### Simulating energies
L_list = [9,10,11,12]
N_sizes = length(L_list)
N_levels = 5 ## Gets until the N'th energie level
Energie_levels = zeros(ComplexF64,(N_sizes,N_levels))

run = true
if run
for (i,L) in enumerate(L_list)

    ###Yin Tang hamiltonian
    H0 = @mpoham (sum(TensorMap((P_inv*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data,Vp←Vp){i} for i in vertices(FiniteChain(L))[1:(end)])) ### Potts
    H1 = @mpoham (sum(TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(FiniteChain(L))[1:(end-1)]) + TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}) ##¨Potts with BC
    H2 =  @mpoham lambda * sum( TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data,Vp⊗Vp←Vp⊗Vp){i,i+1}   for i in vertices(FiniteChain(L))[1:(end - 1)]) + lambda * sum(TensorMap(((P_inv ⊗ P_inv) * sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) *(P⊗P)).data, Vp⊗Vp←Vp⊗Vp ) for j in 1:1:Q-1){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} ###Extra term
    H = H0+H1+H2
    ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)) #;left=Vleft, right=Vright)

    println("start")
    (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    Energie_levels[i,1] = expectation_value(ψ,H)
    states = (ψ, )
    if N_levels !=0
        En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(1),num=N_levels)
        # En , other  = excitations(H,FiniteExcited(gsalg =QuasiparticleAnsatz(maxiter = 200,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))), states,num=N_levels);
        println(En)
        Energie_levels[i,2:end] = En
    end
#     Vleft = Vect[ZNIrrep{5}](1=>1)
#     Vright = Vect[ZNIrrep{5}](3=>1)
#     ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D);left=Vleft, right=Vright)
#     (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-5, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
#     Energie_levels[i,2] = expectation_value(ψ,H)
    


    #### Other Hamiltonian ####
    # H = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
    # +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
    # + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
    # + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
    # ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    # (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    # Energie_levels[i,2] = expectation_value(ψ,H)
end
using JLD2
save_object("QuasiparticleAnsatz-MPSNonHermitian_pottsq$Q excited-N$N_levels,D$D,energies-L$L_list, sector0_lambda.jld2", Energie_levels)
end
