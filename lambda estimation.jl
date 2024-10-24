
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Polynomials
using JLD2

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
    return (tau^k'⊗ identity_e) * (sigma'⊗ sigma)^p + (identity_e ⊗ tau^k') * (sigma'⊗ sigma)^p + (sigma'⊗ sigma)^k * (tau^p'⊗ identity_e) +  (sigma'⊗ sigma)^k * (identity_e ⊗ tau^p')
end

J = 1
h = 1
Q = 5
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

using Optim
Δε = 0.4656 − 0.2245im
ΔL1ε = 1.4656 − 0.2245im
# Δσ = 0.1336 − 0.0205im
#ΔL1σ =1.1336 − 0.0205im
#Cε_primeσσ = 0.0658 + 0.0513im #####################" THIS ONE IS ACTUALLY NOT KNOW BEFOREHAND? how did the authors do this?
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.908 − 0.599im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
#AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)
function lambda_estimation(L,lambda_range)
    gε_prime = zeros(length(lambda_range))
    Eε = zeros(ComplexF64,length(lambda_range))
    Eσ = zeros(ComplexF64,length(lambda_range))
    EL1ε = zeros(ComplexF64,length(lambda_range))
    H0 = @mpoham (sum(TensorMap((P_inv*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data,Vp←Vp){i} for i in vertices(FiniteChain(L))[1:(end)])) + (sum(TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(FiniteChain(L))[1:(end-1)]) + TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}) ##¨Potts with BC
    H1 =  @mpoham sum( TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data,Vp⊗Vp←Vp⊗Vp){i,i+1}   for i in vertices(FiniteChain(L))[1:(end - 1)]) + sum(TensorMap(((P_inv ⊗ P_inv) * sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) *(P⊗P)).data, Vp⊗Vp←Vp⊗Vp ) for j in 1:1:Q-1){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} ###Extra term
    for (i,lambda) in enumerate(lambda_range)
        H = H0 + lambda*H1
        ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)) #;left=Vleft, right=Vright)
        (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=2)
        print(En)
        ΔEε = En[1]
        ΔEL1ε = En[2]
        En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(1),num=1)
        ΔEσ = En[1]
        #ΔEL1σ = En[2]
        fun(x) = abs(x[1]*(ΔEε) - Δε -Cε_primeεε* x[2]) + abs(x[1]*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime*x[2])#+abs(x[1]*(ΔEσ) - Δσ -Cε_primeσσ* x[2]) #+ abs(x[1]*(ΔEL1σ) -ΔL1σ  -Cε_primeσσ* AL1σε_prime*x[2])
        res = optimize(fun, [0.0, 0.0])
        gε_prime[i] = Optim.minimizer(res)[2]
        Eε[i] = ΔEε
        Eσ[i] = ΔEσ
        EL1ε[i] = ΔEL1ε
    end
    save_object("Lambda_est_ge$L.jld2", gε_prime)
    save_object("ΔEε with lambda for $L.jld2",Eε)
    save_object("ΔEσ with lambda for$L.jld2", Eσ)
    save_object("ΔEl1ε with lambda for$L.jld2", EL1ε)
end










N = 8
test_values = zeros(ComplexF64,(N*N))
for i in 1:1:N
    for j in 1:1:N
        test_values[i+(j-1)*N] = 0.12-(0.1/N )* i + (0.02 + 0.1/N * (j-1))*im
    end
end
D= 100
L_list = [6,8,10,12,14]
for L in L_list
    lambda_estimation(L,test_values)
end



  