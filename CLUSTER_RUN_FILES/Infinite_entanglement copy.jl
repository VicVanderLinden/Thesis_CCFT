

Q = 5
using JLD2
## Operators
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using LinearAlgebra 


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
        tau[i,mod1(i + 1, q)] = one(elt)
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
        tau[i,mod1(i + 1, q)] = one(elt)
        identity_e[i,i]= 1
    end
    return (tau^k⊗ identity_e) * (sigma'⊗ sigma)^p + (identity_e ⊗ tau^k) * (sigma'⊗ sigma)^p + (sigma'⊗ sigma)^k * (tau^p⊗ identity_e) +  (sigma'⊗ sigma)^k * (identity_e ⊗ tau^p)
end


Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)
_,_,W = weyl_heisenberg_matrices(Q)
P   = TensorMap(W,ℂ^Q←ℂ^Q)



function infinite_potts(lambda,symytry = true,J=1,h=1,Q=5)
    if symytry
    dat0 = reshape((P*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P').data, (Q,Q))
    dat1 = reshape(((P ⊗ P)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P' ⊗ P')).data, (Q,Q,Q,Q))
    dat2 = reshape(((P ⊗ P) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P'⊗P')).data, (Q,Q,Q,Q))
    H0 = @mpoham (sum(TensorMap(dat0,Vp←Vp){i} for i in -Inf:Inf)) ### Potts
    H1 = @mpoham (sum(TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in -Inf:Inf))
    H2 =  @mpoham lambda* sum(TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){i,i+1} for i in -Inf:Inf)
    return H0 + H1 + H2
    else 
        dat0 = reshape((sum((-h * potts_spin_shift(; q = Q,k=j)') for j in 1:1:Q-1)).data, (Q,Q))
        dat1 = reshape((sum((-J * potts_phase(; q=Q,k=j)') for j in 1:1:Q-1)).data, (Q,Q,Q,Q))
        dat2 = reshape((sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j)' for l in 1:1:Q-1) for j in 1:1:Q-1)).data, (Q,Q,Q,Q))
        H0 = @mpoham (sum(TensorMap(dat0,ℂ^Q←ℂ^Q){i} for i in -Inf:Inf)) ### Potts
        H1 = @mpoham (sum(TensorMap(dat1,ℂ^Q⊗ℂ^Q←ℂ^Q⊗ℂ^Q){i,i+1}  for i in -Inf:Inf))
        H2 =  @mpoham lambda* (sum(TensorMap(dat1,ℂ^Q⊗ℂ^Q←ℂ^Q⊗ℂ^Q){i,i+1} for i in -Inf:Inf))
        return H0 + H1 + H2
    end
    end
function run_sim(Q,D)
    d = D[1]
    ψ_right = InfiniteMPS(Vp,Vect[ZNIrrep{Q}](sector=>d for sector in 0:Q-1)) 
    for d in D
        H = infinite_potts(0.079+0.06im)
        (ψ_right, envir , delta) = find_groundstate(ψ_right, H, VUMPS(maxiter = 2000,tol=1e-8, alg_eigsolve =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        save_object("GS VUMPS for inf,D=$d.jld2",ψ_right)   
        println(ψ_right)
        ψ_right,envs = changebonds(ψ_right,H,MPSKit.VUMPSSvdCut(trscheme = truncdim(5*(d+3))))
    end
end

D = 5:2:7
Q = 5
run_sim(Q,D)

