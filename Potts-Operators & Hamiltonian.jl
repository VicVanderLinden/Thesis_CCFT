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


### model parameters
function Potts_Hamiltonian(L; J=1,h=1,Q=5,lambda=0.079 - 0.060im,sym=true)
    ### symmetry
    if sym       
        T = zeros(ComplexF64,Q,Q)           
        ω = exp(2*π*im/Q)   
        Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)  
        for i in 0:Q-1
            for j in 0:Q-1
                T[i+1,j+1] = (ω^i)^j
            end
        end 
        P = TensorMap(T/sqrt(5),ℂ^Q←ℂ^Q)       
        pspace = ComplexSpace(Q)                     
        P_inv = TensorMap(inv(P.data), pspace ← pspace)        
        H0 = @mpoham (sum(TensorMap((P_inv*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data,Vp←Vp){i} for i in vertices(FiniteChain(L))[1:(end)])) ### Potts
        H1 = @mpoham (sum(TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(FiniteChain(L))[1:(end-1)]) + TensorMap(((P_inv ⊗ P_inv)*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, Vp⊗Vp←Vp⊗Vp){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}) ##¨Potts with BC
        H2 =  @mpoham lambda * sum( TensorMap(((P_inv ⊗ P_inv) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data,Vp⊗Vp←Vp⊗Vp){i,i+1}   for i in vertices(FiniteChain(L))[1:(end - 1)]) + lambda * sum(TensorMap(((P_inv ⊗ P_inv) * sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) *(P⊗P)).data, Vp⊗Vp←Vp⊗Vp ) for j in 1:1:Q-1){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} ###Extra term
        ham = H0+H1+H2
    else
        ham = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
        +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
        + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
        + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
    end    
    return ham
end