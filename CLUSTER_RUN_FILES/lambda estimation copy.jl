
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Polynomials
using JLD2
using LinearAlgebra     
using Optim
## POTTS HAMILTONIAN
Q = 5
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)
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


### model parameters
function Potts_Hamiltonian(L; J=1,h=1,Q=5,lambda=0.079 + 0.060im,sym=true,adjoint=false)
    ### symmetry
    if sym       
        H =  periodic_boundary_conditions(quantum_potts(ZNIrrep{Q};q=Q), L)
        _,_,W = weyl_heisenberg_matrices(Q)
        P   = TensorMap(W,ℂ^Q←ℂ^Q)
        lat = FiniteChain(L)
        # dat0 = reshape((P'*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data, (Q,Q))
        # dat1 = reshape(((P' ⊗ P')*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, (Q,Q,Q,Q))
        dat2 = reshape(((P' ⊗ P') * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data, (Q,Q,Q,Q))
        # H0 = @mpoham (sum(TensorMap(dat0,Vp←Vp){i} for i in vertices(lat)[1:(end)])) ### Potts
        # H1 = @mpoham (sum(TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(lat)[1:(end-1)]) + TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]}) ##¨Potts with BC
        H2 =  @mpoham lambda * sum(TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){i,i+1} for i in vertices(lat)[1:(end - 1)]) + lambda * TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]}
        # ham = H0+H1+H2
        ham = H + H2
    else
        if !adjoint
            ham = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
            +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
            + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
            + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
        else
            ham = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)'){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)'){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
            +sum( -J * potts_phase(; q=Q,k=j)'{vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)'){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
            + lambda' * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l)'{i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
            + lambda' * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l)'{vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
        end
    end
    return ham
end


function lambda_estimation( alg,lambda,L)
    ## sector search

    ψ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))

    H = Potts_Hamiltonian(L;lambda = lambda)
    (ψ, envir , delta)   = find_groundstate(ψ, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    #(ψ, envir , delta)   = find_groundstate(ψ, H, VUMPS(maxiter = 500,tol=1e-6))
    if alg == "QuasiparticleAnsatz"
        En0, st0 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=2)
        En1, st1 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir;  sector=ZNIrrep{5}(1),num=2)
        st = vcat(En0,En1)
        stt = vcat([ψ,expectation_value(ψ,H)],st)
        save_object("PBC L = $L,5statesQP$lambda.jld2", stt)
    end
end



N = 7
v = 1


lambda = 0.06im + 0.07792
## run simulation here 
D = 80

# for L in [6,8,10,12,14,16,18,19,20]
#     lambda_estimation("QuasiparticleAnsatz",lambda,L)
# end
using TensorKit
using TensorOperations
using KrylovKit
using Base
using LinearAlgebra
using MPSKit
using MPSKitModels
using JLD2
## 2N-1 parameter square (for any N), that can be focused on one point
N = 7
v = 1
test_values = zeros(ComplexF64,(2*N-1)^2)

l = length(test_values)

distx = 0.0007## distance from centre in real

disty = 0.0007# distance from centre in imaginary

cent_im = 0.0600im

cent_r = 0.0780

## snake like structure of test_values will allow for faster convergence when recycling ψ (because you don't jump the entire distx after the loop)
for i in 1:1:(2*N-1)
    if div(i,2) == 1
        for j in 1:1:(2*N-1)
            if i <N+1
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+cent_r+ 1im*LinRange(-disty,0.00,N)[j] .+cent_im
                else 
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+cent_r + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+cent_im
                end
            else
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+cent_r  + 1im*LinRange(-disty,0.00,N)[j]  .+cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N])  .+cent_r+ 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N].+cent_im
                end
            end 
        end
    else
        for j in (2*N-1):-1:1
            if i <N+1
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i]) .+cent_r+ 1im*LinRange(-disty,0.00,N)[j] .+cent_im
                else 
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+cent_r + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+cent_im
                end
            else
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+cent_r + 1im*LinRange(-disty,0.00,N)[j] .+cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N])   .+cent_r + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+cent_im
                end
            end 
        end
    end
end





function eye(m)
    return Matrix{ComplexF64}(I,m,m)
end
function tmatrix(Q,L,lambda)
    H = zeros(ComplexF64,(Q^L,Q^L))
    sigma = zeros(ComplexF64,Q,Q)
    tau = zeros(ComplexF64,Q,Q)
    identity_e = zeros(ComplexF64,Q,Q)
    for i in 1:Q
        sigma[i, i] = cis(2*pi*(i-1)/Q)
        tau[i,mod1(i + 1, Q)] = 1
        identity_e[i,i]= 1
    end
  
    for i in 1:L-1
   
        H +=  kron(kron(eye(Q^(i-1)),lambda* sum(sum(kron(tau^k, identity_e) * kron(sigma', sigma)^p + kron(identity_e ,tau^k) * kron(sigma', sigma)^p + kron(sigma', sigma)^k * kron(tau^p, identity_e) +  kron(sigma', sigma)^k * kron(identity_e ,tau^p) for k in 1:1:Q-1) for p in 1:1:Q-1)), eye(Q^(L-i-1)))
        H += -kron(kron(eye(Q^(i-1)),(sum((tau^k) for k in 1:1:Q-1))),eye(Q^(L-i))) 
        H +=  -kron(kron(eye(Q^(i-1)), sum(kron(sigma'^k,sigma^k) for k in 1:Q-1)), eye(Q^(L-i-1)))

    end
    H += -kron(eye(Q^(L-1)),sum((tau^k) for k in 1:1:Q-1))

    #bc
    if L>2
     H += -sum(kron(kron(sigma^k,eye(Q^(L-2))),sigma^k') for k in 1:1:Q-1)
     H += lambda *sum(sum(  kron(kron(sigma^p, eye(Q^(L-2))), tau'^k * sigma'^p) + kron(kron(tau'^k*sigma^p, eye(Q^(L-2))), sigma'^p) + kron(kron(sigma^k, eye(Q^(L-2))), sigma'^k *tau'^p) + kron(kron(sigma^k*tau'^p, eye(Q^(L-2))), sigma'^k)  for k in 1:1:Q-1) for p in 1:1:Q-1)
    end
    eigenvalues = eigvals(H)
    return eigenvalues
end 

L = 5
Q = 5
# eigenvals = zeros(ComplexF64,(Q^L,l))

# runs = true
# if runs == true
#     for (n,parameter) in enumerate(test_values)
#         println(n/l)
#         eig = tmatrix(Q,L,parameter)[1:Q^L]
#         eigenvals[:,n] = eig
#         eig = sort(eig,by = x->real(x),rev=true)
#         println(eig[1])
#     end 
#     save_object("exact diag CCFT $L",eigenvals)
# end

# println(eigenvals[1])