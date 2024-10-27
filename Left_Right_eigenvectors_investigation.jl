using Printf
using MPSKit
using KrylovKit
const Selector = Union{Symbol,EigSorter}
include("Potts-Operators & Hamiltonian.jl")
function ∂∂AC(pos::Int, mps, opp::Union{MPOHamiltonian,SparseMPO,DenseMPO}, cache)
    return MPO_∂∂AC(cache.opp[pos], leftenv(cache, pos, mps), rightenv(cache, pos, mps))
end
import KrylovKit: Arnoldi
using LinearAlgebra
struct MPO_∂∂AC{O,L,R}
o::O
leftenv::L
rightenv::R
end

const maxiter = 200
const tolgauge = 1e-13
const tol = 1e-20
const eigsolver = Arnoldi(; tol, maxiter, eager=true,verbosity=3)
function _schursolve_2sided(A,Astar, x₀, howmany::Int, which::Selector,alg::Arnoldi)
 
    krylovdim = 30
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity - 2)
    numops = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)
    
    # ## do the same for A*
    # initialize arnoldi factorization
    iter_im = ArnoldiIterator(Astar, x₀, alg.orth)
    fact_im = initialize(iter_im; verbosity=alg.verbosity - 2)
    sizehint!(fact_im, krylovdim)
    β_im = normres(fact_im)

    # allocate storage
    HH_im = fill(zero(eltype(fact_im)), krylovdim + 1, krylovdim)
    UU_im = fill(zero(eltype(fact_im)), krylovdim, krylovdim)
    
    # initialize storage, same dimension on decomposition
    K = length(fact) # == 1
    converged = 0
    local T, U,T_im,U_im
    while true
        β = normres(fact)
        β_im = normres(fact_im)
        K = length(fact)
        if β <= tol 
            if K < howmany
                @warn "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`); setting `howmany = $K`."
                howmany = K
            end
        end
        if K == krylovdim 
            ### Rigth
            H = view(HH, 1:K, 1:K)
            U= view(UU, 1:K, 1:K)
            V = fact.V
            f = view(HH, K + 1, 1:K)
            copyto!(U, I) 
            copyto!(H, rayleighquotient(fact)) # Rayleigh quotient in this context is the H in the factorization
            f = fact.H[end-K:end-1]
            ### Left
            H_im = view(HH_im, 1:K, 1:K)
            U_im= view(UU_im, 1:K, 1:K)
            W = fact_im.V
            f_im = fact_im.H[end-K:end-1]
            copyto!(U_im, I)
            copyto!(H_im, rayleighquotient(fact_im)) # Rayleigh quotient in this context is the K in the factorization
            
            ## Changing the rayleighquotients before decomposing
            W_star = [W.basis[i]' for i in 1:length(W.basis)] ### is important that each element becomes adjoint
            V_star = [V.basis[i]' for i in 1:length(V.basis)]
            W_star_V = Matrix{ComplexF64}(undef,krylovdim,krylovdim)
            W_V_star = Matrix{ComplexF64}(undef,krylovdim,krylovdim)
            for i in 1:krylovdim
                for j in 1:krylovdim
                    W_star_V[i,j] =(W_star[i]*V.basis[j])
                    W_V_star[i,j] =(V_star[j]*W.basis[i])
                end
            end
            M = inv(W_star_V)
            M_star = inv(W_V_star)
            W_star_r = Vector{ComplexF64}(undef,krylovdim)
            V_star_r = Vector{ComplexF64}(undef,krylovdim)
            for i in 1:krylovdim
                W_star_r[i] = W_star[i]*fact.r
                V_star_r[i] = V_star[i]*fact_im.r
            end
         
        
            
            ### H and K
            H = H + M *W_star_r*f'            
            H_im = H_im + M_star *V_star_r*f_im'
            
            ## vl+1 and wl+1
            V_M_W_star = sum(sum(V.basis[i]*M[i,j]*W_star[j] for i in 1:krylovdim) for j in 1:krylovdim)
            W_M_star_V_star = sum(sum(W.basis[i]*M_star[i,j]*V_star[j] for i in 1:krylovdim) for j in 1:krylovdim)
            fact.r = (I - V_M_W_star)*fact.r
            fact_im.r = (I - W_M_star_V_star)*fact_im.r
            
            
            # compute dense schur factorization
            T, U, values = KrylovKit.hschur!(H, U)   ### U is Q, T is S in the paper https://ianzwaan.com/assets/pdf/kstwo.pdf
            by, rev =  KrylovKit.eigsort(which)
            p =  KrylovKit.sortperm(values; by=by, rev=rev)
            T, U =  KrylovKit.permuteschur!(T, U, p)
            f = [sum(U'[i,j]*f[i] for i in 1:K) for j in 1:K]
    
            # compute dense schur factorization
            T_im, U_im, values = KrylovKit.hschur!(H_im, U_im)   ### U is Z, T is T in the paper https://ianzwaan.com/assets/pdf/kstwo.pdf
            by, rev =  KrylovKit.eigsort(which)
            p =  KrylovKit.sortperm(values; by=by, rev=rev)
            T_im, U_im =  KrylovKit.permuteschur!(T_im, U_im, p)
            f_im = [sum(U_im'[i,j]*f_im[i] for i in 1:K) for j in 1:K]

        
            ### eigenvalue_criteria
            c = eigvecs(H)
            d = eigvecs(H_im)
            r = [norm(fact.r)*abs(f'*c[:,i]) for i in 1:krylovdim]
            s = [norm(fact_im.r)*abs(f_im'*d[:,i]) for i in 1:krylovdim]
            kappa = [1/abs(W.basis[i]'*V.basis[i]) for i in 1:krylovdim]
            rho = [abs((W.basis[i]' * A * V.basis[i])/(W.basis[i]'*V.basis[i])) for i in 1:krylovdim]
            
            ### checking if the remaining basis vectors are small enough
            converged = 0
         
            while converged < length(fact) && (kappa[converged+1]/rho[converged+1] * max(r[converged+1],s[converged+1])) <= tol
                converged += 1
            end
            
            ### not sure whats happening here, check for singular vectors?
            # if eltype(T) <: Real &&
            #    0 < converged < length(fact) &&
            #    T[converged + 1, converged] != 0
            #     converged -= 1
            # end
            if converged >= howmany
                break
            elseif alg.verbosity > 1
                msg = "Arnoldi schursolve in iter $numiter, krylovdim = $K: "
                msg *= "$converged values converged, normres = ("
                msg *= @sprintf("%.2e", abs((kappa[1]/rho[1] * max(r[1],s[1]))))
                for i in 2:howmany
                    msg *= ", "
                    msg *= @sprintf("%.2e", abs((kappa[i]/rho[i] * max(r[i],s[i]))))
                end
                msg *= ")"
                @info msg
            end





            
            numiter == maxiter && break
         
            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
            if eltype(H) <: Real && H[keep + 1, keep] != 0 # we are in the middle of a 2x2 block
                keep += 1 # conservative choice
                keep >= krylovdim &&
                    error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
            end    



            # Restore Arnoldi form in the first keep columns Set Vm = V`Q1, Hm = S11, hm = Q∗1b`.    Set Wm = W`Z1, Km = T11, km = Z∗1 c`
            vl1 = fact.r
            wl1 = fact_im.r

            #Vl
            B = basis(fact)
            KrylovKit.basistransform!(B, view(U, :, 1:keep))
            B_im = basis(fact_im)
            KrylovKit.basistransform!(B_im, view(U_im, :, 1:keep))
            B[keep+1] = residual(fact)/normres(fact)

           
            vm1 = (I - fact.V.basis[1:keep]'*fact.V.basis[1:keep])* vl1
            wm1 =  (I - fact_im.V.basis[1:keep]'*fact_im.V.basis[1:keep])* wl1
            V_star_vl1_h = Matrix{ComplexF64}(undef,keep,keep)
            W_star_wl1_h =  Matrix{ComplexF64}(undef,keep,keep)
            for i in 1:keep
                for j in 1:keep
                    V_star_vl1_h[i,j] = (fact.V.basis[i]' * vl1)*f[j]'
                    W_star_wl1_h[i,j] = (fact_im.V.basis[i]' * wl1)*f_im[j]'
                end
            end
  

            # H and h (and orthogonalizing)
            copyto!(H[1:keep,1:keep],T[1:keep,1:keep] + V_star_vl1_h )
            copyto!(H[keep+1,1:keep],f[1:keep]*norm(vm1))
            copyto!(H_im[1:keep,1:keep],T_im[1:keep,1:keep] +W_star_wl1_h)
            copyto!(H_im[keep+1,1:keep],f_im[1:keep]*norm(wm1)) 

            #shrinking
            shrink!(fact,keep)
            shrink!(fact_im,keep)
             
            copyto!(rayleighquotient(fact), H[1:keep][1:keep])
            copyto!(rayleighquotient(fact_im), H_im[1:keep][1:keep])
           
            H_repacked = [H[i,j] for j in 1:keep-1 for i in 1:j+1]
            H_repacked = append!(H_repacked, H[keep+1,1:keep])
            H_repacked = append!(H_repacked,norm(vm1))
            fact.H = H_repacked
            H_repacked = [H_im[i,j] for j in 1:keep-1 for i in 1:j+1]
            H_repacked = append!(H_repacked, H_im[keep+1,1:keep])
            H_repacked = append!(H_repacked,norm(wm1))
            fact_im.H = H_repacked
            fact.r = (vm1)/norm(vm1)
            fact_im.r =  (wm1)/norm(wm1)

            K = keep     
            numiter += 1
        end

         # expanding. You give it the existing factorization and the lineair map and parameters in iter
        if K < krylovdim 
            fact = expand!(iter, fact; verbosity=alg.verbosity - 2)
            fact_im = expand!(iter_im, fact_im; verbosity=alg.verbosity - 2)
            numops += 1 
        end
    end
    return T, U,T_im,U_im, converged, numiter, numops
end


D=5
L=5
Q = 5
J = 1
h= 1
lambda = 0.079 +  0.060im
function eye(m)
    return Matrix{ComplexF64}(I,m,m)
end

function H_potts_matrix(L,Q=5)
    Potts_H = zeros(ComplexF64,Q^L,Q^L)
    for i in 1:L-1
        Potts_H +=  kron(kron(eye(Q^(i-1)),sum((-J * potts_phase(; q=Q,k=j)).data for j in 1:1:Q-1) + lambda*sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l).data for l in 1:1:Q-1) for j in 1:1:Q-1)), eye(Q^(L-i-1)))+ kron(kron(eye(Q^(i-1)),sum((-h * potts_spin_shift(; q = Q,k=j)).data for j in 1:1:Q-1)),eye(Q^(L-i))) 
    end
    Potts_H += kron(eye(Q^(L-1)),sum((-h * potts_spin_shift(; q = Q,k=j)).data for j in 1:1:Q-1))
    
    
    
    ## bc
    sigma = zeros(ComplexF64,Q,Q)
    tau = zeros(ComplexF64,Q,Q)
    for i in 1:Q
        sigma[i, i] = cis(2*pi*(i-1)/Q)
        tau[i,mod1(i - 1, Q)] = 1
    end
    Potts_H += sum(kron(kron(sigma^k,eye(Q^(L-2))),sigma^k') for k in 1:1:Q-1)
    Potts_H += lambda *sum(sum(  kron(kron(sigma^p, eye(Q^(L-2))), tau'^k * sigma'^p) + kron(kron(tau'^k*sigma^p, eye(Q^(L-2))), sigma'^p) + kron(kron(sigma^k, eye(Q^(L-2))), sigma'^k *tau'^p) + kron(kron(sigma^k*tau'^k, eye(Q^(L-2))), sigma'^k)  for k in 1:1:Q-1) for p in 1:1:Q-1)
    return Potts_H
end
L = 4
A = H_potts_matrix(4)
x₀ = rand(ComplexF64,Q^L)
x₀ = x₀/sqrt(inner(x₀,x₀))
T, UU,T_im,UU_im,info,_,_ = _schursolve_2sided(A,A',x₀,2,:SR,eigsolver)
println(T[1,1])
println(T_im[1,1])



























































# Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1) 
# (h::MPO_∂∂AC)(x) = ∂AC(x, h.o, h.leftenv, h.rightenv);

### investigation the fact that the right eigenvectors are not the same as the left

# # ### RIGHT
# ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
# envs=environments(ψ₀, Potts_H)
# h = ∂∂AC(1, ψ₀, Potts_H, envs) 
# TT, vecs, vals, info = schursolve(h,ψ₀.AC[1],1,:SR,eigsolver)   
# println("right")
# println(TT)

# ### left                       
# println("left")
# ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
# envs=environments(ψ₀, conj(Potts_H)) ### I don't think you have to do transpose since mpo is in the middle
# h = ∂∂AC(1, ψ₀, conj(Potts_H), envs)
# TT_im, vecs_im, vals_im, info_im = schursolve(h,ψ₀.AC[1],1,:SR,eigsolver)

# println(TT_im)
# Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)
# #ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)); GOING to di this later but it would get even more complicated as you have like sorted dictionaries and shit
# ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
# println(typeof(Potts_H.data))

# TT, vecs, vals, info,_,_ = _schursolve_2sided(Potts_H.data,conj(Potts_H.data),ψ₀,2,:SR,eigsolver) 
# println("done")  
# values = KrylovKit.schur2eigvals(TT)
# vectors = let B = basis(vals)
#     [B * u for u in KrylovKit.cols(vecs, 1:2)]
# end


