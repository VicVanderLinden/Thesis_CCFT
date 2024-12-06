using Printf
using MPSKit
using KrylovKit
const Selector = Union{Symbol,EigSorter}
include("Potts-Operators & Hamiltonian.jl")
include("Krylov-Shur-Decomposition.jl")
using LinearAlgebra
const maxiter = 200
const tolgauge = 1e-13
const tol = 1e-10
const eigsolver = Arnoldi(; tol, maxiter, eager=true,verbosity=3)


D=5
L=4
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

function _schursolve_2sided(A,Astar, x₀, howmany::Int, which::Selector,alg::Arnoldi)
 
    krylovdim = 30
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize krylov-shur factorization
    iter = KrylovShurIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity - 2)
    numops = 1
    # sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim+1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)
    
    # ## do the same for A*
    # initialize krylovshur factorization
    iter_im = KrylovShurIterator(Astar, x₀, alg.orth)
    fact_im = initialize(iter_im; verbosity=alg.verbosity - 2)
    #sizehint!(fact_im, krylovdim)
    β_im = normres(fact_im)

    # allocate storage
    HH_im = fill(zero(eltype(fact_im)), krylovdim+1, krylovdim)
    UU_im = fill(zero(eltype(fact_im)), krylovdim, krylovdim)
    
    # initialize storage, same dimension on decomposition
    K = length(fact) # == 1
    converged = 0
    local eigenvectors,eigenvectors_im,eigenvalues,eigenvalues_im
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
            f = view(HH, K+1,1:K)
     
            copyto!(U, I) 
            copyto!(H, fact.H[1:K,1:K]) # changed since rayleigh coeff is no longer hessenberg
            copyto!(f, fact.H[K+1,1:K]) 

            ### Left
            H_im = view(HH_im, 1:K, 1:K)
            U_im= view(UU_im, 1:K, 1:K)
            W = fact_im.V
            f_im = view(HH_im,K+1,1:K)
            copyto!(U_im, I)
            copyto!(H_im, fact_im.H[1:K,1:K]) # changed since rayleigh coeff is no longer hessenberg
            copyto!(f_im, fact_im.H[K+1,1:K]) 

            ## Changing to the rayleighquotients before decomposing
            W_star_V = Matrix{ComplexF64}(undef,krylovdim,krylovdim)
            V_star_W = Matrix{ComplexF64}(undef,krylovdim,krylovdim)
            for i in 1:krylovdim
                for j in 1:krylovdim
                    W_star_V[i,j] =(W.basis[i]'*V.basis[j])
                    V_star_W[i,j] =(V.basis[i]'*W.basis[j])
                end
            end
            M = inv(W_star_V)
            M_star = inv(V_star_W)
            W_star_r = Vector{ComplexF64}(undef,krylovdim)
            V_star_r = Vector{ComplexF64}(undef,krylovdim)
            for i in 1:krylovdim
                W_star_r[i] = W.basis[i]'*fact.r
                V_star_r[i] = V.basis[i]'*fact_im.r
            end
         
        
            
            ### H and K
            H = H + M *W_star_r*f'      
            H_im = H_im + M_star *V_star_r*f_im'
            
            ## vl+1 and wl+1
            V_M_W_star = sum(sum(V.basis[i]*M[i,j]*W.basis[j]' for i in 1:krylovdim) for j in 1:krylovdim)
            W_M_star_V_star = sum(sum(W.basis[i]*M_star[i,j]*V.basis[j]' for i in 1:krylovdim) for j in 1:krylovdim)
            fact.r = (I - V_M_W_star)*fact.r
            fact_im.r = (I - W_M_star_V_star)*fact_im.r
        
            
            # compute dense schur factorization
            T, U, values = KrylovKit.hschur!(H, U)   ### U is Q, T is S in the paper https://ianzwaan.com/assets/pdf/kstwo.pdf
            by, rev =  KrylovKit.eigsort(which)
            p =  KrylovKit.sortperm(values; by=by, rev=rev)
            T, U =  KrylovKit.permuteschur!(T, U, p)
            f =  [sum(U'[i,j]*f[i] for i in 1:1:krylovdim) for j in 1:1:krylovdim]



            # compute dense schur factorization
            T_im, U_im, values = KrylovKit.hschur!(H_im, U_im)   ### U is Z, T is T in the paper https://ianzwaan.com/assets/pdf/kstwo.pdf
            by, rev =  KrylovKit.eigsort(which)
            p =  KrylovKit.sortperm(values; by=by, rev=rev)
            T_im, U_im =  KrylovKit.permuteschur!(T_im, U_im, p)
            f_im =  [sum(U_im'[i,j]*f_im[i] for i in 1:1:krylovdim) for j in 1:1:krylovdim]


            #Vl
            B = basis(fact)
            KrylovKit.basistransform!(B, view(U, :, 1:K))
            B_im = basis(fact_im)
            KrylovKit.basistransform!(B_im, view(U_im, :, 1:K))
            

            ### eigenvalue_criteria
            c = eigvecs(T) 
            d = eigvecs(T_im)
            
                    
            r = [norm(fact.r)*abs(sum(f'[j]*c[j,i] for j in 1:krylovdim)) for i in 1:krylovdim]
            s = [norm(fact_im.r)*abs(sum(f_im'[j]*d[j,i] for j in 1:krylovdim)) for i in 1:krylovdim]
            kappa = [1/abs(W.basis[i]'*V.basis[i]) for i in 1:krylovdim]
            rho = [abs((W.basis[i]' * A * V.basis[i])/(W.basis[i]'*V.basis[i])) for i in 1:krylovdim]
            ### checking if the remaining basis vectors are small enough

            converged = 0
            while converged < length(fact) && (kappa[converged+1]/abs(rho[converged+1]) * max(r[converged+1],s[converged+1])) <= tol
                converged += 1
            end
            converged = converged
            
            #check for artifacts/vectors that aren't really diagonal?
            if 0 < converged < length(fact) && (T[converged + 1, converged] != 0)
                converged -= 1
            end
            for i in 1:howmany
                if abs(T[i,i] -T_im[i,i]') >= 2*tol
                    converged -= 1
                end
            end

            if converged >= howmany
                eigenvectors  = B[1:howmany]
                eigenvectors_im = B_im[1:howmany]
                eigenvalues = [T[i,i] for i in 1:howmany]
                eigenvalues_im = [T_im[i,i] for i in 1:howmany]
                break
            elseif alg.verbosity > 1
                msg = "Two sided schursolve in iter $numiter, krylovdim = $K: "
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

            ##Set Vm = V`Q1, Hm = S11,.    Set Wm = W`Z1, Km = T11,
            vl1 = fact.r
            wl1 = fact_im.r

          
            ## vl+1
            vm1 = (I - B[1:keep]'*B[1:keep])* vl1
            wm1 =  (I - B_im[1:keep]'*B_im[1:keep])* wl1
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
            while length(B) > keep
                pop!(B)
            end
            while length(B_im) > keep
                pop!(B_im)
            end
            fact.k = keep
            fact_im.k = keep
            fact.r = (vm1)/norm(vm1)
            fact_im.r =  (wm1)/norm(wm1)
            fact.H = H[1:keep+1,1:keep]
            fact_im.H = H_im[1:keep+1,1:keep]
            numiter += 1
        end

         # expanding. You give it the existing factorization and the lineair map and parameters in iter
        if K < krylovdim 
            fact = expand!(iter, fact; verbosity=alg.verbosity - 2)
            fact_im = expand!(iter_im, fact_im; verbosity=alg.verbosity - 2)
            numops += 1 
        end
    end
    return eigenvectors,eigenvectors_im,eigenvalues, eigenvalues_im , converged, numiter, numops
end



D=5
L=4
Q = 3
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

#quick look
# L = 3
# A = H_potts_matrix(L)
# x₀ = rand(ComplexF64,Q^L)
# x₀ = x₀/sqrt(inner(x₀,x₀))
# eigenvectors,eigenvectors_im,eigenvalues, eigenvalues_im,info,_,_ = _schursolve_2sided(A,A',x₀,1,:SR,eigsolver)
# println(eigenvalues[1])
# println(norm(eigenvectors[1]))








# #INVESTIGATING AND PLOTTING
# eig_val = zeros(ComplexF64,6)
# eig_val_im = zeros(ComplexF64,6)
# eig_vec = Vector{ComplexF64}[]
# eig_vec_im = Vector{ComplexF64}[]
# q=3
# for L in 2:1:7
#     A = H_potts_matrix(L,q)
#     x₀ = rand(ComplexF64,q^L)
#     x₀ = x₀/sqrt(inner(x₀,x₀))
#     eigenvectors,eigenvectors_im,eigenvalues, eigenvalues_im,info,_,_ = _schursolve_2sided(A,A',x₀,1,:SR,eigsolver)
#     eig_val[L-1] = eigenvalues[1]
#     eig_val_im[L-1] = eigenvalues_im[1]
#     push!(eig_vec,eigenvectors[1])
#     push!(eig_vec_im,eigenvectors_im[1])
#     print(L)
# end
# using JLD2
# save_object("E_gr Q=3", eig_val)
# save_object("E_gr_im Q=3", eig_val_im)
# save_object("vec_gr Q=3", eig_vec)
# save_object("vec_gr_im Q=3", eig_vec_im)

# using Plots 
# using LaTeXStrings
# using JLD2

# eig_val = load_object("E_gr Q=3")
# eig_val_im = load_object("E_gr_im")
# eig_vec = load_object("vec_gr Q=3")
# eig_vec_im = load_object("vec_gr_im")
# L_list = [2,3,4,5,6,7]

# # p = plot(; xlabel="L", ylabel="real(E0/L)")
# # p = plot!(L_list,real(eig_val./L_list) ; seriestype=:scatter,label = "H")
# # p = plot!(L_list,real(eig_val_im./L_list) ; seriestype=:scatter,label =L"H^{†}")
# # savefig(p,"Eigval_LandR.png")
# # p = plot(; xlabel="L", ylabel="im(E0/L)")
# # p = plot!(L_list,real(-1im.*eig_val./L_list) ; seriestype=:scatter,label = "H0")
# # p = plot!(L_list,real(-1im.*eig_val_im./L_list) ; seriestype=:scatter,label = L"H^{†}")
# # savefig(p,"im Eigval_LandR.png")

# p = plot(title = " Approximate eigenvectors"; xlabel="L", ylabel=L"  $|< ψ_L | ψ_R >|$ ")
# p = plot!(L_list,[abs(inner(eig_vec_im[i],eig_vec[i])) for i in 1:6],seriestype=:scatter,label = L"$|< ψ_L | ψ_R >|$")
# #ylims!((0.99999,1.00001))
# savefig(p,"Approximate eigenvectors inproduct.png")
# # inner(eig_vec_im[end],(H_potts_matrix(7,q)')eig_vec_im[end])
# # println(eig_val[end])

# p = plot(title = " Approximate eigenvectors Q=3"; xlabel="L", ylabel=L"  $ \frac{1}{E_i} |< ψ_i| H_i | ψ_i >|$ ")
# p = plot!(L_list,[abs((1/eig_val[i])*inner(eig_vec[i],H_potts_matrix(i+1,q)*eig_vec[i])) for i in 1:6],seriestype=:scatter,label = L"$\frac{1}{E_i} | < ψ_R| H | ψ_R > |$")
# p = plot!(L_list,[abs((1/eig_val_im[i])*inner(eig_vec_im[i],H_potts_matrix(i+1,q)'*eig_vec_im[i])) for i in 1:6],seriestype=:scatter,label =  L"$\frac{1}{E_i} |< ψ_L |  H^{†} | ψ_L >|$")
# ylims!((0.99999,1.00001))
# savefig(p,"Ground vector_dif Q=3.png")
# inner(eig_vec_im[end],(H_potts_matrix(7,q)')eig_vec_im[end])
# println(eig_val[end])