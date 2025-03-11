##### Complex entanglement entropy for complex conformal field theory #####

## attempt to recreate the entanglement scaling


using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials
using LinearAlgebra 
using JLD2
include("Potts-Operators & Hamiltonian.jl")
Q=5

param = 0.079 -0.06im
Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)

## partial
"""Perform the calculation sum_{i} x[i, i, ..., i]"""
function _tr_all_dims(x::AbstractArray{T,N}) where {T,N}
     sum(i -> x[fill(i, N)...], axes(x, 1))
 end

function tr(x::AbstractArray; dims)
    mapslices(_tr_all_dims, x; dims=dims)
end




using MPSKit: decompose_localmps

function run_sum(L_list,Q,D)
for (i,L) in enumerate(L_list)
    H = Potts_Hamiltonian(L,Q=Q,sym=false)
    H_star = Potts_Hamiltonian(L,Q=Q,sym=false,adjoint=true)
    ψ₀ = FiniteMPS(L,ℂ^Q, ℂ^D);
    #ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](sector=>D for sector in 0:Q-1)) #;left=Vleft, right=Vright)

    (ψ_right, envir , delta) = find_groundstate(ψ₀, H, DMRG2(maxiter = 500,tol=1e-10, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    (ψ_left, envir , delta) = find_groundstate(ψ₀, H_star, DMRG2(maxiter = 500,tol=1e-10, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    
    ############################################### Kawisha method I: exact diagonalization/full state  #################################################
   
    ### contracting alll (non simplified with Z_Q)
    sizesL = size(ψ_left.AL[1].data)
    contracted_left = reshape(ψ_left.AL[1].data, (Int(sizesL[1]/Q),Q))
    sizesR = size(ψ_right.AL[1].data)
    contracted_right = reshape(ψ_right.AL[1].data, (Int(sizesR[1]/Q),Q))
    for i in 2:L
        size_contr_l = size(contracted_left)
        size_contr_r = size(contracted_right)
        sizesL = size(ψ_left.AL[i].data) 
        left = reshape(ψ_left.AL[i].data, (Int(sizesL[1]/(Q*size_contr_l[1])),Q,size_contr_l[1]))
        sizesR = size(ψ_right.AL[i].data)
        right = reshape(ψ_right.AL[i].data, (Int(sizesR[1]/(Q*size_contr_r[1])),Q,size_contr_r[1]))
       
        contracted_old_left = contracted_left
        @tensor contracted_left[a,c,d] := contracted_old_left[e,c]*left[a,d,e]
        contracted_left = reshape(contracted_left,(Int(sizesL[1]/(Q*size_contr_l[1])),Q^(i)))
     
        contracted_old_right = contracted_right
        @tensor contracted_right[b,c,d] := contracted_old_right[e,c]*right[b,d,e]
        contracted_right = reshape(contracted_right,(Int(sizesR[1]/(Q*size_contr_r[1])),Q^(i)))
    end
    contracted_right = contracted_right.*(1/sqrt(contracted_right*contracted_right')[1])
    contracted_left = contracted_left.*(1/sqrt(contracted_left*contracted_left')[1])
    rho_AB = contracted_right'*contracted_left
    rho_AB = rho_AB/((contracted_left*contracted_right')[1])
    println("is it normalized:",sqrt((contracted_right*rho_AB*contracted_right'))) ### properly normaized
    ### Taking subtrace and diagonalizing this (with DMRG?)
    ent = zeros(ComplexF64,(L+1,))
    rho=rho_AB
    x = eigvals(rho)
    s = sum(-lambda*log(lambda) for lambda in x)
    println(s)
    ent[L+1] = s
    for i in 1:L
        rho_old = rho
        rho_old = reshape(rho,Int(size(rho)[1]/(Q)),Q,Int(size(rho)[1]/(Q)),Q)
        rho= Matrix{ComplexF64}(undef,(Q^(L-i),Q^(L-i)))
        @tensor rho[a,b] := rho_old[a,c,b,c]
        if size(rho)[1] < 5^6
     
            #rho_alt = tr(rho_old,dims=(2,4))
            #rho_alt =  reshape(rho_alt,Int(size(rho)[1]),Int(size(rho)[1])) ## both methods are the same 
            x = eigvals(rho)
            s = sum(-lambda*log(lambda) for lambda in x)
            println(s)
            ent[L+1-i] = s
        end
    end
   
   
   
   
   
   
   
   

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   ### other attempts
    # ψ_left.C[i]
    # U, ncr, = tsvd(ψ_left; trunc=SvdCut, alg=TensorKit.SVD())
    # println(U)
    # println(ncr)
    ### if the hamiltonian was hermitian
    ## println(entanglement_spectrum(ψ_right, 2))
    ### println(entanglement_spectrum(ψ_left, 2))
    # wrong -> svd gives real always, its because for SVD^2 represent eigenvalues of Martix' Matrix but here we are working with different states



    ## if the hamiltonian is not hermitian

    # ent = zeros(ComplexF64,(L,))
    # # Eigenvectors and singular vectors are them same if A is a real symmetric matrix .
    # for i in 1:L
    #     sizesL = size(ψ_left.AC[i].data)
    #     left = reshape(ψ_left.AC[i].data, (Int(sizesL[1]/Q),Q,sizesL[2]))
    #     sizesR = size(ψ_right.AC[i].data)
    #     right = reshape(ψ_right.AC[i].data, (Int(sizesR[1]/Q),Q,sizesR[2]))
    #     # rho = zeros(ComplexF64,(Int(sizesL[1]/Q),Int(sizesR[1]/Q),sizesL[2],sizesR[2]))
    #     # @tensor temp[a,b,c,d] =  left[a,e,c]*right[b,e,d]
    #     # rho = reshape(rho, (Int(sizesL[1]/Q)*Int(sizesR[1]/Q),(sizesL[2]*sizesR[2])))
    #     # S = LinearAlgebra.svdvals(rho)
    #     # ent[i] = sum([-s^2*log(s^2) for s in S])
    # end





    ### we would think we could just left gauge it and then it would fall away, however, we are not contracting with the same but with a different vector so its not possible
   ### i think


   ### contracting alll

    ########################## non simplified with Z_Q
    # sizesL = size(ψ_left.AL[1].data)
    # contracted_left = reshape(ψ_left.AL[1].data, (Int(sizesL[1]/Q),Q))
    # sizesR = size(ψ_right.AL[1].data)
    # contracted_right = reshape(ψ_right.AL[1].data, (Int(sizesR[1]/Q),Q))
    # for i in 2:L
    #     println(i)
    #     size_contr_l = size(contracted_left)
    #     size_contr_r = size(contracted_right)
    #     sizesL = size(ψ_left.AL[i].data)
    #     println(sizesL) 
    #     left = reshape(ψ_left.AL[i].data, (Int(sizesL[1]/(Q*size_contr_l[1])),size_contr_l[1],Q))
    #     sizesR = size(ψ_right.AL[i].data)
    #     right = reshape(ψ_right.AL[i].data, (Int(sizesR[1]/(Q*size_contr_r[1])),size_contr_r[1],Q))
       
    #     contracted_old_left = contracted_left
    #     @tensor contracted_left[a,c,d] := contracted_old_left[e,c]*left[a,e,d]
    #     contracted_left = reshape(contracted_left,(Int(sizesL[1]/(Q*size_contr_l[1])),Q^(i)))
     
    #     contracted_old_right = contracted_right
    #     @tensor contracted_right[b,c,d] := contracted_old_right[e,c]*right[b,e,d]
    #     contracted_right = reshape(contracted_right,(Int(sizesR[1]/(Q*size_contr_r[1])),Q^(i)))
    # end


    ##########################################" Simplified with Z_Q? 
    # sizesL = size(ψ_left.AL[1].data)
    # Qs = 1
    # contracted_left = reshape(ψ_left.AL[1].data, (Int(sizesL[1]/Qs),Qs))
    # sizesR = size(ψ_right.AL[1].data)
    # contracted_right = reshape(ψ_right.AL[1].data, (Int(sizesR[1]/Qs),Qs))
    # for i in 2:L
    #     size_contr_l = size(contracted_left)
    #     size_contr_r = size(contracted_right)
    #     sizesL = size(ψ_left.AL[i].data)
    #     left = reshape(ψ_left.AL[i].data, (Int(sizesL[1]/(Qs*size_contr_l[1])),size_contr_l[1],Qs))
    #     sizesR = size(ψ_right.AL[i].data)
    #     right = reshape(ψ_right.AL[i].data, (Int(sizesR[1]/(Qs*size_contr_r[1])),size_contr_r[1],Qs))

    #     contracted_old_left = contracted_left
    #     @tensor contracted_left[a,c,d] := contracted_old_left[e,c]*left[a,e,d]
    #     contracted_left = reshape(contracted_left,(Int(sizesL[1]/(Qs*size_contr_l[1])),Qs^(i)))

    #     contracted_old_right = contracted_right
    #     @tensor contracted_right[b,c,d] := contracted_old_right[e,c]*right[b,e,d]
    #     contracted_right = reshape(contracted_right,Int(sizesR[1]/(Qs*size_contr_r[1])),Qs^(i))
    # end




    ### working with each individually:     https://arxiv.org/pdf/2311.18733
    # println(size(ψ_left.AL[1].data))
    # normalization_factor = sum(ψ_left.AL[i].data'*ψ_right.AL[i].data for i in 1:L)
    # println(normalization_factor)
    # ψ_left = [ψ_left.AL[i] for i in 1:L]
    # ψ_right = [(1/normalization_factor)*ψ_right.AL[i] for i in 1:L]
    # normalization_factor = sum(ψ_left[i].data'*ψ_right[i].data for i in 1:L)
    # println(normalization_factor)
    # ent = zeros(ComplexF64,(L,))
    # for i in 1:L 
    #     ψ_left.C[i]
    #     ψ_right.C[i]
    #     rho =  ψ_left.C[i]'* ψ_right.C[i]
    #     factor =  ψ_right.C[i].data'*ψ_left.C[i].data 
    #     n = 0.999
    #     s = 1/(1-n) * log(tr(factor^n*rho))
    #     ent[i] = s
    # end

    # p = plot(1:L,real(ent))
    # savefig(p,"test ent 1") 
    # p = plot(1:L,real(-1im.*ent))
    # savefig(p,"test ent 1 im") 









    # rho = Matrix{ComplexF64}(undef,(Q^L,Q^L))
    # factor = (contracted_right* contracted_left')[1]
    # println(factor)
    # contracted_left = contracted_left./(factor)
    # contracted_right = contracted_left./(factor')
    # @tensor rho[a,b] = contracted_left'[a,c]*contracted_right[c,b]
    # println(tr(rho))
    # ent = zeros(ComplexF64,(L+1,))
    # println("heere")
    # n = 0.9999
    # s = 1/(1-n) * log(tr(factor^n.*rho))
    # println("heere")
    # ent[1] = s
    # println(s)
    # for i in 1:L
    #     rho_old = rho
    #     rho_old = reshape(rho,Int(size(rho)[1]/(Q)),Q,Int(size(rho)[1]/(Q)),Q)
    #     rho= Matrix{ComplexF64}(undef,(Q^(L-i),Q^(L-i)))
    #     @tensor rho[a,b] = rho_old[a,c,b,c]
    #     s = 1/(1-n) * log(tr(rho^n))
    #     println(s)
    #     ent[i+1] = s
    # end
    
    
    return ent

end

end

L = 5
D = 200
results = run_sum([L,],5,D)
plot(0:L,real(-1im.*results),title="Imaginary entropy of L=$L,D=$D")
plot(0:L,real(results),xlabel = "subsystem size l",ylabel = "Re(S)",title="real entropy of L=$L,D=$D")

   
   
   
   