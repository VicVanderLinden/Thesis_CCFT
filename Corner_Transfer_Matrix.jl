

using TensorKit
using TensorOperations
using KrylovKit
using Base
using LinearAlgebra
using MPSKit
using MPSKitModels
using JLD2





function tmatrix(Q,beta,lambda)
    L = 2
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
        H += -0.5*kron(kron(eye(Q^(i-1)),(sum((tau^k) for k in 1:1:Q-1))),eye(Q^(L-i))) 
        H +=  -kron(kron(eye(Q^(i-1)), sum(kron(sigma'^k,sigma^k) for k in 1:Q-1)), eye(Q^(L-i-1)))

    end
    H += -0.5*kron(eye(Q^(L-1)),sum((tau^k) for k in 1:1:Q-1)) #(not added since this is in the next one)
    
    Transfer_Matrix = exp(-beta*transpose(H) )   #### a little bit wrong since you work with beta/n -> in fractions this is okay but the first two plots are not representative
    V= ℂ^Q
    A_CCFT = TensorMap(Transfer_Matrix, V ⊗ V ← V ⊗ V)   
    return A_CCFT
end
### CMT method (not useful as they don't converge) (only for Q=5)
function construct_tensor_partition(q,beta,chi)
    # test = [-1+exp(beta) for i in 0:q-1]
    # Q = ones((q,q)) + diagm(test)
    # #Q = [exp(beta),exp(-beta),exp(-beta),exp(beta)]
    # #Q = reshape(Q,(q,q))
    # sqrtQ = sqrt(Q)
    # g = zeros((q,q,q,q))
    # for i in 1:q
    #     g[i,i,i,i]=1
    # end
    # a = zeros((q,q,q,q))
    # @tensor begin
    #     a[i,k,m,p]= sqrtQ[i,j]*sqrtQ[k,l]*sqrtQ[m,o]*sqrtQ[p,q]*g[j,l,o,q]
    # end
    # b = zeros((q,q,q,q))
    # for i in 1:q
    #     g[i,i,i,i] = +i
    # end
    # @tensor begin
    #    b[i,k,m,p] = sqrtQ[i,j]*sqrtQ[k,l]*sqrtQ[m,o]*sqrtQ[p,q]*g[j,l,o,q]
    # end

    
    a = reshape(tmatrix(q,beta,0.079+0.06im).data,(q,q,q,q))

    chi0 = q
    C = rand(ComplexF64,(chi0,chi0))
    T = rand(ComplexF64,(chi0,chi0,q))
    S_old = ones(ComplexF64,chi0)
    for i in 1:100000
        println(i)
        s1 = size(a)[1]
        s0 =size(T)[1]
        minsize = s0 * s1
        targetdim = min(chi,minsize)
        Cbig = zeros(ComplexF64,(s0,q,s0,q))
        
        @tensor begin
        Cbig[m,p,l,r] =  C[i,j]*T[i,l,k]*T[m,j,o]*a[p,o,k,r]
        end
        Cbig = reshape(Cbig,(q*size(Cbig)[1],q*size(Cbig)[3]))
        Cbig = 0.5*(Cbig +transpose(Cbig))
        F = eigen(Cbig, sortby =abs)
        S = F.values[end:-1:end-targetdim+1]
        U = F.vectors[:,end:-1:end-targetdim+1]
        #B = reduce(hcat,B)
        U = reshape(U,(s0,s1,size(S)[1]))
        T_old = T
        T = zeros(ComplexF64,(size(S)[1],size(S)[1],q))
        @tensor begin
            T[k,o,q] =  U[i,j,k]*U[l,m,o]*T_old[i,l,p]*a[q,j,p,m]
        end
        T = 1/2 * (T + permutedims(T,(2,1,3)))
        T = normalize(T)
        S = normalize(S)
        C = diagm(S)
        if ((size(S) == size(S_old)))
            println(sum(abs.(abs.(S)-abs.(S_old))))
        end
        if ((size(S) == size(S_old)) && ((sum(abs.(abs.(S)-abs.(S_old)))) < 10^(-10)))
                println("converged")
            break
            end
        S_old = S
    end
    ## calculate partition function and magnetisation
    #@tensor begin 
    #Z = a[a,b,c,d]*T[e,f,a]*T[g,h,b]*T[i,j,c]*T[k,l,d]*C[e,g]*C[h,i]*C[j,k]*C[l,f]
    #k =  b[a,b,c,d]*T[e,f,a]*T[g,h,b]*T[i,j,c]*T[k,l,d]*C[e,g]*C[h,i]*C[j,k]*C[l,f]
    #end"

    # ## calculate correlation length
    M = zeros((chi,chi,chi,chi))
    @tensor begin
         M[a,c,b,d] = T[a,b,e]*T[c,d,e]
    end 
    M = reshape(M,(chi^2,chi^2))
    F = eigen(M)
    lambda_0 = F.values[end]
    lambda_1 = F.values[end-1]
    println(lambda_0)
    println(lambda_1)
 
    # return -1/beta * log(k/Z)
    return (1/log(lambda_0/lambda_1))
end     
function eye(m)
    return Matrix{ComplexF64}(I,m,m)
end
D=55
Q = 5 
construct_tensor_partition(Q,log(1+sqrt(Q)),D,)
# Free energy ifo tempature for different chi

Results =[]
plot()
# plot!([2.269], seriestype="vline")
append!(Results,construct_tensor_partition(Q,log(1+sqrt(Q)),10))
plot!(q, Results,label="chi = 20, T=Tc")
plot!(xlabel = "Q", ylabel = "corrlength", title = "Potts model with Q spins")

# ### calculate F scaling
# dimension = 6:1:14
# F =[]
# for d in dimension
#     append!(F,construct_tensor_partition(2,1/2.269,6))
# end
# plot(dimension,F,label = "scaling" ,seriestype=:scatter) 
# plot!(xlabel = "bond dimension", ylabel = "F", title = "free energy scaling (CMT)")
# plot!(dimension, dimension.^(1/1.3) .- 0.8  .+ 2.3,label = "0.769 power law scaling")


# # attempt to get central charge from bond scaling of correlation length
# dimension = 6:1:25 
# e =[]
# for d in dimension
#      append!(e,construct_tensor_partition(4,1.0986122886681096913952452369225257046474905578227494517346943336,d))
# end
# using CurveFit
# using LaTeXStrings
# a,b = power_fit(dimension,e)
# plot(dimension,e,label = "scaling" ,seriestype=:scatter) 
# plot!(xlabel = "bond dimension", ylabel = "Correlation length", title = "Potts model (q=4) Correlation length (CMT)")
# plot!(dimension, (a-0.02)*dimension.^(1.3440554264387752293026687361882244003012420950658697760608803412),label = "c=1.0000 power law",xaxis=:log, yaxis=:log)      
# println(b)
# println(a)
# plot!(dimension,a*dimension.^b,label = "c = 1.0306 power law" )