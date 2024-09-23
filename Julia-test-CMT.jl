using TensorKit
using TensorOperations
using KrylovKit
using Plots
using Base
using LinearAlgebra

function construct_tensor_partition(q,beta,chi)
    test = [-1+exp(beta) for i in 0:q-1]
    Q = ones((q,q)) + diagm(test)
    #Q = [exp(beta),exp(-beta),exp(-beta),exp(beta)]
    #Q = reshape(Q,(q,q))
    sqrtQ = sqrt(Q)
    g = zeros((q,q,q,q))
    for i in 1:q
        g[i,i,i,i]=1
    end
    a = zeros((q,q,q,q))
    @tensor begin
        a[i,k,m,p]= sqrtQ[i,j]*sqrtQ[k,l]*sqrtQ[m,o]*sqrtQ[p,q]*g[j,l,o,q]
    end
    # b = zeros((q,q,q,q))
    # for i in 1:q
    #     g[i,i,i,i] = +i
    # end
    # @tensor begin
    #    b[i,k,m,p] = sqrtQ[i,j]*sqrtQ[k,l]*sqrtQ[m,o]*sqrtQ[p,q]*g[j,l,o,q]
    # end
    chi0 = q
    C = rand(Float64,(chi0,chi0))
    T = rand(Float64,(chi0,chi0,q))
    S_old = ones(chi0)
    for i in 1:100000
        s1 = size(a)[1]
        s0 =size(T)[1]
        minsize = s0 * s1
        targetdim = min(chi,minsize)
        Cbig = zeros((s0,q,s0,q))
        @tensor begin
        Cbig[m,p,l,r] =  C[i,j]*T[i,l,k]*T[m,j,o]*a[p,o,k,r]
        end
        Cbig = reshape(Cbig,(q*size(Cbig)[1],q*size(Cbig)[3]))
        Cbig = 0.5*(Cbig +transpose(Cbig))
        xo = ones(size(Cbig)[1])
        #S,B,C = eigsolve(Cbig, xo, targetdim,:LR)
        F = eigen(Cbig)
        S = F.values[end:-1:end-targetdim+1]
        U = F.vectors[:,end:-1:end-targetdim+1]
        #B = reduce(hcat,B)
        U = reshape(U,(s0,s1,size(S)[1]))
        T_old = T
        T = zeros((size(S)[1],size(S)[1],q))
        @tensor begin
            T[k,o,q] =  U[i,j,k]*U[l,m,o]*T_old[i,l,p]*a[q,j,p,m]
        end
        T = 1/2 * (T + permutedims(T,(2,1,3)))
        T = normalize(T)
        S = normalize(S)
        C = diagm(S)
        if ((size(S) == size(S_old)) && (sum(broadcast(abs, S-S_old)) < 10^(-10)))
                println("converged")
            break
            end
        S_old = S
    end
    ## calculate partition function and magnetisation
    #@tensor begin 
    #Z = a[a,b,c,d]*T[e,f,a]*T[g,h,b]*T[i,j,c]*T[k,l,d]*C[e,g]*C[h,i]*C[j,k]*C[l,f]
    #k =  b[a,b,c,d]*T[e,f,a]*T[g,h,b]*T[i,j,c]*T[k,l,d]*C[e,g]*C[h,i]*C[j,k]*C[l,f]
    #end

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



## Free energy ifo tempature for different chi
q = 5:10 
Results =[]
plot()
# plot!([2.269], seriestype="vline")
for Q in q
    append!(Results,construct_tensor_partition(Q,log(1+sqrt(Q)),10))
end

plot!(q, Results,label="chi = 30, T=Tc")
plot!(xlabel = "Q", ylabel = "corrlength", title = "Potts model with Q spins")

#### calculate F scaling
# dimension = 6:1:14
# F =[]
# for d in dimension
#     append!(F,construct_tensor_partition(2,1/2.269,6))
#end
# plot(dimension,F,label = "scaling" ,seriestype=:scatter) 
# plot!(xlabel = "bond dimension", ylabel = "F", title = "free energy scaling (CMT)")
# plot!(dimension, dimension.^(1/1.3) .- 0.8  .+ 2.3,label = "0.769 power law scaling")


## attempt to get central charge from bond scaling of correlation length
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