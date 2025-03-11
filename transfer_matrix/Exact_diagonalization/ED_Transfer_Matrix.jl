using TensorKit
using TensorOperations
using KrylovKit
using Base
using LinearAlgebra
using MPSKit
using MPSKitModels
using JLD2
## 2N-1 parameter square (for any N), that can be focused on one point
N = 5
test_values = zeros(ComplexF64,(2*N-1)^2)
l = length(test_values)
distx = 0.1## distance from alleged fixed point 0.079+0.060i in real
disty = 0.1 # distance from alleged fixed point 0.079+0.060i in imaginary

## snake like structure of test_values (to compare)
for i in 1:1:(2*N-1)
    if div(i,2) == 1
        for j in 1:1:(2*N-1)
            if i <N+1
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+0 + 1im*LinRange(-disty,0.00,N)[j] .+0im
                else 
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+0 + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+0im
                end
            else
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+ 0  + 1im*LinRange(-disty,0.00,N)[j]  .+0im
                else
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N])   .+ 0 + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+0im
                end
            end 
        end
    else
        for j in (2*N-1):-1:1
            if i <N+1
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+0 + 1im*LinRange(-disty,0.00,N)[j] .+0im
                else 
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+0 + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+0im
                end
            else
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+ 0  + 1im*LinRange(-disty,0.00,N)[j]  .+0im
                else
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N])   .+ 0 + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+0im
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
    
    Transfer_Matrix = exp(-log(1+sqrt(Q))*H) #### a little bit wrong since you work with beta/n -> in fractions this is okay but the first two plots are not representative
    eigenvalues = eigvals(Transfer_Matrix)
    return eigenvalues
end 



using PlotlyJS
using Polynomials


function plot_difference(results,z_values,dif,test_values,Q,L)
   
    p = PlotlyJS.plot(PlotlyJS.contour(   z=real(z_values),
    x=real(-im*test_values),
        y=real(test_values),fill=true,colorbar=attr(
            title="Re(F)", # title here
            titleside="top",
            titlefont=attr(
                size=14,
                family="Arial, sans-serif"
            )
        )),Layout(title=attr(text = "Exact Free energy of CCFT transfermatrix L=$L,Q=$Q",x = 0.5),xaxis_title="Im(lambda)",yaxis_title="Re(lambda)"))


PlotlyJS.savefig(p,"Q$Q  fig 1 Free energy of CCFT L=$L.png") 
p = PlotlyJS.plot(PlotlyJS.contour(   z=real(-1im*z_values),
    x=real(-im*test_values),
        y=real(test_values),fill=true,colorbar=attr(
            title="Im(F)", # title here
            titleside="top",
            titlefont=attr(
                size=14,
                family="Arial, sans-serif"
            )
        )),Layout(title=attr(text = "Im exact free energy of CCFT transfermatrix L=$L,Q=$Q",x = 0.5),xaxis_title="Im(lambda)",yaxis_title="Re(lambda)"))
PlotlyJS.savefig(p,"Q$Q fig 2 Im Free energy of CCFT L=$L.png") 
p = PlotlyJS.plot(PlotlyJS.contour(   z=dif,
    x=real(-im*test_values),
        y=real(test_values),fill=true,colorbar=attr(
            title="|Im(δx_σ)|", # title here
            titleside="top",
            titlefont=attr( 
                size=14,
                family="Arial, sans-serif"
            )
        )),Layout(title=attr(text = "Exact CCFT|Re(δx_σ)| L=$L,Q=$Q",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)"))
PlotlyJS.savefig(p,"Q$Q  fig 3 exact scaling energy (only real) of CCFT L=$L,.png") 
pp = PlotlyJS.plot(PlotlyJS.contour(   z=log.(real(1 ./ results)),
    x=real(-im*test_values),
        y=real(test_values),fill=true,colorbar=attr(
            title="log(ξ)", # title here
            titleside="top",
            titlefont=attr( 
                size=14,
                family="Arial, sans-serif"
            )
        )),Layout(title=attr(text = "Exact (C)CFT ξ L=$L,Q=$Q",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)"))
PlotlyJS.savefig(pp,"Q$Q fig 4 exact ξ of L=$L.png") 
end

l = length(test_values)
L = 4
Q = 5
eigenvals = zeros(ComplexF64,(Q^L,l))


########################
runs = false
if runs == true
    for (n,parameter) in enumerate(test_values)
        println(n/l)
        eig = tmatrix(Q,L,parameter)[1:Q^L]
        eigenvals[:,n] = eig
        eig = sort(eig,by = x->abs(x),rev=true)
        println(eig[1])
    end 
    save_object("exact diag CCFT_im $L",eigenvals)
end
###########################################



eigenvals = load_object("exact diag CCFT_real $L")
Results = zeros(ComplexF64,(l))
target = 0.134 − 0.021im

#target = 2/15
dif = zeros(Float64,(l))
z_values = zeros(ComplexF64,(l))    
for j in 1:size(eigenvals)[2]
    eig = sort(eigenvals[:,j],by = x->abs(x),rev=true)
    Results[j] = log(((abs(eig[1])/abs(eig[2]))))
    z_values[j] = -1/log(1+sqrt(Q)) * log((eig[1]))/L
    dif[j] = log(abs(real((log(eig[1]/eig[2]) - log(1+sqrt(Q))*(2*pi/L)*target))))
end

plot_difference(Results,z_values,dif,test_values,Q,L)


































#### code that looks at highest eigenvalue evolving
# println(tmatrix(T)) 
# T =normalize(T) 
# size = 200
# eigenvals = zeros(ComplexF64,(25,size)) 
# global A = T    
# for n in 1:size
#     eigenvals[:,n] = tmatrix(A)[1:25]
#     global A=A*T
#     global A =normalize(A)
# end 
# println(eigenvals[:])
# plot()
# plot!(1:size,[sort(real(eigenvals[:,j]),by = abs,rev=true)[4] for j in 1:size],label = "Re lambda" )  
# plot!(1:size,[sort(real(-1im.*eigenvals[:,j]),by = abs,rev=true)[4] for j in 1:size],label = "Im lambda" )  
# plot!(xlabel = "m", ylabel = "eigenvalues", title = "Largest eigenvalue of normalized T^m")




















