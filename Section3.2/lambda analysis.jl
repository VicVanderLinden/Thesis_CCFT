 # ## fitting
        
# fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) +abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1σ) -ΔL1σ  -Cε_primeσσ* AL1σε_prime* (x[2]+1im*x[4]))
  # res = optimize(fun, [0.0, 0.0,0.0,0.0])
        # gε_prime[i] = Optim.minimizer(res)[2]+1im* Optim.minimizer(res)[4]
        # println(gε_prime)
        # fun_2(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) 
        # res_2 = optimize(fun_2, [0.0, 0.0,0.0,0.0])
        # gε_prime_wo_C[i] = Optim.minimizer(res_2)[2]+1im* Optim.minimizer(res_2)[4]
        # Eε[i] = ΔEε
        # Eσ[i] = ΔEσ
        # EL1ε[i] = ΔEL1ε
        # EL1σ[i] = ΔEL1σ7



# ### lambda analysis
using JLD2
using Plots
using LsqFit
N = 7

test_values = zeros(ComplexF64,(2*N-1)^2)

l = length(test_values)


distx = 0.0007## distance from centre in real

disty = 0.0007# distance from centre in imaginary

cent_im = 0.0600im

cent_r = 0.0780

include("Potts-Operators & Hamiltonian.jl")
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

save_object("lambda_range.jld2",test_values)
println(test_values)
# ################## Lambda istelf ###################
using PlotlyJS
using Optim
Δε = 0.465613165838194 - 0.224494536412444im
ΔL1ε = 1.465613165838194 - 0.224494536412444im
Δσ =0.133596708540452 - 0.0204636065293973im
ΔL1σ =1.133596708540452 - 0.0204636065293973im
Cε_primeσσ = 0.0658 + 0.0513im 
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.90830177556852 - 0.598652097099851im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)
errors = []
cost = []
for L in [18,19,20,21,22,23]
    g = []
    cst = []
    for lambda in test_values
        lambda_txt = round(lambda,digits = 6)
       

       ## D 500
       if L ==20 && (lambda_txt == 0.078583+0.06035im || lambda_txt ==  0.078233 + 0.059883im)
        push!(g,0)
    elseif L ==19 && (lambda_txt == 0.078583+0.06035im || lambda_txt == 0.078233 + 0.059883im)
        push!(g,0)    
    elseif L ==18 && (lambda_txt == 077883+0.0593im  || lambda_txt ==  0.078 + 0.060233im)
            push!(g,0)
        elseif  L ==23 && (lambda_txt == 0.077883 +0.060583im || lambda_txt == 0.077883 +0.0607im || lambda_txt == 0.078+0.060467im || lambda_txt == 0.078+0.060583im || lambda_txt == 0.077417+0.0607im)    
        push!(g,0)
    elseif L ==23 && (lambda_txt == 0.077883 +0.060583im || lambda_txt == 0.077883 +0.0607im || lambda_txt == 0.078+0.060467im || lambda_txt == 0.078+0.060583im || lambda_txt == 0.077417+0.0607im)    
            push!(g,0)
        else
        results0 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
        results1 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
        results2 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC1_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
        ΔEε = results1[2][1]
        ΔEL1ε = results1[2][2]
        ΔEσ =  results2[2][end-1]
        ΔEL1σ = results2[2][end]
    
        #D 300/400
        #  if (L ==18 && (lambda_txt ==0.078 + 0.060233im)) || (L == 19  && lambda_txt == 0.078233+0.059883im) || (L ==20 && (lambda_txt ==  0.078583 + 0.06035im || lambda_txt == 077883+0.0593im))
        #     push!(g,0)
        # else
        # results = load_object("Lambda_est_precise/PBC/PBC_D_300/5E_PBC_L=$L"*"_$lambda_txt"*".jld2")
        # ΔEε = results[2]
        # ΔEL1ε = results[3]
        # ΔEσ =  results[end-1]
        # ΔEL1σ = results[end]
        
        #fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4]) -(x[5]+1im*x[6])/L^2 )^2 + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])-(x[7]+1im*x[8])/L^2)^2 +abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4])-(x[9]+1im*x[10])/L^2)^2  ##did i get rid of L-1sigma???
        
        # res = optimize(fun, [0, 0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        #fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4]) -(x[5]+1im*x[6])/L^2 )^2+abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4])-(x[7]+1im*x[8])/L^2)^2  ##did i get rid of L-1sigma???
        
        #res = optimize(fun, [0, 0,0.0,0.0,0.0,0.0,0.0,0.0])
        fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4]))^2 +abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4]))^2 + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4]))^2
        
        res = optimize(fun, [0.0, 0.0,0.0,0.0])
        gε_prime = Optim.minimizer(res)[2]+1im* Optim.minimizer(res)[4]
        push!(cst,Optim.minimum(res))
        push!(g,gε_prime )
        end
    end
    push!(cost,cst)
    push!(errors,g)
end
 
RG_val_final = []
test_alt = []
for (i,L) in enumerate([18,19,20,21,22,23])
    z_values = log.(cost[i])
    RG_values = []
   
    for res in 1:1:length(test_values)
        lambda_txt = round(test_values[res],digits = 6)
        
        if  (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im)
        elseif  (lambda_txt == 0.078583+0.06035im || lambda_txt == 0.078233 + 0.059883im)
        elseif  (lambda_txt == 077883+0.0593im  || lambda_txt ==  0.078 + 0.060233im)
        elseif  (lambda_txt == 0.077883 +0.060583im || lambda_txt == 0.077883 +0.0607im || lambda_txt == 0.078+0.060467im || lambda_txt == 0.078+0.060583im || lambda_txt == 0.077417+0.0607im)    
        else
        push!(RG_values, errors[i][res])
        #push!(z_values,log(abs(errors[i][res])))
        if L == 23 && (lambda_txt != 0.077883 +0.060583im || lambda_txt != 0.077883 +0.0607im || lambda_txt != 0.078+0.060467im || lambda_txt != 0.078+0.060583im || lambda_txt != 0.077417+0.0607im || lambda_txt != 077883+0.0593im  || lambda_txt !=  0.078 + 0.060233im || ambda_txt != 0.078583+0.06035im || lambda_txt !=   0.077883+0.0593im ||  lambda_txt != 0.078583+0.06035im || lambda_txt != 0.078233 + 0.059883im ) 
        push!(test_alt,test_values[res])

        end
        end
    end
    push!(RG_val_final,RG_values)
    p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
         x=real(-im*test_values),
             y=real(test_values),fill=true,colorbar=attr(
                 title="log(|gε'|)", # title here
                titleside="top",
                titlefont=attr(
                  size=14,
                  family="Arial, sans-serif"
              )
             )),Layout(title=attr(text = "Lambda estimation PBC L=$L D = 500",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
     )
    display(p)
    #PlotlyJS.savefig(p,"PBC lambda_estimation_Quasiparticleansatz_D50,L$L,N5.png")  

end
println(length(test_alt))
println(length(RG_val_final[3]))
println(length(test_values))
############### CUT  #######################################"
# using Plots


















# ### real direction
# using Plots
# q = plot(xlabel="real(lambda)", ylabel=" real(EΔσ)*L",title="Energy gap for im(lambda) = 0.060im")
# qa = plot(xlabel="real(lambda)", ylabel=" real(EΔL1σ)*L",title="Energy gap for im(lambda) = 0.060im")

# qb = plot(xlabel="real(lambda)", ylabel=" real(ΔEε )*L",title="Energy gap for im(lambda) = 0.060im")

# qc = plot(xlabel="real(lambda)", ylabel=" real( ΔEL1ε )*L",title="Energy gap for im(lambda) = 0.060im")

# for L in 19:20
#     ΔEε = []
#     ΔEL1ε = []
#     ΔEσ =  []
#     ΔEL1σ = []
#     test_alt = []
#     for res in 1:1:length(test_values)
#         lambda_txt = round(test_values[res],digits = 6)
#         if L ==20 && (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im)
#             push!(ΔEε,0)
#             push!(ΔEL1ε,0)
#             push!(ΔEσ, 0)
#             push!(ΔEL1σ,0)
#         elseif L ==19 && (lambda_txt == 0.078583+0.06035im || lambda_txt == 0.078233 + 0.059883im)
#          push!(ΔEε,0)
#             push!(ΔEL1ε,0)
#             push!(ΔEσ, 0)
#             push!(ΔEL1σ,0)
#         elseif L ==18 && (lambda_txt == 077883+0.0593im  || lambda_txt ==  0.078 + 0.060233im)
#          push!(ΔEε,0)
#             push!(ΔEL1ε,0)
#             push!(ΔEσ, 0)
#             push!(ΔEL1σ,0)
#         else
#         results = load_object("Lambda_est_precise/PBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
#         push!(ΔEε,results[2])
#         push!(ΔEL1ε,results[3])
#         push!(ΔEσ, results[end])
#         push!(ΔEL1σ, results[end-1])
#         end
#     end
#     mid = l÷2-N
#     println([test_values[mid+i] for i in 1:1:(2*N-1)])
#     println(test_values[79:91])
#     plot!(q,real(test_values[79:91]),abs.(L.*ΔEσ[79:91]),seriestype=:scatter,label="L = $L ")
#     plot!(qa,real(test_values[79:91]),abs.(L.*ΔEL1σ[79:91]),seriestype=:scatter,label="L = $L ")
#     plot!(qb,real(test_values[79:91]),abs.(L.*ΔEε[79:91]),seriestype=:scatter,label="L = $L ")
#     plot!(qc,real(test_values[79:91]),abs.(L.*ΔEL1ε[79:91]),seriestype=:scatter,label="L = $L ")
# end
# display(q)
# display(qa)
# display(qb)
# display(qc)


# q = plot(xlabel="Im(lambda)", ylabel=" Im(EΔσ)*L",title="Energy gap for Re(lambda) = 0.0780im")
# qa = plot(xlabel="Im(lambda)", ylabel=" Im(EΔL1σ)*L",title="Energy gap for Re(lambda) = 0.0780im")

# qb = plot(xlabel="Im(lambda)", ylabel=" Im(ΔEε )*L",title="Energy gap for Re(lambda) = 0.0780im")

# qc = plot(xlabel="Im(lambda)", ylabel=" Im( ΔEL1ε )*L",title="Energy gap for Re(lambda) = 0.0780im")

# for L in 19:20
#     ΔEε = []
#     ΔEL1ε = []
#     ΔEσ =  []
#     ΔEL1σ = []
#     test_alt = []
#     for res in 1:1:length(test_values)
#         lambda_txt = round(test_values[res],digits = 6)
#         if L ==20 && (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im)
#             push!(ΔEε,0)
#             push!(ΔEL1ε,0)
#             push!(ΔEσ, 0)
#             push!(ΔEL1σ,0)
#         elseif L ==19 && (lambda_txt == 0.078583+0.06035im || lambda_txt == 0.078233 + 0.059883im)
#             push!(ΔEε,0)
#             push!(ΔEL1ε,0)
#             push!(ΔEσ, 0)
#             push!(ΔEL1σ,0)
#         elseif L ==18 && (lambda_txt == 077883+0.0593im  || lambda_txt ==  0.078 + 0.060233im)
#             push!(ΔEε,0)
#             push!(ΔEL1ε,0)
#             push!(ΔEσ, 0)
#             push!(ΔEL1σ,0)
#         else
#         results = load_object("Lambda_est_precise/PBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
#         push!(ΔEε,results[2])
#         push!(ΔEL1ε,results[3])
#         push!(ΔEσ, results[end])
#         push!(ΔEL1σ, results[end-1])
#         end
#     end
#     x = []

#     for i in 0:(2*(N-1))
#     push!(x,test_values[N + (2*N-1)*i])
    
#     end
#     plot!(q,real(-1im.*x),abs.(-1im.* [L*ΔEσ[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
#     plot!(qa,real(-1im.*x),abs.(-1im.*[L*ΔEL1σ[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
#     plot!(qb,real(-1im.*x),abs.(-1im.*[L*ΔEε[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
#     plot!(qc,real(-1im.*x),abs.(-1im.*[L*ΔEL1ε[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
# end
# display(q)
# display(qa)
# display(qb)
# display(qc)


# ############## no cut but still data collaps
# using PlotlyJS
# E_difΔEε  = []
# E_difΔEL1ε  = []
# E_difΔEσ  = []
# E_difΔEL1σ  = []
# ΔEσ =[]
# ΔEL1σ =[]
# ΔEL1ε =[]
# ΔEε =[]
# L = 18
#     for (i,lambda) in enumerate(test_values)
#             lambda_txt = round(lambda,digits = 6)
          
#             results1 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
#             results2 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC1_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
    
#             # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
     
#         push!(ΔEε , L*results1[2][1])
#         push!(ΔEL1ε , L*results1[2][2])
#         push!( ΔEσ , L*results2[2][end-1])
#         push!(ΔEL1σ, L*results2[2][end])
        
#     end





# L = 19
#     for (i,lambda) in enumerate(test_values)
#             lambda_txt = round(lambda,digits = 6)
          
#             results1 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
#             results2 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC1_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
    
#             # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
#         push!(E_difΔEε, L*results1[2][1])
#         push!(E_difΔEL1ε , L*results1[2][2])
#         push!(E_difΔEσ , L*results2[2][end-1])
#         push!(E_difΔEL1σ , L*results2[2][end])
#         push!(ΔEε , L*results1[2][1])
#         push!(ΔEL1ε , L*results1[2][2])
#         push!( ΔEσ , L*results2[2][end-1])
#         push!(ΔEL1σ, L*results2[2][end])
        
#     end


#     L = 20
#     for (i,lambda) in enumerate(test_values)
#         lambda_txt = round(lambda,digits = 6)
#         if L ==20 && (lambda_txt == 0.078233 +0.059883im)
#             push!(ΔEε , 0)
#             push!(ΔEL1ε , 0)
#             push!( ΔEσ ,0)
#             push!(ΔEL1σ, 0)
#         else
#             results1 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
#             results2 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC1_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
    
#             # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
#        E_difΔEε[i] =  E_difΔEε[i] -L*results1[2][1]
#         E_difΔEL1ε[i] =  E_difΔEL1ε[i] - L*results1[2][2]
#         E_difΔEσ[i] =  E_difΔEσ[i] - L*results2[2][end-1]
#         E_difΔEL1σ[i] =  E_difΔEL1σ[i] -  L*results2[2][end]
#         push!(ΔEε , L*results1[2][1])
#         push!(ΔEL1ε , L*results1[2][2])
#         push!( ΔEσ , L*results2[2][end-1])
#         push!(ΔEL1σ, L*results2[2][end])
#         end
#     end
#     L_list =[18,19,20]
#     f(t, p) =  (p[1]./t.^2) .+ p[2]+ (p[3]./t.^3)
# for (i,lambda) in enumerate(test_values)
# p0 = [0,2*pi*2.8*0.0134,0]  
# fit = curve_fit(f, L_list, real([ΔEσ[i],ΔEσ[l+i],ΔEσ[2*l+i]]), p0)
# fat = curve_fit(f, L_list, real(-1im.*[ΔEσ[i],ΔEσ[l+i],ΔEσ[2*l+i]]), p0)
# println(fit.param[3])
# E_difΔEσ[i] =  abs(E_difΔEσ[i] - (1/19^2 -1/20^2)*(fit.param[1]+1im*fat.param[1])-(1/19^3 -1/20^3)*(fit.param[3]+1im*fat.param[3]))
# fit = curve_fit(f, L_list, real([ΔEε[i],ΔEε[l+i],ΔEε[2*l+i]]), p0)
# fat = curve_fit(f, L_list, real(-1im.*[ΔEε[i],ΔEε[l+i],ΔEε[2*l+i]]), p0)
# E_difΔEε[i] =  abs(E_difΔEε[i] - (1/19^2 -1/20^2)*(fit.param[1]+1im*fat.param[1])-(1/19^3 -1/20^3)*(fit.param[3]+1im*fat.param[3]))
# fit = curve_fit(f, L_list, real([ΔEL1ε[i],ΔEL1ε[l+i],ΔEL1ε[2*l+i]]), p0)
# fat = curve_fit(f, L_list, real(-1im.*[ΔEL1ε[i],ΔEL1ε[l+i],ΔEL1ε[2*l+i]]), p0)
# E_difΔEL1ε[i] =  abs(E_difΔEL1ε[i] - (1/19^2 -1/20^2)*(fit.param[1]+1im*fat.param[1])-(1/19^3 -1/20^3)*(fit.param[3]+1im*fat.param[3]))
# fit = curve_fit(f, L_list, real([ΔEL1σ[i],ΔEL1σ[l+i],ΔEL1σ[2*l+i]]), p0)
# fat = curve_fit(f, L_list, real(-1im.*[ΔEL1σ[i],ΔEL1σ[l+i],ΔEL1σ[2*l+i]]), p0)
# E_difΔEL1σ[i] =  abs(E_difΔEL1σ[i] - (1/19^2 -1/20^2)*(fit.param[1]+1im*fat.param[1])-(1/19^3 -1/20^3)*(fit.param[3]+1im*fat.param[3]))
# x = 18:0.01:20
# beta = round(fit.param[1],digits = 4)

# end
# println(E_difΔEε)
#     z_values = []
#     test_alt = []
#     for res in 1:1:length(test_values)
#         lambda_txt = round(test_values[res],digits = 6)
#         if (lambda_txt !=0.078233 +0.059883im)
#         push!(z_values,log(E_difΔEε[res]))
#         push!(test_alt,test_values[res])
#         end
#     end
#     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
#          x=real(-1im*test_alt),
#              y=real(test_alt),fill=true,colorbar=attr(
#                  title="", # title here
#                 titleside="top",
#                 titlefont=attr(
#                   size=14,
#                   family="Arial, sans-serif"
#               )
#              )),Layout(title=attr(text = "log(|(19*ΔEε(19) - 20*ΔEε(20))-beta(lambda)(1/19^2-1/20^2)  - c(lambda)(1/19^3-1/20^3)|) ",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
#      )
#     display(p)


#     z_values = []
#     test_alt = []
#     for res in 1:1:length(test_values)
#         lambda_txt = round(test_values[res],digits = 6)
#         if (lambda_txt ==0.078233 +0.059883im)
#            else
#         push!(z_values,log(E_difΔEL1ε[res]))
#         push!(test_alt,test_values[res])
#         end
#     end
#     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
#          x=real(-im*test_alt),
#              y=real(test_alt),fill=true,colorbar=attr(
#                  title="", # title here
#                 titleside="top",
#                 titlefont=attr(
#                   size=14,
#                   family="Arial, sans-serif"
#               )
#              )),Layout(title=attr(text = "log(|(19*ΔEL1ε(19) - 20*ΔEL1ε(20)) -beta(lambda)(1/19^2-1/20^2) - c(lambda)(1/19^3-1/20^3)  |",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
#      )
#     display(p)



#     z_values = []
#     test_alt = []
#     for res in 1:1:length(test_values)
#         lambda_txt = round(test_values[res],digits = 6)
#         if (lambda_txt ==0.078233 +0.059883im)
#           else
#         push!(z_values,log(E_difΔEσ[res]))
#         push!(test_alt,test_values[res])
#         end
#     end
#     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
#          x=real(-im*test_alt),
#              y=real(test_alt),fill=true,colorbar=attr(
#                  title="", # title here
#                 titleside="top",
#                 titlefont=attr(
#                   size=14,
#                   family="Arial, sans-serif"
#               )
#              )),Layout(title=attr(text = "log(|(19*ΔEσ(19) - 20*ΔEσ(20))-beta(lambda)(1/19^2-1/20^2) - c(lambda)(1/19^3-1/20^3)  |)",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
#      )
#     display(p)



#     z_values = []
#     test_alt = []
#     for res in 1:1:length(test_values)
#         lambda_txt = round(test_values[res],digits = 6)
#         if (lambda_txt ==0.078233 +0.059883im)
#         else
#         push!(z_values,log(E_difΔEL1σ[res]))
#         push!(test_alt,test_values[res])
#         end
#     end
#     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
#          x=real(-im*test_alt),
#              y=real(test_alt),fill=true,colorbar=attr(
#                  title="", # title here
#                 titleside="top",
#                 titlefont=attr(
#                   size=14,
#                   family="Arial, sans-serif"
#               )
#              )),Layout(title=attr(text = "log(|(19*ΔEL1σ(19) - 20*ΔEL1σ(20))-beta(lambda)(1/19^2-1/20^2) - c(lambda)(1/19^3-1/20^3)  |)",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
#      )
#     display(p)



# # # ######################## second order finite size effects ######################################""
# # using LsqFit
# v = 1
# ΔEσ =[]
# ΔEL1σ =[]
# ΔEL1ε =[]
# ΔEε =[]
# ΔE0 = []
# using Plots
L_list = [6,8,10,12,14,16,18,19,20]
lambda = test_values[84]
lambda_txt = round(test_values[84],digits = 8)
println(lambda)
# for L in [6,8,10,12,14,16]
#     results = load_object("Lambda_est_precise/PBC/PBC_D_400/PBC L = $L"*",5statesQP$lambda_txt.jld2")
#     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")
#     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
#     push!(ΔE0, results[2])
#     println(results[2])
# end


# for L in [18,19,20]
#     lambda_txt = round(lambda,digits = 6)
#     results = load_object("Lambda_est_precise/PBC/PBC_D_400/5E_PBC_L=$L"*"_$lambda_txt.jld2")
#     push!(ΔE0, ((results[1])))
# end
# using CurveFit
# c =  1.13754733664723 - 0.0210687419403234im
# a,b = linear_fit(L_list[end-1:end],ΔE0[end-1:end])
# p = scatter( L_list,real(ΔE0),title="lambda = $lambda_txt")
# display(p)
# p = scatter( L_list,real(-1im.*ΔE0),title="lambda = $lambda_txt")
# display(p)
# println("v = ",6*a/(pi*c))




# for L in [10,12,14,16]
#     lambda_txt = round(lambda,digits = 6)
#     results = load_object("Lambda_est_precise/PBC/PBC_D_300/5E_PB_L = $L"*"$lambda.jld2")
#     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")
#     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
#     push!(ΔEσ, (L/(2*pi*v))*((results[end-1])))
#     push!(ΔEL1ε , (L/(2*pi*v))*results[4])
#     push!(ΔEε , (L/(2*pi*v))*results[3])
#     push!(ΔEL1σ , (L/(2*pi*v))*results[end])
    
# end


# for L in [18,19,20]
#     lambda_txt = round(lambda,digits = 6)
#     results = load_object("Lambda_est_precise/PBC/PBC_D_300/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# end
# L_list = [18,19,20]
# pe = plot(ylabel = "Re( Δσ -(L/2piv)*ΔEσ )",xlabel = "1/L^2",title = "finite size effect at 0.0779 + 0.060im D = 300")
# for L in [18,19,20]
#     lambda_txt = round(lambda,digits = 6)
#     results = load_object("Lambda_est_precise/PBC/PBC_D_300/5E_PBC_L=$L"*"_$lambda_txt.jld2")
#     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")

#     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
  
#     push!(ΔEσ, (L/(2*pi*v))*((results[end-1])) - Δσ)
#     push!(ΔEL1ε , (L/(2*pi*v))*results[3] - ΔL1ε )
#     push!(ΔEε , (L/(2*pi*v))*results[2] -Δε  )
#     push!(ΔEL1σ , (L/(2*pi*v))*results[end] -1.111 + 0.170im )

# end

# println(ΔEσ)


# # # ΔEσ_lower_dim =[]

# # # for L in L_list
# # #     lambda_txt = round(test_values[85],digits = 6)
# # #     results = load_object("Lambda_est_precise/PBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# # #     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")

# # #     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
# # #     push!(ΔEσ_lower_dim, (L/(2*pi*v))*((results[end-1])))

# # end
# f(t, p) =  (p[1]./t.^2) .+ p[2] #+(p[3]./t.^3) 
# p0 = [0, 0.0134]
# fit = curve_fit(f, L_list, real(ΔEσ), p0)
# x = 6:0.001:20
# #fat = curve_fit(f, L_list, real(ΔEσ_lower_dim), p0)
# beta = round(fit.param[1],digits = 4)
# beta_r = fit.param[1]
# # c = round(fit.param[3],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)
# # c = fit.param[3]
# plot!(pe,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# #plot!(pe,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pe, 1 ./L_list.^2,real(ΔEσ),label="data",marker=:cross)
# #plot!(pe, 1 ./L_list.^2,real(ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pe)

# savefig(pe,"Re_ΔEσ_scalng")

# pep = plot(ylabel = "Im( Δσ - (L/2piv)*ΔEσ )",xlabel = "1/L^2",title = "finite size effect at 0.0779 + 0.060im D = 300")
# p0 = [0, -0.0021]
# fit = curve_fit(f, L_list, real(-1im.*ΔEσ), p0)
# beta = round(fit.param[1],digits = 4)
# # c = round(fit.param[3],digits = 4)
# # c_r =fit.param[3]
# scaling_dim = round(fit.param[2],digits = 4)
# p = fit.param[2]
# plot!(pep,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# #plot!(pep,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pep, 1 ./L_list.^2,real(-1im.*ΔEσ),label="data",marker=:cross)
# #(pep, 1 ./L_list.^2,real(-1im.*ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pep)
# savefig(pep,"Im_ΔEσ_scaling")





# # beta_σ = beta_r + 1im*fit.param[1]
# # # c_σ = c_r = 1im*fit.param[3]
# # c =  1.13754733664723 - 0.0210687419403234im
# # delta = 0.5*(0.133596708540452 - 0.0204636065293973im)
# # beta_th_σ = delta^2 - (c+2)/(12) *(delta) 
# # println("BETA σ EXP IS $beta_σ")
# # println("BETA σ Th IS $beta_th_σ")

# # lambda_txt = round(test_values[84],digits = 6)
# pe = plot(ylabel = "Re( ΔEε- (L/2piv)*ΔEε )",xlabel = "1/L^2",title = "finite size effect at $lambda")
# # for L in [18,19,20]
#     lambda_txt = round(test_values[84],digits = 6)
#     results = load_object("Lambda_est_precise/OBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
#     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")

#     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
  
#     push!(ΔEσ, (L/(2*pi*v))*((results[end-1])))
#     push!(ΔEL1ε , L*results[3])
#     push!(ΔEε , L*results[2])
#     push!(ΔEL1σ , L*results[end])

# end

# println(ΔEσ)


# ΔEσ_lower_dim =[]

# for L in L_list
#     lambda_txt = round(test_values[85],digits = 6)
#     results = load_object("Lambda_est_precise/PBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
#     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")

#     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
#     push!(ΔEσ_lower_dim, (L/(2*pi*v))*((results[end-1])))

# # end
# fit = curve_fit(f, L_list, real(ΔEε), p0)
# x = 6:0.1:20
# beta = round(fit.param[1],digits = 4)
# beta_r = round(fit.param[1],digits = 4)
# # c = round(fit.param[3],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)
# # c_r =fit.param[3]
# plot!(pe,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# #plot!(pe,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pe, 1 ./L_list.^2,real(ΔEε),label="data",marker=:cross)
# #plot!(pe, 1 ./L_list.^2,real(ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pe)

# savefig(pe,"Re_ΔEε_scaling")

# pep = plot(ylabel = "Im(Δε -  (L/2piv)*ΔEε )",xlabel = "1/L^2",title = "finite size effect at $lambda")
# fit = curve_fit(f, L_list, real(-1im.*ΔEε), p0)
# beta = round(fit.param[1],digits = 4)

# # c = round(fit.param[3],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)

# plot!(pep,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# #plot!(pep,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pep, 1 ./L_list.^2,real(-1im.*ΔEε),label="data",marker=:cross)
# #(pep, 1 ./L_list.^2,real(-1im.*ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pep)
# savefig(pep,"Im_ΔEε_scaling")

# beta_ε = beta_r + 1im*beta

# # c =  1.13754733664723 - 0.0210687419403234im
# # h= 0.5*(0.465613165838194 - 0.22449453641244im)
# # beta_th_ε = h^2 - ((c+2)/12)*(h) 
# # println("BETA ε EXP IS $beta_ε")
# # println("BETA ε Th IS $beta_th_ε")


# # fr = beta_σ/beta_ε
# # println("fraction EXP = $fr")
# # fr = beta_th_σ/beta_th_ε
# # println("fraction Th = $fr")
# # c_ε = c_r = 1im*c
# ########################" other ones 
# pe = plot(ylabel = "Re( (L/2piv)*ΔEσ')",xlabel = "1/L^2",title = "finite size effect at $lambda")
# # # for L in [18,19,20]
# # #     lambda_txt = round(test_values[84],digits = 6)
# # #     results = load_object("Lambda_est_precise/OBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# # #     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")

# # #     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
  
# # #     push!(ΔEσ, (L/(2*pi*v))*((results[end-1])))
# # #     push!(ΔEL1ε , L*results[3])
# # #     push!(ΔEε , L*results[2])
# # #     push!(ΔEL1σ , L*results[end])

# # # end

# # # println(ΔEσ)


# # # ΔEσ_lower_dim =[]

# # # for L in L_list
# # #     lambda_txt = round(test_values[85],digits = 6)
# # #     results = load_object("Lambda_est_precise/PBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# # #     #results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")

# # #     #push!(ΔEσ, (L/(2*pi*v))*((-results["E0"][k] + results["Eσ"][k])))
# # #     push!(ΔEσ_lower_dim, (L/(2*pi*v))*((results[end-1])))

# # # end
# fit = curve_fit(f, L_list, real(ΔEL1σ), p0)
# x = 6:0.1:20
# beta = round(fit.param[1],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)

# plot!(pe,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# #plot!(pe,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pe, 1 ./L_list.^2,real(ΔEL1σ),label="data",marker=:cross)
# #plot!(pe, 1 ./L_list.^2,real(ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pe)
# savefig(pe,"Re_ΔEσ'_scaling")


# pep = plot(ylabel = "Im( (L/2piv)*ΔEσ')",xlabel = "1/L^2",title = "finite size effect at $lambda")

# fit = curve_fit(f, L_list, real(-1im.*ΔEL1σ), p0)
# beta = round(fit.param[1],digits = 4)
# # c = round(fit.param[3],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)

# plot!(pep,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# #plot!(pep,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pep, 1 ./L_list.^2,real(-1im.*ΔEL1σ),label="data",marker=:cross)
# #(pep, 1 ./L_list.^2,real(-1im.*ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pep)
# savefig(pep,"Im_ΔEσ'_scaling")
# pe = plot(ylabel = "Re( ΔL1ε - (L/2piv)*ΔEL1ε))",xlabel = "1/L^2",title = "finite size effect at $lambda")
# fit = curve_fit(f, L_list, real(ΔEL1ε), p0)
# beta = round(fit.param[1],digits = 4)
# # c = round(fit.param[3],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)
# x = 6:0.1:20
# #fat = curve_fit(f, L_list, real(ΔEσ_lower_dim), p0)
# plot!(pe,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim+$beta/L^2 fit")
# #plot!(pe,1 ./L_list.^2,f(L_list,fat.param),label="1/L^2 fit")
# plot!(pe, 1 ./L_list.^2,real(ΔEL1ε),label="data",marker=:cross)
# #plot!(pe, 1 ./L_list.^2,real(ΔEσ_lower_dim),label="data lower dim",marker=:cross)
# # plot!(p,1 ./L_list.^2,fill(0,length(L_list)),label="σ")
# display(pe)
# savefig(pe,"Re_ΔEL1ε_scaling")


# pep = plot(ylabel = "Im( ΔL1ε - (L/2piv)*ΔEL1ε)",xlabel = "1/L^2",title = "finite size effect at $lambda")
# fit = curve_fit(f, L_list, real(-1im.*ΔEL1ε), p0)
# beta = round(fit.param[1],digits = 4)
# scaling_dim = round(fit.param[2],digits = 4)

# plot!(pep,1 ./L_list.^2,f(L_list,fit.param),label="$scaling_dim + $beta/L^2 fit")
# plot!(pep, 1 ./L_list.^2,real(-1im.*ΔEL1ε),label="data",marker=:cross)
# display(pep)

# savefig(pep,"Im_ΔEL1ε_scaling")



# # ############## no cut but still data collaps
# # using Plots
# # E_0diff  = []
# # E_difΔEε  = []
# # E_difΔEL1ε  = []
# # E_difΔEσ  = []
# # E_difΔEL1σ  = []
# # L = 19
# #     for (i,lambda) in enumerate(test_values)
# #         lambda_txt = round(lambda,digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #             push!(E_0diff,0)
# #             push!(E_difΔEε, 0)
# #             push!(E_difΔEL1ε , 0)
# #             push!(E_difΔEσ , 0)
# #             push!(E_difΔEL1σ ,0)
# #         else
# #         results = load_object("Lambda_est_precise/PBC/PBC_D_400/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# #         # results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")
# #         # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
# #         push!(E_0diff,1/L*results[1])
# #         push!(E_difΔEε, L*results[2])
# #         push!(E_difΔEL1ε , L*results[3])
# #         push!(E_difΔEσ , L*results[end-1])
# #         push!(E_difΔEL1σ , L*results[end])
# #         end
# #     end


# #     L = 20
# #     for (i,lambda) in enumerate(test_values)
# #         lambda_txt = round(lambda,digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #         else
# #         # results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")
# #         # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
# #         results = load_object("Lambda_est_precise/PBC/PBC_D_400/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# #         E_0diff[i] = abs(E_0diff[i]-1/L*results[1])
# #         E_difΔEε[i] =  abs(E_difΔEε[i] - L*results[2] - beta_ε*(1/19^2 - 1/20^2)- c_ε*(1/19^3 - 1/20^3))
# #         E_difΔEL1ε[i] =  abs(E_difΔEL1ε[i] - L*results[3])
# #         E_difΔEσ[i] =  abs(E_difΔEσ[i] - L*results[end-1] - beta_σ*(1/19^2 - 1/20^2)-c_σ*(1/19^3 - 1/20^3))
# #         E_difΔEL1σ[i] =  abs(E_difΔEL1σ[i] - L*results[end])
# #         end
# #     end




# #     z_values = []
# #     test_alt = []
# #     for res in 1:1:length(test_values)
# #         lambda_txt = round(test_values[res],digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #         else
# #         push!(z_values,log(E_0diff[res]))
# #         push!(test_alt,test_values[res])
# #         end
# #     end
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
# #          x=real(-im*test_alt),
# #              y=real(test_alt),fill=true,colorbar=attr(
# #                  title="log(|(19^2*(ΔE0(19) - 20^2*ΔE0(20)) |)", # title here
# #                 titleside="top",
# #                 titlefont=attr(
# #                   size=14,
# #                   family="Arial, sans-serif"
# #               )
# #              )),Layout(title=attr(text = "ΔE0",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# #      )
# #     display(p)
# #     z_values = []
# #     test_alt = []
# #     for res in 1:1:length(test_values)
# #         lambda_txt = round(test_values[res],digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #         else
# #         push!(z_values,log(E_difΔEε[res]))
# #         push!(test_alt,test_values[res])
# #         end
# #     end
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
# #          x=real(-im*test_alt),
# #              y=real(test_alt),fill=true,colorbar=attr(
# #                  title="log(|(19*(ΔEε(19) - 20*ΔEε(20) + beta(1/19^2 - 1/20^2)) |)", # title here
# #                 titleside="top",
# #                 titlefont=attr(
# #                   size=14,
# #                   family="Arial, sans-serif"
# #               )
# #              )),Layout(title=attr(text = "ΔEε",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# #      )
# #     display(p)


# #     z_values = []
# #     test_alt = []
# #     for res in 1:1:length(test_values)
# #         lambda_txt = round(test_values[res],digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #         else
# #         push!(z_values,log(E_difΔEL1ε[res]))
# #         push!(test_alt,test_values[res])
# #         end
# #     end
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
# #          x=real(-im*test_alt),
# #              y=real(test_alt),fill=true,colorbar=attr(
# #                  title="log(|(19*(ΔEε(19) - 20*ΔEε(20) + beta(1/19^2 - 1/20^2)) |)", # title here
# #                 titleside="top",
# #                 titlefont=attr(
# #                   size=14,
# #                   family="Arial, sans-serif"
# #               )
# #              )),Layout(title=attr(text = "ΔEL1ε",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# #      )
# #     display(p)



# #     z_values = []
# #     test_alt = []
# #     for res in 1:1:length(test_values)
# #         lambda_txt = round(test_values[res],digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #         else
# #         push!(z_values,log(E_difΔEσ[res]))
# #         push!(test_alt,test_values[res])
# #         end
# #     end
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
# #          x=real(-im*test_alt),
# #              y=real(test_alt),fill=true,colorbar=attr(
# #                  title="log(|(19*(ΔEε(19) - 20*ΔEε(20) + beta(1/19^2 - 1/20^2)) |)", # title here
# #                 titleside="top",
# #                 titlefont=attr(
# #                   size=14,
# #                   family="Arial, sans-serif"
# #               )
# #              )),Layout(title=attr(text = "ΔEσ",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# #      )
# #     display(p)



# #     z_values = []
# #     test_alt = []
# #     for res in 1:1:length(test_values)
# #         lambda_txt = round(test_values[res],digits = 6)
# #         if (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im || lambda_txt == 0.078233 + 0.059883im)
# #         else
# #         push!(z_values,log(E_difΔEL1σ[res]))
# #         push!(test_alt,test_values[res])
# #         end
# #     end
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
# #          x=real(-im*test_alt),
# #              y=real(test_alt),fill=true,colorbar=attr(
# #                  title="log(|(19*ΔEL1σ(19) - 20*ΔEL1σ(20)) |)", # title here
# #                 titleside="top",
# #                 titlefont=attr(
# #                   size=14,
# #                   family="Arial, sans-serif"
# #               )
# #              )),Layout(title=attr(text = "ΔEL1σ",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# #      )
# #     display(p)

# #########################################################" END Second Order Finite size effects #####################################################"





# # end
# # q = plot(xlabel="Im(lambda)", ylabel=" Im(EΔσ)*L",title="Energy gap for Re(lambda) = 0.0780im")
# # qa = plot(xlabel="Im(lambda)", ylabel=" Im(EΔL1σ)*L",title="Energy gap for Re(lambda) = 0.0780im")

# # qb = plot(xlabel="Im(lambda)", ylabel=" Im(ΔEε )*L",title="Energy gap for Re(lambda) = 0.0780im")

# # qc = plot(xlabel="Im(lambda)", ylabel=" Im( ΔEL1ε )*L",title="Energy gap for Re(lambda) = 0.0780im")

# # for L in 19:20
# #     ΔEε = []
# #     ΔEL1ε = []
# #     ΔEσ =  []
# #     ΔEL1σ = []
# #     test_alt = []
# #     for res in 1:1:length(test_values)
# #         lambda_txt = round(test_values[res],digits = 6)
# #         if L ==20 && (lambda_txt == 0.078583+0.06035im || lambda_txt ==   0.077883+0.0593im)
# #             push!(ΔEε,0)
# #             push!(ΔEL1ε,0)
# #             push!(ΔEσ, 0)
# #             push!(ΔEL1σ,0)
# #         elseif L ==19 && (lambda_txt == 0.078583+0.06035im || lambda_txt == 0.078233 + 0.059883im)
# #             push!(ΔEε,0)
# #             push!(ΔEL1ε,0)
# #             push!(ΔEσ, 0)
# #             push!(ΔEL1σ,0)
# #         elseif L ==18 && (lambda_txt == 077883+0.0593im  || lambda_txt ==  0.078 + 0.060233im)
# #             push!(ΔEε,0)
# #             push!(ΔEL1ε,0)
# #             push!(ΔEσ, 0)
# #             push!(ΔEL1σ,0)
# #         else
# #         results = load_object("Lambda_est_precise/PBC/5E_PBC_L=$L"*"_$lambda_txt.jld2")
# #         push!(ΔEε,results[2])
# #         push!(ΔEL1ε,results[3])
# #         push!(ΔEσ, results[end])
# #         push!(ΔEL1σ, results[end-1])
# #         end
# #     end
# #     x = []

# #     for i in 0:(2*(N-1))
# #     push!(x,test_values[N + (2*N-1)*i])
    
# #     end
# #     plot!(q,real(-1im.*x),real(-1im.* [L*ΔEσ[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
# #     plot!(qa,real(-1im.*x),real(-1im.*[L*ΔEL1σ[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
# #     plot!(qb,real(-1im.*x),real(-1im.*[L*ΔEε[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
# #     plot!(qc,real(-1im.*x),real(-1im.*[L*ΔEL1ε[N + (2*N-1)*i] for i in 0:1:2*(N-1)]),seriestype=:scatter,label="L = $L ")
# # end
# # display(q)
# # display(qa)
# # display(qb)
# # display(qc)
# # im_lm = round(real(-im*test_values[32]), digits = 5)
# # p = plot(; xlabel="real(lambda)", ylabel=" real(EΔε)*L",title="Energy gap scaling with im(lambda) = $im_lm")
# # for L in 6:2:12
# #     data = load_object("data_upto_12/E epsilon with lambda for $L.jld2")
  
# #     plot!(p,real([test_values[24+i] for i in 1:1:8]),real([data[24+i] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# # end
# # plot!(p,[0.079], seriestype="vline",label="real(λc)")
# # savefig(p,"EΔε lambda scaling .png")


# # im_lm = round(real(-im*test_values[32]), digits = 5)
# # p = plot(; xlabel="real(lambda)", ylabel=" real(EΔL-1ε)*L",title="Energy gap scaling with im(lambda) = $im_lm")
# # for L in 6:2:12
# #     data = load_object("data_upto_12/E L1epsilon with lambda for$L.jld2")
  
# #     plot!(p,real([test_values[24+i] for i in 1:1:8]),real([data[24+i] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# # end
# # plot!(p,[0.079], seriestype="vline",label="real(λc)")
# # savefig(p,"EΔL-1ε lambda scaling .png")

# ### imaginary direction
# # im_lm = round(real(test_values[3]),digits = 5)
# # q = plot(; xlabel="im(lambda)", ylabel=" im(EΔσ)*L",title="Energy gap scaling with Re(lambda) = $im_lm")
# # for L in 6:2:12
# #     data = load_object("data_upto_12/E sigma with lambda for$L.jld2")
  
# #     plot!(q,real(-im.*[test_values[3+8*(i-1)] for i in 1:1:8]),real(-im.*[data[3+8*(i-1)] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# # end
# # plot!(q,[0.060], seriestype="vline",label="im(λc)")
# # savefig(q,"Im EΔσ lambda scaling .png")


# # im_lm = round(real(test_values[4]),digits = 5)
# # q = plot(; xlabel="im(lambda)", ylabel=" im(EΔε )*L",title="Energy gap scaling with Re(lambda) = $im_lm")
# # for L in 6:2:12
# #     data = load_object("data_upto_12/E epsilon with lambda for $L.jld2")
  
# #     plot!(q,real(-im.*[test_values[4+8*(i-1)] for i in 1:1:8]),real(-im.*[data[4+8*(i-1)] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# # end
# # plot!(q,[0.060], seriestype="vline",label="im(λc)")
# # savefig(q,"Im EΔε  lambda scaling .png")

# # im_lm = round(real(test_values[4]),digits = 5)
# # q = plot(; xlabel="im(lambda)", ylabel=" im(EΔL-1ε )*L",title="Energy gap scaling with Re(lambda) = $im_lm")
# # for L in 6:2:12
# #     data = load_object("data_upto_12/E L1epsilon with lambda for$L.jld2")
  
# #     plot!(q,real(-im.*[test_values[4+8*(i-1)] for i in 1:1:8]),real(-im.*[data[4+8*(i-1)] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# # end
# # plot!(q,[0.060], seriestype="vline",label="im(λc)")
# # savefig(q,"Im EΔL-1ε lambda scaling .png")







# #### scaling 
# ############### lambda vs Δσ #######################################"
# # using PlotlyJS
# # using Polynomials
# # data = []
# # L_list = [6,8,10,12]
# # for L in L_list
# #     append!(data,load_object("data_upto_12/E L1epsilon with lambda for$L.jld2")  )
# # end

# # results = Vector{Float64}(undef,length(test_values))
# # for i in 1:1:length(test_values)
# #     f = fit(1 ./L_list,real([data[i + 64*j] for j in 0:1:3]), 1)
# #     a = f.coeffs[2]
# #     results[i] = a/(2*pi*2.88)
# # end
# # p = PlotlyJS.plot(PlotlyJS.contour(   z=results.-1.4656,
# #     x=real(-im*test_values),
# #         y=real(test_values),fill=true,colorbar=attr(
# #             title="ΔL-1ε-1.4656", # title here
# #             titleside="top",
# #             titlefont=attr(
# #                 size=14,
# #                 family="Arial, sans-serif"
# #             )
# #         )),Layout(title=attr(text = "ΔL-1ε scaling",x = 0.5), xaxis_title= "Im(λ)",
# #         yaxis_title = "Re(λ)")
# # )
# # PlotlyJS.savefig(p,"ΔL-1ε lambda scaling.png")







# # ### RG flow (of dimensions)
N = 5
using Plots
# tickfontcolor=:white,ytickfontcolor=:white,xtickfontsize=1,ytickfontsize=1
p = Plots.plot(; xlabel="Re(g_e)", ylabel="Im(g_e)",title = "RG flow for L = 18,19,20,21,22,23",legend=false,xguidefontsize=15,yguidefontsize=15,titlefontsize=15)
for (i,lambda) in enumerate(test_alt)
    y = []
    for l in 1:6
        push!(y,RG_val_final[l][i])
    end
    
        
    Plots.plot!(p,[real(y ),],[real(-1im.*y),],color="blue",arrow=true,label=" ")
    
    #quiver!(p,[real(errors[2,i] ),],[real(-1im.*errors[2,i])], quiver = [0.1.*(real(errors[3,i]-errors[2,i]),real(-im.*(errors[3,i]-errors[2,i]))),],color="red")
end
display(p)
Plots.plot!(p,[0.0],[0.0,], seriestype=:scatter)

println("here")




# # lambda_c = [0.0584im+0.076,0.05866im+0.0768,0.05906777im+0.076933,0.0593im+0.0771667,0.0595333im+0.07733333,0.05965im+0.07756667,0.0597667im+0.0776833,0.0777667+0.059888333im,0.06im + 0.078000,0.06im + 0.078000]
# # L = 11:20   
# # using Plots
# # pe = plot(title = "fixed_point_drift L = $L",size=(600,500),xlabel="Im(h)",ylabel="Re(h)",legend=false)
# # plot!(pe,real(-1im.*lambda_c),real(lambda_c),marker=:cross,arrow=true)
# # display(pe)


# # using Plots
# # E_difΔEε  = []
# # E_difΔEσ  = []
# # L = 4
# # results = load_object("exact diag CCFT $L")
# #     for (i,lambda) in enumerate(test_values)
# #         # results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")
# #         # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
      
# #         push!(E_difΔEε, L*(results[3,i] - results[1,i]))
        
# #         push!(E_difΔEσ , L*(results[2,i] - results[1,i]))
# #     end
# #     println((results[1,:]))


# # L = 5
# # results = load_object("exact diag CCFT $L")
# # for (i,lambda) in enumerate(test_values)
# #     # results = load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")
# #     # push!(E_difΔEσ, (L*((-results["E0"][1] + results["Eσ"][1]))))
# #     E_difΔEε[i] =  abs(E_difΔEε[i] - L*(results[3,i] - results[1,i]))
# #     E_difΔEσ[i] =  abs(E_difΔEσ[i] - L*(results[2,i] - results[1,i]))
# # end
# # println((results[1,:]))
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=log.(E_difΔEσ),
# #          x=real(-im*test_values),
# #              y=real(test_values),fill=true,colorbar=attr(
# #                  title="log(|(5*ΔEσ(5) - 4*ΔEσ(4)) |)", # title here
# #                 titleside="top",
# #                 titlefont=attr(
# #                   size=14,
# #                   family="Arial, sans-serif"
# #               )
# #              )),Layout(title=attr(text = "ΔEσ",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# #      )
# #     display(p)
# #     p = PlotlyJS.plot(PlotlyJS.contour(   z=log.(E_difΔEε),
# #     x=real(-im*test_values),
# #         y=real(test_values),fill=true,colorbar=attr(
# #             title="log(|(5*ΔEε(5) - 4*ΔEε(4)))) |)", # title here
# #            titleside="top",
# #            titlefont=attr(
# #              size=14,
# #              family="Arial, sans-serif"
# #          )
# #         )),Layout(title=attr(text = "ΔEε",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
# # )
# # display(p)