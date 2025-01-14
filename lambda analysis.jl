### lambda analysis
using JLD2

N = 5
L = 6
test_values = zeros(ComplexF64,(2*N-1)^2)
### changed this slightly to allow for any parameter N to cross at 0.079 + 0.060i point -> its 2N-1 parameter square now (for any N)
for i in 1:1:(2*N-1)
    for j in 1:1:(2*N-1)
        if i <N+1
            if j<N+1
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(-0.04,0.00,N)[i]) .+ (0.079)  + 1im*LinRange(-0.040,0.00,N)[j] .+ 0.06im 
            else 
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(-0.04,0.00,N)[i]) .+ (0.079)  + 1im*LinRange(0+0.040/(N-1),0.040+0.040/(N-1),N)[j-N] .+ 0.06im 
            end
        else
            if j<N+1
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(0+0.040/(N-1),0.040+0.040/(N-1),N)[i-N]) .+ (0.079)  + 1im*LinRange(-0.040,0.00,N)[j] .+ 0.06im 
            else
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(0+0.040/(N-1),0.040+0.040/(N-1),N)[i-N]) .+ (0.079)  + 1im*LinRange(0+0.040/(N-1),0.040+0.040/(N-1),N)[j-N] .+ 0.06im 
            end
        end 
       
    end
end


################## Lambda istelf ###################
using PlotlyJS
results = load_object("alt_term_Lambda_est_ge6.jld2")
z_values = zeros(length(results))
print(results)
for res in 1:1:length(results)
    z_values[res] = log(abs(results[res]))
end
p = PlotlyJS.plot(PlotlyJS.contour(   z=z_values,
     x=real(-im*test_values),
         y=real(test_values),fill=true,colorbar=attr(
             title="log(|gε'|)", # title here
             titleside="top",
             titlefont=attr(
                 size=14,
                 family="Arial, sans-serif"
             )
         )),Layout(title=attr(text = "L=$L",x = 0.5),xaxis_title="Im(λ)",yaxis_title="Re(λ)")
 )
println(results)
savefig(p,"falt_term_ge__lambda_estimation_D50,L$L,N5.png")  



############### lambda vs ΔEσ/L #######################################"
# using Plots


### real direction
# im_lm = round(real(-im*test_values[32]),digits = 5)
# q = plot(; xlabel="real(lambda)", ylabel=" real(EΔσ)*L",title="Energy gap scaling with im(lambda) = $im_lm")
# for L in 6:2:12
#     data = load_object("data_upto_12/E sigma with lambda for$L.jld2")
  
#     plot!(q,real([test_values[24+i] for i in 1:1:8]),real([data[24+i] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# end
# plot!(q,[0.079], seriestype="vline",label="real(λc)")
# savefig(q,"EΔσ lambda scaling .png")


# im_lm = round(real(-im*test_values[32]), digits = 5)
# p = plot(; xlabel="real(lambda)", ylabel=" real(EΔε)*L",title="Energy gap scaling with im(lambda) = $im_lm")
# for L in 6:2:12
#     data = load_object("data_upto_12/E epsilon with lambda for $L.jld2")
  
#     plot!(p,real([test_values[24+i] for i in 1:1:8]),real([data[24+i] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# end
# plot!(p,[0.079], seriestype="vline",label="real(λc)")
# savefig(p,"EΔε lambda scaling .png")


# im_lm = round(real(-im*test_values[32]), digits = 5)
# p = plot(; xlabel="real(lambda)", ylabel=" real(EΔL-1ε)*L",title="Energy gap scaling with im(lambda) = $im_lm")
# for L in 6:2:12
#     data = load_object("data_upto_12/E L1epsilon with lambda for$L.jld2")
  
#     plot!(p,real([test_values[24+i] for i in 1:1:8]),real([data[24+i] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# end
# plot!(p,[0.079], seriestype="vline",label="real(λc)")
# savefig(p,"EΔL-1ε lambda scaling .png")

### imaginary direction
# im_lm = round(real(test_values[3]),digits = 5)
# q = plot(; xlabel="im(lambda)", ylabel=" im(EΔσ)*L",title="Energy gap scaling with Re(lambda) = $im_lm")
# for L in 6:2:12
#     data = load_object("data_upto_12/E sigma with lambda for$L.jld2")
  
#     plot!(q,real(-im.*[test_values[3+8*(i-1)] for i in 1:1:8]),real(-im.*[data[3+8*(i-1)] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# end
# plot!(q,[0.060], seriestype="vline",label="im(λc)")
# savefig(q,"Im EΔσ lambda scaling .png")


# im_lm = round(real(test_values[4]),digits = 5)
# q = plot(; xlabel="im(lambda)", ylabel=" im(EΔε )*L",title="Energy gap scaling with Re(lambda) = $im_lm")
# for L in 6:2:12
#     data = load_object("data_upto_12/E epsilon with lambda for $L.jld2")
  
#     plot!(q,real(-im.*[test_values[4+8*(i-1)] for i in 1:1:8]),real(-im.*[data[4+8*(i-1)] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# end
# plot!(q,[0.060], seriestype="vline",label="im(λc)")
# savefig(q,"Im EΔε  lambda scaling .png")

# im_lm = round(real(test_values[4]),digits = 5)
# q = plot(; xlabel="im(lambda)", ylabel=" im(EΔL-1ε )*L",title="Energy gap scaling with Re(lambda) = $im_lm")
# for L in 6:2:12
#     data = load_object("data_upto_12/E L1epsilon with lambda for$L.jld2")
  
#     plot!(q,real(-im.*[test_values[4+8*(i-1)] for i in 1:1:8]),real(-im.*[data[4+8*(i-1)] for i in 1:1:8]).*L,seriestype=:scatter,label="L = $L ")
# end
# plot!(q,[0.060], seriestype="vline",label="im(λc)")
# savefig(q,"Im EΔL-1ε lambda scaling .png")







#### scaling 
############### lambda vs Δσ #######################################"
# using PlotlyJS
# using Polynomials
# data = []
# L_list = [6,8,10,12]
# for L in L_list
#     append!(data,load_object("data_upto_12/E L1epsilon with lambda for$L.jld2")  )
# end

# results = Vector{Float64}(undef,length(test_values))
# for i in 1:1:length(test_values)
#     f = fit(1 ./L_list,real([data[i + 64*j] for j in 0:1:3]), 1)
#     a = f.coeffs[2]
#     results[i] = a/(2*pi*2.88)
# end
# p = PlotlyJS.plot(PlotlyJS.contour(   z=results.-1.4656,
#     x=real(-im*test_values),
#         y=real(test_values),fill=true,colorbar=attr(
#             title="ΔL-1ε-1.4656", # title here
#             titleside="top",
#             titlefont=attr(
#                 size=14,
#                 family="Arial, sans-serif"
#             )
#         )),Layout(title=attr(text = "ΔL-1ε scaling",x = 0.5), xaxis_title= "Im(λ)",
#         yaxis_title = "Re(λ)")
# )
# PlotlyJS.savefig(p,"ΔL-1ε lambda scaling.png")







# ### RG flow (of dimensions)
using Plots
L_list = [6,8,10,12]
data_Eε = Matrix{ComplexF64}(undef,length(L_list),(2*N-1)*(2*N-1))
data_Eσ = Matrix{ComplexF64}(undef,length(L_list),(2*N-1)*(2*N-1))
data_g = Matrix{ComplexF64}(undef,length(L_list),(2*N-1)*(2*N-1))

for (i,L) in enumerate(L_list)
    temp = load_object("Vic/ΔEε with lambda for $L.jld2")
    temp2 = load_object("Vic/ΔEσ with lambda for$L.jld2")
    temp3 = load_object("Vic/Lambda_est_ge$L.jld2")
    for j in 1:N*N
        data_g[i,j] = temp3[j]
        data_Eσ[i,j] = temp2[j]
        data_Eε[i,j] = temp[j]
    end
end
p = plot(; xlabel="real(ge)", ylabel="im(ge))",title = "ge evolution L = [6,8,10,12]",legend=false)
for i in 1:1:N*N
    plot!(p,real(data_g[:,i]),real(-im.*data_g[:,i]),color="blue")
    quiver!(p,[real(data_g[2,i]),],[real(-im.*data_g[2,i]);], quiver = [0.1.*(real(data_g[3,i]-data_g[2,i]),real(-im.*(data_g[3,i]-data_g[2,i]))),],color="red")

end
plot!(p,[0.00,],[0.000,], seriestype=:scatter)
# savefig(p,"ge_evolution.png")

println(data_g)
println(test_values)
