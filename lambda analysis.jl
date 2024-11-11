### lambda analysis
using JLD2

L = 8
N = 6
test_values = zeros(ComplexF64,(N*N))
for i in 1:1:N
    for j in 1:1:N
        test_values[i+(j-1)*N] = 0.12-(0.1/N )* i + (0.02 + 0.1/N * (j-1))*im
    end
end



################### Lambda istelf ###################
# using PlotlyJS
# results = load_object("Lambda_est_ge8.jld2")
# p = PlotlyJS.plot(PlotlyJS.contour(   z=broadcast(log,broadcast(abs,results)),
#     x=real(-im*test_values),
#         y=real(test_values),fill=true,colorbar=attr(
#             title="log(|gε'|)", # title here
#             titleside="top",
#             titlefont=attr(
#                 size=14,
#                 family="Arial, sans-serif"
#             )
#         )),Layout(title=attr(text = "L=$L",x = 0.5))
# )
# println(results)
# savefig(p,"lambda_estimation_D100,L$L,N6dif.png")  



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







### RG flow
###
lambda_est = Vector{ComplexF64}[]
L_list = [6,8,10,12]
for L in L_list
    push!(lambda_est,load_object("data_upto_12/Lambda_est_ge$L.jld2"))
end