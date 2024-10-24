
using JLD2
using Plots
using Polynomials
data = load_object("Ground_state_MPSNonHermitian_pottsq5 excited-N0,D100,energies-L[8, 9, 10, 11, 12].jld2")
data2 = load_object("non_sym_Ground_state_MPSNonHermitian_pottsq5 excited-N0,D100,energies-L[8, 9, 10, 11, 12].jld2")


#####################################Energie levels #################
# L_list = 8:1:12
# x_values = 1 ./ L_list
# # Energie_values = zeros(ComplexF64,(length(L_list),5))
# # for i in 1:1:length(L_list)
# #     for j in 1:1:5
# #         Energie_values[i,j] = data[i][j]
# #     end
# # end
# data = Energie_values
# p = plot(; xlabel="L", ylabel="Re(E4-E0)",title = "4th Energy gap of ChepigaAnsatz2 Pottsq5 sector0, D100")
# plot!(p,x_values,real((data[:,5])),seriestype=:scatter,label="real")

# f = fit(x_values,real(data[:,5]), 1)
# c = f.coeffs[2]
# println(c)
# plot!(p,x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"ChepigaAnsatz2-fourth Energy gap Pottsq5D100 sector0.png")



# q = plot(; xlabel="1/L", ylabel="Im(E4-E0)",title="4th Energy gap of ChepigaAnsatz2 Pottsq5 sector0, D100")
# plot!(q,x_values,real(im.*(data[:,4])),seriestype=:scatter,label="imaginary" )
# f = fit(x_values,real(im.*(data[:,4])), 1)
# println(f.coeffs)
# c = f.coeffs[2]
# println(c)
# plot!(q, x_values -> f(x_values); label="fit im(a) = $c")

# savefig(q,"Im ChepigaAnsatz2-fourth Energy  gap Pottsq5 D100 sector0.png")












######################################### Energie levels scaling dimension##################
# x_values = 1 ./ L_list
# f = fit(x_values , real(E_gap), 1)
# xn = f.coeffs[2]/ (2 * pi)
# p = plot(; xlabel="1/L", ylabel="real(E1-E0)",xlimits =(0,0.2))
# plot!(p, x_values  -> f(x_values ); label="fit real(xn) = $xn")
# p = plot!(1 ./L_list,real(E_gap); seriestype=:scatter)




# f = fit(x_values , real(-im*E_gap), 1)
# xn = f.coeffs[2]/ (2 * pi)
# q = plot(; xlabel="1/L", ylabel="im(E1-E0)",xlimits =(0,0.2))
# plot!(q,x_values  -> f(x_values ); label="fit im(xn) = $xn")
# q = plot!(x_values ,real(-im*E_gap); seriestype=:scatter)
# savefig(p,"Real Energy Difference scaling L=[8,9,10,11,12,13], D= $D.png")
# savefig(q,"Imaginary Energy Difference scaling L=[8,9,10,11,12,13], D= $D.png")
























################################## central charge ########################################
# L_list = 8:1:12
# D=100
# x_values = 1 ./L_list.^2
# divided_energies = similar(L_list,ComplexF64)
# E_gap = similar(L_list,ComplexF64)
# for j in 1:1:length(L_list)
#     divided_energies[j] = data[j][1]/L_list[j]
# end

# f = fit(x_values, real(divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Re(E0/L)", xlim = (0,0.025), ylim = (-4.25,-4.21))
# p = plot!(x_values,real(divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"Real Energy scaling L=[8,9,10,11,12,13], D = $D.png")


# f = fit(x_values, real(-im*divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Im(E0/L)", xlim = (0,0.025), ylim = (-0.284,-0.274))
# p = plot!(x_values,real(-im.*divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"Imaginary Energy scaling L=[8,9,10,11,12,13], D= $D.png")



### Model comparison
# L_list = 8:1:12
# D=100
# x_values = 1 ./L_list.^2
# energie_diff = similar(L_list,ComplexF64)

# for j in 1:1:length(L_list)
#     energie_diff[j]  = (data[j][1]-data2[j][1])
# end

# p = plot(; xlabel="L", ylabel="Re(E0_z5 - E0)")
# p = plot!(L_list,real(energie_diff) ; seriestype=:scatter)
# savefig(p,"E diff between Z5 symmetric L=[8,9,10,11,12], D = $D.png")

# q = plot(; xlabel="L", ylabel="im(E0_z5 - E0)")
# q = plot!(L_list,real(-im.*energie_diff) ; seriestype=:scatter)
# savefig(q,"Im E diff between Z5 symmetric L=[8,9,10,11,12], D = $D.png")