
using JLD2
using Plots
using Polynomials
data = load_object("QuasiparticleAnsatz-MPSNonHermitian_pottsq5 excited-N5,D100,energies-L[9, 10, 11, 12], sector0_lambda.jld2")
#####################################Energie levels #################
L_list = 8:1:12
x_values = 1 ./ L_list
p = plot(; xlabel="L", ylabel="Re(E6-E0)",title = "6th Energy gap of quasiparticle Pottsq5 sector0, D50")
plot!(p,x_values,real((data[:,6])),seriestype=:scatter,label="real")

f = fit(x_values,real(data[:,6]), 1)
c = f.coeffs[2]
println(c)
plot!(p,x_values -> f(x_values); label="fit real(a) = $c")
savefig(p,"QuasiparticleAnsatz-6th Energy gap Pottsq5D50 sector0, lambda.png")



q = plot(; xlabel="1/L", ylabel="Im(E6-E0)",title="6th Energy gap of Pottsq5 sector0, D50")
plot!(q,x_values,real(-im.*(data[:,6])),seriestype=:scatter,label="imaginary" )
f = fit(x_values,real(-im.*(data[:,6])), 1)
println(f.coeffs)
c = f.coeffs[2]
println(c)
plot!(q, x_values -> f(x_values); label="fit im(a) = $c")

savefig(q,"Im QuasiparticleAnsatz-sixth Energy gap Pottsq5 D50 sector0,lambda.png")












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
























################################### central charge ########################################
# L_list = 8:1:13
# D=40
# x_values = 1 ./L_list.^2
# println(energies)
# divided_energies = similar(L_list,ComplexF64)
# E_gap = similar(L_list,ComplexF64)
# for j in 1:1:length(L_list)
#     divided_energies[j] = energies[j]/L_list[j]
#     E_gap[j] = (energies[j])
# end

# f = fit(x_values, real(divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Re(E0/L)")
# p = plot!(x_values,real(divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"Real Energy scaling L=[8,9,10,11,12,13], D = $D.png")


# f = fit(x_values, real(-im*divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Im(E0/L)")
# p = plot!(x_values,real(-im.*divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"Imaginary Energy scaling L=[8,9,10,11,12,13], D= $D.png")



