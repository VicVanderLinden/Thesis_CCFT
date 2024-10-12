
using JLD2
using Plots
energies = load_object("MPSNonHermitian_pottsq5VicVanderLinden-N1,D60,energies.jld2")

# #####################################Energie levels all #################
# # 

# p = plot(; xlabel="1/L", ylabel="real(En - E0)",title = "Energy gap of Pottsq5 using naive DMRG")
# for (d,L) in enumerate(L_List)
#     plot!(p,1 ./(zeros(10).+L),real(data[d,2:end].-real(data[d,1])),seriestype=:scatter,legend =false )
# end

# savefig(p,"Real 10 level Energy scaling L=[8,9,10], D= 25.png")

# q = plot(; xlabel="1/L", ylabel="IM(En-E0)",title="Energy gap of Pottsq5 using naive DMRG")
# for (d,L) in enumerate(L_List)
#     plot!(q,1 ./(zeros(10).+L),real(-im.*(data[d,2:end].-data[d,1])),seriestype=:scatter,legend=false)
# end
# savefig(q,"Imaginary 10 level Energy scaling L=[8,9,10], D= 25.png")












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
L_list = 8:1:13
D=40
x_values = 1 ./L_list.^2
println(energies)
divided_energies = similar(L_list,ComplexF64)
E_gap = similar(L_list,ComplexF64)
for j in 1:1:length(L_list)
    divided_energies[j] = energies[j]/L_list[j]
    E_gap[j] = (energies[j])
end

f = fit(x_values, real(divided_energies), 1)
c = f.coeffs[2]
println(c)
p = plot(; xlabel="1/L²", ylabel="Re(E0/L)")
p = plot!(x_values,real(divided_energies) ; seriestype=:scatter)
plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
savefig(p,"Real Energy scaling L=[8,9,10,11,12,13], D = $D.png")


f = fit(x_values, real(-im*divided_energies), 1)
c = f.coeffs[2]
println(c)
p = plot(; xlabel="1/L²", ylabel="Im(E0/L)")
p = plot!(x_values,real(-im.*divided_energies) ; seriestype=:scatter)
plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
savefig(p,"Imaginary Energy scaling L=[8,9,10,11,12,13], D= $D.png")



