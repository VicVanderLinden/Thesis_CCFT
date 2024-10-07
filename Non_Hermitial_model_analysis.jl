
using JLD2
using Plots
data = [-33.212827480601305 + 2.791500707592999im -32.91404308182004 + 2.5567487132950015im -32.91419279561889 + 2.562498939180486im -32.90996665066797 + 2.568002392891223im -32.9077506429289 + 2.567002486958397im -32.918900070735816 + 2.583833397730901im -32.932557426919814 + 2.5942830212332746im -32.92573918461274 + 2.595582300537449im -32.93114874680414 + 2.5965323938489546im -32.936935591638786 + 2.6066204467991674im -32.94319260048343 + 2.6078670370653048im ; -37.31544446022326 + 3.1188523610037073im -37.051826752493604 + 2.9236197485168742im -37.04617335277802 + 2.941443394996058im -37.04601823083093 + 2.9296630699261406im -37.05710217282382 + 2.9425715495356277im -37.050237961463495 + 2.9394260461626858im -37.05464051317381 + 2.9539218577261157im -37.057454059820145 + 2.954477837438376im -37.06610520216612 + 2.9598783626307212im -37.072738403023536 + 2.953763407149434im -37.07752384509823 + 2.9598822787491232im ; -41.42155337276354 + 3.447591797746254im -41.18566825938927 + 3.2852041988618033im -41.181070320291575 + 3.2829969031737294im -41.180235820842384 + 3.299586923608662im -41.17605508834979 + 3.3055954385650126im -41.18510786542924 + 3.3097248680066826im -41.177703192316905 + 3.3150386795908604im -41.18827981462501 + 3.3093430215143296im -41.20342230504647 + 3.305596062596196im -41.19507612993433 + 3.3107243420206105im -41.21065705526494 + 3.308213651702952im]
#####################################Energie levels all #################
# 

p = plot(; xlabel="1/L", ylabel="real(E0 - En)",title = "Energy gap of Pottsq5 using naive DMRG")
for (d,L) in enumerate(L_List)
    plot!(p,1 ./(zeros(10).+L),real(data[d,2:end].-real(data[d,1])),seriestype=:scatter,legend =false )
end

savefig(p,"Real 10 level Energy scaling L=[8,9,10], D= 25.png")

q = plot(; xlabel="1/L", ylabel="IM(En-E0)",title="Energy gap of Pottsq5 using naive DMRG")
for (d,L) in enumerate(L_List)
    plot!(q,1 ./(zeros(10).+L),real(-im.*(data[d,2:end].-real(data[d,1]))),seriestype=:scatter,legend=false )
end
savefig(q,"Imaginary 10 level Energy scaling L=[8,9,10], D= 25.png")












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
# x_values = 1 ./L_list.^2
# divided_energies = similar(L_list,ComplexF64)
# E_gap = similar(L_list,ComplexF64)
# for j in 1:1:length(L_list)
#     divided_energies[j] = energies[j]/L_list[j]
#     E_gap[j] = (energies_E1[j]-energies[j])
# end

# f = fit(x_values, real(divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Re(E0/L)")
# p = plot!(x_values,real(divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(c) = $c")
# savefig(p,"Real Energy scaling L=[8,9,10,11,12,13], D = $D.png")


# f = fit(x_values, real(-im*divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Im(E0/L)")
# p = plot!(x_values,real(-im.*divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(c) = $c")
# savefig(p,"Imaginary Energy scaling L=[8,9,10,11,12,13], D= $D.png")



