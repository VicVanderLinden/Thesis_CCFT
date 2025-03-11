
using JLD2
using Plots
using Polynomials
test_values = LinRange(-0.010,0.010,5) .+ (0.079-0.06im)
data_E0 = Vector{ComplexF64}[]
data_Eσ = Vector{ComplexF64}[]
L_list = [8,9,10,11,12]
# for L in L_list
#     push!(data_E0,load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")["E0"])
#     push!(data_Eσ,load("Ground_state_andEsigma_L6-24/Energy_$L.jld2")["Eσ"])
# end

data = load_object("MPS_altlambda_pottsq5 excited-N5,D50,energies-L[8, 9, 10, 11, 12].jld2")
println(data)
# ####################################Energie levels #################
L_list = 8:1:12
x_values = 1 ./ L_list
N = 4
Energie_values = zeros(ComplexF64,(length(L_list),N+1))
for i in 1:1:length(L_list)
    for j in 1:1:N+1
        if j==1
        Energie_values[i,j] = data[i][j]/L_list[i]
        else 
        Energie_values[i,j] = data[i][j]
        end
    end
end
level = 1
p = plot(; xlabel="1/L^2", ylabel="Re(E0/L)",title = " alt lambbda GS Energy Pottsq5, D100")
plot!(p,x_values.^2,real((Energie_values[:,level])),seriestype=:scatter,label="real")
f = fit(x_values.^2,real((Energie_values[:,level])), 1)
c = f.coeffs[2]
plot!(p,x_values -> f(x_values); label="fit real(a) = $c")
savefig(p,"q5alt_lambda_QuasiparticleAnsatz-$level Energy gap Pottsq3D100.png")

dim_unscaled = zeros(ComplexF64,N,1)
for (i,level) in enumerate(2:1:5)
println(Energie_values[level,1])
p = plot(; xlabel="1/L", ylabel="Re(E$(level-1)-E0)",title = "alt lambda $(level-1) Energy gap Pottsq5, D100")
plot!(p,x_values,real((Energie_values[:,level])),seriestype=:scatter,label="real")
f = fit(x_values,real((Energie_values[:,level])), 1)
c = f.coeffs[2]
plot!(p,x_values -> f(x_values); label="fit real(a) = $c")
savefig(p,"q5alt_lambda_QuasiparticleAnsatz-$level Energy gap Pottsq3D100.png")

end

# q = plot(; xlabel="1/L", ylabel="Im(E$level-E0)",title="$level Energy gap Pottsq5 sector0, D100")
# plot!(q,x_values,real(im.*(Energie_values[:,level])),seriestype=:scatter,label="imaginary" )
# f = fit(x_values,real(im.*(Energie_values[:,level])), 1)
# d = f.coeffs[2]
# plot!(q, x_values -> f(x_values); label="fit im(a) = $d")
# savefig(q,"Im ChepigaAnsatz2-$level Energy  gap Pottsq5 D100 sector0.png")

# dim_unscaled[i] =c+1im*d
# end













# ######################################### Energie levels scaling dimension##################
# x_values = [0,0,0,0]

# p = plot(; xlabel="spin", ylabel="Re(Δ)",xlimits =(-0.5,0.2))
# println(dim_unscaled)
# plot!(p, x_values, real(dim_unscaled[:,1]) ./(2*pi*2.6284),seriestype=:scatter,label="operators")

# q = plot(; xlabel="spin", ylabel="Im(Δ)",xlimits =(-0.5,0.2))

# plot!(q, x_values, real(-im*dim_unscaled[:,1]) ./(2*pi*4.32329234552),seriestype=:scatter,label="operators")




















# # 

# ################################## central charge ########################################
# D=64
# x_values = 1 ./L_list.^2
# divided_energies = similar(L_list,ComplexF64)
# E_gap = similar(L_list,ComplexF64)
# println(size(data))
# for j in 1:1:length(L_list)
#     divided_energies[j] = data[j][1]/L_list[j]
# end

# f = fit(x_values, real(divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Re(E0/L)", xlim = (0,0.025))
# p = plot!(x_values,real(divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"alt Real Energy scaling L=$L_list, D = $D.png")


# f = fit(x_values, real(im*divided_energies), 1)
# c = f.coeffs[2]
# println(c)
# p = plot(; xlabel="1/L²", ylabel="Im(E0/L)", xlim = (0,0.025))
# p = plot!(x_values,real(im.*divided_energies) ; seriestype=:scatter)
# plot!(p, x_values -> f(x_values); label="fit real(a) = $c")
# savefig(p,"alt Imaginary Energy scaling $L_list, D= $D.png")













### Model comparison
# L_list = 8:1:12
# D=100
# x_values = 1 ./L_list.^2
# energie_diff = similar(L_list,ComplexF64)

# for j in 1:1:length(L_list)
#     energie_diff[j]  = (Energie_values[j][1]-Energie_values2[j][1])
# end

# p = plot(; xlabel="L", ylabel="Re(E0_z5 - E0)")
# p = plot!(L_list,real(energie_diff) ; seriestype=:scatter)
# savefig(p,"E diff between Z5 symmetric L=[8,9,10,11,12], D = $D.png")

# q = plot(; xlabel="L", ylabel="im(E0_z5 - E0)")
# q = plot!(L_list,real(-im.*energie_diff) ; seriestype=:scatter)
# savefig(q,"Im E diff between Z5 symmetric L=[8,9,10,11,12], D = $D.png")