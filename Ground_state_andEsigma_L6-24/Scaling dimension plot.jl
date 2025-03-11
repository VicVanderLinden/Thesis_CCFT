# ##Scaling dimension plot

using JLD2
using Polynomials

# datasector0 = [5.4796-8.1251im, 21.984-9.6931im, 23.4532-11.1809im]
# datasector1 = [2.16395-1.63599im, 17.2026-5.14224im, 18.076-11.41066im]


# v = datasector1[end]/(2*pi*(1.1336−0.0205im))
# v = 2.8810 −0.7091im
# println(v)
# testsector0 = datasector0/(2*pi*v)
# testsector1 = datasector1/(2*pi*v)

# actual_sector0 = [0.4656−0.2245im, 1.4656−0.2245im, 1.908-0.599im]
# actual_sector1 = [0.1336−0.0205im, 1.111 −0.170im, 1.1336−0.0205im]

# q = plot(; xlabel="charge", ylabel="Real Δn",title="Dimension scaling")
# plot!(q,[0,0,0],real(testsector0),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
# plot!(q,[0,0,0],real(actual_sector0),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
# plot!(q,[1,1,1],real(testsector1),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
# plot!(q,[1,1,1],real(actual_sector1),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
# savefig(q,"Real QuasiparticleAnsatz 3Dimensionscaling D50_v.png")

# p=plot(; xlabel="charge", ylabel="Im Δn",title="Dimension scaling")
# plot!(p,[0,0,0],real(-im.*testsector0),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
# plot!(p,[0,0,0],real(-im*actual_sector0),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
# plot!(p,[1,1,1],real(-im*testsector1),seriestype=:scatter,label="results",markeralpha=[0.5,0.5,0.5])
# plot!(p,[1,1,1],real(-im*actual_sector1),seriestype=:scatter,label="theoretical",markeralpha=[0.5,0.5,0.5])
# savefig(p,"Im QuasiparticleAnsatz 3Dimensionscaling D50_v.png")




λ_c = 0.079-0.06im # complex conjugate of Tang et al.
Δσ = 0.134-0.021im # theoretical prediction
v = 2.8810 - 0.7091im # fit from Tang et al.
c = 1.1375 - 0.0211im # theoretical prediction

λ_list = LinRange(-0.010,0.010,5) .+ (λ_c)
L_list = [6,8,10,12,14,16,20,24]

using CairoMakie

##
loc = "Ground_state_andEsigma_L6-24/"

E0_list = zeros(Complex{Float64},length(L_list),length(λ_list))
gap_list = zeros(Complex{Float64},length(L_list),length(λ_list))

# data consists of E0 and Eσ values for various L and λ values
# rows are different λs, columns are different Ls
for (i,L) in enumerate(L_list)
    E0_list[i,:] = load(loc*"Energy_$(L).jld2")["E0"]
    gap_list[i,:] = load(loc*"Energy_$(L).jld2")["Eσ"] - load(loc*"Energy_$(L).jld2")["E0"]
end

# 3rd column is critical value, largest L is smallest finite-size correction
data = (gap_list .* L_list .- gap_list[end,3] * L_list[end])
coeff = 2π * Δσ * v
data2 = data ./ coeff

function RGplot()
    f = Figure(size = (800, 800),fontsize=30)
    ax = Axis(f[1,1],xlabel=L"Re(g)", ylabel=L"Im(g)")
    for (i,λ) in enumerate(λ_list)
        scatter!(ax, real(data[:,i]), imag(data[:,i]), label = "λ = $(round(λ,digits=3))",markersize=12)
    end
    scatter!(ax, real(data[1,:]), imag(data[1,:]),marker=:cross,color=:black,label=L"L=6",markersize=14)
    scatter!(ax, real(data[end,:]), imag(data[end,:]),marker=:cross,color=:red,label="L=$(L_list[end])",markersize=14)
    axislegend(ax, position = :rt)
    f
end

# include finite-size corrections from irrelevant operators
ΔT2 = 4
ep = 1.5
function fscplot(dat=data, all_lambdas=true)
    f = Figure(size = (800, 800),fontsize=30)
    y_label = dat == data2 ? L"(E_\sigma - E_0) * L / 2πvΔ_σ" : L"(E_\sigma - E_0) * L"
    ax = Axis(f[1,1],xlabel=L"1/L^{Δ_{T^2} - 2}", ylabel=y_label)
    if !all_lambdas # just the critical value
        scatter!(ax, 1 ./ L_list .^(ep), real(dat[:,3]), label = "λ = $λ_c",markersize=12)
    else
        for (i,λ) in enumerate(λ_list)
            scatter!(ax, 1 ./ L_list .^(ep), real(dat[:,i]), label = "λ = $(round(λ,digits=3))",markersize=12)
        end
    end
    lines!(ax, 1 ./ L_list .^(ep), real(dat[:,3]))
    axislegend(ax, position = :rt)
    display(f)
    return linear_fit(1 ./ L_list .^(ep), real(dat[:,3]))
end
fscplot(data,false)