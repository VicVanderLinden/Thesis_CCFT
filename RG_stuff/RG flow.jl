using CairoMakie
using JLD2
using CurveFit

λ_c = 0.079-0.06im # complex conjugate of Tang et al.
Δσ = 0.134-0.021im # theoretical prediction
v = 2.8810 - 0.7091im # fit from Tang et al.
c = 1.1375 - 0.0211im # theoretical prediction

λ_list = LinRange(-0.010,0.010,5) .+ (λ_c)
L_list = [6,8,10,12,14,16,20,24]

E0_list = zeros(Complex{Float64},length(L_list),length(λ_list))
gap_list = zeros(Complex{Float64},length(L_list),length(λ_list))

loc = "Thesis_CCFT/RG_stuff/RGdata/"

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
function fscplot(dat=data, all_lambdas=true)
    f = Figure(size = (800, 800),fontsize=30)
    y_label = dat == data2 ? L"(E_\sigma - E_0) * L / 2πvΔ_σ" : L"(E_\sigma - E_0) * L"
    ax = Axis(f[1,1],xlabel=L"1/L^{Δ_{T^2} - 2}", ylabel=y_label)
    if !all_lambdas # just the critical value
        scatter!(ax, 1 ./ L_list .^2, real(dat[:,3]), label = "λ = $λ_c",markersize=12)
    else
        for (i,λ) in enumerate(λ_list)
            scatter!(ax, 1 ./ L_list .^2, real(dat[:,i]), label = "λ = $(round(λ,digits=3))",markersize=12)
        end
    end
    lines!(ax, 1 ./ L_list .^2, real(dat[:,3]))
    axislegend(ax, position = :rt)
    display(f)
    return linear_fit(1 ./ L_list .^2, real(dat[:,3]))
end

# using v from Tang et al. for c
start = 1
fit = linear_fit(1 ./ L_list[start:end].^2, (E0_list[start:end,3]) ./ L_list[start:end])
c_est = -fit[2]*6/v/π

# fit v to Δσ
v_fit = gap_list[end,3] * L_list[end] / (2π * Δσ)
c_est_fit = -fit[2]*6/v_fit/π

# same fit, averaging over lengths
vlinearfit = linear_fit(1 ./ L_list.^2, gap_list[:,3] .* L_list)
vfit = vlinearfit[1] / (2π*Δσ)
clinear_estfit = -fit[2]*6/vfit/π

function fscplot2(all_lambdas=true)
    f = Figure(size = (2000, 2000),fontsize=30)
    ax = Axis(f[1,1],xlabel=L"1/L^2", ylabel=L"Re(E_\sigma - E_0)")
    ax2 = Axis(f[1,2],xlabel=L"1/L", ylabel=L"Re(E_\sigma - E_0)L")
    ax3 = Axis(f[2,1],xlabel=L"1/L^2", ylabel=L"Im(E_\sigma - E_0)")
    ax4 = Axis(f[2,2],xlabel=L"1/L", ylabel=L"Im(E_\sigma - E_0)L")
    if !all_lambdas # just the critical value
        scatter!(ax, 1 ./ L_list .^ 2, real(gap_list[:,3]), label = "λ = $λ_c",markersize=12)
        scatter!(ax2, 1 ./ L_list, real(gap_list[:,3]) .* L_list, label = "λ = $λ_c",markersize=12)
        scatter!(ax3, 1 ./ L_list .^ 2, imag(gap_list[:,3]), label = "λ = $λ_c",markersize=12)
        scatter!(ax4, 1 ./ L_list, imag(gap_list[:,3]) .* L_list, label = "λ = $λ_c",markersize=12)
    else
        for (i,λ) in enumerate(λ_list)
            scatter!(ax, 1 ./ L_list .^ 2, real(gap_list[:,i]), label = "λ = $(round(λ,digits=3))",markersize=12)
            scatter!(ax2, 1 ./ L_list, real(gap_list[:,i]) .* L_list, label = "λ = $(round(λ,digits=3))",markersize=12)
            scatter!(ax3, 1 ./ L_list .^ 2, imag(gap_list[:,i]), label = "λ = $(round(λ,digits=3))",markersize=12)
            scatter!(ax4, 1 ./ L_list, imag(gap_list[:,i]) .* L_list, label = "λ = $(round(λ,digits=3))",markersize=12)
        end
    end
    axislegend(ax, position = :lt)
    axislegend(ax2, position = :rt)
    axislegend(ax3, position = :lt)
    axislegend(ax4, position = :rt)
    f
end

function cfit()
    f = Figure(size = (1200, 800),fontsize=30)
    ax = Axis(f[1,1],xlabel=L"1/L^2", ylabel=L"Re(E_0 /L)")
    ax2 = Axis(f[1,2],xlabel=L"1/L^2", ylabel=L"Im(E_0 /L)")
    scatter!(ax, 1 ./ L_list .^ 2, real(E0_list[:,3]) ./ L_list, label = "Re(E_0 /L)",markersize=12)
    scatter!(ax2, 1 ./ L_list .^ 2, imag(E0_list[:,3]) ./ L_list, label = "Im(E_0 /L)",markersize=12)
    fit_real = linear_fit(1 ./ L_list.^2, real(E0_list[:,3]) ./ L_list)
    fit_imag = linear_fit(1 ./ L_list.^2, imag(E0_list[:,3]) ./ L_list)
    c_real = -fit_real[2]*6/π/v
    c_imag = -fit_imag[2]*6/π/v
    @show c_real, c_imag
    axislegend(ax, position = :lt)
    axislegend(ax2, position = :rt)
    f
end
