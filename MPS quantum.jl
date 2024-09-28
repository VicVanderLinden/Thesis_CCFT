### MPS quantum
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials

function scaling_simulations_potts(ψ₀,H,Ds)
    entropies = similar(Ds,Float64)
    correlations = similar(Ds,Float64)
    ψ, envs = find_groundstate(ψ₀, H, VUMPS(maxiter = 500,tol=1e-7, alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
    entropies[1] = real(entropy(ψ)[1])
    correlations[1] = correlation_length(ψ)
    for (i,d) in enumerate(diff(Ds))
        ψ,envs = changebonds(ψ, H, OptimalExpand(; trscheme=truncdim(d)), envs)
        ψ, envs = find_groundstate(ψ, H, VUMPS(maxiter = 500,tol=1e-7, alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
        entropies[i+1] = real(entropy(ψ)[1])
        correlations[i+1] = correlation_length(ψ)
    end
    return entropies,correlations
end

Q = [2,3,4,5,10]
for q in Q
    println(q)

    ##intialize hamiltonian
    H0 = quantum_potts(InfiniteChain();q=q);
    ##### Contruct the ansatz
    ψ₀ = InfiniteMPS(ℂ^q, ℂ^25)
    
    bond_dimensions = 10:1:30 
    Ss, ξs = scaling_simulations_potts(ψ₀, H0, bond_dimensions)
    f = fit(log.(ξs), 6 * Ss, 1)
    c = f.coeffs[2]
    println(c)
    p = plot(; xlabel="logarithmic correlation length", ylabel="entanglement entropy")
    p = plot(log.(ξs), Ss; seriestype=:scatter, label=nothing)
    plot!(p, ξ -> f(ξ) / 6; label="fit c = $c")
    plot!(xlabel = "Correlation length", ylabel = "Entropy", title = "VUMPS-Potts model (q=$q) scaling hypothesis")
    savefig(p,"Entropy_scaling_Potts$q from_D10-30.png")
    p2 = plot(bond_dimensions,ξs)
    plot!(ylabel = "Correlation length", xlabel = "bond_dimensions", title = "VUMPS-Potts model (q=$q) correlation length")
    savefig(p2,"Correlation_length_Potts$q from_D10-30.png")
end