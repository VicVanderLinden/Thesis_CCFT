### MPS quantum
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Plots
using Polynomials


function potts_spin_shift end
potts_spin_shift(; kwargs...) = potts_spin_shift(ComplexF64, Trivial; kwargs...)
potts_spin_shift(elt::Type{<:Number}; kwargs...) = potts_spin_shift(elt, Trivial; kwargs...)
function potts_spin_shift(symmetry::Type{<:Sector}; kwargs...)
    return potts_spin_shift(ComplexF64, symmetry; kwargs...)
end


function potts_spin_shift(elt::Type{<:Number}, ::Type{Trivial}; q=3, k=1)
    pspace = ComplexSpace(q)
    tau = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        tau[i,mod1(i - 1, q)] = one(elt)
    end
    return tau^k
end
function potts_phase end
potts_phase(; kwargs...) = potts_phase(ComplexF64, Trivial; kwargs...)
potts_phase(elt::Type{<:Number}; kwargs...) = potts_phase(elt, Trivial; kwargs...)
function potts_phase(symmetry::Type{<:Sector}; kwargs...)
    return potts_phase(ComplexF64, symmetry; kwargs...)
end


function potts_phase(elt::Type{<:Number}, ::Type{Trivial}; q=3,k=1)
    pspace = ComplexSpace(q)
    sigma = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        sigma[i, i] = cis(2*pi*(i-1)/q)
    end
    return (sigma'⊗ sigma)^k
end






function scaling_simulations_potts(ψ₀,H,Ds)
    entropies = similar(Ds,Float64)
    correlations = similar(Ds,Float64)
    ψ, envs = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
    entropies[1] = real(entropy(ψ)[1])
    correlations[1] = correlation_length(ψ, tol=1e-6, num_vals=2,eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))
    for (i,d) in enumerate(diff(Ds))
        ψ,envs = changebonds(ψ, H, OptimalExpand(; trscheme=trunc(d)), envs) ### truncr !!
        ψ, envs = find_groundstate(ψ, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
        entropies[i+1] = real(entropy(ψ)[1])
        correlations[i+1] = correlation_length(ψ,tol=1e-6, num_vals=2,eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false))
    end
    return entropies,correlations
end
function truncr_scaling_simulations_potts(ψ₀,H,amount)
    entropies = zeros(amount)
    correlations = zeros(amount)
    ψ, envs = find_groundstate(ψ₀, H, VUMPS(maxiter = 500,tol=1e-5, alg_eigsolve =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
    entropies[1] = real(entropy(ψ)[1])
    correlations[1] = correlation_length(ψ)
    for i in 1:1:amount-1
        ψ,envs = changebonds(ψ, H, OptimalExpand(; trscheme=truncerr(1e-5)), envs)
        ψ, envs = find_groundstate(ψ, H, VUMPS(maxiter = 500,tol=1e-5, alg_eigsolve =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)));
        entropies[i+1] = real(entropy(ψ)[1])
        correlations[i+1] = correlation_length(ψ)
    end
    return entropies,correlations
end






                                    ##### Simulating energies
Q_list = [5,]
J = 1
h=1
    


D = 10 ### begin bond dimension
amount = 5
for Q in Q_list
    
    ##intialize hamiltonian
    H0 = @mpoham sum( sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1){i} + sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1){i,p}  for (i,p) in nearest_neighbours(InfiniteChain(1)))
    ##### Contruct the ansatz
    ψ₀ = InfiniteMPS(ℂ^Q, ℂ^D);
    Ss, ξs = truncr_scaling_simulations_potts(ψ₀, H0, amount)
    f = fit(log.(ξs), 6 * Ss, 1)
    c = f.coeffs[2]
    println(c)
    p = plot(; xlabel="logarithmic correlation length", ylabel="entanglement entropy")
    p = plot(log.(ξs), Ss; seriestype=:scatter, label=nothing)
    plot!(p, ξ -> f(ξ) / 6; label="fit c = $c")
    plot!(xlabel = "Correlation length", ylabel = "Entropy", title = "truncr VUMPS-Potts model (q=$Q) scaling hypothesis")
    savefig(p,"truncr($amount) Entropy_scaling_Potts$Q.png")
    # p2 = plot(bond_dimensions,ξs)
    # plot!(ylabel = "Correlation length", xlabel = "bond_dimensions", title = "truncr VUMPS-Potts model (q=$Q) correlation length")
    # savefig(p2,"truncr Correlation_length_Potts$Q from_D10-30.png")
end


