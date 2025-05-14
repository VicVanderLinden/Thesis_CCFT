### <psi| (H^dag-\lambda*)(H-\lambda) |psi>

using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Polynomials
using JLD2
using Optim
using LoggingExtras
include("Potts-Operators & Hamiltonian.jl") 
J = 1
h = 1
Q = 5
D=50
lambda = 0.079 + 0.060im
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)
using MPSKit: AbstractFiniteMPS, find_groundstate,calc_galerkin,LoggingExtras,updatetol,∂∂AC,fixedpoint, IterLog, MPSKit.loginit!, MPSKit.logfinish!, MPSKit.logcancel!, MPSKit.logiter!

## franks idea (i think)
function find_groundstate_fr(ψ::AbstractFiniteMPS,H, L, alg::DMRG, envs=environments(ψ, H))
    H = Potts_Hamiltonian(L;lambda = lambda)
    H_adj = Potts_Hamiltonian(L;lambda = lambda')
    ϵs = map(pos -> calc_galerkin(ψ, pos, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")   
    alt_ground_energy = 0 + 0im
    Ground_energy = 0+0im
    LoggingExtras.withlevel(; alg.verbosity) do 
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)
            envs=environments(ψ, H)
            zerovector!(ϵs)
            
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                h = ∂∂AC(pos, ψ, H, envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(ψ, pos, envs))
                ψ.AC[pos] = vec
            end
            ϵ = maximum(ϵs)
            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

            envs = environments(ψ, (H_adj-Ground_energy')*(H-Ground_energy))
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                h = ∂∂AC(pos, ψ, (H_adj-Ground_energy')*(H-Ground_energy), envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(ψ, pos, envs))
                ψ.AC[pos] = vec
            end
            ϵ = maximum(ϵs)
            ψ, envs = alg.finalize(iter, ψ, (H_adj-Ground_energy')*(H-Ground_energy), envs)::Tuple{typeof(ψ),typeof(envs)}

            if ϵ <= alg.tol
                  ### extra minimazation to find complex form, it doesn't really give an eigenvector now, nor does it indicate a direciton prior to converging
                  #Ground_energy = expectation_value(ψ,H)
                  #fun(x) = abs(expectation_value(ψ,H_adj*H) - (real(Ground_energy)+1im.*x[1])*expectation_value(ψ,H_adj) -(real(Ground_energy)-1im.*x[1])expectation_value(ψ,H) +abs(Ground_energy)^2)
                  #res = optimize(fun, [real(-im.*Ground_energy)])
                  #alt_ground_energy = real(Ground_energy)  + 1im*Optim.minimizer(res)[1]
                  # doesnt really optimize 
                  @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 2 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ,Ground_energy
end
Frank_terms = zeros(ComplexF64,length(8:12))
for (i,L) in enumerate(8:12)
    ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    H = Potts_Hamiltonian(L;lambda = lambda)
    H_adj = Potts_Hamiltonian(L;lambda = lambda')
    (ψ, envir , delta,ge)   = find_groundstate_fr(ψ₀, H,L, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    Frank_terms[i] = ge
end
print(Frank_terms)