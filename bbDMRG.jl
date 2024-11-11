### bbDMR   G

# Krylov-Schur-restarted Arnoldi    diagonalization technique?
struct bbDMRG{A,F} <: Algorithm
    tol::Float64
    maxiter::Int
    eigalg::A
    verbosity::Int
    finalize::F
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::bbDMRG, envs=environments(ψ, H))
    # ϵs = map(pos -> calc_galerkin(ψ, pos, envs), 1:length(ψ))
    # ϵ = maximum(ϵs)
    log = IterLog("bbDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)

            zerovector!(ϵs)


            ### loop over positions
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                h = ∂∂AC(pos, ψ, H, envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(ψ, pos, envs))
                ψ.AC[pos] = vec
            end


            
            ϵ = maximum(ϵs)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ
end
