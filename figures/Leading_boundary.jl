using MPSKit, MPSKitModels
using TensorKit
using KrylovKit
using TensorOperations
using MPSKitModels
# using LinearAlgebra
using Plots
using LaTeXStrings
using JLD2

using LoggingExtras
using MPSKit:
    updatetol,
    ∂∂AC,
    normalize!,
    add!,
    MPOHamiltonian,
    IterLog,
    loginit!,
    logiter!,
    logfinish!,
    logcancel!,
    fixedpoint,
    Algorithm,
    FiniteQP,
    LinearCombination,
    ProjectionOperator




### DMRG ADAPTATION to largest magnitude
@kwdef struct FiniteExcited_alt{A} <: Algorithm
    "optimization algorithm"
    gsalg::A = DMRG()
    "energy penalty for enforcing orthogonality with previous states"
    weight::Float64 = 1.0 ################THIS IS VARIABLE IN MY CODE -> because largest magn
end
function excitations_alt(
    H,
    alg::FiniteExcited_alt,
    states::Tuple{T,Vararg{T}},
    energies; #
    init = FiniteMPS([copy(first(states).AC[i]) for i = 1:length(first(states))]),
    num = 1,
) where {T<:FiniteMPS}
    num == 0 && return (scalartype(T)[], T[])
    super_op = LinearCombination(tuple(H, ProjectionOperator.(states)...), energies)
    envs = environments(init, super_op)
    ne, _ = find_groundstate_alt(
        init,
        super_op,
        DMRG(
            maxiter = 500,
            tol = 1e-6,
            eigalg = MPSKit.Defaults.alg_eigsolve(; ishermitian = false),
        ),
        envs,
    )
    E = expectation_value(ne, H)
    nstates = (states..., ne)
    energies = (energies..., -E)
    println(energies)
    ens, excis = excitations_alt(H, alg, nstates, energies; init = init, num = num - 1) ## bit redudant, but don't want to screw up format

    push!(ens, expectation_value(ne, H))
    push!(excis, ne)

    return ens, excis
end
function excitations_alt(H, alg::FiniteExcited_alt, ψ::FiniteMPS, energies; kwargs...) #,energies::Tuple{Float64}
    return excitations_alt(H, alg, (ψ,), energies; kwargs...)
end



function calc_galerkin_alt(
    pos::Int,
    above::Union{InfiniteMPS,FiniteMPS,WindowMPS},
    operator,
    below,
    envs,
)
    AC´ = ∂∂AC(pos, above, operator, envs) * above.AC[pos]
    normalize!(AC´)
    out = add!(AC´, below.AL[pos] * below.AL[pos]' * AC´, -1)
    return norm(out)
end

function find_groundstate_alt(ψ::FiniteMPS, H, alg, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin_alt(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter = 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)

            zerovector!(ϵs)
            for pos in [1:(length(ψ)-1); length(ψ):-1:2]
                h = ∂∂AC(pos, ψ, H, envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :LM, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin_alt(pos, ψ, H, ψ, envs))
                ψ.AC[pos] = vec
            end
            ϵ = maximum(ϵs)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}
            if ϵ <= 1e-5 ### I manually coded this because for some reason it would be turne dto 1e-10 for exications
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

function cal_lnz(data)
    lnz = 0
    for (i, z) in enumerate(data)
        lnz += log(z) * 2.0^(1 - i)
    end
    return lnz
end

function rough_lnz(O)
    scheme = TRG(O)
    data = run!(scheme,truncdim(16),maxiter(40))
    return cal_lnz(data)
end