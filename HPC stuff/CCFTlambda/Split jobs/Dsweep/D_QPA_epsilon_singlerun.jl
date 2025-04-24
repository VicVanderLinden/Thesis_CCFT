include("lambdadephelper.jl")

using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using LinearAlgebra
using JLD2
using ArgParse

BLAS.set_num_threads(8) # try with 1
@info "number of BLAS threads: " BLAS.get_num_threads()

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
        tau[i,mod1(i + 1, q)] = one(elt)
    end
    return (tau^k)                    ### nonzero elements on the parts that intermingle between sectors
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
    return ((sigma'⊗ sigma)^k)
end

function potts_phase_shift_combined end
potts_phase_shift_combined(; kwargs...) = potts_phase_shift_combined(ComplexF64, Trivial; kwargs...)
potts_phase_shift_combined(elt::Type{<:Number}; kwargs...) = potts_phase_combined(elt, Trivial; kwargs...)
function potts_phase_shift_combined(symmetry::Type{<:Sector}; kwargs...)
    return potts_phase_combined(ComplexF64, symmetry; kwargs...)
end


function potts_phase_shift_combined(elt::Type{<:Number}, ::Type{Trivial}; q=3,k=1,p=1)
    pspace = ComplexSpace(q)
    sigma = TensorMap(zeros, elt, pspace ← pspace)
    tau = TensorMap(zeros, elt, pspace ← pspace)
    identity_e = TensorMap(zeros, elt, pspace ← pspace)
    for i in 1:q
        sigma[i, i] = cis(2*pi*(i-1)/q)
        tau[i,mod1(i + 1, q)] = one(elt)
        identity_e[i,i]= 1
    end
    return (tau^k⊗ identity_e) * (sigma'⊗ sigma)^p + (identity_e ⊗ tau^k) * (sigma'⊗ sigma)^p + (sigma'⊗ sigma)^k * (tau^p⊗ identity_e) +  (sigma'⊗ sigma)^k * (identity_e ⊗ tau^p)
end


### model parameters
## we target epsilon since the most sensitive
function Potts_Hamiltonian(L; J=1,h=1,Q=5,lambda=0.079 + 0.060im,sym=true,adjoint=false)
    ### symmetry
    if sym    
        if !adjoint   
            H = periodic_boundary_conditions(quantum_potts(ZNIrrep{Q};q=Q), L)
            _,_,W = weyl_heisenberg_matrices(Q)
            P   = TensorMap(W,ℂ^Q←ℂ^Q)
            lat = FiniteChain(L)
            # dat0 = reshape((P'*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data, (Q,Q))
            # dat1 = reshape(((P' ⊗ P')*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, (Q,Q,Q,Q))
            dat2 = reshape(((P' ⊗ P') * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data, (Q,Q,Q,Q))
            # H0 = @mpoham (sum(TensorMap(dat0,Vp←Vp){i} for i in vertices(lat)[1:(end)])) ### Potts
            # H1 = @mpoham (sum(TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(lat)[1:(end-1)]) + TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]}) ##¨Potts with BC
            H2 =  @mpoham lambda * sum(TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){i,i+1} for i in vertices(lat)[1:(end - 1)]) + lambda * TensorMap(dat2, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]} ###Extra term
            # ham = H0+H1+H2
            ham = H + H2
        else 
            _,_,W = weyl_heisenberg_matrices(Q)
            P   = TensorMap(W,ℂ^Q←ℂ^Q)
            lat = FiniteChain(L)
            dat0 = reshape((P*sum((-h * potts_spin_shift(; q = Q,k=j)') for j in 1:1:Q-1)*P').data, (Q,Q))
            dat1 = reshape(((P ⊗ P)*sum((-J * potts_phase(; q=Q,k=j)') for j in 1:1:Q-1)*(P' ⊗ P')).data, (Q,Q,Q,Q))
            dat2 = reshape(((P ⊗ P) * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j)' for l in 1:1:Q-1) for j in 1:1:Q-1) *(P'⊗P')).data, (Q,Q,Q,Q))
            H0 = @mpoham (sum(TensorMap(dat0,Vp←Vp){i} for i in vertices(lat)[1:(end)])) ### Potts
            H1 = @mpoham (sum(TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(lat)[1:(end-1)]) + TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]}) ##¨Potts with BC
            H2 =  @mpoham lambda'* sum(TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){i,i+1} for i in vertices(lat)[1:(end - 1)]) + lambda' * TensorMap(dat2, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]} ###Extra term
            ham = H0+H1+H2
        end
        
    else
        if !adjoint
            ham = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
            +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
            + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
            + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
        else
            ham = @mpoham (sum(sum((-J * conjugate(potts_phase(; q=Q,k=j))){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)'){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
            +sum( -J * conjugate(potts_phase(; q=Q,k=j)){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * conjugate(potts_spin_shift(; q = Q,k=j))){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
            + lambda' * sum( sum(sum(conjugate(potts_phase_shift_combined)(;q=Q,k=j,p=l)'{i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
            + lambda' * sum(sum(conjugate(potts_phase_shift_combined(;q=Q,k=j,p=l)){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
        end
    end
    return ham
end

function qpa_epsilon(D,lambda,Q )
    ψ = jldopen(readpath, "r") do file
        return file["ψ"]
    end
      
    # ψ2 = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    H = Potts_Hamiltonian(L;lambda = lambda)
    (ψ, _ , _)   = find_groundstate(ψ, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
    jldsave(path; ψ)

    En0, _ = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=1)
    expe = expectation_value(ψ,H)
    st = vcat(expe,En0)
    jldsave(savepath; st)
end

s = ArgParseSettings();

@add_arg_table s begin
    "--D"	
        arg_type = Int
        default = 10
end

parsed_args = parse_args(s)
D = parsed_args["D"]
L = 24 
lambda = 0.0790 + 0.060im
Q = 5
Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)

readpath = joinpath("Data", "Groundstates", "5GS_PBC_D$(D)_L=$(L)_$(lambda).jld2")
savepath = joinpath("Data", "Epsilon", "QPA_PBC_D$(D)_L=$(L)_$(lambda).jld2")

t = time()
qpa_epsilon(D,lambda,5)
dt = time() - t
@info "bond dimension = $D,time = $dt"
