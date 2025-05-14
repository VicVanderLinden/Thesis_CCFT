
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Polynomials
using JLD2
using LinearAlgebra     
using Optim
## POTTS HAMILTONIAN


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
function Potts_Hamiltonian(L; J=1,h=1,Q=5,lambda=0.079 + 0.060im,sym=true,adjoint=false)
    ### symmetry
    if sym       
        H = periodic_boundary_conditions(quantum_potts(ZNIrrep{Q};q=Q), L)
        _,_,W = weyl_heisenberg_matrices(Q)
        P   = TensorMap(W,ℂ^Q←ℂ^Q)
        lat = FiniteChain(L)
        # dat0 = reshape((P'*sum((-h * potts_spin_shift(; q = Q,k=j)) for j in 1:1:Q-1)*P).data, (Q,Q))
        # dat1 = reshape(((P' ⊗ P')*sum((-J * potts_phase(; q=Q,k=j)) for j in 1:1:Q-1)*(P ⊗ P)).data, (Q,Q,Q,Q))
        dat2 = reshape(((P' ⊗ P') * sum(sum(potts_phase_shift_combined(;q=Q,k=l,p=j) for l in 1:1:Q-1) for j in 1:1:Q-1) *(P⊗P)).data, (Q,Q,Q,Q))
        # H0 = @mpoham (sum(TensorMap(dat0,Vp←Vp){i} for i in vertices(lat)[1:(end)])) ### Potts
        # H1 = @mpoham (sum(TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){i,i+1}  for i in vertices(lat)[1:(end-1)]) + TensorMap(dat1, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]}) ##¨Potts with BC
        H2 =  @mpoham lambda * sum(TensorMap(dat2,Vp⊗Vp←Vp⊗Vp){i,i+1} for i in vertices(lat)[1:(end - 1)]) + lambda * TensorMap(dat2, Vp⊗Vp←Vp⊗Vp){vertices(lat)[end],vertices(lat)[1]} #
        # ham = H0+H1+H2
        ham = H + H2
    else
        if !adjoint
            ham = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
            +sum( -J * potts_phase(; q=Q,k=j){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
            + lambda * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
            + lambda * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l){vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
        else
            ham = @mpoham (sum(sum((-J * potts_phase(; q=Q,k=j)'){i,i+1} + (-h * potts_spin_shift(; q = Q,k=j)'){i} for j in 1:1:Q-1) for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" potts model
            +sum( -J * potts_phase(; q=Q,k=j)'{vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]}+ (-h * potts_spin_shift(; q = Q,k=j)'){vertices(FiniteChain(L))[end]} for j in 1:1:Q-1) ##potts model periodic bc
            + lambda' * sum( sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l)'{i,i+1} for l in 1:1:Q-1) for j in 1:1:Q-1)   for i in vertices(FiniteChain(L))[1:(end - 1)]) ##" additional non hermitian model
            + lambda' * sum(sum(potts_phase_shift_combined(;q=Q,k=j,p=l)'{vertices(FiniteChain(L))[end],vertices(FiniteChain(L))[1]} for l in 1:1:Q-1) for j in 1:1:Q-1)); ## non hermitian model periodic bc
        end
    end
    return ham
end

J = 1
h = 1 
Q = 5
### symmetry
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)


Δε = 0.4656 − 0.2245im
ΔL1ε = 1.4656 − 0.2245im
Δσ = 0.1336 − 0.0205im
ΔL1σ =1.1336 − 0.0205im
Cε_primeσσ = 0.0658 + 0.0513im 
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.9083 − 0.5987im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)


function lambda_estimation(D,L,lambda_range,alg)
    ## sector search
    ψ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    if alg == "DMRG"
    ψ2 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D),left = Vect[ZNIrrep{Q}](0=>0,1=>1,2=>0,3=>0,4=>0),right = Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    ψ3 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D), left = Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0),right = Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    ψ4 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D),left = Vect[ZNIrrep{Q}](0=>0,1=>1,2=>0,3=>0,4=>0),right = Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    ψ5 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D),left = Vect[ZNIrrep{Q}](0=>0,1=>1,2=>0,3=>0,4=>0),right = Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    ψ6 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D),left= Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0) , right=Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    end
    for (i,lambda) in enumerate(lambda_range)
        H = Potts_Hamiltonian(L;lambda = lambda)
        (ψ, envir , delta)   = find_groundstate(ψ, H, DMRG(maxiter = 500,tol=1e-8, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        #(ψ, envir , delta)   = find_groundstate(ψ, H, VUMPS(maxiter = 500,tol=1e-6))
        ΔEε = 0
        ΔEL1ε = 0
        ΔEσ = 0
        ΔEL1σ = 0
        if alg == "QuasiparticleAnsatz"
            En0, st0 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=2)
            En1, st1 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir;  sector=ZNIrrep{5}(1),num=2)
            En = vcat(En1,En0)
            ΔEε = En[3]
            ΔEL1ε = En[4]
            ΔEσ = En[1]
            ΔEL1σ = En[2]
            expe = expectation_value(ψ,H)
            st = vcat(expe,st0)
            save_object("PBC L = $L,5statesQP$lambda.jld2", st)
        elseif alg == "ChepigaAnsatz"
            println("Doesn't work, only targets sector 0 for some reason")
            Ground_energy = expectation_value(ψ,H)
            En, st = excitations(H, ChepigaAnsatz(), ψ, envir;num=5)
            println(En.-Ground_energy)
            ΔEε = En[2]-Ground_energy
            ΔEL1ε = En[5]-Ground_energy
            ΔEσ = En[1]-Ground_energy
            ΔEL1σ = En[3]-Ground_energy
            stt = vcat([ψ,],st)
            save_object("5stateschepiga$lambda.jld2", stt)
        elseif alg == "DMRG" 
            Ground_energy = expectation_value(ψ,H)
            ψ2 ,st2,delta  = find_groundstate( ψ2,H,DMRG(maxiter = 500,tol=1e-8, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
            ΔEσ = expectation_value(ψ2,H)-Ground_energy
            En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=1e-8,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,),init = ψ3,num=1)
            ψ3 = st2[1]
            ΔEε = En[1]-Ground_energy
            En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=1e-8,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ2,),init = ψ4,num=1)
            ψ4 = st2[1]
            ΔEσprime = En[1]-Ground_energy
            En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=1e-6,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ2,ψ4,),init = ψ5,num=1)
            ψ5 = st2[1]
            ΔEL1σ = En[1]-Ground_energy
            # En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=1e-10,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,ψ3,),init = ψ6,num=1)
            # ψ6 = st2[1]
            # ΔEL1ε = En[1]-Ground_energy
            save_object("6energies$L,$alg,$lambda.jld2", [Ground_energy,ΔEσ,ΔEε,ΔEσprime,ΔEL1σ,]) #ΔEL1ε
        else
            pritnln("$alg not supported")
        end
    end
end

N = 6
v = 1
test_values = zeros(ComplexF64,(2*N+1)^2)

l = length(test_values)

distx = 0.07885 ## distance from centre in real

disty = 0.06025# distance from centre in imaginary

cent_im = 0.06025im

cent_r = 0.07885

for i in 1:1:(2*N+1)
        for j in 1:1:(2*N+1)
            if i <N+2
                if j<N+2
                    # println("here")
                    # println(length(LinRange(-distx,0.00,N+1)))
                    test_values[i+(j-1)*(2*N+1)] =  (LinRange(-distx,0.00,N+1)[i])  .+cent_r+ 1im*LinRange(-disty,0.00,N+1)[j] .+cent_im
                else 
                    test_values[i+(j-1)*(2*N+1)] =  (LinRange(-distx,0.00,N+1)[i])  .+cent_r + 1im*LinRange(disty/(N),disty+disty/(N),N)[j-N-1] .+cent_im
                end
            else
                if j<N+2
                    test_values[i+(j-1)*(2*N+1)] =  (LinRange(distx/(N),distx+distx/(N),N)[i-N-1]) .+cent_r  + 1im*LinRange(-disty,0.00,N+1)[j]  .+cent_im
                else
                    test_values[i+(j-1)*(2*N+1)] =  (LinRange(distx/(N),distx+distx/(N),N)[i-N-1])  .+cent_r+ 1im*LinRange(disty/(N),disty+disty/(N),N)[j-N-1].+cent_im
                end
            end 
        end
end
test_values_2 = zeros(ComplexF64,(2*N+1)^2)

l = length(test_values_2)

distx = 0.07885 ## distance from centre in real

disty = 0.06025# distance from centre in imaginary

cent_im = -0.06025im

cent_r = 0.07885

for i in 1:1:(2*N+1)
        for j in 1:1:(2*N+1)
            if i <N+2
                if j<N+2
                    # println("here")
                    # println(length(LinRange(-distx,0.00,N+1)))
                    test_values_2[i+(j-1)*(2*N+1)] =  (LinRange(-distx,0.00,N+1)[i])  .+cent_r+ 1im*LinRange(disty,0.00,N+1)[j] .+cent_im
                else 
                    test_values_2[i+(j-1)*(2*N+1)] =  (LinRange(-distx,0.00,N+1)[i])  .+cent_r - 1im*LinRange(disty/(N),disty+disty/(N),N)[j-N-1] .+cent_im
                end
            else
                if j<N+2
                    test_values_2[i+(j-1)*(2*N+1)] =  (LinRange(distx/(N),distx+distx/(N),N)[i-N-1]) .+cent_r  + 1im*LinRange(disty,0.00,N+1)[j]  .+cent_im
                else
                    test_values_2[i+(j-1)*(2*N+1)] =  (LinRange(distx/(N),distx+distx/(N),N)[i-N-1])  .+cent_r - 1im*LinRange(disty/(N),disty+disty/(N),N)[j-N-1].+cent_im
                end
            end 
        end
end
println(test_values[14])
test_values = vcat(test_values,test_values_2[14:end])
using Plots
scatter(real(test_values),imag(test_values))
println(test_values)
# ## run simulation here 
# D = 80
# test_values =[0.07885+0.06025im,]
# for L in [12,13,14,15]
#     lambda_estimation(D,L,test_values,"DMRG")
# end
