
using MPSKit
using MPSKitModels
using TensorKit
using TensorOperations
using Polynomials
using JLD2
include("Potts-Operators & Hamiltonian.jl") 
J = 1
h = 1
Q = 5
### symmetry
Vp = Vect[ZNIrrep{Q}](0=>1,1=>1,2=>1,3=>1,4=>1)

using LinearAlgebra     
using Optim


Δε = 0.4656 − 0.2245im
ΔL1ε = 1.4656 − 0.2245im
Δσ = 0.1336 − 0.0205im
ΔL1σ =1.1336 − 0.0205im
Cε_primeσσ = 0.0658 + 0.0513im #####################" THIS ONE IS ACTUALLY NOT KNOW BEFOREHAND? how did the authors do this?
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.908 − 0.599im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)
function lambda_estimation(L,lambda_range)
    gε_prime = zeros(ComplexF64,length(lambda_range))
    gε_prime_wo_C = zeros(ComplexF64,length(lambda_range))
    Eε = zeros(ComplexF64,length(lambda_range))
    Eσ = zeros(ComplexF64,length(lambda_range))
    EL1ε = zeros(ComplexF64,length(lambda_range))
    EL1sigma = zeros(ComplexF64, length(lambda_range))
    ψ₀ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)) #;left=Vleft, right=Vright) 
    for (i,lambda) in enumerate(lambda_range)
        H = Potts_Hamiltonian(L;lambda = lambda)
        (ψ, envir , delta)   = find_groundstate(ψ₀, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=2)
        print(En)
        ΔEε = En[1]
        ΔEL1ε = En[2]
        En, st = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(1),num=2)
        ΔEσ = En[1]
        ΔEL1σ = En[2]
        fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) +abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1σ) -ΔL1σ  -Cε_primeσσ* AL1σε_prime* (x[2]+1im*x[4]))
        res = optimize(fun, [0.0, 0.0,0.0,0.0])
        gε_prime[i] = Optim.minimizer(res)[2]+1im* Optim.minimizer(res)[4]

        fun_2(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) 
        res_2 = optimize(fun_2, [0.0, 0.0,0.0,0.0])
        gε_prime_wo_C[i] = Optim.minimizer(res_2)[2]+1im* Optim.minimizer(res_2)[4]
        Eε[i] = ΔEε
        Eσ[i] = ΔEσ
        EL1ε[i] = ΔEL1ε
        EL1sigma[i] = ΔEL1σ
    end
    save_object("Lambda_est_ge_woc$L.jld2", gε_prime_wo_C)
    save_object("Lambda_est_ge$L.jld2", gε_prime)
    save_object("ΔEε with lambda for $L.jld2",Eε)
    save_object("ΔEσ with lambda for$L.jld2", Eσ)
    save_object("ΔEl1ε with lambda for$L.jld2", EL1ε)
end

N=5 #### make sure it has a zero in linspace (5 or 9) depending on accuracy
D = 50
test_values = zeros(ComplexF64,(N*N))
for i in 1:1:N
    for j in 1:1:N
        test_values[i+(j-1)*N] =  LinRange(-0.040,0.040,N)[i] .+ (0.079)  + 1im*LinRange(-0.040,0.040,N)[j] .+ 0.06im 
    end
end
for L in [6,8,10,12]
    lambda_estimation(L,test_values)
end


  