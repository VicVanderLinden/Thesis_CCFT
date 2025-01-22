
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
T = zeros(ComplexF64,Q,Q)           
ω = exp(2*π*im/Q)   
for i in 0:Q-1
    for j in 0:Q-1
        T[i+1,j+1] = (ω^i)^j
    end
end
P = TensorMap(T/sqrt(5),ℂ^Q←ℂ^Q)
pspace = ComplexSpace(Q)                     
P_inv = TensorMap(inv(P.data), pspace ← pspace)  


using LinearAlgebra     
using Optim


Δε = 0.4656 − 0.2245im
ΔL1ε = 1.4656 − 0.2245im
Δσ = 0.1336 − 0.0205im
ΔL1σ =1.1336 − 0.0205im
Cε_primeσσ = 0.0658 + 0.0513im 
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.908 − 0.599im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)


function lambda_estimation(L,lambda_range,alg="QuasiparticleAnsatz")
    gε_prime = zeros(ComplexF64,length(lambda_range))
    gε_prime_wo_C = zeros(ComplexF64,length(lambda_range))
    Eε = zeros(ComplexF64,length(lambda_range))
    Eσ = zeros(ComplexF64,length(lambda_range))
    EL1ε = zeros(ComplexF64,length(lambda_range))
    EL1σ = zeros(ComplexF64, length(lambda_range))

    ## sector search
    ψ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D)) #;left=Vleft, right=Vright) 
   
    for (i,lambda) in enumerate(lambda_range)
        H = Potts_Hamiltonian(L;lambda = lambda)
        (ψ, envir , delta)   = find_groundstate(ψ, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        #(ψ, envir , delta)   = find_groundstate(ψ, H, VUMPS(maxiter = 500,tol=1e-6))
       
        if alg == "QuasiparticleAnsatz"
            println("here")
            En0, st0 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=2)
            En1, st1 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(1),num=2)
            En = vcat(En1,En0)
            println(En)
            st = vcat(st1,st0)
        end
        if alg == "ChepigaAnsatz"
            En, st = excitations(H, ChepigaAnsatz(), ψ, envir; num=4)
        end
        # if alg == "DMRG"
        #      En ,st  = excitations(H,FiniteExcited(),ψ,init=ψ1,num=2); 
        #      En ,st  = excitations(H,FiniteExcited(),ψ,init=ψ2,num=2); 
        # end
        ΔEε = En[3]
        ΔEL1ε = En[4]
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
        EL1σ[i] = ΔEL1σ
    end
    save_object("Lambda_est_ge_woc$L.jld2", gε_prime_wo_C)
    save_object("Lambda_est_ge$L.jld2", gε_prime)
    save_object("ΔEε with lambda for $L.jld2",Eε)
    save_object("ΔEσ with lambda for$L.jld2", Eσ)
    save_object("ΔEl1ε with lambda for$L.jld2", EL1ε)
    save_object("ΔEL1σ with lambda for$L.jld2", EL1σ)
end

N=5 
D = 50
test_values = zeros(ComplexF64,(2*N-1)^2)
distx = 0.04 ## distance from alleged fixed point 0.079+0.060i in real
disty = 1 # distance from alleged fixed point 0.079+0.060i in imaginary
### changed this slightly to allow for any parameter N to cross at 0.079 + 0.060i point -> its 2N-1 parameter square now (for any N)
for i in 1:1:(2*N-1)
    for j in 1:1:(2*N-1)
        if i <N+1
            if j<N+1
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i]) .+ (0.079)  + 1im*LinRange(-disty,0.00,N)[j] .+ 0.06im 
            else 
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i]) .+ (0.079)  + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+ 0.06im 
            end
        else
            if j<N+1
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+ (0.079)  + 1im*LinRange(-disty,0.00,N)[j] .+ 0.06im 
            else
                test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+ (0.079)  + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+ 0.06im 
            end
        end 
       
    end
end




## run simulation here -> chose length scales
for L in [6]
    lambda_estimation(L,test_values)
end
