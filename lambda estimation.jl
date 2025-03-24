
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
Cε_primeσσ = 0.0658 + 0.0513im 
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.908 − 0.599im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)


function lambda_estimation(L,lambda_range,alg)
    gε_prime = zeros(ComplexF64,length(lambda_range))
    gε_prime_wo_C = zeros(ComplexF64,length(lambda_range))
    Eε = zeros(ComplexF64,length(lambda_range))
    Eσ = zeros(ComplexF64,length(lambda_range))
    EL1ε = zeros(ComplexF64,length(lambda_range))
    EL1σ = zeros(ComplexF64, length(lambda_range))

    ## sector search
    ψ = FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    ψ2 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    ψ3 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D),left= Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0) , right=Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    ψ4 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    ψ5 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D))
    ψ6 =FiniteMPS(L,Vp,Vect[ZNIrrep{Q}](0=>D,1=>D,2=>D,3=>D,4=>D),left= Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0) , right=Vect[ZNIrrep{Q}](0=>1,1=>0,2=>0,3=>0,4=>0))
    for (i,lambda) in enumerate(lambda_range)
        H = Potts_Hamiltonian(L;lambda = lambda)
        (ψ, envir , delta)   = find_groundstate(ψ, H, DMRG(maxiter = 500,tol=1e-6, eigalg =MPSKit.Defaults.alg_eigsolve(; ishermitian=false)))
        #(ψ, envir , delta)   = find_groundstate(ψ, H, VUMPS(maxiter = 500,tol=1e-6))
        ΔEε = 0
        ΔEL1ε = 0
        ΔEσ = 0
        ΔEL1σ = 0
        if alg == "QuasiparticleAnsatz"
            En0, st0 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir; sector=ZNIrrep{5}(0),num=2)
            En1, st1 = excitations(H, QuasiparticleAnsatz(ishermitian=false), ψ, envir;  sector=ZNIrrep{5}(1),num=2)
            En = vcat(En1,En0)
            println(En)
            st = vcat(st1,st0)
            stt = vcat([ψ,],st)
            save_object("5statesQP$lambda.jld2", stt)
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
            println("Doesn't work, only targets sector 0 for some reason")
            println("Doesn't work (bad convergence and |><| requires two states to be calculated for each level)")
             En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=5e-6,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,),init = ψ2,num=5)
             ψ2 = st2[1]
             ΔEσ = En[1]-Ground_energy
             En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=6e-6,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,ψ2),init = ψ3,num=1)
             ψ3 = st2[1]
             ΔEε = En[1]-Ground_energy
             En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=7.5e-6,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,ψ2,ψ3),init = ψ4,num=1)
             ψ4 = st2[1]
             ΔEL1σ = En[1]-Ground_energy
             En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=2.5e-5,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,ψ2,ψ3,ψ4),init = ψ5,num=1)
             ψ5 = st2[1]
            #  ΔEL1ε = En[1]
             En ,st2  = excitations(H,FiniteExcited(gsalg = DMRG(tol=2.5e-5,eigalg =MPSKit.Defaults.alg_eigsolve(maxiter = 500,ishermitian=false))), (ψ,ψ2,ψ3,ψ4,ψ5),init = ψ6,num=1)
             ψ6 = st2[1]
             ΔEL1ε = En[1]-Ground_energy
             save_object("5states$alg $lambda.jld2", [ ψ, ψ2, ψ3, ψ4, ψ5,ψ6])
        else
            pritnln("$alg not supported")
        end

        ## fitting
        
        fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) +abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1σ) -ΔL1σ  -Cε_primeσσ* AL1σε_prime* (x[2]+1im*x[4]))
        res = optimize(fun, [0.0, 0.0,0.0,0.0])
        gε_prime[i] = Optim.minimizer(res)[2]+1im* Optim.minimizer(res)[4]
        println(gε_prime)
        fun_2(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) 
        res_2 = optimize(fun_2, [0.0, 0.0,0.0,0.0])
        gε_prime_wo_C[i] = Optim.minimizer(res_2)[2]+1im* Optim.minimizer(res_2)[4]
        Eε[i] = ΔEε
        Eσ[i] = ΔEσ
        EL1ε[i] = ΔEL1ε
        EL1σ[i] = ΔEL1σ
    end
    save_object("$alg Lambda_est_ge_woc$L.jld2", gε_prime_wo_C)
    save_object("$alg Lambda_est_ge$L.jld2", gε_prime)
    save_object("$alg ΔEε with lambda for $L.jld2",Eε)
    save_object("$alg ΔEσ with lambda for$L.jld2", Eσ)
    save_object("$alg ΔEl1ε with lambda for$L.jld2", EL1ε)
    save_object("$alg ΔEL1σ with lambda for$L.jld2", EL1σ)
end

N = 2
D = 50
test_values = zeros(ComplexF64,(2*N-1)^2)
l = length(test_values)
distx = 0.01## distance from alleged fixed point 0.079+0.060i in real
disty = 0.01# distance from alleged fixed point 0.079+0.060i in imaginary
cent_im = 0.060im
cent_r = 0.079

## snake like structure of test_values will allow for faster convergence when recycling ψ (because you don't jump the entire distx after the loop)
for i in 1:1:(2*N-1)
    if div(i,2) == 1
        for j in 1:1:(2*N-1)
            if i <N+1
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+cent_r+ 1im*LinRange(-disty,0.00,N)[j] .+cent_im
                else 
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+cent_r + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+cent_im
                end
            else
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+cent_r  + 1im*LinRange(-disty,0.00,N)[j]  .+cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N])  .+cent_r+ 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N].+cent_im
                end
            end 
        end
    else
        for j in (2*N-1):-1:1
            if i <N+1
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i]) .+cent_r+ 1im*LinRange(-disty,0.00,N)[j] .+cent_im
                else 
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(-distx,0.00,N)[i])  .+cent_r + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+cent_im
                end
            else
                if j<N+1
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N]) .+cent_r + 1im*LinRange(-disty,0.00,N)[j] .+cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] =  (LinRange(distx/(N-1),distx+distx/(N-1),N)[i-N])   .+cent_r + 1im*LinRange(disty/(N-1),disty+disty/(N-1),N)[j-N] .+cent_im
                end
            end 
        end
    end
end




## run simulation here -> chose length scales
for L in [8,]
    lambda_estimation(L,test_values,"DMRG")
end
