
using Optim
using JLD2
using Plots
Δε = 0.4656 − 0.2245im
ΔL1ε = 1.4656 − 0.2245im
Δσ = 0.1336 − 0.0205im
ΔL1σ =1.1336 − 0.0205im
Cε_primeσσ = 0.0658 + 0.0513im 
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.9083 − 0.5987im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)
errors = []

for L in [6,8,10,12,14,16,20,24]
    g = []
    
        results = load("lambda_est_inital/Ground_state_andEsigma_L6-24/Energy_$L.jld2")
        println(results)
        # H = Potts_Hamiltonian(L,lambda=lambda)
        E0 =  results["E0"][1]
        ΔEε = results["E0"][2]-E0
        ΔEL1ε = results["E0"][3]-E0
        ΔEσ =   results["Eσ"][1]-E0
        ΔEL1σ = results["Eσ"][2]-E0
        
        fun(x) = abs((x[1]+1im*x[3])*(ΔEε) - Δε -Cε_primeεε* (x[2]+1im*x[4])) + abs((x[1]+1im*x[3])*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[2]+1im*x[4])) +abs((x[1]+1im*x[3])*(ΔEσ) - Δσ -Cε_primeσσ*  (x[2]+1im*x[4])) 
        #fun(x) = abs(alpha_fix*(ΔEε) - Δε -Cε_primeεε* (x[1]+1im*x[2])) + abs(alpha_fix*(ΔEL1ε) -ΔL1ε  -Cε_primeεε* AL1εε_prime* (x[1]+1im*x[2])) +abs(alpha_fix*(ΔEσ) - Δσ -Cε_primeσσ*  (x[1]+1im*x[2])) + abs(alpha_fix*(ΔEL1σ) -ΔL1σ  -Cε_primeσσ* AL1σε_prime* (x[1]+1im*x[2]))
        
        res = optimize(fun, [0.0, 0.0,0.0,0.0])
        gε_prime = Optim.minimizer(res)[2]+1im* Optim.minimizer(res)[4]
        push!(g,gε_prime )
 
end

p = plot(; xlabel="Re(g_e)", ylabel="Im(g_e)",title = " error evolution higer D for L = 18,19,20",legend=false,xguidefontsize=15,yguidefontsize=15,titlefontsize=15)
for (i,lambda) in enumerate(test_values[20:150])
    y = []
    for l in 1:3
        push!(y,errors[l][i])
    end
    if abs(y[end-1]-y[end]) < 0.001 && abs(y[2]-y[1]) < 0.001 
        
    plot!(p,[real(y ),],[real(-1im.*y),],color="blue",arrow=true,label=" ")
    end
    #quiver!(p,[real(errors[2,i] ),],[real(-1im.*errors[2,i])], quiver = [0.1.*(real(errors[3,i]-errors[2,i]),real(-im.*(errors[3,i]-errors[2,i]))),],color="red")
end
display(p)
plot!(p,[0.0],[0.0,], seriestype=:scatter)
savefig(p,"ge_evolution.png")

println("here")