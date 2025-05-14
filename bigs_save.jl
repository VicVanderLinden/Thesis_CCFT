### SAVING THE DATA IN ONE BIG FILE ###################"" 
using JLD2
test_values = zeros(ComplexF64,(2*N-1)^2)

l = length(test_values)

distx = 0.0007## distance from centre in real

disty = 0.0007# distance from centre in imaginary

cent_im = 0.0600im

cent_r = 0.0780


include("Potts-Operators & Hamiltonian.jl")
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




# ################## Lambda istelf ###################
using PlotlyJS
using Optim
Δε = 0.465613165838194 - 0.224494536412444im
ΔL1ε = 1.465613165838194 - 0.224494536412444im
Δσ =0.133596708540452 - 0.0204636065293973im
ΔL1σ =1.133596708540452 - 0.0204636065293973im
# Cε_primeσσ = 0.0658 + 0.0513im 
Cε_primeεε = 0.8791 − 0.1404im
Δε_prime = 1.90830177556852 - 0.598652097099851im
AL1εε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δε)
AL1σε_prime = 1+Δε_prime*(Δε_prime -2)/(4*Δσ)
BIG_SAVE = zeros(ComplexF64,(3,length(test_values)))
BIG_SAVE1 = zeros(ComplexF64,(3,length(test_values)))
BIG_SAVE2 = zeros(ComplexF64,(3,length(test_values)))
BIG_SAVE3 = zeros(ComplexF64,(3,length(test_values)))
for (i,L) in enumerate([18,19,20])
    g = []
    for (j,lambda) in enumerate(test_values)
        lambda_txt = round(lambda,digits = 6)
       

       ## D 500
        if L ==20 && (lambda_txt == 0.078233 +0.059883im)
        else
        results0 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
        results1 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC0_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
        results2 = load_object("Lambda_est_precise/PBC/PBC_D_500/L$L/5EXC1_PBC_L=$L"*"_$lambda_txt"*"_D100.jld2")
        BIG_SAVE[i,j] = results1[2][1]
        BIG_SAVE1[i,j] = results1[2][2]
        BIG_SAVE2[i,j]=  results2[2][end-1]
        BIG_SAVE3[i,j] = results2[2][end]
        end
    end
   

end
result = Dict("Δε th"=>Δε,"ΔL1ε th" => ΔL1ε, "Δσ th " =>Δσ ,"ΔL1σ th"=> ΔL1σ,"Cε'εε" => Cε_primeεε,"AL1εε'"=>1+Δε_prime*(Δε_prime -2)/(4*Δε), "AL1σε'"=>AL1σε_prime ,"lambda_values" => test_values,"L_values" => [18,19,20],"Eε"=>BIG_SAVE,"EL1ε"=>BIG_SAVE1,"Eσ"=>BIG_SAVE2,"Eσ'/EL1σ"=>BIG_SAVE3)

save("D500results.jld2",result)