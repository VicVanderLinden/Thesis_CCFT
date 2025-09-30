include("Potts-Operators & Hamiltonian.jl")
Q = 5
Vp = Vect[ZNIrrep{Q}](sector=>1 for sector in 0:Q-1)
L = 11
using JLD2
function lambda_estimation(lambda,L)
    H = Potts_Hamiltonian(L;lambda = lambda)
    E1 = exact_diagonalization(H,sector=ZNIrrep{5}(0),num= 6,alg =MPSKit.Arnoldi())[1]
    E2 = exact_diagonalization(H,sector=ZNIrrep{5}(1),num= 3,alg =MPSKit.Arnoldi())[1]
    save_object("ED 9states,L = $L $lambda.jld2", [E1,E2])
end

lambda = 0.079 + 0.060im 
#### Define L  outside of scope
lambda_estimation(lambda,L)

