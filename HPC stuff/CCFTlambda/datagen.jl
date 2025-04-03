using CSV, DataFrames, DelimitedFiles

N = 7
test_values = zeros(ComplexF64, (2 * N - 1)^2)
l = length(test_values)
distx = 0.0007## distance from centre in real
disty = 0.0007# distance from centre in imaginary
cent_im = 0.0600im
cent_r = 0.0780

## snake like structure of test_values will allow for faster convergence when recycling ψ (because you don't jump the entire distx after the loop)
for i in 1:1:(2*N-1)
    if div(i, 2) == 1
        for j in 1:1:(2*N-1)
            if i < N + 1
                if j < N + 1
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(-distx, 0.00, N)[i]) .+ cent_r + 1im * LinRange(-disty, 0.00, N)[j] .+ cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(-distx, 0.00, N)[i]) .+ cent_r + 1im * LinRange(disty / (N - 1), disty + disty / (N - 1), N)[j-N] .+ cent_im
                end
            else
                if j < N + 1
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(distx / (N - 1), distx + distx / (N - 1), N)[i-N]) .+ cent_r + 1im * LinRange(-disty, 0.00, N)[j] .+ cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(distx / (N - 1), distx + distx / (N - 1), N)[i-N]) .+ cent_r + 1im * LinRange(disty / (N - 1), disty + disty / (N - 1), N)[j-N] .+ cent_im
                end
            end
        end
    else
        for j in (2*N-1):-1:1
            if i < N + 1
                if j < N + 1
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(-distx, 0.00, N)[i]) .+ cent_r + 1im * LinRange(-disty, 0.00, N)[j] .+ cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(-distx, 0.00, N)[i]) .+ cent_r + 1im * LinRange(disty / (N - 1), disty + disty / (N - 1), N)[j-N] .+ cent_im
                end
            else
                if j < N + 1
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(distx / (N - 1), distx + distx / (N - 1), N)[i-N]) .+ cent_r + 1im * LinRange(-disty, 0.00, N)[j] .+ cent_im
                else
                    test_values[i+(j-1)*(2*N-1)] = (LinRange(distx / (N - 1), distx + distx / (N - 1), N)[i-N]) .+ cent_r + 1im * LinRange(disty / (N - 1), disty + disty / (N - 1), N)[j-N] .+ cent_im
                end
            end
        end
    end
end
Ls = [18, 19, 20]
W = (2 * N - 1)^2
M = Matrix{Any}(undef, length(Ls) * W, 3)
# writedlm("test_values.csv", test_values)
reals = round.(real(test_values); digits=6)
imags = round.(imag(test_values); digits=6)

for (index, L) in enumerate(Ls)
    M[1+(index-1)*W:1:index*W, 1] = reals
    M[1+(index-1)*W:1:index*W, 2] = imags
    M[1+(index-1)*W:1:index*W, 3] = Int.([L for _ in 1:1:W])
end
M
writedlm("CCFTlambda/test_values.csv", M, ',')