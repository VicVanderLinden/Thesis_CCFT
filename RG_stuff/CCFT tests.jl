using SpecialFunctions

function m(Q)
    return 2*pi / (acos(Q/2 - 1 - 0im)) # complex input for complex output, minus sign for correct branch
end

function g(Q) # plus sign for other CCFT
    return 4 + im*(2/pi)*log((Q-2+sqrt(Q*(Q-4)))/2)
end

@assert g(5) ≈ 4 + 4/m(5)

function cpotts(Q)
    return 13 - 6*(g(Q)/4 + 4/g(Q))
end

function Δϵprime(Q) # (1,3) operator in tricritical case, (3,1) in critical case
    return 16/g(Q) - 2
end

# https://arxiv.org/pdf/1303.3015 eq A.5
function C_DF(n1,m1,n2,m2,n3=1,m3=3)
    ρ = m(5)/(m(5)+1)
    s = 1/2 * (n1 + n2 - n3 - 1)
    t = 1/2 * (m1 + m2 - m3 - 1)
    γ(val) = gamma(val)/gamma(one(ComplexF64)-val)
    fancyprod(x) = foldl(*, x, init = one(ComplexF64))
    line1 = sqrt((γ(ρ-1) * γ(m1-n1/ρ) * γ(m2-n2/ρ) * γ(-m3+n3/ρ)) / (γ(1-1/ρ) * γ(-n1+m1*ρ) * γ(-n2+m2*ρ) * γ(n3-m3*ρ)))
    line2 = fancyprod(((i-j*ρ) * (i+n3-(j+m3)*ρ) * (i-n1-(j-m1)*ρ) * (i-n2-(j-m2)*ρ))^(-2) for i=1:s, j=1:t)
    line3 = fancyprod(γ(i/ρ) * γ(-m3+(i+n3)/ρ) * γ(m1+(i-n1)/ρ) * γ(m2+(i-n2)/ρ) for i=1:s)
    line4 = fancyprod(γ(j*ρ) * γ(-n3+(j+m3)*ρ) * γ(n1+(j-m1)*ρ) * γ(n2+(j-m2)*ρ) for j=1:t)
    return line1 * line2 * line3 * line4 * ρ^(4*s*t + 2*t -2*s - 1)
end

# ϵ = (1,2) tricritical
# ϵ' = (1,3) tricritical
# ϵ'' = (1,4) tricritical

#Im flip
imag(Δϵprime(5)) /(C_DF(1,2,1,2,1,3))
-sqrt(3)/(2pi)

