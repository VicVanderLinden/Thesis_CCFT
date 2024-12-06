
# Orthogonal(!) krylov-schur decomposition based on https://epubs.siam.org/doi/pdf/10.1137/S0895479800371529
using KrylovKit: KrylovIterator, KrylovFactorization, OrthonormalBasis, Orthogonalizer, apply, zerovector, PackedHessenberg
mutable struct KrylovShurFactorization{T,S} <: KrylovFactorization{T,S}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    H::Matrix{S} # stores the rayleighquotient in matrix form (changed -> to allow for non-triangulity while expanding)
    r::T # residual
end

Base.length(F::KrylovShurFactorization) = F.k
Base.sizehint!(F::KrylovShurFactorization, n) = begin
    sizehint!(F.V, n)
    sizehint!(F.H, (n * n + 3 * n) >> 1)
    return F
end
Base.eltype(F::KrylovShurFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:KrylovShurFactorization{<:Any,S}}) where {S} = S

basis(F::KrylovShurFactorization) = F.V
rayleighquotient(F::KrylovShurFactorization) = F.H ### changed
residual(F::KrylovShurFactorization) = F.r
@inbounds normres(F::KrylovShurFactorization) = abs(F.H[end])
rayleighextension(F::KrylovShurFactorization) = SimpleBasisVector(F.k, F.k)
struct KrylovShurIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::T
    orth::O
end
KrylovShurIterator(A, x₀) = KrylovShurIterator(A, x₀, KrylovDefaults.orth)

Base.IteratorSize(::Type{<:KrylovShurIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:KrylovShurIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::KrylovShurIterator)
    state = initialize(iter)
    return state, state
end



function Base.iterate(iter::KrylovShurIterator, state)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end


function initialize(iter::KrylovShurIterator; verbosity::Int=0)
    # initialize without using eltype
    x₀ = iter.x₀
    β₀ = norm(x₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    Ax₀ = apply(iter.operator, x₀)
    α = inner(x₀, Ax₀) / (β₀ * β₀) ### this makes sense, its the first index in the shur component
    T = typeof(α) # scalar type of the Rayleigh quotient
    # this line determines the vector type that we will henceforth use
    # vector scalar type can be different from `T`, e.g. for real inner products
   
    v = add!!(scale(Ax₀, zero(α)), x₀, 1 / β₀)
    if typeof(Ax₀) != typeof(v)
        r = add!!(zerovector(v), Ax₀, 1 / β₀)
    else
        r = scale!!(Ax₀, 1 / β₀)
    end
    βold = norm(r)
    r = add!!(r, v, -α)
    β = norm(r)
    # possibly reorthogonalize
    if iter.orth isa Union{ClassicalGramSchmidt2,ModifiedGramSchmidt2}
        dα = inner(v, r)
        α += dα
        r = add!!(r, v, -dα)
        β = norm(r)
    elseif iter.orth isa Union{ClassicalGramSchmidtIR,ModifiedGramSchmidtIR}
        while eps(one(β)) < β < iter.orth.η * βold
            βold = β
            dα = inner(v, r)
            α += dα
            r = add!!(r, v, -dα)
            β = norm(r)
        end
    end
    V = OrthonormalBasis([v])
    H = Matrix{ComplexF64}(undef, (2,1)) ## maybe have to change this
    H[1,1] = α
    H[2,1] = β
    if verbosity > 0
        @info "Krylov Shur iteration step 1: normres = $β"
    end
    return state = KrylovShurFactorization(1, V, H, r)
end
# function initialize!(iter::KrylovShurIterator, state::KrylovShurFactorization; verbosity::Int=0)
#     x₀ = iter.x₀
#     V = state.V
#     while length(V) > 1
#         pop!(V)
#     end
#     H = empty!(state.H)

#     V[1] = scale!!(V[1], x₀, 1 / norm(x₀))
#     w = apply(iter.operator, V[1])
#     r, α = orthogonalize!!(w, V[1], iter.orth)
#     β = norm(r)
#     state.k = 1
#     push!(H, α, β) 
#     state.r = r
#     if verbosity > 0
#         @info "Krylov iteration step 1: normres = $β"
#     end
#     return state
# end   
# KrylovShurr recurrence: use provided orthonormalization routines, but keeps new vector?
function KrylovShurrecurrence!!(operator,
    V::OrthonormalBasis,
    h::AbstractVector,
    orth::Orthogonalizer)
w = apply(operator, last(V))
r, h = orthogonalize!!(w, V, h, orth)
return r, norm(r)
end

### not changed much -> the expansion is the same (we only need to do in the end some householder to bring it in triangular form) 
### only we allow to have matrices and thus changed the H indices
function expand!(iter::KrylovShurIterator, state::KrylovShurFactorization; verbosity::Int=0)
    state.k += 1
    k = state.k
    V = state.V
    H = state.H
    r = state.r
    push!(V, scale(r, 1 / norm(r))) 
    m = size(H)[1] ### this allows for any matrix to exist (usually it will be l+1 x l matrix)
    n = size(H)[2]
    ### resizing the matrix explcitely
    H_temp = zeros(ComplexF64,(m+1,n+1))
    for i in 1:m
        for j in 1:n
            H_temp[i,j] = H[i,j]
        end
    end
    H_temp[m+1,1:n] .=0 ## the way it works, expansions are still with almost zero row at the bottom. I could probably not drag this allong but in the shrinking it does require a whole hamiltonian
    r, β = KrylovShurrecurrence!!(iter.operator, V, view(H_temp, 1:m,n+1), iter.orth) 
    H_temp[m + 1,n+1] = β
    state.H = H_temp
    state.r = r/β
    if verbosity > 0
        @info "Krylov Shur iteration step $k: normres = $β"
    end
    return state
end
# function shrink!(state::ArnoldiFactorization, k)
#     length(state) <= k && return state #### Put the entire two sided shrink in here
#     V = state.V
#     H = state.H
#     while length(V) > k + 1
#         pop!(V)
#     end
#     r = pop!(V)
#     resize!(H, (k * k + 3 * k) >> 1)
#     state.k = k
#     state.r = scale!!(r, normres(state))
#     return state
# end
