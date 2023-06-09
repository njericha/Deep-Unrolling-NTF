"""
Matrix-Vector Factorization Algorithms
"""

using Random
using LinearAlgebra
using Einsum

"""Contract the 3rd index of a tensor with a vector"""
function ×₃(T::AbstractArray{U, 3}, v::Vector{U}) where U
    @einsum M[i,j] := T[i,j,k]*v[k]
    return M
end

exp_update(x) = (exp(x)-1) / (exp(1)-1)
smooth_power(i) = 2-1/i#tanh(i-1)+1#3 - 2000/(i+999)
ReLU(x) = max(x,0)

"""Matrix-Vector Factor X=ATb using multipicative updates"""
function mvf(X, T; power=1, maxiter=800, tol=1e-3, λA=0, λb=0, ϵA=1e-8, ϵb=1e-8)
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    
    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
    i = 1
    error = zeros((maxiter,))
    normX = norm(X)
    error[i] = norm(X - A*(T×₃b))/normX
    B = (T×₃b)

    # Updates
    while (error[i] > tol) && (i < maxiter)
        power = smooth_power(i)
        
        # Precompute Matricies
        AX = A'X
        AATb = A'A*B
        
        # Update b
        for q ∈ eachindex(b)
            Tq = @view T[:,:,q]
            b[q] *= (sum(Tq .* AX) / (sum(Tq .* AATb) + ϵb + λb*b[q]))^power
        end

        #normalize!(b)

        # Precompute Matrix
        B = T×₃b

        # Update A
        A .*= ((X*B') ./ (A*B*B' .+ ϵA .+ λA.*A)).^power

        # Find relative error
        i += 1
        error[i] = norm(X - A*B)/normX
    end

    error = error[1:i] # Chop off excess
    # Normalize b
    bnorm = norm(b)
    A .*= bnorm
    b ./= bnorm
    return (A, b, error)
end

"""Alternating least squares to solve X=ATb"""
function als(X, T; maxiter=800, tol=1e-3)
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    
    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
    i = 1
    error = zeros((maxiter,))
    normX = norm(X)
    error[i] = norm(X - A*(T×₃b))/normX

    # Precompute
    @einsum XT[i,j,q] := X[i,l]*T[j,l,q] #access XTq = @view XT[:,:,q]
    @einsum TT[i,j,k,q] := T[i,l,k]*T[j,l,q] #access TTkq = @view TT[:,:,k,q]
    @einsum c[q] := A[i,j]*XT[i,j,q]
    @einsum D[k,q] := A[l,i]*A[l,j]*TT[i,j,k,q]

    # Updates
    while (error[i] > tol) && (i < maxiter)
        # Update b
        b = ReLU.(D \ c)

        # Precompute Matrix
        B = T×₃b

        # Update A
        A = ReLU.(X / B)

        # Find relative error
        i += 1
        error[i] = norm(X - A*B)/normX

        # Precompute
        @einsum c[q] = A[i,j]*XT[i,j,q] # note = rather than := becuase the memory is already allocated
        @einsum D[k,q] = A[l,i]*A[l,j]*TT[i,j,k,q]
    end

    error = error[1:i] # Chop off excess
    # Normalize b
    bnorm = norm(b)
    A .*= bnorm
    b ./= bnorm
    return (A, b, error)
end

function als(X, T; maxiter=800, tol=1e-3, λA=0, λb=0, ϵA=0, ϵb=0) #TODO Combine into one function with the previous version
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    
    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
    i = 1
    error = zeros((maxiter,))
    normX = norm(X)
    error[i] = norm(X - A*(T×₃b))/normX

    # Precompute
    @einsum XT[i,j,q] := X[i,l]*T[j,l,q] #access XTq = @view XT[:,:,q]
    @einsum TT[i,j,k,q] := T[i,l,k]*T[j,l,q] #access TTkq = @view TT[:,:,k,q]
    @einsum c[q] := A[i,j]*XT[i,j,q]
    @einsum D[k,q] := A[l,i]*A[l,j]*TT[i,j,k,q]

    # Updates
    while (error[i] > tol) && (i < maxiter)
        # Update b
        b = ReLU.((D + λb*I) \ (c .- ϵb))

        # Precompute Matrix
        B = T×₃b

        # Update A
        A = ReLU.((X*B' .- ϵA) / (B*B' + λA*I))

        # Find relative error
        i += 1
        error[i] = norm(X - A*B)/normX

        # Precompute
        @einsum c[q] = A[i,j]*XT[i,j,q] # note = rather than := becuase the memory is already allocated
        @einsum D[k,q] = A[l,i]*A[l,j]*TT[i,j,k,q]
    end

    error = error[1:i] # Chop off excess
    # Normalize b
    bnorm = norm(b)
    A .*= bnorm
    b ./= bnorm
    return (A, b, error)
end

iseven(n::Integer) = (n % 2 == 0)

function H(A, λ, γ; δ=1e-8)
    if γ==0 #quick exit
        return λ*I
    end
    colnorms = norm.(eachcol(A)) .+ δ
    G = Diagonal(colnorms)^(-1)
    return λ*I + γ*G
end

function max_norm_reg(b, μ)
    j = argmax(b)
    v = zero(b)
    v[j] = μ
    return v
end

function n_norm_reg(b, ν)
    v = 1:length(b)
    return ν*v
end

function double_M_norm_reg(b, ν)
    p = length(b)÷2
    M = Diagonal([1:p; 1:p])^2
    return ν*M
end

function M_norm_reg(p, ν)
    M = Diagonal(1:p)^2
    return ν*M
end

function als_seperate(X, T; maxiter=800, tol=1e-3, λA=0, λb=0, ϵA=0, ϵb=0, γA=0, μb=0)
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    @assert iseven(r) "size(T)[1] = $r is not even"
    @assert iseven(p) "size(T)[3] = $p is not even"

    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
    v = zero(b)

    # Rescaling step
    b1 = 1 ./(1:p÷2) + 0.1*abs.(randn((p÷2,)))
    b2 = 1 ./(1:p÷2) + 0.1*abs.(randn((p÷2,)))
    b1 ./= b1[1] #ensure first entry of b is 1 and rescale appropriately
    b2 ./= b2[1] #ensure first entry of b is 1 and rescale appropriately
    b = [b1;b2]
    #println(b)

    i = 1
    error = zeros((maxiter,))
    normX = norm(X)
    error[i] = norm(X - A*(T×₃b))/normX

    # Precompute
    @einsum XT[i,j,q] := X[i,l]*T[j,l,q] #access XTq = @view XT[:,:,q]
    @einsum TT[i,j,k,q] := T[i,l,k]*T[j,l,q] #access TTkq = @view TT[:,:,k,q]
    @einsum c[q] := A[i,j]*XT[i,j,q]
    @einsum D[k,q] := A[l,i]*A[l,j]*TT[i,j,k,q]

    # Precompute Matrix
    B = T×₃b

    # Updates
    while (error[i] > tol) && (i < maxiter)
        # Update A
        A = ReLU.((X*B' .- ϵA) / (B*B' + H(A, λA, γA)))

        # Precompute
        @einsum c[q] = A[i,j]*XT[i,j,q] # note "=" is used rather than := becuase the memory is already allocated
        @einsum D[k,q] = A[l,i]*A[l,j]*TT[i,j,k,q]

        # Update b
        v = 0#[n_norm_reg(b[1:p÷2], μb); n_norm_reg(b[p÷2+1:end], μb)]
        #M = double_M_norm_reg(b, μb)
        #b = ReLU.((D + λb*I + M) \ (c .- ϵb .- v))

        # Update b fixing b[1] = b[q+1] = 1
        q = p÷2
        M = @view M_norm_reg(q, μb)[2:end,2:end]
        D1 = @view D[2:q, 2:q]
        d1 = @view D[2:q, 1]
        D2 = @view D[q+2:end, q+2:end]
        d2 = @view D[q+2:end, q+1]
        c1 = @view c[2:q]
        c2 = @view c[q+2:end]
        b1 = ReLU.((D1 + λb*I + M) \ (c1 .- d1 .- ϵb))
        b2 = ReLU.((D2 + λb*I + M) \ (c2 .- d2 .- ϵb))
        b = [1;b1;1;b2]
        #println(length(b1),length(b2),length(b))

        # Ensure first entry of b is 1 and rescale appropriately
        #println(b);normalize!(@view b[1:p÷2]);normalize!(@view b[p÷2+1:end])
        #b[1:p÷2] ./= (b[1] + 0.1)
        #b[p÷2+1:end] ./= (b[p÷2+1] + 0.1)

        # Precompute Matrix
        B = T×₃b

        # Find relative error
        i += 1
        error[i] = norm(X - A*B)/normX
    end

    error = error[1:i] # Chop off excess
    # Normalize b
    #bnorm = norm(b)
    #A .*= bnorm
    #b ./= bnorm
   
    b1, b2 = b[1:p÷2], b[p÷2+1:end]
    A1, A2 = A[:, 1:r÷2], A[:, r÷2+1:end];

    return (A1, A2, b1, b2, error)
end

# Generate a data set all at once
function generate_dataset((m,n),(r,N,p),dataset_size=25)
    @assert n==N "Missmatch between the second dimentions n={$n} and N={$N}"
    As = [randn((m,r)) for _ ∈ 1:dataset_size]
    bs = [randn((p,)) for _ ∈ 1:dataset_size]
    Ts = [randn((r,N,p)) for _ ∈ 1:dataset_size]
    Xs = As .* (Ts .×₃ bs)
    return zip(Xs, As, Ts, bs)
end


# Define an iterator to generate a data set one element at a time
struct MyDataSet
    Xsize::Tuple{Int,Int}
    Asize::Tuple{Int,Int}
    Tsize::Tuple{Int,Int,Int}
    bsize::Tuple{Int}
    dataset_length::Int
end

function MyDataSet((m,n,r,p),dataset_length=1)
    Xsize=(m,n)
    Asize=(m,r)
    Tsize=(r,n,p)
    bsize=(p,)
    return MyDataSet(Xsize,Asize,Tsize,bsize,dataset_length)
end

Base.length(D::MyDataSet)=D.dataset_length
Base.eltype(::Type{MyDataSet})=Tuple{Matrix{Float64},Matrix{Float64},Array{Float64},Vector{Float64}}

function Base.iterate(D::MyDataSet, state=1)
    if state > D.dataset_length 
        return nothing
    else
        A = abs.(randn(D.Asize))
        b = normalize(abs.(randn(D.bsize)))
        T = abs.(randn(D.Tsize))
        X = A * (T ×₃ b)
        ((X,A,T,b), state+1)
    end
end