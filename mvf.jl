"""
Matrix-Vector Factorization Algorithms
"""

using Random
using LinearAlgebra
using Einsum

"""Contract the 3rd index of a tensor with a vector"""
function ×₃(T::Array{Float64, 3}, v::Vector{Float64})
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

function als_seperate(X, T; maxiter=800, tol=1e-3, λA=0, λb=0, ϵA=0, ϵb=0)
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    
    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))

    # Rescaling step
    b1 = 1 ./(1:p÷2) + 0.5*abs.(randn((p÷2,)))
    b2 = 1 ./(1:(p-p÷2)) + 0.5*abs.(randn((p-p÷2,)))
    b1 ./= b1[1] #ensure first entry of b is 1 and rescale appropriately
    b2 ./= b2[1] #ensure first entry of b is 1 and rescale appropriately
    b = [b1;b2]

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

        # Ensure first entry of b is 1 and rescale appropriately
        b[1:p÷2] ./= b[1]
        b[p÷2+1:end] ./= b[p÷2+1]

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