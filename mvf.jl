"""
Matrix-Vector Factorization Algorithms
"""

using Random
using LinearAlgebra

using Einsum
using LBFGSB
using ForwardDiff

"""Contract the 3rd index of a tensor with a vector"""
function ×₃(T::AbstractArray{U, 3}, v::AbstractVector{U}) where U
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

function H(A, λ, γ; δ=1e-10)
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
    M = Diagonal(1:p)
    return ν*M
end

rel_error(x, xhat) = abs(x - xhat) / x

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
    #b1 = 1 ./(1:p÷2).^0.5 + 0.1*abs.(randn((p÷2,)))
    #b2 = 1 ./(1:p÷2) + 0.1*abs.(randn((p÷2,)))
    #b1 ./= b1[1] #ensure first entry of b is 1 and rescale appropriately
    #b2 ./= b2[1] #ensure first entry of b is 1 and rescale appropriately
    #b2 = zero(b1)
    #b2[1] = 1
    q = p÷2
    b1 = abs.(randn((p÷2,)))
    b2 = abs.(randn((p÷2,)))
    b = [b1;b2]
    #println(b)

    #=
    n = 1:q
    spectrum1 = @. 1/n
    spectrum2 = @. 0*1/n^4
    spectrum2[1] = 1
    spectrum2[3] = 1/2
    spectrum2[5] = 1/3
    b = [spectrum1;spectrum2]
    =#

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

    # Updates, i==1 to ensure one step is performed, run until the error doesn't improve by tol
    while (i == 1) || ((rel_error(error[i], error[i-1]) > tol) && (i < maxiter))
        # Update A
        A = ReLU.((X*B' .- ϵA) / (B*B' + H(A, λA, γA)))

        # Precompute
        @einsum c[q] = A[i,j]*XT[i,j,q] # note "=" is used rather than := becuase the memory is already allocated
        @einsum D[k,q] = A[l,i]*A[l,j]*TT[i,j,k,q]

        # Update b
        #v = 0#[n_norm_reg(b[1:p÷2], μb); n_norm_reg(b[p÷2+1:end], μb)]
        #M = double_M_norm_reg(b, μb)
        #b = ReLU.((D + λb*I + M) \ (c .- ϵb .- v))

        # Update b fixing b[1] = b[q+1] = 1
        #M = @view M_norm_reg(q, μb)[2:end,2:end]
        M = M_norm_reg(q, μb)
        D += λb*I + [M zero(M);zero(M) M]
        D1 = @view D[1:q, 2:q]
        d1 = @view D[1:q, 1]
        D2 = @view D[q+1:end, q+2:end]
        d2 = @view D[q+1:end, q+1]
        c1 = @view c[1:q]
        c2 = @view c[q+1:end]
        #b1 = ReLU.((D1 + λb*I + M) \ (c1 .- d1 .- ϵb))
        #b2 = ReLU.((D2 + λb*I + M) \ (c2 .- d2 .- ϵb))
        b1 = ReLU.(D1 \ (c1 .- d1 .- ϵb))
        b2 = ReLU.(D2 \ (c2 .- d2 .- ϵb))
        b = [1;b1;1;b2]
        #b = [spectrum1;spectrum2]
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

norm_21(A) = sum(norm.(eachcol(A)))
make_indexes(p) = repeat(1:(p÷2),2) # for example, p=6 -> [1,2,3,1,2,3]
function norm_n(b)
    p = length(b)
    sum(make_indexes(p) .* (b .^ 2))
end

#=function b_sep(b)
    q = length(b)÷2
    b1 = @view b[1:q]
    b2 = @view b[q+1:end]
    return b1'b2 #inner product
end=#
#=
function b_sep_grad(b,j)
    q = length(b)÷2
    return j ≤ q ? b[j+q] : b[j-q]
end=#

function b_sep(b)
    q = length(b)÷2
    b1 = @view b[1:q]
    b2 = @view b[q+1:end]
    return abs(b1'b2 / (norm(b1)*norm(b2))) #normalized inner product
end

function b_sep_grad(b,j)
    q = length(b)÷2
    b1 = @view b[1:q]
    b2 = @view b[q+1:end]
    norm_b1 = norm(b1)
    norm_b2 = norm(b2)
    inner_prod = b1'b2
    if j ≤ q
        g = (b[j+q] - b[j]*inner_prod/norm_b1^2) / (norm_b1 * norm_b2)
    else
        g = (b[j-q] - b[j]*inner_prod/norm_b2^2) / (norm_b1 * norm_b2)
    end
    return sign(b_sep(b))*g # TODO optimize computation, only should evaluate the sign once
end

f(A, b, X, T, ϵA,γA,λA,μb,δb) = 0.5*norm(X-A*(T×₃b))^2 + ϵA*norm(vec(A),1) + γA*norm_21(A) +
    0.5*λA*norm(A)^2 + 0.5*μb*norm_n(b) + δb*b_sep(b)

function grad_A!(Z, A, b, X, T, ϵA,γA,λA) # Gradient w.r.t. A
    B = T×₃b
    norm_21_matrix = A .* repeat((norm.(eachcol(A)))',size(A)[1],1) .^ (-1)
    Z[:,:] = (A*B .- X)*B' .+ ϵA .+ γA .* norm_21_matrix .+ λA .* A
    # (A*B .- X)*B' .+ ϵA .+ γA .* norm_21_matrix .+ λA .* A #not sure which one is faster
end

function grad_b!(v, A, b, X, T, μb,δb) # Gradient w.r.t. b
    #AX = A'X # Precompute Matricies
    #AATb = A'A*(T×₃b)
    AATbX = A'*(A*(T×₃b)-X)
    indexes = make_indexes(length(b))
    for (q, idx) ∈ zip(eachindex(b), indexes)
        Tq = @view T[:,:,q]
        #v[q] = sum(Tq .* AATb) - sum(Tq .* AX) + μb*idx*b[q]
        v[q] = sum(Tq .* AATbX) + μb*idx*b[q] + δb*b_sep_grad(b,idx)
    end
    v[1] = 0 #ensure b[1],b[p÷2] are not updated from 1
    v[length(b)÷2+1] = 0
end

"""
    find_best_scale(x, y)

finds the best constant c s.t. c * x = y.
More exactly, c where norm(c .* x - y) is minimized
"""
function find_best_scale(x, y)
    xx = sum(x .* x)
    xy = sum(x .* y)
    return xy / xx
end

function nnls_seperate(X, T; maxiter=25, tol=1e-3, λA=0, ϵA=0, γA=0, μb=0, δb=0)
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    @assert iseven(r) "size(T)[1] = $r is not even"
    @assert iseven(p) "size(T)[3] = $p is not even"

    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
    b[1] = 1 # fix the first entry of the spectrums to 1
    b[p÷2+1] = 1
    i = 1
    error = zeros((maxiter,))
    norm_grad_A = zeros((maxiter,))
    norm_grad_b = zeros((maxiter,))
    error[i] = norm(X - A*(T×₃b))
    G = copy(A); grad_A!(G, A, b, X, T, ϵA,γA,λA)
    g = copy(b); grad_b!(g, A, b, X, T, μb,δb)
    norm_grad_A[i] = norm(G)
    norm_grad_b[i] = norm(g)

    # Forward maps and gradients
    #A -> f(A, b, X, T, ϵA,γA,λA,μb)
    #b -> f(A, b, X, T, ϵA,γA,λA,μb)
    #A -> grad_A(A, b, X, T, ϵA,γA,λA)
    #b -> grad_b(A, b, X, T, μb)

    mat(a) = reshape(a, m, r)

    function update_A(A, b) #TODO add smart lbfgsb
        a_guess = vec(A)#abs.(randn(length(vec(A))))
         _, a = lbfgsb(a -> f(mat(a), b, X, T, ϵA,γA,λA,μb,δb),
                       (z, a) -> grad_A!(mat(z),mat(a), b, X, T, ϵA,γA,λA),
                       a_guess, lb=0,iprint=0) # LBFGSB only accepts vector not matrix inputs
        #_, a = smartest_lbfgsb(a -> f(mat(a), b, X, T, ϵA,γA,λA,μb),
        #                vec(A), lb=0,iprint=0) # LBFGSB only accepts vector not matrix inputs
        return mat(a)
    end

    function update_b(A, b)#TODO add smart lbfgsb
        b_guess = b#abs.(randn(length(b)))
        #b[1] = 1 # fix the first entry of the spectrums to 1
        #b[p÷2+1] = 1
         _, b = lbfgsb(b -> f(A, b, X, T, ϵA,γA,λA,μb,δb),
                       (v, b) -> grad_b!(v, A, b, X, T, μb,δb),
                       b_guess, lb=0,iprint=0)
        # _, b = smartest_lbfgsb(b -> f(A, b, X, T, ϵA,γA,λA,μb),
        #               b, lb=0,iprint=0)
        return b
    end

    #not_converged(error, i) = (rel_error(error[i], error[i-1]) > tol)
    function not_converged(norm_grad_A, norm_grad_b, i)
        norm_grad = sqrt(norm_grad_A[i]^2 + norm_grad_b[i]^2)
        init_grad = sqrt(norm_grad_A[1]^2 + norm_grad_b[1]^2)
        return (norm_grad/init_grad) > tol
    end

    # Updates
    # i==1 to ensure the first step is performed
    # run while the error improves by tol and too many iterations haven't past
    while (i == 1) || (not_converged(norm_grad_A, norm_grad_b, i) && (i < maxiter))
        # Updates
        A = update_A(A, b)
        b = update_b(A, b)

        # Find error
        i += 1
        error[i] = norm(X - A*(T×₃b))

        # Save gradients
        G = copy(A); grad_A!(G, A, b, X, T, ϵA,γA,λA)
        g = copy(b); grad_b!(g, A, b, X, T, μb,δb)
        norm_grad_A[i] = norm(G)
        norm_grad_b[i] = norm(g)
    end

    # Rescale the learned factorization to best fit Y = c .* Yhat
    # This is needed since regularizes like L1 reduce the overall
    # amplitude of the learned result
    Xhat = A*(T×₃b)
    c = find_best_scale(Xhat, X)
    A .*= c
    @show c

    error = error[1:i] ./ norm(X) # Chop off excess and standardize error
    norm_grad_A = norm_grad_A[1:i] #./ norm_grad_A[1] # Chop off and standardize by initial gradient
    norm_grad_b = norm_grad_b[1:i] #./ norm_grad_b[1] # Chop off and standardize by initial gradient

    b1, b2 = b[1:p÷2], b[p÷2+1:end]
    A1, A2 = A[:, 1:r÷2], A[:, r÷2+1:end];

    return (A1, A2, b1, b2, error, norm_grad_A, norm_grad_b)
end

function nnls_vec(X, T; maxiter=25, tol=1e-3, λA=0, ϵA=0, γA=0, μb=0, δb=0)
    # Extract Sizes
    m, n = size(X)
    r, N, p = size(T)
    @assert n==N "Missmatch between the second dimention of X and T"
    @assert iseven(r) "size(T)[1] = $r is not even"
    @assert iseven(p) "size(T)[3] = $p is not even"

    mat(a) = reshape(a, m, r)

    # Make a single vector v representing all unknowns
    vectorize(A,b) = [vec(A) ; b]
    matricize(v) = (mat(@view v[1:m*r]), @view v[m*r+1:end])
    # v = vectorize(A,b)
    # A, b = matricize(v)

    # Initilization
    A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
    b[1] = 1 # fix the first entry of the spectrums to 1
    b[p÷2+1] = 1
    v_init = vectorize(A,b)

    function f_vec(v)
        A, b = matricize(v)
        return f(A, b, X, T, ϵA,γA,λA,μb,δb)
    end

    function grad_vec!(u, v)
        Z, z = matricize(u)
        A, b = matricize(v)
        grad_A!(Z, A, b, X, T, ϵA,γA,λA)
        grad_b!(z, A, b, X, T, μb,δb)
    end

    # All the work is done by lbfgsb
    _, v_out = lbfgsb(f_vec, grad_vec!, v_init, lb=0, iprint=0, factr=1e10)
    A, b = matricize(v_out)

    function update_A(A, b)
         _, a = lbfgsb(a -> f(mat(a), b, X, T, ϵA,γA,λA,μb,δb),
                       (z, a) -> grad_A!(mat(z),mat(a), b, X, T, ϵA,γA,λA),
                        vec(A), lb=0,iprint=0, factr=1e8)
        return mat(a)
    end

    function update_b(A, b)
         _, b = lbfgsb(b -> f(A, b, X, T, ϵA,γA,λA,μb,δb),
                       (v, b) -> grad_b!(v, A, b, X, T, μb,δb),
                       b, lb=0,iprint=0, factr=1e8)
        return b
    end
    # Single stage refinement
    b = update_b(A, b)
    A = update_A(A, b)

    b1, b2 = b[1:p÷2], b[p÷2+1:end]
    A1, A2 = A[:, 1:r÷2], A[:, r÷2+1:end]
    return (A1, A2, b1, b2)
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

"""
lbfgsb that computes the gradient of f using ForwardDiff
"""
function smart_lbfgsb(f, varargs...)
    function g!(z,x)
        z[:] = ForwardDiff.gradient(f,x)
    end
    return lbfgsb(f, g!, varargs...)
end

"""
lbfgsb that computes the gradient of f using ForwardDiff
"""
function smartest_lbfgsb(f, x0; m=10, lb=[-Inf for i in x0], ub=[Inf for i in x0], kwargs...)
    function g!(z,x)
        z[:] = ForwardDiff.gradient(f,x)
    end
    return lbfgsb(f, g!, x0; m=m, lb=lb, ub=ub, kwargs...)
end
