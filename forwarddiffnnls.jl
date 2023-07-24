using Random
using LinearAlgebra
using LBFGSB
using ForwardDiff

"""
lbfgsb that computes the gradient of f using ForwardDiff
"""
function smart_lbfgsb(f; varargs...)
    function g!(z,x)
        z[:] = ForwardDiff.gradient(f,x)
    end
    return lbfgsb(f, g!; varargs...)
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


# Test to solve Y = AX where A is (fixed and known) random, Y given, X unknown
# And X has positive entries
m,r,n=100,2,3
A = randn((m,r))
X_true = abs.(randn((r,n)))
Y = A*X_true

f(X) = 0.5*norm(Y - A*X)^2

function g!(Z, X)
    Z[:,:] = -A'*(Y - A*X)
end

# Initilization
X = abs.(randn((r, n)))

# LBFGSB only accepts vector not matrix inputs, so we define the inverse of vec()
mat(x) = reshape(x, r, n)

function solve_for_X(X)
    f_out, x_out = lbfgsb(x->f(mat(x)),(z,x)->g!(mat(z),mat(x)),vec(X), lb=0,iprint=0)
    return f_out, mat(x_out)
end

@time f_out, X_out = solve_for_X(X)

function smart_solve_for_X(X)
    f_out, x_out = smartest_lbfgsb(x->f(mat(x)),vec(X), lb=0,iprint=0) # LBFGSB only accepts vector not matrix inputs
    return f_out, mat(x_out)
end

@time f_out, X_out = smart_solve_for_X(X)
