using Random
using LinearAlgebra
using LBFGSB

# Test to solve Y = AX where A is (fixed and known) random, Y given, X unknown
# And X has positive entries
m,r,n=4,2,3
A = randn((m,r))
X_true = abs.(randn((r,n)))
Y = A*X_true

f(X) = 0.5*norm(Y - A*X)^2
function g!(Z, X)
    Z[:,:] = -A'*(Y - A*X)
end

# Initilization
X = abs.(randn((r, n)))

mat(x) = reshape(x, r, n)

function solve_for_X(X)
    f_out, x_out = lbfgsb(x->f(mat(x)),(z,x)->g!(mat(z),mat(x)),vec(X), lb=0,iprint=0) # LBFGSB only accepts vector not matrix inputs
    return f_out, mat(x_out)
end

f_out, X_out = solve_for_X(X)

function smart_solve_for_X(X)
    f_out, x_out = smartest_lbfgsb(x->f(mat(x)),vec(X), lb=0,iprint=0) # LBFGSB only accepts vector not matrix inputs
    return f_out, mat(x_out)
end

f_out, X_out = smart_solve_for_X(X)

# Test to solve Y = AX where A is (fixed and known) random, Y given, X unknown
# And X has positive entries
n=5
a = randn(n)
x = abs.(randn(n))
y = a'x

f(x) = 1e10*0.5*(y - a'x)^2
function g!(z, x)
    z[:] = 1e10*(y - a'x)*a
end

# Initilization
x = abs.(randn(n))

function solve_for_x(x)
    f_out, x_out = lbfgsb(x->f(x),(z,x)->g!(z,x),x, lb=0, ub=Inf, m=5, factr=1e7, pgtol=1e-7, iprint=0, maxfun=15000, maxiter=15000) # LBFGSB only accepts vector not matrix inputs
    return f_out, x_out
end
x=fill(Cdouble(3e0),n)
f_out, x_out = solve_for_x(x)




####

function f(x)
    y = 0.25 * (x[1] - 1)^2
    for i = 2:length(x)
        y += (x[i] - x[i-1]^2)^2
    end
    4y
end

# and its gradient function that maps a vector x to a vector z
function g!(z, x)
    n = length(x)
    t₁ = x[2] - x[1]^2
    z[1] = 2 * (x[1] - 1) - 1.6e1 * x[1] * t₁
    for i = 2:n-1
        t₂ = t₁
        t₁ = x[i+1] - x[i]^2
        z[i] = 8 * t₂ - 1.6e1 * x[i] * t₁
    end
    z[n] = 8 * t₁
end

# the first argument is the dimension of the largest problem to be solved
# the second argument is the maximum number of limited memory corrections

n = 25  # the dimension of the problem
x = fill(Cdouble(3e0), n)  # the initial guess
# set up bounds

lb = [isodd(i) ? 1e0 : -1e2 for i in 1:n]
ub = 1e2

fout, xout = lbfgsb(f, g!, x, lb=lb, ub=ub, m=5, factr=1e7, pgtol=1e-5, iprint=0, maxfun=15000, maxiter=15000)