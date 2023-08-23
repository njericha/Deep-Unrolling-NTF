using LinearAlgebra
using Plots
using LogExpFunctions #softmax
#using Zygote

a=10000
blend(x,y) = log(a, a^x+a^y)

#f(x, y) = (10-x*y)^2 + x^2 + y^2
f(x, y) = blend(x^2+(y-2)^2, (x-2)^2+y^2) #min at
gx(x, y)= (4* a^(4* x))/(a^(4* x) + a^(4* y)) + 2* x - 4
gy(x, y)= 2* y - (4* a^(4* x))/(a^(4* x) + a^(4* y))
fxx(x, y) = (16* log(a)* a^(4 *(x + y)))/(a^(4* x) + a^(4* y))^2 + 2
fyy(x, y) = (16 *log(a) *a^(4 *(x + y)))/(a^(4* x) + a^(4 *y))^2 + 2
#gx(x, y) = 2 *(x - 10 *y + x *y^2) #gradient(x->f(x,y), x)[1]
#gy(x, y) = 2 *(-10 *x + y + x^2 *y)#gradient(y->f(x,y), y)[1]

g(x, y) = [gx(x, y), gy(x, y)]
h(x, y) = [fxx(x, y), fyy(x, y)]

#x = range(1.5, 4.5, length=100)
#y = range(2.5, 5, length=100)

x = range(0, 2, length=100)
y = range(0, 1.2, length=100)

z = @. f(x', y)
Lx = maximum(abs.(fxx.(x', y)))
Ly = maximum(abs.(fyy.(x', y)))
L = maximum(norm.(h.(x', y)))
contour(x, y, z; title="f(x,y) = smoothmax(x^2+(y-2)^2, (x-2)^2+y^2)\nsmoothmax(a,b,q)=log_q(q^a+q^b)\nBudget of 128 iterations; q=10^4",aspect_ratio=:equal)

plot!([1],[1],label="minimum",shape=:star)
#plot!([3],[3],label="minimum",shape=:star)


gradient_step_x(x, y) = x - gx(x, y)/Lx
gradient_step_y(x, y) = y - gy(x, y)/Ly
gradient_step(x, y) = [x, y] .- g(x, y) ./ L

#solve_for_x(x,y) = 10y/(1+y^2)
#solve_for_y(x,y) = 10x/(1+x^2)

solve_for_x(x,y) = y
solve_for_y(x,y) = x

N = 64 # number of outer iterations
Kx = 1 # number of inner iterations
Ky = 1 # number of inner iterations
total_N = (N * (Kx + Ky))
#x0, y0 = 4,5
x0, y0 = 1.9,.2

#################
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0

@time begin
for i in 1:total_N
    x, y = gradient_step(x, y)
    push!(path, (x, y))
end
end
plot!(collect(zip(path)); color=:green, label="Gradient Descent")

###############
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0
@time begin
for i in 1:N
    for _ in 1:Kx
        x = gradient_step_x(x, y)
        push!(path, (x, y))
    end
    for _ in 1:Ky
        y = gradient_step_y(x, y)
        push!(path, (x, y))
    end
end
end
plot!(collect(zip(path)); color=:blue, label="Coordinate Descent")


###############
N /= 2 # number of outer iterations
Kx *= 2 # number of inner iterations
Ky *= 2 # number of inner iterations
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0
@time begin
for i in 1:N
    for _ in 1:Kx
        x = gradient_step_x(x, y)
        push!(path, (x, y))
    end
    for _ in 1:Ky
        y = gradient_step_y(x, y)
        push!(path, (x, y))
    end
end
end
plot!(collect(zip(path)); color=:red, label="2-step CD")

###############
N /= 2 # number of outer iterations
Kx *= 2 # number of inner iterations
Ky *= 2 # number of inner iterations
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0
@time begin
for i in 1:N
    for _ in 1:Kx
        x = gradient_step_x(x, y)
        push!(path, (x, y))
    end
    for _ in 1:Ky
        y = gradient_step_y(x, y)
        push!(path, (x, y))
    end
end
end
plot!(collect(zip(path)); color=:orange, label="4-step CD")

###############
N /= 2 # number of outer iterations
Kx *= 2 # number of inner iterations
Ky *= 2 # number of inner iterations
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0
@time begin
for i in 1:N
    for _ in 1:Kx
        x = gradient_step_x(x, y)
        push!(path, (x, y))
    end
    for _ in 1:Ky
        y = gradient_step_y(x, y)
        push!(path, (x, y))
    end
end
end
plot!(collect(zip(path)); color=:purple, label="8-step CD")

###############
N /= 2 # number of outer iterations
Kx *= 2 # number of inner iterations
Ky *= 2 # number of inner iterations
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0
@time begin
for i in 1:N
    for _ in 1:Kx
        x = gradient_step_x(x, y)
        push!(path, (x, y))
    end
    for _ in 1:Ky
        y = gradient_step_y(x, y)
        push!(path, (x, y))
    end
end
end
plot!(collect(zip(path)); color=:brown, label="16-step CD")

###############
N /= 2 # number of outer iterations
Kx *= 2 # number of inner iterations
Ky *= 2 # number of inner iterations
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0
@time begin
for i in 1:N
    for _ in 1:Kx
        x = gradient_step_x(x, y)
        push!(path, (x, y))
    end
    for _ in 1:Ky
        y = gradient_step_y(x, y)
        push!(path, (x, y))
    end
end
end
plot!(collect(zip(path)); color=:black, label="32-step CD")
#=
###############
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0

@time begin
for i in 1:(total_N÷2)
    x = solve_for_x(x,y)
    push!(path, (x, y))
    y = solve_for_y(x,y)
    push!(path, (x, y))
end
end
#plot!(collect(zip(path)); color=:black, label="Alternating Direct Solve")
=#
