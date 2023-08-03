using LinearAlgebra
using Plots
using Zygote

f(x, y) = (10-x*y)^2 + x^2 + y^2

gx(x, y) = 2 *(x - 10 *y + x *y^2) #gradient(x->f(x,y), x)[1]
gy(x, y) = 2 *(-10 *x + y + x^2 *y)#gradient(y->f(x,y), y)[1]

g(x, y) = [gx(x, y), gy(x, y)]

x = range(2, 4, length=100)
y = range(2, 5, length=100)
z = @. f(x', y)
Lx = maximum(gx.(x, y))
Ly = maximum(gx.(x, y))
L = maximum(norm.(g.(x, y)))
contour(x, y, z; title="f(x,y) = (10-x*y)^2 + x^2 + y^2",aspect_ratio=:equal)

plot!([3],[3],label="minimum",shape=:star)


gradient_step_x(x, y) = x - gx(x, y)/Lx
gradient_step_y(x, y) = y - gy(x, y)/Ly
gradient_step(x, y) = [x, y] .- g(x, y) ./ L

N = 32 # number of outer iterations
Kx = 1 # number of inner iterations
Ky = 1 # number of inner iterations
x0, y0 = 4,5
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

#################
path = []

x, y = x0, y0
push!(path, (x, y))
i, j = 0, 0

@time begin
for i in 1:(N * (Kx + Ky))
    x, y = gradient_step(x, y)
    push!(path, (x, y))
end
end
plot!(collect(zip(path)); color=:green, label="Gradient Descent")
