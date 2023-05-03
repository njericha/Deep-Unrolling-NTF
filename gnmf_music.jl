### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ee730960-e9f1-11ed-052d-f5fa1dab785c
begin
	using Einsum
	using Random
	using LinearAlgebra
end

# ╔═╡ d574d519-0dba-4879-8ef4-ed4984d222b7
begin
	function ×₃(T::Array{Float64, 3}, v::Vector{Float64})
		@einsum M[i,j] := T[i,j,k]*v[k] #contract the 3rd index of a tensor with a vector
		return M
	end
	
	function mu_gnmf(X, T; maxiter=800, tol=5e-4, λ=0, ϵ=1e-8) #multipicative updated nonnegative matrix factorization
		#using LinearAlgebra
		
		# Initilization
		m, n = size(X)
		r, N, p = size(T)
		@assert n==N "Missmatch between the second dimention of X and T"
		A = abs.(randn((m, r)))
		b = abs.(randn((p,)))
		i = 0
	
		# Updates
		while (norm(X - A*(T×₃b))/norm(X) > tol) && (i < maxiter)
			i += 1
			@show i
			#@einsum b[q] = b[q] * T[u,v,q]*A[w,u]*X[w,v] / (T[i,j,q]*A[l,i]*A[l,s]*T[s,j,t]*b[t] + ϵ + λ*b[q]) #(W'*V ) ./ (W'*W*H  .+ ϵ + λ.*H) #update b
			#b_old = b
			for q ∈ 1:p
				TT = T[:,:,q]
				b[q] *= tr(TT'*A'*X) / (tr(TT'*A'*A*(T×₃b)) + ϵ + λ*b[q]) #can replace b with b_old?
			end
			B = (T×₃b)
			A .*= (X *B') ./ (A *B*B' .+ ϵ + λ.*A) #update A
		end
	
		return (A, b, i)
	end
end

# ╔═╡ 7fae4483-bb30-4236-a080-adbb83f458e3
begin
	#Random.seed!(314)
	m,n,r,p = (100,100,10,5)
	A = abs.(randn((m, r)))
    b = abs.(randn((p,)))
	T = abs.(randn((r,n,p)))
	T[T.>=1] .= 1
	T[T.< 1] .= 0
	X = A*(T×₃b)
	(AA, bb, i) = mu_gnmf(X, T)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Einsum = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Einsum = "~0.4.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "4ad79eff2e7822e7856a1d31097ad65f1d309aa8"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═ee730960-e9f1-11ed-052d-f5fa1dab785c
# ╠═d574d519-0dba-4879-8ef4-ed4984d222b7
# ╠═7fae4483-bb30-4236-a080-adbb83f458e3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
