### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 76472511-92f0-471f-89d9-83dbe39003b5
begin
	using Einsum
	using Random
	using LinearAlgebra
end

# ╔═╡ 6ffa1770-9c95-4f1a-bba1-d0c03b330f62
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

# ╔═╡ be1c0889-2eb3-40c7-84ce-f0ffb165b2cf
AA

# ╔═╡ 25da844c-0793-48ab-b619-f91ac111fc10
A

# ╔═╡ 310756d6-9429-46a2-8449-7cb0e33e1902
b

# ╔═╡ 420a33e5-05c3-478d-8d99-5f8cb4b8b276
bb

# ╔═╡ 34fd689f-44d6-4a70-9788-f3f3b978bbda
T

# ╔═╡ 57aff24b-93ca-4016-bb8d-a6bac1cc0cad
X

# ╔═╡ 20ce07da-1ae0-4c09-97f4-8c6b3fd35e30
AA*(T×₃bb)

# ╔═╡ d930924e-52eb-4261-85ff-c680927a567c
(T×₃bb)

# ╔═╡ fb5b8b19-f504-48c7-a7f7-889bdd6d57fe
(T×₃b)

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
# ╠═76472511-92f0-471f-89d9-83dbe39003b5
# ╠═6ffa1770-9c95-4f1a-bba1-d0c03b330f62
# ╠═be1c0889-2eb3-40c7-84ce-f0ffb165b2cf
# ╠═25da844c-0793-48ab-b619-f91ac111fc10
# ╠═310756d6-9429-46a2-8449-7cb0e33e1902
# ╠═420a33e5-05c3-478d-8d99-5f8cb4b8b276
# ╠═34fd689f-44d6-4a70-9788-f3f3b978bbda
# ╠═57aff24b-93ca-4016-bb8d-a6bac1cc0cad
# ╠═20ce07da-1ae0-4c09-97f4-8c6b3fd35e30
# ╠═d930924e-52eb-4261-85ff-c680927a567c
# ╠═fb5b8b19-f504-48c7-a7f7-889bdd6d57fe
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
