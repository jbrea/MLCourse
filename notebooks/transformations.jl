### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 34beb7d6-fba9-11eb-15b8-e34afccc9f88
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	using CSV, DataFrames, Distributions, Plots, PlutoUI, MLJ, MLJLinearModels, Random
	PlutoUI.TableOfContents()
end

# ╔═╡ e688a9de-2dba-4fab-b4e6-9803c5361a62
begin
	using MLCourse
	MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
md"# Transformations of the Input

## Vector Features"

# ╔═╡ b66c7efc-8fd8-4152-9b0f-9332f760b51b
function xor_generator(; n = 200)
	x = 2 * rand(n, 2) .- 1
	DataFrame(X1 = x[:, 1], X2 = x[:, 2],
		      y = coerce((x[:, 1] .> 0) .⊻ (x[:, 2] .> 0), Binary))
end

# ╔═╡ ebe38042-5160-469b-8c07-484b7c019063
xor_data = xor_generator()

# ╔═╡ fa7cf2ae-9777-4ba9-9fe0-df545b94c5d8
begin
	scatter(xor_data.X1, xor_data.X2, c = int.(xor_data.y) .+ 1,
            label = nothing, xlabel = "X₁", ylabel = "X₂")
	hline!([0], label = nothing, c = :black)
	vline!([0], label = nothing, c = :black, size = (400, 300))
end

# ╔═╡ 99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
begin
	lin_mach = fit!(machine(LogisticClassifier(penalty = :none),
			                select(xor_data, Not(:y)),
			                xor_data.y))
	lin_pred = predict_mode(lin_mach, select(xor_data, Not(:y)))
	(training_misclassification_rate = mean(lin_pred .!= xor_data.y),)
end

# ╔═╡ a7f12901-e155-4f61-b252-c5af6669bfee
function vector_features(data, vectors...)
	q = length(vectors) # number of vector features
	names = [Symbol("H", i) for i in 1:q]
	feature_representation = [max.(0, data.X1 * v[1] .+ data.X2 * v[2])
		                      for v in vectors]
	DataFrame(feature_representation, names)
end

# ╔═╡ ad2cd0c5-e308-4911-aba9-3b56eda8a2e2
transformed_xor_input = vector_features(xor_data, [1, 1], [1, -1], [-1, 1], [-1, -1])

# ╔═╡ 1b8239fb-3894-47d6-87bf-4592b2fd078d
begin
	lin_mach_f = fit!(machine(LogisticClassifier(penalty = :none),
			                  transformed_xor_input,
			                  xor_data.y))
	lin_pred_f = predict_mode(lin_mach_f, transformed_xor_input)
	(training_misclassification_rate = mean(lin_pred_f .!= xor_data.y),)
end

# ╔═╡ 764466ab-ed67-48b7-98a0-b2cb49a80c83
md"## Splines"

# ╔═╡ 026de033-b9b2-4165-80e7-8603a386381c
function spline_features(data, degree, knots)
	q = length(knots) + degree
	names = [Symbol("H", i) for i in 1:q]
	features = [data.x .^ d for d in 1:degree]
	append!(features, [max.(0, (data.x .- k).^degree) for k in knots])
	DataFrame(features, names)
end

# ╔═╡ 2121906c-48de-4669-accb-00dc1f763065
spline_data = (x = -4:.1:4,)

# ╔═╡ 2e1c3fa4-e424-4844-82c1-53d02271e7a6
begin
	knots = -3:2:3
    spline_input = spline_features(spline_data, 3, knots)
end

# ╔═╡ 0815ba19-5c7d-40d6-b948-8edccfb5c386
begin
	plot(spline_data.x, Array(spline_input) * 2*randn(ncol(spline_input)))
	vline!(knots, linestyle = :dash)
end

# ╔═╡ cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
md"# Transformations of the Output"

# ╔═╡ 5e6d6155-c254-4ae9-a0e1-366fc6ce6403
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), DataFrame);

# ╔═╡ ed239411-185a-4d9f-b0ec-360400f24dc7
normal_fit = Distributions.fit(Normal, weather.LUZ_wind_peak)

# ╔═╡ 7c04a091-93af-476e-8769-b14c08d3ae10
begin
	histogram(weather.LUZ_wind_peak, normalize = :pdf,
		      xlabel = "LUZ_wind_peak", label = nothing)
	plot!(x -> pdf(normal_fit, x), w = 2, label = "fitted Normal distribution")
end

# ╔═╡ 3c3aa56f-cb19-4115-9268-b73337fb6a49
lognormal_fit = Distributions.fit(Normal, log.(weather.LUZ_wind_peak))

# ╔═╡ 15faf5e7-fa72-4519-a83b-3007486450dd
begin
	histogram(log.(weather.LUZ_wind_peak), normalize = :pdf,
		      xlabel = "log(LUZ_wind_peak)", label = nothing)
	plot!(x -> pdf(lognormal_fit, x), w = 2, label = "fitted Normal distribution")
end

# ╔═╡ 74f5178c-a308-4184-bc94-4a04f6f9ffdc
gamma_fit = Distributions.fit_mle(Gamma, weather.LUZ_wind_peak)

# ╔═╡ 62491145-48b9-4ac5-8e7b-0676e9616fe9
begin
	histogram(weather.LUZ_wind_peak, normalize = :pdf,
		      xlabel = "LUZ_wind_peak", label = nothing)
	plot!(x -> pdf(gamma_fit, x), w = 2, label = "fitted Gamma distribution")
end

# ╔═╡ Cell order:
# ╟─4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
# ╠═b66c7efc-8fd8-4152-9b0f-9332f760b51b
# ╠═ebe38042-5160-469b-8c07-484b7c019063
# ╠═fa7cf2ae-9777-4ba9-9fe0-df545b94c5d8
# ╠═99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
# ╠═a7f12901-e155-4f61-b252-c5af6669bfee
# ╠═ad2cd0c5-e308-4911-aba9-3b56eda8a2e2
# ╠═1b8239fb-3894-47d6-87bf-4592b2fd078d
# ╟─764466ab-ed67-48b7-98a0-b2cb49a80c83
# ╠═026de033-b9b2-4165-80e7-8603a386381c
# ╠═2121906c-48de-4669-accb-00dc1f763065
# ╠═2e1c3fa4-e424-4844-82c1-53d02271e7a6
# ╠═0815ba19-5c7d-40d6-b948-8edccfb5c386
# ╟─cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
# ╠═5e6d6155-c254-4ae9-a0e1-366fc6ce6403
# ╠═ed239411-185a-4d9f-b0ec-360400f24dc7
# ╟─7c04a091-93af-476e-8769-b14c08d3ae10
# ╠═3c3aa56f-cb19-4115-9268-b73337fb6a49
# ╟─15faf5e7-fa72-4519-a83b-3007486450dd
# ╠═74f5178c-a308-4184-bc94-4a04f6f9ffdc
# ╟─62491145-48b9-4ac5-8e7b-0676e9616fe9
# ╟─e688a9de-2dba-4fab-b4e6-9803c5361a62
# ╟─34beb7d6-fba9-11eb-15b8-e34afccc9f88
