### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 2b747e4c-ab56-11ec-3e1b-2be59046c02d
begin
    using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using Plots , MLJ, MLJFlux, Flux, MLJMultivariateStatsInterface, Images, DataFrames, StatsBase, Random, SpecialFunctions, Distributions
end

# ╔═╡ 96701def-57bb-4140-b27a-f2d3f2fb9443
using MLCourse; MLCourse.footer()

# ╔═╡ d3bd1a08-e8e5-46a0-a184-092e2000c818
import Flux: binarycrossentropy

# ╔═╡ 1a4e4ffc-2bde-4e5b-a6f5-9458b97ba29d
md"# Non-linear Encoders, Decoders and Generators

In this notebook we look at a simple data generator that produces images of crosses at different positions and with different gray level. Although the images consist of ``6\times6 = 36`` pixels, allowing for ``2^{36}`` different black-and-white images, the data is effectively low-dimensional: the cross can be only at one of 16 positions and all pixels of the cross have the same gray-level.
"

# ╔═╡ 91c25364-754c-43af-b0fa-0b077f6652bc
function generate_input(; x = rand(1:4), y = rand(1:4))
	img = zeros(Float32, 6, 6)
	c = .75*rand() .+ .2
	for i in x:x+2
		img[i, y+1] = c
	end
	for j in y:y+2
		img[x+1, j] = c
	end
	img
end

# ╔═╡ c1a2b203-96a7-4bde-888e-fda3bb20cfb8
function plot_samples(func)
	plot([plot(Gray.(reshape(func(), 6, 6)), ticks = false)
		 for _ in 1:15]..., layout = (3, 5))
end

# ╔═╡ d9a30eea-6d78-4615-9e19-f759dfab55b2
plot_samples(generate_input)

# ╔═╡ 4796369c-c2b5-4c26-9996-5ace5503d8f4
begin
	images = [reshape(generate_input(), :) for _ in 1:500]
	df = DataFrame(hcat(images...)', :auto)
end;

# ╔═╡ a0acfe55-0a38-4089-95a5-310a348368a6
begin
	mach = machine(PCA(), df)
	fit!(mach, verbosity = 0)
	report(mach)
end

# ╔═╡ 9d412310-c503-4ef7-b3e6-b7a62ac51980
md"## Auto-Encoder"

# ╔═╡ 50f477ef-9355-4479-8124-87ed86ee9d79
begin
	encoder = Chain(Dense(36, 32, relu),
				    Dense(32, 2))
	decoder = Chain(Dense(2, 32, relu),
					Dense(32, 36, sigmoid))
	auto_encoder = Chain(encoder, decoder)
	opt = ADAM()
	loss = x -> Flux.Losses.binarycrossentropy(auto_encoder(x), x)
	for epoch in 1:200
		Flux.train!(loss, Flux.params(auto_encoder), images, opt)
		# @show mean(loss.(images))
	end
end

# ╔═╡ 3c021353-18f1-4424-9c93-008d63897aac
let img = rand(images)
	p1 = plot(Gray.(reshape(img, 6, 6)), title = "input")
	p2 = plot(Gray.(reshape(auto_encoder(img), 6, 6)),
		      title = "output of auto-encoder")
	plot(p1, p2, layout = (1, 2))
end

# ╔═╡ 4966bd8d-a524-4254-8e78-f255cb9ff7cd
encoder(rand(images))

# ╔═╡ a2545995-6357-4fe7-8039-1878eead39cc
let unique_images = union(images)
	enc = encoder.(unique_images)
	f1 = first.(enc)
	f2 = last.(enc)
	xlim = 1.2 .* (minimum(f1), maximum(f1))
	ylim = 1.2 .* (minimum(f2), maximum(f2))
	r = (x, y) -> ((x - xlim[1])/(xlim[2] - xlim[1]),
	               1-(y - ylim[1])/(ylim[2] - ylim[1]))
	scatter(f1, f2; label = false, xlim = xlim, ylim = ylim,
	        xlabel = "feature dimension 1", ylabel = "feature dimension 2")
	for i in eachindex(enc)
		plot!(Gray.(reshape(unique_images[i], 6, 6)), ticks = false,
		      inset = (1, bbox(r(enc[i]...)..., .1, .1)), subplot = i+1)
        i == 20 && break
	end
	plot!()
end

# ╔═╡ 1dc388c3-2a71-4a1b-a218-d56e8030dfec
plot_samples(() -> decoder(5*randn(2)))

# ╔═╡ 5b91d824-c96c-48b7-81b2-40be50e1116b
md"## Variational Auto-Encoder"

# ╔═╡ 1e4f8da7-343f-45e4-b62b-f2de59d0031c
begin
	struct VAEEncoder
		trunc
		μ
		logvar
	end
	Flux.@functor VAEEncoder
	function (vae::VAEEncoder)(x)
		h = vae.trunc(x)
		vae.μ(h), vae.logvar(h)
	end
	encoder_vae = VAEEncoder(Dense(36, 32, relu),
		                     Dense(32, 2),
		                     Dense(32, 2))
	decoder_vae = Chain(Dense(2, 32, relu),
					    Dense(32, 36, sigmoid))
	opt_vae = ADAM()
	function loss_vae(x; β = .02)
		μ, logvar = encoder_vae(x)
		z = μ .+ exp.(.5 * logvar) .* randn(size(μ))
		Flux.Losses.binarycrossentropy(decoder_vae(z), x) + β * mean(@. (exp(logvar) + μ^2 -1 - logvar))
	end
	for epoch in 1:300
		Flux.train!(loss_vae, Flux.params([encoder_vae, decoder_vae]), images, opt_vae)
		# @show mean(loss_vae.(images))
	end
end

# ╔═╡ ff87b90d-bc1c-41f9-952c-4a737938784e
let img = rand(images)
	p1 = plot(Gray.(reshape(img, 6, 6)), title = "input")
	p2 = plot(Gray.(reshape(decoder_vae(encoder_vae(img)[1]), 6, 6)),
		      title = "output of VAE")
	plot(p1, p2, layout = (1, 2))
end

# ╔═╡ 43ae2ef9-e64f-44ff-b359-cdbeb97e6bc0
encoder_vae(rand(images))

# ╔═╡ afb8b465-8568-4cc5-a421-f173511f4a4d
let unique_images = union(images)
	enc = first.(encoder_vae.(unique_images))
	f1 = first.(enc)
	f2 = last.(enc)
	xlim = 1.2 .* (minimum(f1), maximum(f1))
	ylim = 1.2 .* (minimum(f2), maximum(f2))
	r = (x, y) -> ((x - xlim[1])/(xlim[2] - xlim[1]),
	               1-(y - ylim[1])/(ylim[2] - ylim[1]))
	scatter(f1, f2; label = false, xlim = xlim, ylim = ylim,
	        xlabel = "feature dimension 1", ylabel = "feature dimension 2")
	for i in eachindex(enc)
		plot!(Gray.(reshape(unique_images[i], 6, 6)), ticks = false,
		      inset = (1, bbox(r(enc[i]...)..., .1, .1)), subplot = i+1)
        i == 20 && break
	end
	plot!()
end

# ╔═╡ 594b2459-52f1-4032-9282-59b7ae19e428
plot_samples(() -> decoder_vae(randn(2)))

# ╔═╡ c940ef9f-37de-4b9a-a525-1b8e1cc149c1
md"## P(X₁, X₂, X₃) = P(X₁)P(X₂|X₁)P(X₃|X₁, X₂) and Recurrent Neural Networks"

# ╔═╡ 129b5f42-85cc-4669-a417-dc3fc0fd027e
begin
	rnn = Chain(Dense(1, 32, relu), LSTM(32, 32), Dense(32, 2, softplus))
	negllbeta(ab, x) = (1-ab[1])*log(x + eps()) + (1-ab[2])*log(1-x) +
	                   logbeta(ab[1], ab[2])
	function loss_rnn(x)
		loss = 0.
		for i in 1:length(x)-1
			loss += negllbeta(rnn(x[i:i]), x[i+1])
		end
		Flux.reset!(rnn)
		loss
	end
	opt_rnn = ADAM()
	for epoch in 1:200
		Flux.train!(loss_rnn, Flux.params(rnn), images, opt)
		# @show mean(loss_rnn.(images))
	end
end

# ╔═╡ 660f8163-7b97-47b3-89ae-b888b4f77743
function sample_rnn()
	result = [0.]
	for _ in 1:35
		push!(result, rand(Beta(rnn([Float32(result[end])])...)))
	end
	Flux.reset!(rnn)
	result
end

# ╔═╡ 00f9e58f-a064-43ff-a9d4-0448d20fb105
plot_samples(sample_rnn)

# ╔═╡ 90726f54-a523-4d38-b6f0-6278cf9791cc
md"## Generative Adversarial Networks"

# ╔═╡ f85ef269-6e71-403c-a12b-c8a2c4875a68
begin
	Random.seed!(123)
	generator = Chain(Dense(2, 32, relu), Dense(32, 36, sigmoid))
	discriminator = Chain(Dense(2*36, 64, relu), Dense(64, 1, sigmoid))
	generate_fakes() = reshape(generator(randn(2, 2)), :)
	generator_loss(::Any) = binarycrossentropy(discriminator(generate_fakes()), 1)
	function discriminator_loss(x)
		binarycrossentropy(discriminator(x), 1) + # real
		binarycrossentropy(discriminator(generate_fakes()), 0) # fake
	end
	opt_gen = Flux.Optimiser(ClipValue(1e-1), ADAM(1e-3))
	opt_disc = Flux.Optimiser(ClipValue(1e-1), ADAM(1e-3))
	for it in 1:3e4
		Flux.train!(discriminator_loss, Flux.params(discriminator),
			        [vcat(rand(images, 2)...) for _ in 1:20], opt_disc)
		Flux.train!(generator_loss, Flux.params(generator), 1:20, opt_gen)
		if it % 500 == 0
			# @show mean(discriminator_loss.([vcat(rand(images, 2)...) for _ in 1:length(images)]))
			# @show mean(generator_loss.(1:100))
		end
	end
end

# ╔═╡ e25aaf77-2e9b-4380-b73d-55cb4d1fd295
plot_samples(() -> generator(randn(2)))

# ╔═╡ Cell order:
# ╠═2b747e4c-ab56-11ec-3e1b-2be59046c02d
# ╠═d3bd1a08-e8e5-46a0-a184-092e2000c818
# ╟─1a4e4ffc-2bde-4e5b-a6f5-9458b97ba29d
# ╠═91c25364-754c-43af-b0fa-0b077f6652bc
# ╠═c1a2b203-96a7-4bde-888e-fda3bb20cfb8
# ╠═d9a30eea-6d78-4615-9e19-f759dfab55b2
# ╠═4796369c-c2b5-4c26-9996-5ace5503d8f4
# ╠═a0acfe55-0a38-4089-95a5-310a348368a6
# ╟─9d412310-c503-4ef7-b3e6-b7a62ac51980
# ╠═50f477ef-9355-4479-8124-87ed86ee9d79
# ╠═3c021353-18f1-4424-9c93-008d63897aac
# ╠═4966bd8d-a524-4254-8e78-f255cb9ff7cd
# ╠═a2545995-6357-4fe7-8039-1878eead39cc
# ╠═1dc388c3-2a71-4a1b-a218-d56e8030dfec
# ╟─5b91d824-c96c-48b7-81b2-40be50e1116b
# ╠═1e4f8da7-343f-45e4-b62b-f2de59d0031c
# ╠═ff87b90d-bc1c-41f9-952c-4a737938784e
# ╠═43ae2ef9-e64f-44ff-b359-cdbeb97e6bc0
# ╟─afb8b465-8568-4cc5-a421-f173511f4a4d
# ╠═594b2459-52f1-4032-9282-59b7ae19e428
# ╟─c940ef9f-37de-4b9a-a525-1b8e1cc149c1
# ╠═129b5f42-85cc-4669-a417-dc3fc0fd027e
# ╠═660f8163-7b97-47b3-89ae-b888b4f77743
# ╠═00f9e58f-a064-43ff-a9d4-0448d20fb105
# ╟─90726f54-a523-4d38-b6f0-6278cf9791cc
# ╠═f85ef269-6e71-403c-a12b-c8a2c4875a68
# ╠═e25aaf77-2e9b-4380-b73d-55cb4d1fd295
# ╟─96701def-57bb-4140-b27a-f2d3f2fb9443
