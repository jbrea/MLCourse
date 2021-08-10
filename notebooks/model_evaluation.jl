### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 736856ce-f490-11eb-3349-057c86edfe7e
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI, MLJ, MLJLinearModels, Plots, DataFrames, Random
    gr()
    PlutoUI.TableOfContents()
end

# ╔═╡ c693088f-7f80-4cdd-b9b5-65a50da732ac
begin
    using MLCourse
    import MLCourse: poly, poly2
    MLCourse.list_notebooks(@__FILE__)
end


# ╔═╡ 7eb6a060-f948-4d85-881a-4909e74c15bd
md"# Validation Set Approach"

# ╔═╡ 7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
begin
    f(x) = 0.3 + 2x - 0.8x^2 - 0.4x^3
    function data_generator(; n = 500, seed = 12)
        rng = MersenneTwister(seed)
        x = randn(rng, n)
        DataFrame(x = x, y = f.(x) .+ randn(rng, n))
    end
    function data_split(data;
                        idx_train = 1:50,
                        idx_valid = 51:100,
                        idx_test = 101:500)
        (train = data[idx_train, :],
         valid = data[idx_valid, :],
         test = data[idx_test, :])
    end
end;

# ╔═╡ fa9b4b9e-4e97-4e5c-865d-ad3ef288e4cf
data1 = data_split(data_generator(seed = 1))

# ╔═╡ 751880ec-1a82-4142-b875-177d436bbc72
begin
    preprocessor(degree) = X -> MLCourse.poly(select(X, Not(:y)), degree)
	mse(mach, data, p) = mean((predict(mach, p(data)) .- data.y).^2)
    function fit_and_evaluate(model, data, degree)
        p = preprocessor(degree)
        mach = fit!(machine(model, p(data.train), data.train.y), verbosity = 0)
        (train = √(mse(mach, data.train, p)),
		 valid = √(mse(mach, data.valid, p)),
		 test = √(mse(mach, data.test, p)))
    end
end

# ╔═╡ 2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
losses1 = [fit_and_evaluate(LinearRegressor(), data1, degree) for degree in 1:10]

# ╔═╡ 91eecd2b-af18-4a63-9684-28950e604d1a
let validlosses = getproperty.(losses1, :valid), i = argmin(validlosses)
    plot(1:10, validlosses, label = nothing)
    scatter!([i], [validlosses[i]], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ b45b32b1-8a65-4823-b3bb-f0b7cc57604b
md"For this seed of the random number generator the optimal degree (x coordinate of the red point in the figure above) found by the validation set approach is close to the actual degree of the data generating process.
Let us now look at other seeds."

# ╔═╡ ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
losses = [fit_and_evaluate(LinearRegressor(),
                           data_split(data_generator(seed = seed)),
                           degree)
          for degree in 1:10, seed in 1:20]

# ╔═╡ 26362233-b006-423d-8fb5-7cd9150405b4
let validlosses = getproperty.(losses, :valid), i = argmin(validlosses, dims = 1)
    plot(validlosses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 5cff19c7-3426-4d91-b555-f059f5f41886
md"Here we see a large variability of the optimal degree (x coordinates of all the points in the figure above).

In the following cell we compare the validation set estimate of the test error to the test set estimation of the test error. We can confirm that the validation set estimate is much lower than the test set estimate."

# ╔═╡ df16313f-76be-4d8b-88e7-7cc618ff49d4
let validlosses = getproperty.(losses, :valid),
	testlosses = getproperty.(losses, :test),
	winners = argmin(validlosses, dims = 1)
	(mean_valid_winner = mean(validlosses[winners]),
	 mean_test_winner = mean(testlosses[winners]))
end

# ╔═╡ 46bffe01-1a7c-405b-b3ec-bfe7570b8a3c
md"# Cross-Validation"

# ╔═╡ b3aed705-5072-4d9b-bb8f-865ac1561bf6
begin
    function cross_validation_sets(idx, K)
        n = length(idx)
        r = n ÷ K
        [let idx_valid = idx[(i-1)*r+1:(i == K ? n : i*r)]
             (idx_valid = idx_valid, idx_train = setdiff(idx, idx_valid))
         end
         for i in 1:K]
    end
    function cross_validation(model, data, degree; K = 5)
        losses = [fit_and_evaluate(model,
                                   data_split(data; idxs...),
                                   degree)
                  for idxs in cross_validation_sets(1:100, K)]
        (train = mean(getproperty.(losses, :train)),
         valid = mean(getproperty.(losses, :valid)),
         test = mean(getproperty.(losses, :test)))
    end
end

# ╔═╡ 6916273a-d116-4173-acaa-2bcac1d1753b
cross_validation_sets(1:100, 4)

# ╔═╡ c14c9e49-a993-456e-9115-97da86f8e498
losses1_cv10 = [cross_validation(LinearRegressor(),
		                         data_generator(seed = 1),
					 		     degree,
		                         K = 10) for degree in 1:10]

# ╔═╡ 4b435518-2f12-4921-bb1f-fdd049ddfaed
let validlosses = getproperty.(losses1_cv10, :valid), i = argmin(validlosses)
    plot(1:10, validlosses, label = nothing)
    scatter!([i], [validlosses[i]], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 1e584a38-2fef-4877-87f6-92237d71c4b3
losses_cv10 = [cross_validation(LinearRegressor(),
                                data_generator(seed = seed),
                                degree,
                                K = 10)
               for degree in 1:10, seed in 1:20]

# ╔═╡ a7c88b3f-92cb-4253-a889-c78683722c1d
let validlosses = getproperty.(losses_cv10, :valid), i = argmin(validlosses, dims = 1)
    plot(validlosses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ eaacf529-1727-4743-941b-360c53088b1d
md"Instead of our own cross-validation function we can also use the builtin functions of MLJ."

# ╔═╡ 967a6e08-f0e6-45b7-988b-d0df237f3ddf
MLJ.MLJBase.train_test_pairs(CV(nfolds = 10), 1:100)

# ╔═╡ 8ae790b3-3987-4f41-8e21-adbb71081eb9
let data = data_generator(seed = 1)
	evaluate(LinearRegressor(), preprocessor(4)(data), data.y,
		     resampling = CV(nfolds = 10))
end

# ╔═╡ Cell order:
# ╟─7eb6a060-f948-4d85-881a-4909e74c15bd
# ╠═7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
# ╠═fa9b4b9e-4e97-4e5c-865d-ad3ef288e4cf
# ╠═751880ec-1a82-4142-b875-177d436bbc72
# ╠═2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
# ╟─91eecd2b-af18-4a63-9684-28950e604d1a
# ╟─b45b32b1-8a65-4823-b3bb-f0b7cc57604b
# ╠═ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
# ╟─26362233-b006-423d-8fb5-7cd9150405b4
# ╟─5cff19c7-3426-4d91-b555-f059f5f41886
# ╠═df16313f-76be-4d8b-88e7-7cc618ff49d4
# ╟─46bffe01-1a7c-405b-b3ec-bfe7570b8a3c
# ╠═b3aed705-5072-4d9b-bb8f-865ac1561bf6
# ╠═6916273a-d116-4173-acaa-2bcac1d1753b
# ╠═c14c9e49-a993-456e-9115-97da86f8e498
# ╟─4b435518-2f12-4921-bb1f-fdd049ddfaed
# ╠═1e584a38-2fef-4877-87f6-92237d71c4b3
# ╟─a7c88b3f-92cb-4253-a889-c78683722c1d
# ╟─eaacf529-1727-4743-941b-360c53088b1d
# ╠═967a6e08-f0e6-45b7-988b-d0df237f3ddf
# ╠═8ae790b3-3987-4f41-8e21-adbb71081eb9
# ╟─c693088f-7f80-4cdd-b9b5-65a50da732ac
# ╟─736856ce-f490-11eb-3349-057c86edfe7e
