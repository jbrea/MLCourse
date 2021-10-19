### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 736856ce-f490-11eb-3349-057c86edfe7e
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLJ, MLJLinearModels, Plots, DataFrames, Random
end

# ╔═╡ c693088f-7f80-4cdd-b9b5-65a50da732ac
begin
    using MLCourse
    import MLCourse: poly
    MLCourse.list_notebooks(@__FILE__)
end


# ╔═╡ e1077092-f72a-42af-b6e0-a616f938cba8
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 7eb6a060-f948-4d85-881a-4909e74c15bd
md"# Validation Set Approach

In the following cell we define a `data_generator` that creates data sets with a polynomial relationship of degree 3 between the input x and the average output f(x). We split these data sets into a training set, a validation set and a test set. The training set will be used to find the parameters for a machine learning method with given hyper-parameters, the validation set will be used to find the hyper-parameters and the test set will be used to evaluate all our models.
"

# ╔═╡ 7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
begin
    f(x) = 0.3 + 2x - 0.8x^2 - 0.4x^3
    function data_generator(; n = 500, seed = 12)
        rng = MersenneTwister(seed)
        x = randn(rng, n)
        DataFrame(x = x, y = f.(x) .+ randn(rng, n))
    end
    function data_split(data;
                        shuffle = false,
                        idx_train = 1:50,
                        idx_valid = 51:100,
                        idx_test = 101:500)
        idxs = if shuffle
                randperm(size(data, 1))
            else
                1:size(data, 1)
            end
        (train = data[idxs[idx_train], :],
         valid = data[idxs[idx_valid], :],
         test = data[idxs[idx_test], :])
    end
    function fit_and_evaluate(model, data)
        mach = fit!(machine(model, select(data.train, :x), data.train.y),
                    verbosity = 0)
        (train = rmse(predict(mach, select(data.train, :x)), data.train.y),
         valid = rmse(predict(mach, select(data.valid, :x)), data.valid.y),
         test = rmse(predict(mach, select(data.test, :x)), data.test.y))
    end
end;

# ╔═╡ fa9b4b9e-4e97-4e5c-865d-ad3ef288e4cf
data1 = data_split(data_generator(seed = 1))

# ╔═╡ 751880ec-1a82-4142-b875-177d436bbc72
md"To simplify our training and test procedure we define a new type of a machine called `PolynomialRegressor` that takes as keyword argument the `degree` of the polynomial. You do not need to understand the code in the following hidden cell. But if you are interested in how one can define custom machines in the `MLJ` framework you are welcome to look at the code and the [MLJ Documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/simple_user_defined_models/); it is actually quite simple to write a custom machine."

# ╔═╡ da2bbb4d-d45d-4d2a-9c8a-21395bcbb851
begin
    # Use the Live Docs to learn more about Base.@kwdef
    Base.@kwdef mutable struct PolynomialRegressor{T} <: Deterministic
        degree::Int = 3
        regressor::T = LinearRegressor()
    end
    function MLJ.MLJBase.fit(model::PolynomialRegressor, verbosity, X, y)
        Xpoly = poly(X, model.degree) # here we transform the input
        MLJ.MLJBase.fit(model.regressor, verbosity, Xpoly, y)
    end
    function MLJ.MLJBase.predict(model::PolynomialRegressor, fitresult, Xnew)
        Xpoly = poly(Xnew, model.degree) # here we transform the input
        MLJ.MLJBase.predict(model.regressor, fitresult, Xpoly)
    end
end

# ╔═╡ 2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
losses1 = [fit_and_evaluate(PolynomialRegressor(; degree), data1) for degree in 1:10]

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
losses = [fit_and_evaluate(PolynomialRegressor(; degree),
                           data_split(data_generator(seed = seed)))
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
md"# Cross-Validation

In the following cell we implement cross-validation ourselves."

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
    function cross_validation(model, data; K = 5)
        losses = [fit_and_evaluate(model,
                                   data_split(data; idxs...))
                  for idxs in cross_validation_sets(1:100, K)]
        (train = mean(getproperty.(losses, :train)),
         valid = mean(getproperty.(losses, :valid)),
         test = mean(getproperty.(losses, :test)))
    end
end

# ╔═╡ 6916273a-d116-4173-acaa-2bcac1d1753b
cross_validation_sets(1:100, 4)

# ╔═╡ c14c9e49-a993-456e-9115-97da86f8e498
losses1_cv10 = [cross_validation(PolynomialRegressor(; degree),
		                         data_generator(seed = 1),
		                         K = 10) for degree in 1:10]

# ╔═╡ 4b435518-2f12-4921-bb1f-fdd049ddfaed
let validlosses = getproperty.(losses1_cv10, :valid), i = argmin(validlosses)
    plot(1:10, validlosses, label = nothing)
    scatter!([i], [validlosses[i]], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 1e584a38-2fef-4877-87f6-92237d71c4b3
losses_cv10 = [cross_validation(PolynomialRegressor(; degree),
                                data_generator(seed = seed),
                                K = 10)
               for degree in 1:10, seed in 1:20]

# ╔═╡ a7c88b3f-92cb-4253-a889-c78683722c1d
let validlosses = getproperty.(losses_cv10, :valid), i = argmin(validlosses, dims = 1)
    plot(validlosses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 0ab1238e-8279-43d5-9803-848cb00156c2
let validlosses = getproperty.(losses_cv10, :valid),
	testlosses = getproperty.(losses_cv10, :test),
	winners = argmin(validlosses, dims = 1)
	(mean_valid_winner = mean(validlosses[winners]),
	 mean_test_winner = mean(testlosses[winners]))
end


# ╔═╡ eaacf529-1727-4743-941b-360c53088b1d
md"Instead of our own cross-validation function we can also use the builtin functions of MLJ. The `train_test_pairs` function does basically the same as our `cross_validation_sets` function."

# ╔═╡ 967a6e08-f0e6-45b7-988b-d0df237f3ddf
MLJ.MLJBase.train_test_pairs(CV(nfolds = 10), 1:100)

# ╔═╡ 535a11d9-833e-4237-949e-9a3e289c600b
md"`MLJ` has the very useful function `evaluate!`. Have a look at the Live docs to learn more about this function. In the following you see an example where we perform 10-fold cross-validation on the `rmse` measure to estimate the expected test error for a polynomial regressor of degree 4."

# ╔═╡ 8ae790b3-3987-4f41-8e21-adbb71081eb9
let data = data_generator(seed = 1, n = 100)
    evaluate!(machine(PolynomialRegressor(degree = 4), select(data, :x), data.y),
              resampling = CV(nfolds = 10), measure = rmse)
end

# ╔═╡ 134df37f-b737-4c80-a5f9-1149aeec970c
md"We can use this now to find the best degree with 10-fold cross-validation for 100 different seeds. You will see that - even though not perfect - the hyper-parameters found with 10-fold cross-validation have lower variance and are usually closer to the true value than with the validation set approach."

# ╔═╡ abb71af3-8aae-4806-9d5a-d144c15d22ef
losses_mlj_cv10 = [let data = data_generator(seed = seed)[1:100, :]
                       evaluate!(machine(PolynomialRegressor(; degree),
                                         select(data, :x),
                                         data.y),
                                resampling = CV(nfolds = 10),
                                measure = rmse,
                                verbosity = 0).measurement[]
                   end
                   for degree in 1:10, seed in 1:20]

# ╔═╡ 6599d2b4-68f4-4c22-8e40-bf3722597692
let validlosses  = losses_mlj_cv10, i = argmin(validlosses, dims = 1)
    plot(validlosses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 29683c99-6a6a-4f65-bea2-d592895d887e
md"# Model Tuning

Finding good hyper-parameters (tuning) is such an important step in the process of finding good machine learning models that there exist some nice utility functions to tune the hyper-parameters ([tuning section in the MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/)). In the cell below you see an example where a `PolynomialRegressor` is automatically tuned on a given dataset by performing 10-fold cross-validation on all degrees on a \"grid\", i.e. of all degrees from 1 to 17 are tested."

# ╔═╡ f93f20db-4fed-481f-b085-ca744b68fa8f
begin
    model = PolynomialRegressor()
    data2 = data_generator(seed = 2, n = 100)
    self_tuning_model = TunedModel(model = model,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(),
                                   range = range(model, :degree, values = 1:17),
                                   measure = rmse)
    self_tuning_mach = machine(self_tuning_model, select(data2, :x), data2.y) |> fit!
end

# ╔═╡ d72609a1-f93f-4ca6-8759-727662233e97
md"The self-tuned model actually found the true degree 3 for this data set, as we can see from the report of the tuned machine. The estimated test error can also be found in the report under `best_history_entry.measurement`."

# ╔═╡ 59245b3e-ddfc-46c4-ba44-86ce191672ae
report(self_tuning_mach)

# ╔═╡ f8b48f50-09cb-498a-89d1-9ac9b5722d0c
md"The result of the tuned machine looks quite good here: the reducible error (average distance between red and green curve) looks small."

# ╔═╡ 47a2c14b-a02d-43d7-ac8a-4d95a0d91fa8
let x = -3:.1:3
    scatter(data2.x, data2.y, label = "data", legend = :bottomleft)
    plot!(f, label = "generator", w = 2, xlims = (-3, 3))
    plot!(x, predict(self_tuning_mach, DataFrame(x = x)), label = "self tuning fit", w = 2)
end


# ╔═╡ ee89c448-1e69-4fd1-a4b8-7297a09f2685
md"# Exercises

## Conceptual
1. We review k-fold cross-validation.
    - Explain how k-fold cross-validation is implemented.
    - What are the advantages and disadvantages of k-fold cross-validation relative to:
        - The validation set approach?
        - LOOCV?
2. Suppose you receive the following email of a colleague of yours. Write an answer to this email.
```
    Hi

    In my internship I am doing this research project with a company that
    manufactures constituents for medical devices. We need to assure high
    quality and detect the pieces that are not fully functional. To do so we
    perform 23 measurements on each piece. My goal is to find a machine
    learning method that detects defective pieces automatically. Both, my
    training set and my test set consist of 4000 pieces that were fine and
    30 defective ones.  I ran logistic regression on that data and found a
    training error of 0.1% and a test error of 0.3%, which is already pretty
    good, I think, no? But with kNN classification it is even better. For
    k = 7 I found a test error of 0.05% which is by far lower than all test
    errors I obtained with other values of k. I was really impressed. Now
    we have a method that predicts with 99.95% accuracy whether a piece is
    defective or not!

    Because you are taking this machine learning class now, I wanted to ask you
    for advice. Does it all make sense to you what I described? If not, do you
    have any suggestion to get even better results?

    Ã bientôt
    Jamie
```



## Applied
1. Take the `classification_data` in our notebook on
   \"flexibility and bias-variance-decomposition notebook\" and find with 10-fold
   cross-validation the optimal number ``k`` of neighbors of kNN
   classification, using the AUC measure. Hint: `MLJ` has the builtin function `auc`.
   Plot the validation AUC for ``k = 1, \ldots, 50``.
2. In this exercise you apply our \"recipe for supervised learning\" (see slides). The goal is to predict the miles a car can drive per gallon fuel (mpg) as a function of its horsepower. You can download a dataset with `using OpenML; cars = DataFrame(OpenML.load(455))`. In the cleaning step we will remove all rows that contain missing values (you can use the function `dropmissing`). We select the machine learning methods `PolynomialRegressor` and `KNNRegressor` and we take as measure the `rmse`. Make sure to go trough the steps 2, 5, 9 of the recipe. Plot the predictions of the best method you found.
"

# 1. Perform k-nearest neighbors regression on data generated with our `data_generator`
#    defined in the first cell of this notebook and find the optimal number k of neighbors
#    with k-fold cross validation.


# ╔═╡ 0651292e-3f4e-4263-8235-4caa563403ec
MLCourse.footer()

# ╔═╡ Cell order:
# ╠═736856ce-f490-11eb-3349-057c86edfe7e
# ╟─7eb6a060-f948-4d85-881a-4909e74c15bd
# ╠═7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
# ╠═fa9b4b9e-4e97-4e5c-865d-ad3ef288e4cf
# ╟─751880ec-1a82-4142-b875-177d436bbc72
# ╟─da2bbb4d-d45d-4d2a-9c8a-21395bcbb851
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
# ╠═0ab1238e-8279-43d5-9803-848cb00156c2
# ╟─eaacf529-1727-4743-941b-360c53088b1d
# ╠═967a6e08-f0e6-45b7-988b-d0df237f3ddf
# ╟─535a11d9-833e-4237-949e-9a3e289c600b
# ╠═8ae790b3-3987-4f41-8e21-adbb71081eb9
# ╟─134df37f-b737-4c80-a5f9-1149aeec970c
# ╠═abb71af3-8aae-4806-9d5a-d144c15d22ef
# ╟─6599d2b4-68f4-4c22-8e40-bf3722597692
# ╟─29683c99-6a6a-4f65-bea2-d592895d887e
# ╠═f93f20db-4fed-481f-b085-ca744b68fa8f
# ╟─d72609a1-f93f-4ca6-8759-727662233e97
# ╠═59245b3e-ddfc-46c4-ba44-86ce191672ae
# ╟─f8b48f50-09cb-498a-89d1-9ac9b5722d0c
# ╟─47a2c14b-a02d-43d7-ac8a-4d95a0d91fa8
# ╟─ee89c448-1e69-4fd1-a4b8-7297a09f2685
# ╟─c693088f-7f80-4cdd-b9b5-65a50da732ac
# ╟─e1077092-f72a-42af-b6e0-a616f938cba8
# ╟─0651292e-3f4e-4263-8235-4caa563403ec
