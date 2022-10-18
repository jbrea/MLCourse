### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 736856ce-f490-11eb-3349-057c86edfe7e
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLJ, MLJLinearModels, Plots, DataFrames, Random
end

# ╔═╡ c693088f-7f80-4cdd-b9b5-65a50da732ac
begin
    using MLCourse
    import MLCourse: poly, Polynomial
    MLCourse.list_notebooks(@__FILE__)
end


# ╔═╡ e1077092-f72a-42af-b6e0-a616f938cba8
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 7eb6a060-f948-4d85-881a-4909e74c15bd
md"# Validation Set Approach

In the following cell we define a `data_generator` that creates data sets with a polynomial relationship of degree 3 between the input x and the average output f(x).
"

# ╔═╡ 7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
begin
    f(x) = 0.3 + 2x - 0.8x^2 - 0.4x^3
    function data_generator(; n = 500, seed = 12)
        rng = MersenneTwister(seed)
        x = randn(rng, n)
        DataFrame(x = x, y = f.(x) .+ randn(rng, n))
    end
end;

# ╔═╡ 5049407c-bf93-41a5-97bd-863a69d7016a
md"## Test Error Estimation

We will estimate here the test error with the validation set approach for fixed hyper-parameters."

# ╔═╡ 46827e21-0eea-4ec9-83ed-05e41dda1502
data1 = data_generator(seed = 1);

# ╔═╡ 6cfc32b3-40c0-4149-ba67-2ec67b3938f3
md"split seed = $(@bind split_seed Slider(1:20, show_value = true))"

# ╔═╡ 1320b9b0-d6fc-4e61-8d62-de3988b8b21d
idxs = shuffle(MersenneTwister(split_seed), 1:500) # shuffle the indices

# ╔═╡ 5dc21667-d94f-4c2a-b217-80b9a2262d8c
begin
	data1_train = data1[idxs[1:250], :]
	data1_test = data1[idxs[251:end], :]
	mach = machine(Polynomial(degree = 8) |> LinearRegressor(),
	               select(data1_train, :x), data1_train.y)
	fit!(mach, verbosity = 0)
	(training_error = rmse(predict(mach), data1_train.y),
	 test_error = rmse(predict(mach, select(data1_test, :x)), data1_test.y))
end

# ╔═╡ 1fe3f6d1-6336-417e-885e-de3fc5821bdf
let
	scatter(data1_train.x, data1_train.y, label = "training set")
	scatter!(data1_test.x, data1_test.y, label = "test set")
	xgrid = -4:.1:4
	pred = predict(mach, DataFrame(x = xgrid))
	plot!(xgrid, pred, label = "fit", ylim = (-6, 5), xlim = (-4, 4))
end

# ╔═╡ b79fae90-87f3-4f89-933b-d82e76c94d81
md"We see that the resulting curve and the test error depends strongly on the training set, i.e. the indices that are selected for the training set. The training error is always close to the irreducible error in this case."

# ╔═╡ 7d30f043-2915-4bba-aad5-cb7dbe76a3e5
md"## Hyper-parameter tuning

Now we will use a nested validation set approach to find the optimal degree in polynomial regression and estimate the test error. For this we will split the data into training, validation and test set. For each hyper-parameter value we will fit the parameters on the training set and estimate the test error on the validation set. When we have found the best hyper-parameters based on the test error computed on the validation set, we will estimate the test error of the best model using the test set."

# ╔═╡ ffffa3d0-bffb-4ff8-9966-c3312f952ac5
begin
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
end

# ╔═╡ 2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
losses1 = [fit_and_evaluate(Polynomial(; degree) |> LinearRegressor(),
	                        data_split(data1))
           for degree in 1:10]

# ╔═╡ 91eecd2b-af18-4a63-9684-28950e604d1a
let validlosses = getproperty.(losses1, :valid), i = argmin(validlosses)
    plot(1:10, validlosses, label = nothing)
    scatter!([i], [validlosses[i]], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ b45b32b1-8a65-4823-b3bb-f0b7cc57604b
md"For this seed of the random number generator the optimal degree (x coordinate of the red point in the figure above) found by the validation set approach is close to the actual degree of the data generating process. The estimated test loss of the model with degree $(argmin((x -> x.valid).(losses1))) is $(losses1[argmin((x -> x.valid).(losses1))].test)."

# ╔═╡ 23e91f72-0124-48d4-8b53-bd5da67b7ac7
losses1[4]

# ╔═╡ 4461d5b0-bc9b-4c89-aa76-70524a5caa7c
md"Let us repeat the process above for other splits of the data."

# ╔═╡ ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
losses = [fit_and_evaluate(Polynomial(; degree) |> LinearRegressor(),
                           data_split(data1, shuffle = true))
          for degree in 1:10, _ in 1:20]

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

Here we would like to find the optimal hyper-parameters with cross-validation.
We will not estimate the final test error of the model with the best hyper-parameters.

In the following cell we implement cross-validation."

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
    function cross_validation(model, data; K = 5, shuffle = false)
		idxs = 1:size(data, 1)
		if shuffle
            idxs = Random.shuffle(idxs)
		end
        losses = [fit_and_evaluate(model,
                                   data_split(data;
									          idx_test = [], # no test set
									          idxs...)) # training and validation
                  for idxs in cross_validation_sets(idxs, K)]
        (train = mean(getproperty.(losses, :train)),
         valid = mean(getproperty.(losses, :valid)))
    end
end

# ╔═╡ 6916273a-d116-4173-acaa-2bcac1d1753b
cross_validation_sets(1:100, 4)

# ╔═╡ c14c9e49-a993-456e-9115-97da86f8e498
losses1_cv10 = [cross_validation(Polynomial(; degree) |> LinearRegressor(),
		                         data1,
		                         K = 10) for degree in 1:10]

# ╔═╡ 4b435518-2f12-4921-bb1f-fdd049ddfaed
let validlosses = getproperty.(losses1_cv10, :valid), i = argmin(validlosses)
    plot(1:10, validlosses, label = nothing)
    scatter!([i], [validlosses[i]], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 1e584a38-2fef-4877-87f6-92237d71c4b3
losses_cv10 = [cross_validation(Polynomial(; degree) |> LinearRegressor(),
                                data1,
                                K = 10, shuffle = true)
               for degree in 1:10, _ in 1:20]

# ╔═╡ a7c88b3f-92cb-4253-a889-c78683722c1d
let validlosses = getproperty.(losses_cv10, :valid), i = argmin(validlosses, dims = 1)
    plot(validlosses, label = nothing, ylims = (0.5, 2))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 0b81c3d5-277a-4fe6-889b-550e2f83c39d
md"With cross-validation we find often degree = 3 as the optimal one."

# ╔═╡ 5747d116-1e61-45f0-a87b-89372c6f270f
md"### Tuning Hyper-parameters with cross-validation and estimating the test error with the validation set approach

Instead of using all data for hyper-parameter search using cross-validation, we could set some data aside as a test set and use this to estimate the test error of the model with the optimal hyper-parameters."

# ╔═╡ 8204d9da-855d-4c66-b4dd-2c18a17b539b
losses2_cv10 = [cross_validation(Polynomial(; degree) |> LinearRegressor(),
		                         data1[1:100, :],
		                         K = 10) for degree in 1:10]

# ╔═╡ 96b3c070-0c72-44c7-ac98-ff6cb7e82380
md"The optimal degree found with cross-validation on the first 100 data points is:"

# ╔═╡ bae635b1-6eba-4ba4-80e9-547770612843
argmin((x -> x.valid).(losses2_cv10))

# ╔═╡ d9b04582-3a7a-4e19-a0da-d363810883f7
md"Let us refit the model with the optimal degree on all data that was used for cross-validation and estimate it's test error on data points 101 to 500."

# ╔═╡ 9042e811-9a34-42b8-8069-877f3b3f1e75
let 
	model = Polynomial(degree = 4) |> LinearRegressor()
	mach = machine(model, select(data1[1:100, :], :x), data1[1:100, :y])
	fit!(mach, verbosity = 0)
	testset = data1[101:end, :]
	(test_error = rmse(predict(mach, select(testset, :x)), testset.y),)
end

# ╔═╡ e2658128-4053-4484-8e1e-229eceb755ab
md"# Resampling Strategies with MLJ"

# ╔═╡ eaacf529-1727-4743-941b-360c53088b1d
md"Instead of our own cross-validation function we can also use the builtin functions of MLJ. The `train_test_pairs` function does basically the same as our `cross_validation_sets` function."

# ╔═╡ 967a6e08-f0e6-45b7-988b-d0df237f3ddf
MLJ.MLJBase.train_test_pairs(CV(nfolds = 10), 1:100)

# ╔═╡ 535a11d9-833e-4237-949e-9a3e289c600b
md"`MLJ` has the very useful function `evaluate!`. Have a look at the Live docs to learn more about this function. In the following you see an example where we perform 10-fold cross-validation on the `rmse` measure to estimate the expected test error for a polynomial regressor of degree 4."

# ╔═╡ 8ae790b3-3987-4f41-8e21-adbb71081eb9
evaluate!(machine(Polynomial(degree = 4) |> LinearRegressor(),
	              select(data1, :x), data1.y),
          resampling = CV(nfolds = 10), measure = rmse)

# ╔═╡ 134df37f-b737-4c80-a5f9-1149aeec970c
md"We can use this now to find the best degree with 10-fold cross-validation for 100 different seeds. You will see that - even though not perfect - the hyper-parameters found with 10-fold cross-validation have lower variance and are usually closer to the true value than with the validation set approach."

# ╔═╡ abb71af3-8aae-4806-9d5a-d144c15d22ef
losses_mlj_cv10 = [evaluate!(machine(Polynomial(; degree) |> LinearRegressor(),
                                     select(data1, :x), data1.y),
                             resampling = CV(nfolds = 10, shuffle = true),
                             measure = rmse,
                             verbosity = 0).measurement[]
                   for degree in 1:10, seed in 1:20]

# ╔═╡ 6599d2b4-68f4-4c22-8e40-bf3722597692
let validlosses  = losses_mlj_cv10, i = argmin(validlosses, dims = 1)
    plot(validlosses, label = nothing, ylims = (0.5, 2))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ fc07009a-eeac-4c6b-9559-1e99ae68a6c3
md"MLJ implements the following resampling strategies. With `Holdout` we use the validation set approach."

# ╔═╡ 42511b31-6d70-495e-8362-01a29143b96e
subtypes(ResamplingStrategy)

# ╔═╡ 29683c99-6a6a-4f65-bea2-d592895d887e
md"# Model Tuning

Finding good hyper-parameters (tuning) is such an important step in the process of finding good machine learning models that there exist some nice utility functions to tune the hyper-parameters ([tuning section in the MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/)). In the cell below you see an example where the degree of polynomial regression is automatically tuned on a given dataset by performing 10-fold cross-validation on all degrees on a \"grid\", i.e. of all degrees from 1 to 17 are tested."

# ╔═╡ f93f20db-4fed-481f-b085-ca744b68fa8f
begin
    model = Polynomial() |> LinearRegressor()
    data2 = data_generator(seed = 2, n = 100)
    self_tuning_model = TunedModel(model = model, # the model to be tuned
                                   resampling = CV(nfolds = 10), # how to evaluate
                                   range = range(model, :(polynomial.degree),
                                                 values = 1:17), # see below
                                   measure = rmse) # evaluation measure
    self_tuning_mach = machine(self_tuning_model, select(data2, :x), data2.y)
	fit!(self_tuning_mach, verbosity = 0)
end

# ╔═╡ a41302ae-e2d8-4e5d-8dac-198bed16217b
md"The `range(model, :hyper, values = ...)` function in the cell above returns a `NominalRange(polynomial.degree = 1, 2, 3, ...)` to the tuner. This specifies the degrees of the polynomial that should be tested during tuning.

In general, for any given model you may first want to find out, how the parameters are called that you would like to tune. This can be done by inspecting the model.
In the output below you see that the model is a `DeterministicPipeline`, with a `polynomial` and a `linear_regressor`. To tune the degree of the `polynomial` we choose `:(polynomial.degree) in the range function above."

# ╔═╡ 7cab21da-6ccc-474a-adb6-a544ba7269e8
model

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


# ╔═╡ cf3841ba-3963-4716-8fc9-8cce1dc4a5fa
md"# Nested Cross-Validation

If we want to tune the hyper-parameters with cross-validation and care about the estimated test error of the model found with cross-validation, we can use nested cross-validation, where hyper-parameters are optimized with cross-validation for each fold of the outer cross-validation. This happens when we `evaluate!` with cross-validation a self-tuning machine that itself uses cross-validation for hyper-parameter tuning. If we used `resampling = Holdout(fraction_train = 0.5)` we would use the validation set approach to estimate the test error with 50% of the data in the training set. If we used `resampling = Holdout(fraction_train = 0.5)` in the self-tuning machine, we would use the validation set approach for tuning the hyper-parameters.
"

# ╔═╡ 3e9c6f1c-4cb6-48d8-8119-07a4b03c2e4b
nested_cv = evaluate!(machine(self_tuning_model, select(data1, :x), data1.y),
                      resampling = CV(nfolds = 5), measure = rmse)

# ╔═╡ eba90af0-bf67-48f8-8d3c-f831b4fc289a
md"Have a look at the best models found for each fold:"

# ╔═╡ c8d26ab6-86ec-4b37-9540-0f785fd8cdc2
nested_cv.report_per_fold

# ╔═╡ ee89c448-1e69-4fd1-a4b8-7297a09f2685
md"# Exercises

## Conceptual
#### Exercise 1
We review k-fold cross-validation.
- Explain how k-fold cross-validation is implemented.
- What are the advantages and disadvantages of k-fold cross-validation relative to:
    - The validation set approach?
    - LOOCV?
#### Exercise 2
You are given a model with unknown hyper-parameters and the goal is to find the optimal hyper-parameters and estimate the test error of the optimal model. Suppose of colleague of yours argues that this should be done in the following way: \"Take all the data and run cross-validation to find the best hyper-parameters. Taking all data is better, than taking only a subset, because one has more data to estimate the parameters and estimate the test error on the validation sets. Once the best hyper-parameters are found: estimate the test error by running again cross-validation on all the data with the hyper-parameter fixed to the best value.\" Do you agree with your colleague? If not, explain where you disagree and where you agree.
#### Exercise 3
Suppose you receive the following email of a colleague of yours. Write an answer to this email.
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
#### Exercise 4
Take the `classification_data` in our notebook on
   \"flexibility and bias-variance-decomposition notebook\" and find with 10-fold
   cross-validation the optimal number ``k`` of neighbors of kNN
   classification, using the AUC measure. Hint: `MLJ` has the builtin function `auc`.
   Plot the validation AUC for ``k = 1, \ldots, 50``.
"

# 1. Perform k-nearest neighbors regression on data generated with our `data_generator`
#    defined in the first cell of this notebook and find the optimal number k of neighbors
#    with k-fold cross validation.


# ╔═╡ 186fa41b-5e74-4191-bc2d-e8d865606fc1
md"
#### Exercise 5
With the same data as in the previous exercise, use the `MLJ` function `evaluate!` and a self tuning machine to estimate with the validation set approach the test error of kNN classifier whose hyper-parameter is tuned with 5 fold cross-validation. Use one quarter of the data for the test set.
"

# ╔═╡ 6f3fea58-8587-4b03-a11a-bac8a46abe67
md"
#### Exercise 6
In this exercise you apply our \"recipe for supervised learning\" (see slides). The goal is to predict the miles a car can drive per gallon fuel (mpg) as a function of its horsepower. You can download a dataset with `using OpenML; cars = DataFrame(OpenML.load(455))`. In the cleaning step we will remove all rows that contain missing values (you can use the function `dropmissing`). We select the machine learning methods polynomial regression and k nearest neighbors regression and we take as measure the `rmse`. Make sure to go trough the steps 2, 5, 9 of the recipe. Plot the predictions of the best method you found. *Hint:* use `TuningModels` and the `range` function (see examples above) to tune the hyper-parameters.
"

# ╔═╡ 0651292e-3f4e-4263-8235-4caa563403ec
MLCourse.footer()

# ╔═╡ Cell order:
# ╠═736856ce-f490-11eb-3349-057c86edfe7e
# ╟─7eb6a060-f948-4d85-881a-4909e74c15bd
# ╠═7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
# ╟─5049407c-bf93-41a5-97bd-863a69d7016a
# ╠═46827e21-0eea-4ec9-83ed-05e41dda1502
# ╟─6cfc32b3-40c0-4149-ba67-2ec67b3938f3
# ╠═1320b9b0-d6fc-4e61-8d62-de3988b8b21d
# ╠═5dc21667-d94f-4c2a-b217-80b9a2262d8c
# ╟─1fe3f6d1-6336-417e-885e-de3fc5821bdf
# ╟─b79fae90-87f3-4f89-933b-d82e76c94d81
# ╟─7d30f043-2915-4bba-aad5-cb7dbe76a3e5
# ╠═ffffa3d0-bffb-4ff8-9966-c3312f952ac5
# ╠═2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
# ╟─91eecd2b-af18-4a63-9684-28950e604d1a
# ╟─b45b32b1-8a65-4823-b3bb-f0b7cc57604b
# ╠═23e91f72-0124-48d4-8b53-bd5da67b7ac7
# ╟─4461d5b0-bc9b-4c89-aa76-70524a5caa7c
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
# ╟─0b81c3d5-277a-4fe6-889b-550e2f83c39d
# ╟─5747d116-1e61-45f0-a87b-89372c6f270f
# ╠═8204d9da-855d-4c66-b4dd-2c18a17b539b
# ╟─96b3c070-0c72-44c7-ac98-ff6cb7e82380
# ╠═bae635b1-6eba-4ba4-80e9-547770612843
# ╟─d9b04582-3a7a-4e19-a0da-d363810883f7
# ╠═9042e811-9a34-42b8-8069-877f3b3f1e75
# ╟─e2658128-4053-4484-8e1e-229eceb755ab
# ╟─eaacf529-1727-4743-941b-360c53088b1d
# ╠═967a6e08-f0e6-45b7-988b-d0df237f3ddf
# ╟─535a11d9-833e-4237-949e-9a3e289c600b
# ╠═8ae790b3-3987-4f41-8e21-adbb71081eb9
# ╟─134df37f-b737-4c80-a5f9-1149aeec970c
# ╠═abb71af3-8aae-4806-9d5a-d144c15d22ef
# ╟─6599d2b4-68f4-4c22-8e40-bf3722597692
# ╟─fc07009a-eeac-4c6b-9559-1e99ae68a6c3
# ╠═42511b31-6d70-495e-8362-01a29143b96e
# ╟─29683c99-6a6a-4f65-bea2-d592895d887e
# ╠═f93f20db-4fed-481f-b085-ca744b68fa8f
# ╟─a41302ae-e2d8-4e5d-8dac-198bed16217b
# ╠═7cab21da-6ccc-474a-adb6-a544ba7269e8
# ╟─d72609a1-f93f-4ca6-8759-727662233e97
# ╠═59245b3e-ddfc-46c4-ba44-86ce191672ae
# ╟─f8b48f50-09cb-498a-89d1-9ac9b5722d0c
# ╟─47a2c14b-a02d-43d7-ac8a-4d95a0d91fa8
# ╟─cf3841ba-3963-4716-8fc9-8cce1dc4a5fa
# ╠═3e9c6f1c-4cb6-48d8-8119-07a4b03c2e4b
# ╟─eba90af0-bf67-48f8-8d3c-f831b4fc289a
# ╠═c8d26ab6-86ec-4b37-9540-0f785fd8cdc2
# ╟─ee89c448-1e69-4fd1-a4b8-7297a09f2685
# ╟─186fa41b-5e74-4191-bc2d-e8d865606fc1
# ╟─6f3fea58-8587-4b03-a11a-bac8a46abe67
# ╟─c693088f-7f80-4cdd-b9b5-65a50da732ac
# ╟─e1077092-f72a-42af-b6e0-a616f938cba8
# ╟─0651292e-3f4e-4263-8235-4caa563403ec
