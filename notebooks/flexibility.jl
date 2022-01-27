### A Pluto.jl notebook ###
# v0.17.7

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

# ╔═╡ 8fa836a6-1133-4a54-b996-a02083fc6bba
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using Random, Statistics, DataFrames, Plots, MLJ, MLJLinearModels, MLCourse
	import MLCourse: fitted_linear_func
end

# ╔═╡ 269a609c-74af-4e7e-86df-e2279096a7a6
using NearestNeighborModels

# ╔═╡ 12942f34-efb1-11eb-3eb4-c1a38396cfb8
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 12942f50-efb1-11eb-01c0-055b6be166e0
md"# When Linear Methods Are Not Flexible Enough

We will use the function ``f(x) = 0.3 \sin(10x) + 0.7 x`` to design two artificial
data sets: one for regression with a single predictor and one for classification
with two predictors.

In this section we will see that standard linear regression or classification is not flexible enough to fit the artifical data sets.
"

# ╔═╡ 12942f5a-efb1-11eb-399a-a1300d636217
begin
    f(x) = .3 * sin(10x) + .7x
    # f(x) = .0015*(12x-6)^4 -.035(12x-6)^2 + .5x + .2  # an alternative
	σ(x) = 1 / (1 + exp(-x))
    function regression_data_generator(; n, rng = Random.GLOBAL_RNG)
        x = rand(rng, n)
        DataFrame(x = x, y = f.(x) .+ .1*randn(rng, n))
    end
    function classification_data_generator(; n, rng = Random.GLOBAL_RNG)
        X1 = rand(rng, n)
        X2 = rand(rng, n)
        df = DataFrame(X1 = X1, X2 = X2,
                       y = categorical(σ.(20(f.(X1) .- X2)) .> rand(rng, n),
					                   levels = [false, true], ordered = true))
    end
end;

# ╔═╡ 12942f62-efb1-11eb-3f59-f981bc32f308
regression_data = regression_data_generator(n = 50, rng = MersenneTwister(3))

# ╔═╡ 12942f6e-efb1-11eb-0a49-01a6a2d0196f
m1 = machine(LinearRegressor(), select(regression_data, :x), regression_data.y) |> fit!;

# ╔═╡ 12942f6e-efb1-11eb-35c0-19a7c4d6b44e
begin
    scatter(regression_data.x, regression_data.y, label = "training data")
    plot!(fitted_linear_func(m1), w = 3, xlabel = "X", ylabel = "Y",
          label = "linear regression", legend = :topleft)
end

# ╔═╡ 12942f94-efb1-11eb-234e-51492b51e583
classification_data = classification_data_generator(n = 400, rng = MersenneTwister(8))

# ╔═╡ 12942fa0-efb1-11eb-01c8-6dae80c55fb8
m2 = machine(LogisticClassifier(penalty = :none),
             select(classification_data, Not(:y)),
             classification_data.y) |> fit!;

# ╔═╡ 12942f94-efb1-11eb-2c48-a3418b53b886
begin
    xgrid = MLCourse.grid(0:.01:1, 0:.02:1, names = (:X1, :X2))
    scatter(xgrid.X1, xgrid.X2, color = coerce(predict_mode(m2, xgrid), Count),
            markersize = 2, label = nothing, markerstrokewidth = 0)
    scatter!(classification_data.X1, classification_data.X2,
		     xlabel = "X₁", ylabel = "X₂",
             color = coerce(classification_data.y, Count), label = nothing)
end

# ╔═╡ 2ae86454-1877-4972-9cf6-24ef9350a296
md"The small dots in the background indicate how logistic regression would
would classify test data (with decision threshold 0.5; blue = false, red = true)
and the large points are the training data."

# ╔═╡ d487fcd9-8b45-4237-ab2c-21f82ddf7f7c
md"## Polynomial Regression

One way to increase the flexibility is to fit a polynomial. This can be achieved by running linear regression on a transformed data set. The transformation consists of computing higher powers of the original input. In this course we do this transformation with the static transformation machine `Polynomial`.
"

# ╔═╡ c50ed135-68d5-43bc-9c26-9c265702a1f0
MLJ.transform(machine(Polynomial(degree = 5)), regression_data)

# ╔═╡ 75c5471d-926d-4b05-9002-28b14e1dd428
md"To create a machine that transforms the input to a polynomial representation and run linear regression on this transformed representation, we use an `MLJ.Pipeline`.
Pipelines in MLJ can be written explicitely with `Pipeline(Polynomial(degree = 5), LinearRegressor())` or with the pipe operator `|>`, i.e. `Polynomial(degree = 5) |> LinearRegressor()`.
"

# ╔═╡ 710d3104-e197-44c1-a10b-de1098d57dd6
md"degree = $(@bind degree Slider(1:17, default = 4, show_value = true))"

# ╔═╡ ad50d244-c644-4f61-bd8b-995d0110811d
m3 = machine(Polynomial(degree = degree) |> LinearRegressor(),
             select(regression_data, Not(:y)),
             regression_data.y);

# ╔═╡ e9b0ea86-a5f5-43fa-aa16-5a0240f298dd
fit!(m3, verbosity = 0);

# ╔═╡ 0272460a-5b9f-4728-a531-2b497b26c512
function compute_loss(mach, data, lossfunc; operation = predict)
    pred = operation(mach, select(data, Not(:y)))
    lossfunc(pred, data.y)
end;

# ╔═╡ cfcb8f61-af91-40dd-951a-09e8dbf17e30
begin
    mse(ŷ, y) = mean((ŷ .- y).^2)
    regression_test_data = regression_data_generator(n = 10^4)
    plosses = hcat([let m = fit!(machine(Polynomial(degree = d) |> LinearRegressor(),
                                         select(regression_data, Not(:y)),
                                         regression_data.y), verbosity = 0)
                        [compute_loss(m, regression_data, mse),
                         compute_loss(m, regression_test_data, mse)]
                   end
                   for d in 1:17]...)
end;

# ╔═╡ 6fa9b644-d4a6-4c53-9146-9d978207bfd0
begin
    scatter(regression_data.x, regression_data.y, label = "training data")
    p5 = plot!(0:.01:1, predict(m3, (x = 0:.01:1,)), w = 3,
		       xlabel = "X", ylabel = "Y",
               label = "$degree-polynomial regression", legend = :topleft)
	plot!(f, color = :green, label = "data generator", w = 2)
    p6 = plot(1:17, plosses[1, :], color = :blue, label = "training loss")
    plot!(1:17, plosses[2, :], color = :red, label = "test loss")
    hline!([minimum(plosses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    hline!([.1^2], color = :green, style = :dash, label = "irreducible error")
    scatter!([degree], [plosses[1, degree]], color = :blue, label = nothing)
    scatter!([degree], [plosses[2, degree]], color = :red, label = nothing)
    vline!([argmin(plosses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility (degree)", ylabel = "mse")
    plot(p5, p6, layout = (1, 2), size = (700, 400))
end

# ╔═╡ edfb269d-677e-4687-8bff-0aa9ae6e64c3
md"## Polynomial Classification"

# ╔═╡ 5ea1b31d-91e5-4c8f-93d6-5d31816fdbf5
MLJ.transform(machine(Polynomial(degree = 2, predictors = (:X1, :X2))),
	          classification_data)

# ╔═╡ 59acced5-16eb-49b8-8cf2-0c43a88d838e
md"degree = $(@bind degree2 Slider(1:17, default = 3, show_value = true))"

# ╔═╡ 2fa54070-e261-462d-bd63-c225b92fa876
m4 = machine(Polynomial(degree = degree2, predictors = (:X1, :X2)) |> LogisticClassifier(penalty = :none),
             select(classification_data, Not(:y)),
             classification_data.y);

# ╔═╡ e0acbf00-f6de-483b-902b-31db99298da7
fit!(m4, verbosity = 0);

# ╔═╡ 16f0d1b3-bd97-407d-9a79-25b0fb05bbeb
begin
    classification_test_data = classification_data_generator(n = 10^4)
    cplosses = hcat([let m = fit!(machine(Polynomial(degree = d,
                                                     predictors = (:X1, :X2)) |>
                                          LogisticClassifier(penalty = :none),
                                          select(classification_data, Not(:y)),
                                          classification_data.y), verbosity = 0)
                         [mean(compute_loss(m, classification_data, log_loss)),
                          mean(compute_loss(m, classification_test_data, log_loss))]
                   end
                   for d in 1:17]...)
    c_irred_error = let data = classification_test_data,
                        p = σ.(20(f.(data.X1) .- data.X2))
        mean(log_loss(UnivariateFinite([false, true], p,
			                            augment = true, pool = missing),
			 data.y))
    end
end;

# ╔═╡ ed62cb94-8187-4d50-a6f9-6967893dd021
begin
    scatter(xgrid.X1, xgrid.X2, color = coerce(predict_mode(m4, xgrid), Count),
            markersize = 2, label = nothing, markerstrokewidth = 0,
            xlabel = "X1", ylabel = "X2")
    p7 = scatter!(classification_data.X1, classification_data.X2,
                  color = coerce(classification_data.y, Count), label = nothing)
    plot(1:17, cplosses[1, :], color = :blue, label = "training loss")
    plot!(1:17, cplosses[2, :], color = :red, label = "test loss")
    hline!([c_irred_error], color = :green, style = :dash, label = "irreducible error")
    hline!([minimum(cplosses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    scatter!([degree2], [cplosses[1, degree2]], color = :blue, label = nothing)
    scatter!([degree2], [cplosses[2, degree2]], color = :red, label = nothing)
    p8 = vline!([argmin(cplosses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility (degree)", ylabel = "negative loglikelihood")
    plot(p7, p8, layout = (1, 2), size = (700, 400))
end

# ╔═╡ 5c984615-f123-47fc-8330-66694ab1cb9f
md"# K-Nearest-Neighbor Regression

K = $(@bind K Slider(1:50, show_value = true))
"

# ╔═╡ 12942f82-efb1-11eb-2827-df957759b02c
m12 = machine(KNNRegressor(K = K),
	          select(regression_data, :x),
	          regression_data.y);

# ╔═╡ e1477620-dc57-4e9a-b342-8798cf6aeffe
fit!(m12);

# ╔═╡ 8ba77b77-1016-4f5d-9f9e-76b2ad1f9eac
begin
    losses = hcat([let m = fit!(machine(KNNRegressor(K = k),
                                        select(regression_data, Not(:y)),
                                        regression_data.y), verbosity = 0)
                       [compute_loss(m, regression_data, mse),
                        compute_loss(m, regression_test_data, mse)]
                   end
                   for k in 1:50]...)
end;

# ╔═╡ 542327e9-1599-4a14-805f-5d2a3c4eae14
begin
    scatter(regression_data.x, regression_data.y, label = "training data")
    p1 = plot!(0:.001:1, predict(m12, (x = 0:.001:1,)), w = 3,
               label = "$K-NN fit", legend = :topleft, xlabel = "x", ylabel = "y")
    plot(1 ./ (1:50), losses[1, :], color = :blue, label = "training loss")
    plot!(1 ./ (1:50), losses[2, :], color = :red, label = "test loss")
    hline!([minimum(losses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    hline!([.1^2], color = :green, style = :dash, label = "irreducible error")
    scatter!([1 / K], [losses[1, K]], color = :blue, label = nothing)
    scatter!([1 / K], [losses[2, K]], color = :red, label = nothing)
    p2 = vline!([1 / argmin(losses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility 1/K", ylabel = "mse")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end

# ╔═╡ 12942f8c-efb1-11eb-284c-393f6a694818
md"# K-Nearest-Neighbor Classification

K = $(@bind Kc Slider(1:100, show_value = true))"

# ╔═╡ 12942fc8-efb1-11eb-3180-dff1921c5bf9
m14 = machine(KNNClassifier(K = Kc),
             select(classification_data, Not(:y)),
             classification_data.y);

# ╔═╡ 87468757-49c0-474f-8bea-bd5b37d10161
fit!(m14);

# ╔═╡ 0a57f15b-c292-4c64-986d-f046260da66e
begin
    closses = hcat([let m = fit!(machine(KNNClassifier(K = k),
                                         select(classification_data, Not(:y)),
                                         classification_data.y), verbosity = 0)
                        [mean(compute_loss(m, classification_data, log_loss)),
                         mean(compute_loss(m, classification_test_data, log_loss))]
                   end
                   for k in 1:100]...)
end;

# ╔═╡ 12942fc8-efb1-11eb-0c02-0150ef55ae98
begin
    scatter(xgrid.X1, xgrid.X2, color = coerce(predict_mode(m14, xgrid), Count),
            markersize = 2, label = nothing, markerstrokewidth = 0,
            xlabel = "X1", ylabel = "X2")
    p3 = scatter!(classification_data.X1, classification_data.X2,
                  color = coerce(classification_data.y, Count), label = nothing)
    plot(1 ./ (1:100), closses[1, :], color = :blue, label = "training loss")
    plot!(1 ./ (1:100), closses[2, :], color = :red, label = "test loss")
    hline!([c_irred_error], color = :green, style = :dash, label = "irreducible error")
    hline!([minimum(closses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    scatter!([1 / Kc], [closses[1, Kc]], color = :blue, label = nothing)
    scatter!([1 / Kc], [closses[2, Kc]], color = :red, label = nothing)
    p4 = vline!([1 / argmin(closses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility 1/K", ylabel = "negative loglikelihood")
    plot(p3, p4, layout = (1, 2), size = (700, 400))
end

# ╔═╡ cc8ed1de-beab-43e5-979e-e83df23f96ae
md"## Application to Handwritten Digit Recognition (MNIST)

In the following we are fitting a first nearest-neigbor classifier to the MNIST
data set.

WARNING: The following code takes more than 10 minutes to run.
Especially the prediction is slow, because for every test image the closest out
of 60'000 training images has to be found.

```julia
using OpenML
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => Continuous)
    df[:, 1:end-1] ./ 255,
    df.class
end
m5 = fit!(machine(KNNClassifier(K = 1), mnist_x[1:60000, :], mnist_y[1:60000]))
mnist_errorrate = mean(predict_mode(m5, mnist_x[60001:70000, :]) .!= mnist_y[60001:70000])
```
We find a misclassification rate of approximately 3%. This is clearly better than the
approximately 7.8% obtained with Multinomial Logistic Regression.
"

# ╔═╡ f6093c98-7e89-48ba-95c9-4d1f60a25033
md"# Bias-Variance Decomposition

Below we define a function that fits a polynomial regression of a given `degree` to some `training_data` and evaluates it on some `test_data`. The result is returned in form of a `DataFrame` with a single row.
"

# ╔═╡ 376d62fc-2859-422d-9944-bd9c7929f942
function fit_and_evaluate(degree, training_data, test_data)
    m = fit!(machine(Polynomial(; degree) |> LinearRegressor(),
                     select(training_data, Not(:y)),
                     training_data.y), verbosity = 0)
    ŷ = predict(m, select(test_data, Not(:y)))
    DataFrame(degree = degree,
              training_loss = compute_loss(m, training_data, mse),
              test_loss = compute_loss(m, test_data, mse),
              prediction = Ref(ŷ)) # we use Ref to store the reference to the vector of predictions instead of inserting the predicted values as different rows into the DataFrame.
end;

# ╔═╡ 021f812a-b52a-46a3-b81a-fa1bfefff295
md"Now we will run this function on 9 different polynomial degrees and 100 different training sets. We use always the same (large) test set."

# ╔═╡ 1fd15c67-a196-4324-94f6-9e4ccdc92482
results = vcat([fit_and_evaluate(degree,
                         regression_data_generator(n = 50,
                                                   rng = MersenneTwister(seed)),
                         regression_test_data)
                for degree in 1:9, seed in 1:100]...)

# ╔═╡ bdaa48d6-30bb-4d04-b5dd-cc1825cb69e3
result = combine(groupby(results, :degree),
	             :training_loss => mean,
                 :test_loss => mean,
                 :prediction => (x -> mean(var(x))) => :variance,
                 :prediction => (x -> mean((mean(x) .- f.(regression_test_data.x)).^2)) => :bias)

# ╔═╡ 5ac9bcb4-c5a5-4b53-bb45-73dde74b7f60
begin
    plot(result.degree, result.training_loss_mean,
         label = "training loss", w = 2, ylabel = "loss",
         xlabel = "flexibility (degree)")
	plot!(result.degree, result.test_loss_mean, label = "test loss", w = 2)
    plot!(result.degree, result.bias, label = "bias^2", w = 2)
    plot!(result.degree, result.variance, label = "variance", w = 2)
    plot!(result.degree, result.variance .+ result.bias .+ .1^2, label = "bias^2 + variance + Var(ϵ)", color = :black, w = 2, linestyle = :dash)
    hline!([.1^2], label = "Var(ϵ)")
end

# ╔═╡ ae86ee9c-3645-43d2-9a42-79658521c3fb
md"# Exercises

## Conceptual
1. For each of example below, indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.
   + The sample size n is extremely large, and the number of predictors ``p`` is small.
   + The number of predictors ``p`` is extremely large, and the number of observations ```n`` is small.
   + The relationship between the predictors and response is highly non-linear.
   + The variance of the error terms, i.e. ``\sigma^2 = Var(\epsilon)``, is extremely high.
1. The table below provides a training data set containing six observations, three predictors, and one qualitative response variable. Suppose we wish to use this data set to make a prediction for ``Y`` when ``X_1 = X_2 = X_3 = 0`` using K-nearest neighbors.
   - Compute the Euclidean distance between each observation and the test point, X1 = X2 = X3 = 0.
   - What is our prediction with K = 1? Why?
   - What is our prediction with K = 3? Why?
   - If the decision boundary at threshold 0.5 in this problem is highly non- linear, then would we expect the best value for K to be large or small? Why?
|Obs.|``X_1``|``X_2``|``X_3``|``Y``|
|:----|-------|-------|-------|-----:|
|1 | 0 | 3 | 0 | Red|
|2 | 2 | 0 | 0 | Red|
|3 | 0 | 1 | 3 | Red|
|4 | 0 | 1 | 2 | Green|
|5 | -1| 0 | 1 | Green|
|6 | 1 | 1 | 1 | Red|
3. Try without looking at the figure above:
   - Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The x-axis should represent the amount of flexibility in the method, and the y-axis should represent the values for each curve. There should be five curves. Make sure to label each one.
   - Explain why each of the five curves has its particular shape.
2. Suppose that we take a data set with mutually distinct inputs ``x_i\neq x_j`` for ``i\neq j``, divide it into equally-sized training and test sets, and then try out two different classification procedures. First we use logistic regression and get an error rate of 20% on the training data and 30% on the test data. Next we use 1-nearest neighbors (i.e. ``K = 1``) and get an average error rate (averaged over both test and training data sets) of 18%. Based on these results, which method should we prefer to use for classification of new observations? Why?

## Applied
1. Apply K-nearest neighbors regression to the weather data set. Use as input all predictors except `:time` and `:LUZ_wind_peak`.
    * Compute the training and the test loss for ``K = 5, 10, 20, 50, 100``.
    * Which value of the hyper-parameter ``K`` should we prefer to make predictions on new data?
    * Should we prefer K-nearest neighbors with optimal ``K`` or multiple linear regression to make predictions on new data? *Hint*: Remember that we found a training error (RMSE) of approximately 8.1 and a test error of 8.9.
2. In this exercise we review the error-decomposition and the bias-variance decomposition.
* Write a data generator where the mean of the output depends through the non-linear function ``f(x) = x^2 * \sin(x) + 4 * \tanh(10x)`` on the input and normally distributed noise ``\epsilon`` with mean 0 and standard deviation 1.5.
    * Take the linear function ``\hat f(x) = 2x`` and estimate its reducible error at input point ``x = 0`` and at input point ``x = 2`` in two ways:
        * Using directly ``f``.
        * Using ``10^5`` samples from the data generator and your knowledge about the irreducible error.
    * Generate ``10^4`` training sets of 100 data points with input ``x`` normally distributed with standard deviation 2 and mean 0 and estimate the bias of linear regression at ``x = 4``  in two ways:
        * Using directly ``f``.
        * Using ``10^4`` samples from the data generator, your knowledge about the irreducible error and your estimate of the variance of linear regression.
"

# ╔═╡ efda845a-4390-40bc-bdf2-89e555d3b1b2
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 2320f424-7652-4e9f-83ef-fc011b722dcc
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─12942f50-efb1-11eb-01c0-055b6be166e0
# ╠═8fa836a6-1133-4a54-b996-a02083fc6bba
# ╠═12942f5a-efb1-11eb-399a-a1300d636217
# ╠═12942f62-efb1-11eb-3f59-f981bc32f308
# ╠═12942f6e-efb1-11eb-0a49-01a6a2d0196f
# ╟─12942f6e-efb1-11eb-35c0-19a7c4d6b44e
# ╠═12942f94-efb1-11eb-234e-51492b51e583
# ╠═12942fa0-efb1-11eb-01c8-6dae80c55fb8
# ╟─12942f94-efb1-11eb-2c48-a3418b53b886
# ╟─2ae86454-1877-4972-9cf6-24ef9350a296
# ╟─d487fcd9-8b45-4237-ab2c-21f82ddf7f7c
# ╠═c50ed135-68d5-43bc-9c26-9c265702a1f0
# ╟─75c5471d-926d-4b05-9002-28b14e1dd428
# ╟─710d3104-e197-44c1-a10b-de1098d57dd6
# ╠═ad50d244-c644-4f61-bd8b-995d0110811d
# ╠═e9b0ea86-a5f5-43fa-aa16-5a0240f298dd
# ╟─6fa9b644-d4a6-4c53-9146-9d978207bfd0
# ╠═0272460a-5b9f-4728-a531-2b497b26c512
# ╠═cfcb8f61-af91-40dd-951a-09e8dbf17e30
# ╟─edfb269d-677e-4687-8bff-0aa9ae6e64c3
# ╠═5ea1b31d-91e5-4c8f-93d6-5d31816fdbf5
# ╟─59acced5-16eb-49b8-8cf2-0c43a88d838e
# ╠═2fa54070-e261-462d-bd63-c225b92fa876
# ╠═e0acbf00-f6de-483b-902b-31db99298da7
# ╟─ed62cb94-8187-4d50-a6f9-6967893dd021
# ╠═16f0d1b3-bd97-407d-9a79-25b0fb05bbeb
# ╟─5c984615-f123-47fc-8330-66694ab1cb9f
# ╠═269a609c-74af-4e7e-86df-e2279096a7a6
# ╠═12942f82-efb1-11eb-2827-df957759b02c
# ╠═e1477620-dc57-4e9a-b342-8798cf6aeffe
# ╟─542327e9-1599-4a14-805f-5d2a3c4eae14
# ╟─8ba77b77-1016-4f5d-9f9e-76b2ad1f9eac
# ╟─12942f8c-efb1-11eb-284c-393f6a694818
# ╠═12942fc8-efb1-11eb-3180-dff1921c5bf9
# ╠═87468757-49c0-474f-8bea-bd5b37d10161
# ╟─12942fc8-efb1-11eb-0c02-0150ef55ae98
# ╟─0a57f15b-c292-4c64-986d-f046260da66e
# ╟─cc8ed1de-beab-43e5-979e-e83df23f96ae
# ╟─f6093c98-7e89-48ba-95c9-4d1f60a25033
# ╠═376d62fc-2859-422d-9944-bd9c7929f942
# ╟─021f812a-b52a-46a3-b81a-fa1bfefff295
# ╠═1fd15c67-a196-4324-94f6-9e4ccdc92482
# ╠═bdaa48d6-30bb-4d04-b5dd-cc1825cb69e3
# ╟─5ac9bcb4-c5a5-4b53-bb45-73dde74b7f60
# ╟─ae86ee9c-3645-43d2-9a42-79658521c3fb
# ╟─efda845a-4390-40bc-bdf2-89e555d3b1b2
# ╟─12942f34-efb1-11eb-3eb4-c1a38396cfb8
# ╟─2320f424-7652-4e9f-83ef-fc011b722dcc
