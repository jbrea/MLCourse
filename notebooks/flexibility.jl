### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 12942f34-efb1-11eb-3eb4-c1a38396cfb8
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI, Plots, MLJ, MLJLinearModels, NearestNeighborModels, DataFrames, Random, Statistics
    gr()
    PlutoUI.TableOfContents()
end

# ╔═╡ 6d84c382-27d5-47bd-97a9-88153d20b2fc
begin
    using MLJOpenML
    mnist_x, mnist_y = let df = MLJOpenML.load(554) |> DataFrame
		coerce!(df, :class => Multiclass)
        coerce!(df, Count => Continuous)
		df[:, 1:end-1] ./ 255,
        df.class
	end
end;

# ╔═╡ efda845a-4390-40bc-bdf2-89e555d3b1b2
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 12942f50-efb1-11eb-01c0-055b6be166e0
md"# When Linear Methods Are Not Flexible Enough

We will use the function ``f(x) = 0.3 \sin(10x) + 0.7 x`` to design two artificial
data sets: one for regression with a single predictor and one for classification
with two predictors.
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
                       y = σ.(20(f.(X1) .- X2)) .> rand(rng, n))
        coerce!(df, :y => Multiclass)
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

"

# ╔═╡ 48bcf292-4d5a-45e8-a495-26404d221bd9
begin
    polysymbol(x, d) = Symbol(d == 0 ? "" : d == 1 ? "$x" : "$x^$d")
    function poly(data, degree)
        res = DataFrame([data.x .^ k for k in 1:degree],
                        [polysymbol("x", k) for k in 1:degree])
        if hasproperty(data, :y)
            res.y = data.y
        end
        res
    end
end;

# ╔═╡ c50ed135-68d5-43bc-9c26-9c265702a1f0
poly(regression_data, 5)

# ╔═╡ 710d3104-e197-44c1-a10b-de1098d57dd6
md"degree = $(@bind degree Slider(1:17, default = 4, show_value = true))"

# ╔═╡ ad50d244-c644-4f61-bd8b-995d0110811d
m3 = machine(LinearRegressor(),
             select(poly(regression_data, degree), Not(:y)),
             regression_data.y) |> fit!;

# ╔═╡ 0272460a-5b9f-4728-a531-2b497b26c512
function loss(mach, data, lossfunc; operation = predict)
    pred = operation(mach, select(data, Not(:y)))
    lossfunc(pred, data.y)
end;

# ╔═╡ cfcb8f61-af91-40dd-951a-09e8dbf17e30
begin
    mse(ŷ, y) = mean((ŷ .- y).^2)
    regression_test_data = regression_data_generator(n = 10^4)
    plosses = hcat([let m = fit!(machine(LinearRegressor(),
                                         select(poly(regression_data, d), Not(:y)),
                                         regression_data.y))
                        [loss(m, poly(regression_data, d), mse),
                         loss(m, poly(regression_test_data, d), mse)]
                   end
                   for d in 1:17]...)
end;

# ╔═╡ 6fa9b644-d4a6-4c53-9146-9d978207bfd0
begin
    scatter(regression_data.x, regression_data.y, label = "training data")
    p5 = plot!(0:.01:1, predict(m3, poly((x = 0:.01:1,), degree)), w = 3,
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

# ╔═╡ 0b246590-9b1f-4e15-9ff2-7e2dd1121518
function poly2(data, degree)
    res = DataFrame([data.X1 .^ d1 .* data.X2 .^ d2
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree],
                    [Symbol(polysymbol("x₁", d1), polysymbol("x₂", d2))
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end;

# ╔═╡ 5ea1b31d-91e5-4c8f-93d6-5d31816fdbf5
poly2(classification_data, 3)

# ╔═╡ 59acced5-16eb-49b8-8cf2-0c43a88d838e
md"degree = $(@bind degree2 Slider(1:17, default = 3, show_value = true))"

# ╔═╡ 2fa54070-e261-462d-bd63-c225b92fa876
m4 = machine(LogisticClassifier(penalty = :none),
             select(poly2(classification_data, degree2), Not(:y)),
             classification_data.y) |> fit!;

# ╔═╡ 16f0d1b3-bd97-407d-9a79-25b0fb05bbeb
begin
    classification_test_data = classification_data_generator(n = 10^4)
    cplosses = hcat([let m = fit!(machine(LogisticClassifier(penalty = :none),
                                          select(poly2(classification_data, d), Not(:y)),
                                          classification_data.y))
                         [mean(loss(m, poly2(classification_data, d), log_loss)),
                          mean(loss(m, poly2(classification_test_data, d), log_loss))]
                   end
                   for d in 1:17]...)
    c_irred_error = let data = classification_test_data,
                        p = σ.(20(f.(data.X1) .- data.X2))
        mean(log_loss(UnivariateFinite([false, true], p, augment = true), data.y))
    end
end;

# ╔═╡ ed62cb94-8187-4d50-a6f9-6967893dd021
begin
    scatter(xgrid.X1, xgrid.X2, color = coerce(predict_mode(m4, poly2(xgrid, degree2)), Count),
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
m12 = machine(KNNRegressor(K = K), select(regression_data, :x), regression_data.y) |> fit!;

# ╔═╡ 8ba77b77-1016-4f5d-9f9e-76b2ad1f9eac
begin
    losses = hcat([let m = fit!(machine(KNNRegressor(K = k),
                                        select(regression_data, Not(:y)),
                                        regression_data.y), verbosity = 0)
                       [loss(m, regression_data, mse),
                        loss(m, regression_test_data, mse)]
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
             classification_data.y) |> fit!;

# ╔═╡ 0a57f15b-c292-4c64-986d-f046260da66e
begin
    closses = hcat([let m = fit!(machine(KNNClassifier(K = k),
                                         select(classification_data, Not(:y)),
                                         classification_data.y), verbosity = 0)
                        [mean(loss(m, classification_data, log_loss)),
                         mean(loss(m, classification_test_data, log_loss))]
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
"

# ╔═╡ b26387d0-36b6-4001-8718-afa3a2b49db1
m5 = fit!(machine(KNNClassifier(K = 1), mnist_x[1:5000, :], mnist_y[1:5000]));

# ╔═╡ be295d99-17e1-4689-bc62-91495787edf9
mnist_errorrate = mean(predict_mode(m5, mnist_x[5001:10^4, :]) .!= mnist_y[5001:10^4])

# ╔═╡ 69f82636-1371-46ee-8617-6d475e2e366a
md"The average mis-classification rate of 1-NN on MNIST is approximately $mnist_errorrate."

# ╔═╡ f6093c98-7e89-48ba-95c9-4d1f60a25033
md"# Bias-Variance Decomposition"

# ╔═╡ 376d62fc-2859-422d-9944-bd9c7929f942
function fit_and_evaluate(degree, training_data, test_data)
    training_data_poly = poly(training_data, degree)
    test_data_poly = poly(test_data, degree)
    m = fit!(machine(LinearRegressor(),
                     select(training_data_poly, Not(:y)),
                     training_data_poly.y), verbosity = 0)
    ŷ = predict(m, select(test_data_poly, Not(:y)))
    DataFrame(degree = degree,
              training_loss = loss(m, training_data_poly, mse),
              test_loss = loss(m, test_data_poly, mse),
              prediction = Ref(ŷ))
end;

# ╔═╡ 1fd15c67-a196-4324-94f6-9e4ccdc92482
results = vcat([fit_and_evaluate(degree,
                         regression_data_generator(n = 50,
                                                   rng = MersenneTwister(seed)),
                         regression_test_data)
                for degree in 1:9, seed in 1:1000]...)

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
1. Apply K-nearest neighbors regression to the weather data set and compare the result you obtain to the one of linear regression.
"

# ╔═╡ Cell order:
# ╟─12942f50-efb1-11eb-01c0-055b6be166e0
# ╠═12942f5a-efb1-11eb-399a-a1300d636217
# ╠═12942f62-efb1-11eb-3f59-f981bc32f308
# ╠═12942f6e-efb1-11eb-0a49-01a6a2d0196f
# ╟─12942f6e-efb1-11eb-35c0-19a7c4d6b44e
# ╠═12942f94-efb1-11eb-234e-51492b51e583
# ╠═12942fa0-efb1-11eb-01c8-6dae80c55fb8
# ╟─12942f94-efb1-11eb-2c48-a3418b53b886
# ╟─2ae86454-1877-4972-9cf6-24ef9350a296
# ╟─d487fcd9-8b45-4237-ab2c-21f82ddf7f7c
# ╟─48bcf292-4d5a-45e8-a495-26404d221bd9
# ╠═c50ed135-68d5-43bc-9c26-9c265702a1f0
# ╟─710d3104-e197-44c1-a10b-de1098d57dd6
# ╠═ad50d244-c644-4f61-bd8b-995d0110811d
# ╟─6fa9b644-d4a6-4c53-9146-9d978207bfd0
# ╟─0272460a-5b9f-4728-a531-2b497b26c512
# ╟─cfcb8f61-af91-40dd-951a-09e8dbf17e30
# ╟─edfb269d-677e-4687-8bff-0aa9ae6e64c3
# ╟─0b246590-9b1f-4e15-9ff2-7e2dd1121518
# ╠═5ea1b31d-91e5-4c8f-93d6-5d31816fdbf5
# ╟─59acced5-16eb-49b8-8cf2-0c43a88d838e
# ╠═2fa54070-e261-462d-bd63-c225b92fa876
# ╟─ed62cb94-8187-4d50-a6f9-6967893dd021
# ╟─16f0d1b3-bd97-407d-9a79-25b0fb05bbeb
# ╟─5c984615-f123-47fc-8330-66694ab1cb9f
# ╠═12942f82-efb1-11eb-2827-df957759b02c
# ╟─542327e9-1599-4a14-805f-5d2a3c4eae14
# ╟─8ba77b77-1016-4f5d-9f9e-76b2ad1f9eac
# ╟─12942f8c-efb1-11eb-284c-393f6a694818
# ╠═12942fc8-efb1-11eb-3180-dff1921c5bf9
# ╟─12942fc8-efb1-11eb-0c02-0150ef55ae98
# ╟─0a57f15b-c292-4c64-986d-f046260da66e
# ╟─cc8ed1de-beab-43e5-979e-e83df23f96ae
# ╠═6d84c382-27d5-47bd-97a9-88153d20b2fc
# ╠═b26387d0-36b6-4001-8718-afa3a2b49db1
# ╠═be295d99-17e1-4689-bc62-91495787edf9
# ╟─69f82636-1371-46ee-8617-6d475e2e366a
# ╟─f6093c98-7e89-48ba-95c9-4d1f60a25033
# ╠═376d62fc-2859-422d-9944-bd9c7929f942
# ╠═1fd15c67-a196-4324-94f6-9e4ccdc92482
# ╠═bdaa48d6-30bb-4d04-b5dd-cc1825cb69e3
# ╟─5ac9bcb4-c5a5-4b53-bb45-73dde74b7f60
# ╟─ae86ee9c-3645-43d2-9a42-79658521c3fb
# ╟─efda845a-4390-40bc-bdf2-89e555d3b1b2
# ╟─12942f34-efb1-11eb-3eb4-c1a38396cfb8
