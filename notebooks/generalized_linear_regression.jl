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

# ╔═╡ 94f8e29e-ef91-11eb-1ae9-29bc46fa505a
begin
	using Pkg
	joinpath(Pkg.devdir(), "MLCourse")
	using PlutoUI, Plots, DataFrames, Random, CSV, MLJ, MLJGLMInterface
    import MLJLinearModels
    gr()
	PlutoUI.TableOfContents()
end

# ╔═╡ 20c5c7bc-664f-4c04-8215-8f3a9a2095c9
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 8217895b-b120-4b08-b18f-d921dfdddf10
md"# Linear Regression

## Wind speed prediction with one predictor
"

# ╔═╡ 9f84bcc5-e5ab-4935-9076-c19e9bd668e8
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), DataFrame);

# ╔═╡ 12d5824c-0873-49a8-a5d8-f93c73b633ae
md"The julia machine learning framework `MLJ` supports different implementations
of linear regression. In the previous course we used the one from `MLJLinearModels`.
Here we will use the one from `MLJGLMInterface`.
"

# ╔═╡ 34e527f2-ef80-4cb6-be3a-bee055eca125
begin
    training_set1 = (X = (LUZ_pressure = weather.LUZ_pressure[1:end-5],),
                     y = weather.LUZ_wind_peak[6:end])
    m1 = machine(LinearRegressor(), training_set1.X, training_set1.y) |> fit!
end;

# ╔═╡ 006fc1eb-50d5-4206-8c87-53e873f158f4
begin
    scatter(training_set1.X.LUZ_pressure[1:10:end],
            training_set1.y[1:10:end], label = "data")
    plot!(fitted_linear_func(m1), label = "linear fit", w = 3)
end

# ╔═╡ e4712ebe-f395-418b-abcc-e10ada4b05c2
md"Let us inspect the results.
First we look at the fitted parameters.
We find that there is a negative correlation between the pressure in Luzern
and the wind speed 5 hours later in Luzern.
"

# ╔═╡ 8c9ae8f7-81b2-4d60-a8eb-ded5364fe0cc
fitted_params(m1)

# ╔═╡ f4f890b6-0ad4-4155-9321-15d673e15489
md"Next we predict for different pressure values the distribution of
    wind speeds. In the probabilistic interpretation of supervised learning,
    standard linear regression finds conditional normal distributions with
    input-dependent mean ``\mu = \hat y = \theta_0 + \theta_1 x`` and
    constant standard deviation σ.
"

# ╔═╡ 9b62c374-c26e-4990-8ffc-790928e62e88
predict(m1, (LUZ_pressure = [930., 960., 990.],))

# ╔═╡ 7923a0a8-3033-4dde-91e8-22bf540c9866
md"We use the root-mean-squared-error (`rmse`) = ``\sqrt{\frac1n\sum_{i=1}^n(y_i - \hat y_i)^2}`` to evaluate our training error. To compute the test error, we use hourly data from 2019 to 2020.
"

# ╔═╡ 57f352dc-55ee-4e14-b68d-698938a97d92
rmse(predict_mean(m1, training_set1.X), training_set1.y)

# ╔═╡ b0de002f-3f39-4378-8d68-5c4606e488b7
begin
    weather_test = CSV.read(joinpath(@__DIR__, "..", "data", "weather2019-2020.csv"), DataFrame);
    test_set1 = (X = (LUZ_pressure = weather_test.LUZ_pressure[1:end-5],),
                     y = weather_test.LUZ_wind_peak[6:end])
end

# ╔═╡ ce536f60-68b3-4901-bd6a-c96378054b12
rmse(predict_mean(m1, test_set1.X), test_set1.y)

# ╔═╡ c65b81bd-395f-4461-a73b-3535903cb2d7
md"## Multiple Linear Regression

In multiple linear regression there are multiple predictors ``x_1, x_2, \ldots, x_p``
and the response is ``\hat y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
`` (we use here ``\beta`` instead of ``\theta`` for the parameters).
With ``p = 2`` predictors we can visualize linear regression as the plane
that is closest to the data. Use the sliders below to get a feeling for the
parameters. You can also change the viewing angle by clicking and dragging in
the figure."

# ╔═╡ 51c9ea74-3110-4536-a4af-7cc73b45a4a6
md"β₀ = $(@bind β₀ Slider(-1:.02:1, default = .4, show_value = true))

β₁ = $(@bind β₁ Slider(-1:.02:1, default = .2, show_value = true))

β₂ = $(@bind β₂ Slider(-1:.02:1, default = -.6, show_value = true))
"

# ╔═╡ 0f544053-1b7a-48d6-b18b-72092b124305
begin
    Random.seed!(3)
	X = DataFrame(X1 = randn(20), X2 = randn(20))
    f0(X1, X2, β₀, β₁, β₂) = β₀ + β₁ * X1 + β₂ * X2
    f0(β₀, β₁, β₂) = (X1, X2) -> f0(X1, X2, β₀, β₁, β₂)
	data_generator(X1, X2; β₀, β₁, β₂, σ = 0.8) = f0(X1, X2, β₀, β₁, β₂) + σ * randn()
	y = data_generator.(X.X1, X.X2, β₀ = .4, β₁ = .5, β₂ = -.6)
end;


# ╔═╡ d541a8cd-5aa4-4c2d-bfdf-5e6297bb65a8
begin
    plotly()
    p1 = scatter3d(X.X1, X.X2, y, markersize = 1,
                   xlims = (-3, 3), xlabel = "X1",
                   ylims = (-3, 3), ylabel = "X2",
                   zlims = (-4, 4), zlabel = "y", label = "data")
    wireframe!(-3:.1:3, -3:.1:3, f0(β₀, β₁, β₂),
               label = "function", title = "data & function", color = :green)
    plot_residuals!(X.X1, X.X2, y, f0(β₀, β₁, β₂))
    p2 = contour(-1:.1:1, -1:.1:1, (β₀, β₁) -> mean((β₀ .+ β₁ .* X.X1 .+ β₂ .* X.X2 .- y).^2), levels = 100, ylabel = "β₁", cbar = false, title = "loss")
    scatter!([β₀], [β₁], label = nothing)
    p3 = contour(-1:.1:1, -1:.1:1, (β₀, β₂) -> mean((β₀ .+ β₁ .* X.X1 .+ β₂ .* X.X2 .- y).^2), levels = 100, xlabel = "β₀", ylabel = "β₂")
    scatter!([β₀], [β₂], label = "current loss")
    plot(p1, plot(p2, p3, layout = (2, 1)), layout = (1, 2),
         size = (700, 400), legend = false)
end

# ╔═╡ da6462d8-3343-41d8-82dd-48770176d4ba
md"## Wind speed prediction with multiple predictors"

# ╔═╡ 753ec309-1363-485d-a2bd-b9fa100d9058
m2 = machine(LinearRegressor(), select(weather[1:end-5,:], Not([:LUZ_wind_peak, :time])),
             weather.LUZ_wind_peak[6:end]) |> fit!;

# ╔═╡ 618ef3c7-0fda-4970-88e8-1dac195545de
sort!(DataFrame(predictor = names(select(weather, Not([:LUZ_wind_peak, :time]))),
                value = fitted_params(m2).coef), :value)

# ╔═╡ 2d25fbb6-dc9b-40ad-bdce-4c952cdad077
rmse(predict_mean(m2, select(weather[1:end-5,:], Not([:LUZ_wind_peak, :time]))),
     weather.LUZ_wind_peak[6:end])

# ╔═╡ c9f10ace-3299-45fb-b98d-023a35dd405a
rmse(predict_mean(m2, select(weather_test[1:end-5,:], Not([:LUZ_wind_peak, :time]))),
     weather_test.LUZ_wind_peak[6:end])

# ╔═╡ 99a371b2-5158-4c42-8f50-329352b6c1f2
md"# Error Decomposition

"

# ╔═╡ f10b7cad-eda3-4ec9-99ee-d43ed013a057
begin
    f(x) = sin(2x) + 2*(x - .5)^3 - .5x
    conditional_generator(x; n = 50) = f.(x) .+ .2*randn(n)
end;

# ╔═╡ 05354df5-a803-422f-87a3-1c56a34e8a48
f̂(x) = 0.1 + x

# ╔═╡ 9e61b4c3-1a9f-41a7-9882-25ed797a7b8d
expected_error(f, x) = mean((conditional_generator(x, n = 10^6) .- f(x)).^2);

# ╔═╡ c6a59b85-d031-4ad4-9e24-691494d08cde
expected_error(f̂, .1)

# ╔═╡ e50b8196-e804-473a-b3b5-e22fdb9d2f45
(f(.1) - f̂(.1))^2

# ╔═╡ f413ea94-36ca-4afc-8ca8-9a7e88101980
expected_error(f, .1)

# ╔═╡ dbf7fc72-bfd0-4c57-a1a9-fb5881e16e7e
let x = rand(100), grid = 0:.05:1
    gr()
    p1 = scatter(x, vcat(conditional_generator.(x, n = 1)...), label = "samples")
    plot!(f, label = "f")
    plot!(f̂, label = "f̂")
    p2 = plot(grid, expected_error.(f̂, grid), label = "expected error f̂", w = 3)
    plot!(grid, (f.(grid) .- f̂.(grid)).^2, label = "reducible error", w = 3)
    hline!([.2^2], label = "irreducible error", ylims = (0, .15), w = 3, xlabel = "x")
    plot(p1, p2, layout = (2, 1), legend = :right, ylabel = "y")
end

# ╔═╡ ad5b293d-c0f4-4693-84f4-88308639a501
md"# Logistic Regression

## Preparing the spam data

The text in our spam data set is already preprocessed. But we do not yet have
a format similar to our weather prediction data set with a fixed number ``p`` of
predictors for each email. In this section we create a very simple feature
representation of our emails:
1. We create a lexicon of words that are neither very frequent nor very rare.
2. For each email we count how often every word in this lexicon appears.
3. Our feature matrix will consist of ``n`` rows (one for each email) and ``p``
    predictors (one for each word in the lexicon) with ``x_{ij}`` measuring how
    often word ``j`` appears in document ``i``, normalized by the number of
    lexicon words in each email (such that the elements in every row sum to 1).
"

# ╔═╡ 210b977d-7136-407f-a1c9-eeea869d0312
begin
    spamdata = CSV.read(joinpath(@__DIR__, "..", "data", "spam.csv"), DataFrame)
    dropmissing!(spamdata) # remove entries without any text (missing values).
end

# ╔═╡ 72969aca-b203-4d83-8923-74e523aa1c01
import TextAnalysis: Corpus, StringDocument, DocumentTermMatrix, lexicon,
                     update_lexicon!, tf

# ╔═╡ 4cbb3057-01f4-4e80-9029-4e80d6c9e5e6
md"In the next cell we create the full lexicon of words appearing in the first
2000 emails. Each lexicon entry is of the form `\"word\" => count`."

# ╔═╡ c50c529f-d393-4854-b5ed-91e90d557d12
begin
    crps = Corpus(StringDocument.(spamdata.text[1:2000]))
    update_lexicon!(crps)
    lexicon(crps)
end

# ╔═╡ 72b50cee-d436-42ce-add9-07b0c012cb31
md"Now we select only those words of the full lexicon that appear at least 100
times and at most 10^3 times. These numbers are pulled out of thin air (like
all the design choses of this very crude feature engineering).
"

# ╔═╡ bf4110a9-31a4-48a3-bd6d-85c404d0e72d
begin
    small_lex = Dict(k => lexicon(crps)[k]
                     for k in findall(x -> 100 <= last(x) <= 10^3, lexicon(crps)))
    m = DocumentTermMatrix(crps, small_lex)
end

# ╔═╡ 534681d5-71d8-402a-b455-f491cfbb353e
begin
    spam_or_ham = coerce(spamdata.label[1:2000], Binary)
    normalized_word_counts = float.(DataFrame(tf(m), :auto))
end

# ╔═╡ ec1c2ea5-29ce-4371-be49-08798305ff50
Markdown.parse("Here we go: now we have a matrix of size
 $(join(size(normalized_word_counts), " x ")) as input and a vector of binary label as
 output. We will be able to use this as input in multiple logistic regression.
 For future usage we save this preprocessed representation of the spam data to
 a file.")

# ╔═╡ 681cb7b9-f041-4aea-907e-4d85135c005a
CSV.write(joinpath(dirname(pathof(MLCourse)), "..", "data", "spam_preprocessed.csv"),
          [normalized_word_counts DataFrame(spam_or_ham = spam_or_ham)])

# ╔═╡ f7117513-283f-4e32-a2a1-3594c794c94d
md"## Multiple Logistic Regression

In the top row of the figure below we see in two different ways (once as a 3D
plot and once as a contour plot) the probability of class A for the selected
parameter values. The bottom row shows samples (large points, red = class A)
obtained with this probability distribution and predictions (small points)
at decision threshold 0.5.
Play with the parameters to get a feeling for how they affect the probability
and the samples.

θ₀ = $(@bind θ₀ Slider(-3:3, default = 0, show_value = true))

θ₁ = $(@bind θ₁ Slider(-8:8, default = 3, show_value = true))

θ₂ = $(@bind θ₂ Slider(-8:8, default = 0, show_value = true))
"

# ╔═╡ fd4165dc-c3e3-4c4c-9605-167b5b4416da
md"## Confusion Matrix, ROC and AUC"

# ╔═╡ 7738c156-8e1b-4723-9818-fba364822171
md"s = $(@bind s Slider(-4:.1:4, default = 0, show_value = true))

seed = $(@bind seed Slider(1:100, show_value = true))

threshold = $(@bind threshold Slider(.01:.01:.99, default = 0.5, show_value = true))
"

# ╔═╡ 0fcfd7d2-6ea3-4c75-bad3-7d0fdd6fde11
begin
    logistic(x) = 1/(1 + exp(-x))
    logodds(p) = log(p/(1-p))
    function error_rates(x, y, t)
        P = sum(y)
        N = length(y) - P
        pos_pred = y[(logistic.(x) .> t)]
        TP = sum(pos_pred)
        FP = sum(1 .- pos_pred)
        FP/N, TP/P
    end
end;

# ╔═╡ 4f89ceab-297f-4c2c-9029-8d2d7fad084f
let f(x1, x2) = logistic(θ₀ + θ₁ * x1 + θ₂ * x2)
    p1 = wireframe(-3:.1:3, -3:.1:3, f, zlims = (0, 1))
    p2 = contour(-3:.1:3, -3:.1:3, f, contour_labels = true, levels = 20, cbar = false)
	plotly()
    samples = (X1 = 6 * rand(200) .- 3, X2 = 6 * rand(200) .- 3)
    labels = f.(samples.X1, samples.X2) .> rand(200)
    xgrid = MLCourse.grid(-3:.2:3, -3:.2:3, names = (:X1, :X2))
    scatter(xgrid.X1, xgrid.X2, color = (f.(xgrid.X1, xgrid.X2) .> .5) .+ 1,
            markersize = 2, markerstrokewidth = 0, label = nothing)
    p3 = scatter!(samples.X1, samples.X2, color = labels .+ 1, xlabel = "X1")
    plot(p1, p2, plot(), p3, layout = (2, 2), size = (700, 600),
         ylabel = "X2", legend = false)
end


# ╔═╡ 285c6bfc-5f29-46e0-a2c1-8abbec74501b
begin
    Random.seed!(seed)
    auc_samples_x = 2 * randn(200)
end;

# ╔═╡ c98524b5-d6b3-469c-82a1-7d231cc792d6
begin
    auc_samples_y = logistic.(2.0^s * auc_samples_x) .> rand(200)
    auc = [error_rates(2.0^s * auc_samples_x, auc_samples_y, t)
           for t in .01:.01:.99]
    push!(auc, (0., 0.))
    prepend!(auc, [(1., 1.)])
end;

# ╔═╡ 3336ab15-9e9b-44af-a7d5-1d6472241e62
let
    gr()
    p1 = scatter(auc_samples_x, auc_samples_y, markershape = :vline, label = nothing, color = :black)
    plot!(x -> logistic(2.0^s * x), color = :blue, label = nothing, xlims = (-8, 8))
    vline!([1/(2.0^s) * logodds(threshold)], w = 3, color = :red,
           label = nothing, xlabel = "x", ylabel = "y")
    p2 = plot(first.(auc), last.(auc), title = "ROC", label = nothing)
    fp, tp = auc[floor(Int, threshold * 100)]
    scatter!([fp], [tp], color = :red, xlims = (-.01, 1.01), ylims = (-.01, 1.01),
            labels = nothing, ylabel = "true positive rate",
            xlabel = "false positive rate")
    plot(p1, p2, size = (700, 400))
end


# ╔═╡ 62ad57e5-1366-4635-859b-ccdab2efd3b8
md"## Multiple Logistic Regression on the spam data"

# ╔═╡ 29e1d9ff-4375-455a-a69b-8dd0c2cac57d
m3 = fit!(machine(LinearBinaryClassifier(),
                  normalized_word_counts,
                  spam_or_ham));

# ╔═╡ 1d1a24c6-c166-49a2-aa21-7acf50b55a66
predict(m3, normalized_word_counts)

# ╔═╡ 32bafa9e-a35e-4f54-9857-d269b47f95c3
confusion_matrix(predict_mode(m3, normalized_word_counts), spam_or_ham)

# ╔═╡ 4e4f4adf-364f-49b9-9391-5050a4c1286a
md"With our simple features, logistic regression can classify the training data
almost always correctly. Let us see how well this works for test data.
"

# ╔═╡ 50c035e6-b892-4157-a52f-824578366977
begin
    test_crps = Corpus(StringDocument.(spamdata.text[2001:4000]))
    test_input = float.(DataFrame(tf(DocumentTermMatrix(test_crps, small_lex)), :auto))
    test_labels = coerce(spamdata.label[2001:4000], Binary)
    confusion_matrix(predict_mode(m3, test_input), test_labels)
end

# ╔═╡ ba4b5683-5932-415e-8772-8b3eef5eb63d
md"We save also the test data for future usage."

# ╔═╡ 9e1076f0-c7c3-4d04-8c00-317d39c5340b
CSV.write(joinpath(dirname(pathof(MLCourse)), "..", "data",
                   "spam_preprocessed_test.csv"),
          [test_input DataFrame(spam_or_ham = test_labels)])

# ╔═╡ a30578dd-aecb-46eb-b947-f009282cf2fc
md"Let us evaluate the fit in terms of commonly used losses for binary classification."

# ╔═╡ 8ed39cdc-e99e-48ff-9973-66df41aa0f78
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

# ╔═╡ dd463687-b73d-4e70-b2cf-97a56a0ad409
losses(m3, normalized_word_counts, spam_or_ham)

# ╔═╡ 935adbcd-48ab-4a6f-907c-b04137ca3abe
losses(m3, test_input, test_labels)

# ╔═╡ 8b0451bf-59b0-4e71-be84-549e23b5bfe7
md"# Exercises

## Conceptual

1. Suppose we have a data set with three predictors, ``X_1`` = Final Grade, ``X_2`` = IQ, ``X_3`` = Level (1 for College and 0 for High School).  The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get ``\hat\beta_0 = 50, \hat\beta_1 = 20, \hat\beta_2 = 0.07, \hat\beta_3 = 35``.
   - Which answer is correct, and why?
      - For a fixed value of IQ and Final Grade, high school graduates earn more, on average, than college graduates.
      - For a fixed value of IQ and Final Grade, college graduates earn more, on average, than high school graduates.
   - Predict the salary of a college graduate with IQ of 110 and a Final Grade of 4.0.
2. Suppose we collect data for a group of students in a machine learning class with variables ``X_1 =`` hours studied, ``X_2 =`` grade in statistics class, and ``Y =`` receive a 6 in the machine learning class. We fit a logistic regression and produce estimated coefficients, ``\hat{\beta}_0 = -6``, ``\hat{\beta}_1 = 0.025``, ``\hat{\beta}_2 = 1``.
   - Estimate the probability that a student who studies for 75 hours and had a 4 in the statistics class gets a 6 in the machine learning class.
   - How many hours would the above student need to study to have a 50% chance of getting an 6 in the machine learning class?

## Applied
1. In the multiple linear regression of the weather data set above we used all
   available predictors. We do not know if all of them are relevant. Try to find
   here the model with at most 4 predictors that has the lowest test rmse.
   How much higher is the test error compared to the fit with all available predictors?
2. In this exercise we perform linear classification of the MNIST handwritten digits
   dataset.
   - Load the MNIST data set (Hint: see notebook \"supervised learning\").
   - Usually the first 60'000 images are taken as training set, but for this exercise I recommend to use fewer rows, e.g. the first 5000.
   - Coerce the output to multiclass (for classification) and the input to real numbers  with the command `coerce!(mnist, :class => Multiclass); coerce!(mnist, Count => Continuous)`.
   - Scale the input values to the interval [0, 1) with `mnist[:, 1:784] ./= 255`
   - Fit a `MLJLinearModels.MultinomialClassifier(penalty = :none)` to the data.
     Hint: this may take a long time. Just use e.g. the first 5000 rows of the data
     if you want to wait less.
   - Compute the misclassification rate and the confusion matrix on the training set.
   - Use as test data rows 60001 to 70000 and compute the misclassification rate
     and the confusion matrix on this test set.
   - Plot some of the wrongly classified training and test images.
     Are they also difficult for you to classify?
"

# ╔═╡ Cell order:
# ╟─8217895b-b120-4b08-b18f-d921dfdddf10
# ╠═9f84bcc5-e5ab-4935-9076-c19e9bd668e8
# ╟─12d5824c-0873-49a8-a5d8-f93c73b633ae
# ╠═34e527f2-ef80-4cb6-be3a-bee055eca125
# ╠═006fc1eb-50d5-4206-8c87-53e873f158f4
# ╟─e4712ebe-f395-418b-abcc-e10ada4b05c2
# ╠═8c9ae8f7-81b2-4d60-a8eb-ded5364fe0cc
# ╟─f4f890b6-0ad4-4155-9321-15d673e15489
# ╠═9b62c374-c26e-4990-8ffc-790928e62e88
# ╟─7923a0a8-3033-4dde-91e8-22bf540c9866
# ╠═57f352dc-55ee-4e14-b68d-698938a97d92
# ╠═b0de002f-3f39-4378-8d68-5c4606e488b7
# ╠═ce536f60-68b3-4901-bd6a-c96378054b12
# ╟─c65b81bd-395f-4461-a73b-3535903cb2d7
# ╟─51c9ea74-3110-4536-a4af-7cc73b45a4a6
# ╟─d541a8cd-5aa4-4c2d-bfdf-5e6297bb65a8
# ╟─0f544053-1b7a-48d6-b18b-72092b124305
# ╟─da6462d8-3343-41d8-82dd-48770176d4ba
# ╠═753ec309-1363-485d-a2bd-b9fa100d9058
# ╠═618ef3c7-0fda-4970-88e8-1dac195545de
# ╠═2d25fbb6-dc9b-40ad-bdce-4c952cdad077
# ╠═c9f10ace-3299-45fb-b98d-023a35dd405a
# ╟─99a371b2-5158-4c42-8f50-329352b6c1f2
# ╠═f10b7cad-eda3-4ec9-99ee-d43ed013a057
# ╠═05354df5-a803-422f-87a3-1c56a34e8a48
# ╠═9e61b4c3-1a9f-41a7-9882-25ed797a7b8d
# ╠═c6a59b85-d031-4ad4-9e24-691494d08cde
# ╠═e50b8196-e804-473a-b3b5-e22fdb9d2f45
# ╠═f413ea94-36ca-4afc-8ca8-9a7e88101980
# ╟─dbf7fc72-bfd0-4c57-a1a9-fb5881e16e7e
# ╟─ad5b293d-c0f4-4693-84f4-88308639a501
# ╠═210b977d-7136-407f-a1c9-eeea869d0312
# ╠═72969aca-b203-4d83-8923-74e523aa1c01
# ╟─4cbb3057-01f4-4e80-9029-4e80d6c9e5e6
# ╠═c50c529f-d393-4854-b5ed-91e90d557d12
# ╟─72b50cee-d436-42ce-add9-07b0c012cb31
# ╠═bf4110a9-31a4-48a3-bd6d-85c404d0e72d
# ╠═534681d5-71d8-402a-b455-f491cfbb353e
# ╟─ec1c2ea5-29ce-4371-be49-08798305ff50
# ╠═681cb7b9-f041-4aea-907e-4d85135c005a
# ╟─f7117513-283f-4e32-a2a1-3594c794c94d
# ╟─4f89ceab-297f-4c2c-9029-8d2d7fad084f
# ╟─fd4165dc-c3e3-4c4c-9605-167b5b4416da
# ╟─7738c156-8e1b-4723-9818-fba364822171
# ╠═0fcfd7d2-6ea3-4c75-bad3-7d0fdd6fde11
# ╟─3336ab15-9e9b-44af-a7d5-1d6472241e62
# ╟─285c6bfc-5f29-46e0-a2c1-8abbec74501b
# ╟─c98524b5-d6b3-469c-82a1-7d231cc792d6
# ╟─62ad57e5-1366-4635-859b-ccdab2efd3b8
# ╠═29e1d9ff-4375-455a-a69b-8dd0c2cac57d
# ╠═1d1a24c6-c166-49a2-aa21-7acf50b55a66
# ╠═32bafa9e-a35e-4f54-9857-d269b47f95c3
# ╟─4e4f4adf-364f-49b9-9391-5050a4c1286a
# ╠═50c035e6-b892-4157-a52f-824578366977
# ╟─ba4b5683-5932-415e-8772-8b3eef5eb63d
# ╠═9e1076f0-c7c3-4d04-8c00-317d39c5340b
# ╟─a30578dd-aecb-46eb-b947-f009282cf2fc
# ╠═8ed39cdc-e99e-48ff-9973-66df41aa0f78
# ╠═dd463687-b73d-4e70-b2cf-97a56a0ad409
# ╠═935adbcd-48ab-4a6f-907c-b04137ca3abe
# ╟─8b0451bf-59b0-4e71-be84-549e23b5bfe7
# ╟─20c5c7bc-664f-4c04-8215-8f3a9a2095c9
# ╟─94f8e29e-ef91-11eb-1ae9-29bc46fa505a
