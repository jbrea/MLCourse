### A Pluto.jl notebook ###
# v0.19.0

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

# ╔═╡ 94f8e29e-ef91-11eb-1ae9-29bc46fa505a
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using Plots, DataFrames, Random, CSV, MLJ, MLJLinearModels, MLCourse
	import PlutoPlotly as PP
	import Distributions: Poisson
    import MLCourse: fitted_linear_func
end

# ╔═╡ 12d5824c-0873-49a8-a5d8-f93c73b633ae
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 8217895b-b120-4b08-b18f-d921dfdddf10
md"# Linear Regression

## Wind speed prediction with one predictor
"

# ╔═╡ 9f84bcc5-e5ab-4935-9076-c19e9bd668e8
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"),
                   DataFrame);

# ╔═╡ d14244d4-b54e-42cf-b812-b149d9fa59e0
training_set1 = DataFrame(LUZ_pressure = weather.LUZ_pressure[1:end-5],
                          wind_peak_in5h = weather.LUZ_wind_peak[6:end])

# ╔═╡ 34e527f2-ef80-4cb6-be3a-bee055eca125
m1 = machine(LinearRegressor(),
			 select(training_set1, :LUZ_pressure),
	         training_set1.wind_peak_in5h);

# ╔═╡ dbe563b8-2d0d-40b7-8ccf-7efa40f640f2
fit!(m1, verbosity = 0);

# ╔═╡ 006fc1eb-50d5-4206-8c87-53e873f158f4
begin
    scatter(training_set1.LUZ_pressure,
            training_set1.wind_peak_in5h,
		    xlabel = "pressure [hPa]",
		    ylabel = "wind speed [km/h]",
		    label = "data")
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
md"Next we predict the expected wind peak for different pressure values."

# ╔═╡ 9b62c374-c26e-4990-8ffc-790928e62e88
predict(m1, (LUZ_pressure = [930., 960., 990.],))

# ╔═╡ 7923a0a8-3033-4dde-91e8-22bf540c9866
md"We use the root-mean-squared-error (`rmse`) = ``\sqrt{\frac1n\sum_{i=1}^n(y_i - \hat y_i)^2}`` to evaluate our training error. To compute the test error, we use hourly data from 2019 to 2020.
"

# ╔═╡ 57f352dc-55ee-4e14-b68d-698938a97d92
rmse(predict(m1), training_set1.wind_peak_in5h)

# ╔═╡ b0de002f-3f39-4378-8d68-5c4606e488b7
begin
    weather_test = CSV.read(joinpath(@__DIR__, "..", "data", "weather2019-2020.csv"),
		                    DataFrame);
    test_set1 = DataFrame(LUZ_pressure = weather_test.LUZ_pressure[1:end-5],
                          wind_peak_in5h = weather_test.LUZ_wind_peak[6:end])
end;

# ╔═╡ ce536f60-68b3-4901-bd6a-c96378054b12
rmse(predict(m1, select(test_set1, :LUZ_pressure)), test_set1.wind_peak_in5h)

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
let
    f = f0(β₀, β₁, β₂)
    p1 = PP.scatter3d(x = X.X1, y = X.X2, z = y,
                      marker_size = 3, mode = "markers", name = "data")
    l1 = PP.Layout(scene1 = PP.attr(xaxis = PP.attr(title_text = "X₁",
                                                    title_standoff = 0,
                                                    range = (-3, 3)),
                                    yaxis = PP.attr(range = (-3, 3),
                                                    title_text = "X₂",
                                                    title_standoff = 0),
                                    zaxis = PP.attr(range = (-4, 4),
                                                    title = "Y"),
		                            domain = PP.attr(x = [-.1, .6], y = [0, 1]),
                                    uirevision = 1),
                   showlegend = false, showscale = false,
                   xaxis2_domain = [.7, 1], xaxis4_domain = [.7, 1],
                   xaxis2_title_text = "β₀", yaxis2_title_text = "β₁",
                   xaxis4_title_text = "β₀", yaxis4_title_text = "β₂",
                   xaxis2_title_standoff = 0, yaxis2_title_standoff = 0,
                   xaxis4_title_standoff = 0, yaxis4_title_standoff = 0
                  )
    xgrid = -3:1:3; ygrid = -3:1:3
    p2 = PP.mesh3d(x = repeat(xgrid, length(ygrid)),
                      y = repeat(ygrid, inner = length(xgrid)),
                      z = reshape(f.(xgrid, ygrid'), :), opacity = .5,
                      name = "function", title = "data & function",
                      color = "green")
    res = [PP.scatter3d(x = [x1, x1], y = [x2, x2], z = [z, f(x1, x2)],
	                    mode = "lines", line_color = "red", name = nothing)
           for (x1, x2, z) in zip(X.X1, X.X2, y)]
    plot1 = PP.Plot([p1, p2, res...])
#     plot_residuals!(X.X1, X.X2, y, f0(β₀, β₁, β₂))
    lossf1 = (β₀, β₁) -> mean((β₀ .+ β₁ .* X.X1 .+ β₂ .* X.X2 .- y).^2)
    xgrid = -1:.1:1; ygrid = -1:.1:1
    p3 = PP.contour(x = xgrid, y = ygrid, z = lossf1.(xgrid, ygrid'),
                    ncontours = 50, ylabel = "β₁", showscale = false, title = "loss")
    p4 = PP.scatter(x = [β₀], y = [β₁], mode = "markers", marker_color = "red")
    plot2 = PP.Plot([p3, p4], PP.Layout(title = "loss function"))
    lossf2 = (β₀, β₂) -> mean((β₀ .+ β₁ .* X.X1 .+ β₂ .* X.X2 .- y).^2)
    p5 = PP.contour(x = xgrid, y = ygrid,
                    z = lossf2.(xgrid, ygrid'),
                    ncontours = 50, xlabel = "β₀", ylabel = "β₂")
    p6 = PP.scatter(x = [β₀], y = [β₂], mode = "markers", marker_color = "red")
    plot3 = PP.Plot([p5, p6])
    plot = [plot1 plot2; PP.Plot() plot3]
    PP.relayout!(plot, l1)
    PP.PlutoPlot(plot)
end

# ╔═╡ da6462d8-3343-41d8-82dd-48770176d4ba
md"## Wind speed prediction with multiple predictors

Our weather data set contains multiple measurements.
With `names(weather)` we see all the columns of the the weather data frame.
Try it in a new cell!

With `DataFrame(schema(weather))` we get additionally information about the type of data. `MLJ` distinguished between the [scientific type](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types) and the number type of a column. This distinction is useful, because e.g. categorical variables (class 1, class 2, class 3, ...) can be represented as an integer but also count variables (e.g. number of items) can be represented as an integer, but suitability of a given machine learning method depends more on the scientific type than on the number type.
"

# ╔═╡ 25e1f89f-a5b2-4d5c-a6e8-a990b3bebddb
DataFrame(schema(weather))

# ╔═╡ 8307e205-3bcd-4e68-914e-621fd8d29e43
md"When loading data one should check if the scientific type is correctly detected by the data loader. For the weather data the wind direction is automatically detected as a count variable, because the angle of the wind is measured in integer degrees. But it makes more sense to interpret the angle as a continuous variable. Therefore we use the `MLJ` function `coerce!` to transform all `Count` columns to `Continuous`."

# ╔═╡ eeb9db2b-1ba3-4cc4-ac6f-b3249e9f1146
coerce!(weather, Count => Continuous);

# ╔═╡ 48a329d3-19ef-4883-b25f-71b44e824ab6
md"Now we will try to predict the wind peak based on all measurements, except the response variable and the time variable."

# ╔═╡ 753ec309-1363-485d-a2bd-b9fa100d9058
m2 = machine(LinearRegressor(),
	         select(weather[1:end-5,:], Not([:LUZ_wind_peak, :time])),
             weather.LUZ_wind_peak[6:end]);

# ╔═╡ 93c39214-1498-4ebd-a7a6-cdb0438da14d
fit!(m2, verbosity = 0);

# ╔═╡ fac51c63-f227-49ac-89f9-205bf03e7c08
md"Let us have a look at the fitted parameters. For example, we can conclude that an increase of pressure in Luzern by 1hPa correlates with a decrease of almost 3km/h of the expected wind peaks - everything else being the same. This result is consistent with our expectation that high pressure locally is usually accompanied by good weather without strong winds. On the other hand, an increase of pressure in Genève by 1hPa correlates with an increase of more than 3km/h of the expected wind peaks. This result is also consistent with our expectations, given that stormy west wind weather occurs usually when there is a pressure gradient from south west to north east of Switzerland. Note also that the parameter for pressure in Pully (50 km north east from Genève) is -2km/h⋅hPa, i.e. when the pressure in Pully is the same as in Genève these two contributions to the prediction almost compensate each other."

# ╔═╡ 618ef3c7-0fda-4970-88e8-1dac195545de
sort!(DataFrame(predictor = names(select(weather, Not([:LUZ_wind_peak, :time]))),
                value = last.(fitted_params(m2).coefs)), :value)

# ╔═╡ 2d25fbb6-dc9b-40ad-bdce-4c952cdad077
rmse(predict(m2, select(weather[1:end-5,:], Not([:LUZ_wind_peak, :time]))),
     weather.LUZ_wind_peak[6:end])

# ╔═╡ c9f10ace-3299-45fb-b98d-023a35dd405a
rmse(predict(m2, select(weather_test[1:end-5,:], Not([:LUZ_wind_peak, :time]))),
     weather_test.LUZ_wind_peak[6:end])

# ╔═╡ 2724c9b7-8caa-4ba7-be5c-2a9095281cfd
md"We see that both the training and the test error is lower, when using multiple predictors instead of just one."

# ╔═╡ 99a371b2-5158-4c42-8f50-329352b6c1f2
md"# Error Decomposition

In the following cells we look at error decomposition discussed in the slides.
"

# ╔═╡ f10b7cad-eda3-4ec9-99ee-d43ed013a057
begin
    f(x) = sin(2x) + 2*(x - .5)^3 - .5x
    conditional_generator(x; n = 50) = f.(x) .+ .2*randn(n)
end;

# ╔═╡ 05354df5-a803-422f-87a3-1c56a34e8a48
f̂(x) = 0.1 + x

# ╔═╡ 3d77d753-b247-4ead-a385-7cbbcfc3190b
md"The expected error of a function `f` at point `x` for our `conditional_generator` can be estimated by computing the mean squared error for many samples obtained from this generator."

# ╔═╡ 9e61b4c3-1a9f-41a7-9882-25ed797a7b8d
expected_error(f, x) = mean((conditional_generator(x, n = 10^6) .- f(x)).^2);

# ╔═╡ c6a59b85-d031-4ad4-9e24-691494d08cde
expected_error(f̂, .1) # estimated total expected error

# ╔═╡ e50b8196-e804-473a-b3b5-e22fdb9d2f45
(f(.1) - f̂(.1))^2 # reducible error

# ╔═╡ f413ea94-36ca-4afc-8ca8-9a7e88101980
expected_error(f, .1) # estimated irreducible error

# ╔═╡ 2bfa1a57-b171-44c3-b0d7-b8dda48d26d7
md"Instead of using estimates for the irreducible error we could just compute it in this simple example: it is ``\sigma^2 = 0.04``. In the figure below we look at the expected, reducible and irreducible errors as a function of ``x``."

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

# ╔═╡ db2c6bd4-ee6f-4ba9-b6ec-e7cf94389f93
md"With machine learning we cannot remove the irreducible error. But in cases where accurate prediction is the goal, we should try to minimize the reducible error for all inputs ``x`` we care about."

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
    spam_or_ham = coerce(String.(spamdata.label[1:2000]), OrderedFactor)
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

In the figure below we see on the left the probability of class A for the selected
parameter values and the decision threshold as a red plane. On the right we see samples (large points, red = class A)
obtained with this probability distribution and predictions (small points)
at the given decision threshold.
Play with the parameters to get a feeling for how they affect the probability
and the samples.

θ₀ = $(@bind θ₀ Slider(-3:3, default = 0, show_value = true))

θ₁ = $(@bind θ₁ Slider(-8:8, default = 3, show_value = true))

θ₂ = $(@bind θ₂ Slider(-8:8, default = 0, show_value = true))

decision threshold = $(@bind thresh Slider(.01:.01:1, default = .5, show_value = true))
"

# ╔═╡ 26d957aa-36d4-4b90-9b91-2d9d883877ea
begin
	Random.seed!(123)
	samples = (X1 = 6 * rand(200) .- 3, X2 = 6 * rand(200) .- 3)
end;

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
        pos_pred = y[x .> t]
        TP = sum(pos_pred)
        FP = sum(1 .- pos_pred)
        FP/N, TP/P
    end
end;

# ╔═╡ 4f89ceab-297f-4c2c-9029-8d2d7fad084f
let f(x1, x2) = logistic(θ₀ + θ₁ * x1 + θ₂ * x2)
    Random.seed!(17)
    xgrid = -3:.25:3; ygrid = -3:.25:3
    wireframe = [[PP.scatter3d(x = fill(x, length(ygrid)),
                               y = ygrid, z = f.(x, ygrid),
                               mode = "lines", line_color = "blue")
                      for x in xgrid];
                 [PP.scatter3d(x = xgrid,
                               y = fill(y, length(xgrid)),
                               z = f.(xgrid, y), mode = "lines",
                               line_color = "blue")
                  for y in ygrid];
                 PP.mesh3d(opacity = .2, color = "red",
                           x = repeat(xgrid, length(ygrid)),
                           y = repeat(ygrid, inner = length(xgrid)),
                           z = fill(thresh, length(xgrid)*length(ygrid)))]
    l1 = PP.Layout(scene1 = PP.attr(xaxis_title_text = "X₁",
                                    xaxis_title_standoff = 0,
                                    yaxis_title_text = "X₂",
                                    yaxis_title_standoff = 0,
                                    zaxis_title_text = "probability of A",
                                    zaxis_title_standoff = 0,
                                    camera_eye = PP.attr(x = -1, y = 2.2, z = .2),
                                    domain = PP.attr(x = [-.1, .65], y = [-.1, 1.1])
                                   ),
                   xaxis2_domain = [.65, 1], yaxis2_domain = [.1, .9],
                   xaxis2_title_text = "X₁", yaxis2_title_text = "X₂",
                   xaxis2_title_standoff = 0, yaxis2_title_standoff = 0,
                   uirevision = 1, showlegend = false)
    labels = f.(samples.X1, samples.X2) .> rand(200)
    grid = MLCourse.grid(-3:.2:3, -3:.2:3, names = (:X1, :X2))
    plabels = f.(grid.X1, grid.X2) .> thresh
    pdata = [PP.scatter(x = grid.X1[plabels],
                        y = grid.X2[plabels],
                        mode = "markers", marker_color = "green",
                        marker_size = 3),
             PP.scatter(x = grid.X1[(!).(plabels)],
                        y = grid.X2[(!).(plabels)],
                        mode = "markers", marker_color = "red",
                        marker_size = 3),
             PP.scatter(x = samples.X1[labels], y = samples.X2[labels],
                        mode = "markers",
                        marker_color = "green"),
             PP.scatter(x = samples.X1[(!).(labels)], y = samples.X2[(!).(labels)],
                        mode = "markers",
                        marker_color = "red")]
    plot = hcat(PP.Plot(wireframe), PP.Plot(pdata))
    PP.relayout!(plot, l1)
    PP.PlutoPlot(plot)
#     p2 = contour(-3:.1:3, -3:.1:3, f, contour_labels = true, levels = 20, cbar = false)
# 	plotly()
#     samples = (X1 = 6 * rand(200) .- 3, X2 = 6 * rand(200) .- 3)
#     labels = f.(samples.X1, samples.X2) .> rand(200)
#     xgrid = MLCourse.grid(-3:.2:3, -3:.2:3, names = (:X1, :X2))
#     scatter(xgrid.X1, xgrid.X2, color = (f.(xgrid.X1, xgrid.X2) .> .5) .+ 1,
#             markersize = 2, markerstrokewidth = 0, label = nothing)
#     p3 = scatter!(samples.X1, samples.X2, color = labels .+ 1, xlabel = "X1")
#     plot(p1, p2, plot(), p3, layout = (2, 2), size = (700, 600),
#          ylabel = "X2", legend = false)
end


# ╔═╡ 285c6bfc-5f29-46e0-a2c1-8abbec74501b
begin
    Random.seed!(seed)
    auc_samples_x = 2 * randn(200)
end;

# ╔═╡ c98524b5-d6b3-469c-82a1-7d231cc792d6
begin
    auc_samples_y = logistic.(2.0^s * auc_samples_x) .> rand(200)
    errs = [error_rates(auc_samples_x, auc_samples_y, 1/(2.0^s) * logodds(t))
           for t in .01:.01:.99]
    push!(errs, (0., 0.))
    prepend!(errs, [(1., 1.)])
end;

# ╔═╡ 3336ab15-9e9b-44af-a7d5-1d6472241e62
let
	gr()
    p1 = scatter(auc_samples_x, auc_samples_y, markershape = :vline, label = nothing, color = :black)
    plot!(x -> logistic(2.0^s * x), color = :blue, label = nothing, xlims = (-8, 8))
    vline!([1/(2.0^s) * logodds(threshold)], w = 3, color = :red,
           label = nothing, xlabel = "x", ylabel = "y")
    p2 = plot(first.(errs), last.(errs), title = "ROC", label = nothing)
    fp, tp = errs[floor(Int, threshold * 100)]
    scatter!([fp], [tp], color = :red, xlims = (-.01, 1.01), ylims = (-.01, 1.01),
            labels = nothing, ylabel = "true positive rate",
            xlabel = "false positive rate")
    plot(p1, p2, size = (700, 400))
end

# ╔═╡ f1a48773-2971-4069-a240-fd1e10aeb1ed
confusion_matrix(auc_samples_x .> 1/(2.0^s) * logodds(threshold),
                 categorical(auc_samples_y, levels = [false, true], ordered = true))

# ╔═╡ 62ad57e5-1366-4635-859b-ccdab2efd3b8
md"## Multiple Logistic Regression on the spam data"

# ╔═╡ 29e1d9ff-4375-455a-a69b-8dd0c2cac57d
m3 = fit!(machine(LogisticClassifier(penalty = :none),
                  normalized_word_counts,
                  spam_or_ham), verbosity = 0);

# ╔═╡ 1d1a24c6-c166-49a2-aa21-7acf50b55a66
predict(m3)

# ╔═╡ 21b66582-3fda-401c-9421-73ae2f455a75
predict_mode(m3)

# ╔═╡ 32bafa9e-a35e-4f54-9857-d269b47f95c3
confusion_matrix(predict_mode(m3), spam_or_ham)

# ╔═╡ 4e4f4adf-364f-49b9-9391-5050a4c1286a
md"With our simple features, logistic regression can classify the training data
almost always correctly. Let us see how well this works for test data.
"

# ╔═╡ 50c035e6-b892-4157-a52f-824578366977
begin
    test_crps = Corpus(StringDocument.(spamdata.text[2001:4000]))
    test_input = float.(DataFrame(tf(DocumentTermMatrix(test_crps, small_lex)), :auto))
    test_labels = coerce(String.(spamdata.label[2001:4000]), OrderedFactor)
    confusion_matrix(predict_mode(m3, test_input), test_labels)
end

# ╔═╡ ef9489c3-2bff-431b-92c4-f1b9778040cf
md"In the following we use the functions `roc_curve` and `auc` to plot the ROC curve and compute the area under the curve."

# ╔═╡ e7d48a13-b4e6-4633-898c-c13b3e7f68ea
let
	fprs1, tprs1, _ = roc_curve(predict(m3), spam_or_ham)
    fprs2, tprs2, _ = roc_curve(predict(m3, test_input), test_labels)
	plot(fprs1, tprs1, label = "training ROC")
	plot!(fprs2, tprs2, label = "test ROC", legend = :bottomright)
end

# ╔═╡ 8b851c67-0c6e-4081-a8ed-b818c2902c2f
(training_auc = auc(predict(m3), spam_or_ham),
 test_auc = auc(predict(m3, test_input), test_labels))

# ╔═╡ a30578dd-aecb-46eb-b947-f009282cf2fc
md"Let us evaluate the fit in terms of commonly used losses for binary classification."

# ╔═╡ 8ed39cdc-e99e-48ff-9973-66df41aa0f78
function losses(machine, input, response)
    (negative_loglikelihood = sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = auc(predict(machine, input), response)
	)
end;

# ╔═╡ dd463687-b73d-4e70-b2cf-97a56a0ad409
losses(m3, normalized_word_counts, spam_or_ham)

# ╔═╡ 57dcadc0-2da2-4521-aeaf-6fd01f4bd82b
spam_or_ham

# ╔═╡ 935adbcd-48ab-4a6f-907c-b04137ca3abe
losses(m3, test_input, test_labels)

# ╔═╡ b6689b27-e8a2-44e4-8791-ce237767ee63
md"# Poisson Regression

In this section we have a look at a regression problem where the response is a count variable. As an example we use a Bike sharing data set. In this data set, the number of rented bikes in Washington D.C. at a given time is recorded together with the weather condition. Remove the semicolon in the cell below, if you want to learn more about this dataset.

In this section our goal will be to predict the number of rented bikes at a given temperature and windspeed.
"

# ╔═╡ 310999a0-f212-4e69-a4cb-346b3f49f202
OpenML.describe_dataset(42712);

# ╔═╡ 6384a36d-1dac-4d72-9d7b-84511f20e6ca
bikesharing = OpenML.load(42712) |> DataFrame

# ╔═╡ b4e180b4-5582-4491-89d3-1f730c4f2fbf
DataFrame(schema(bikesharing))

# ╔═╡ b9ba1df0-5086-4c0f-a2c9-200c2be27294
md"Above we see that the `:count` column is detected as `Continuous`, whereas it should be `Count`. We therefore coerce it to the correct scientific type."

# ╔═╡ 1bf98d4b-ffbf-4157-9496-38e2c8e92c94
coerce!(bikesharing, :count => Count);

# ╔═╡ 6acda59d-cb26-4ec0-8a91-2ec5a71bb2b7
md"For counts, i.e. non-negative integers, it is not natural to assume a normally distributed noise. Instead, a common choice is to model a count variable ``Y`` with a Poisson distribution defined by
```math
P(Y = k|f(x)) = \frac{e^{-f(x)}f(x)^k}{k!}
```
where ``f(x)`` is the expected value of ``Y`` conditioned on ``x``.
In the cell below you can see the distribution of counts for different values of ``f(x)``.
"

# ╔═╡ 058d6dff-d6a5-47db-bba2-756d67b873d4
md"``f(x)`` = $(@bind λ Slider(0:.1:15, show_value = true))"

# ╔═╡ 2767c63e-eea4-4227-9d54-293826091e70
scatter(pdf.(Poisson(λ), 0:25), xlabel = "k", ylabel = "probability", legend = false)

# ╔═╡ 63e5e978-3357-43db-b379-78e8dec4c224
md"For count variables we can use Poisson regression. Following the standard recipe, we parametrize ``f(x) = \theta_0 + \theta_1 x_1 + \cdots +\theta_d x_d``, plug this into the formula of the Poisson distribution and fit the parameters ``\theta_0, \ldots, \theta_d`` by maximizing the log-likelihood. In `MLJ` this is done by the `CountRegressor()`."

# ╔═╡ 81c55206-bf59-4c4e-ac5e-77a46e31bec7
import MLJGLMInterface: LinearCountRegressor

# ╔═╡ 96d0366e-dd40-4214-9fce-ec76b8496958
m4 = machine(LinearCountRegressor(),
	         select(bikesharing, [:temp, :humidity]),
             bikesharing.count);

# ╔═╡ 368b2d2c-393a-4b25-9a7c-0404a41e5873
fit!(m4, verbosity = 0);

# ╔═╡ d7af774f-5d58-40a4-a5cd-905fdbe460a3
fitted_params(m4)

# ╔═╡ 6ea40424-22a0-42b9-bfab-8d4903ab8d64
md"Not suprisingly, at higher temperatures more bikes are rented than at lower temperatures, but humitidy coefficient is negative.

In the next cell, we see that the predictions with this machine are Poisson distributions."

# ╔═╡ f071c985-7be7-454e-8541-28416400882f
predict(m4)

# ╔═╡ aa96bbe7-49f4-4244-9c71-8d9b2b3ee065
md"And we can obtain the mean or the mode of this conditional distribution with:"

# ╔═╡ d5b394ac-b243-4825-a5c1-b30146500ef6
predict_mean(m4)

# ╔═╡ caa11dd3-577d-4692-b889-3a38d0bf61e0
predict_mode(m4)

# ╔═╡ 9ec91fbc-b756-4074-a623-1d47925c8239
md"Side remark: `MLJ` distinguishes between models that make deterministic point predictions, like `LinearRegressor()` that predicts the expected response (which is the same as the mean of the conditional normal distribution in the probabilistic view), and models that predict probability distributions, like `LogisticClassifier()` or `LinearCountRegressor()`. If you are unsure about the prediction type of a model you can use the function `prediction_type`:"

# ╔═╡ 41e5133c-db89-4407-9501-70e869616e9d
prediction_type(LinearRegressor())

# ╔═╡ b8bb7c85-0be8-4a87-96da-4e1b37aea96d
prediction_type(LogisticClassifier())

# ╔═╡ d3d7fa67-ca7d-46e1-b705-e30ec9b09f6a
prediction_type(LinearCountRegressor())

# ╔═╡ 8b0451bf-59b0-4e71-be84-549e23b5bfe7
md"""# Exercises

## Conceptual

1. Suppose we have a data set with three predictors, ``X_1`` = Final Grade, ``X_2`` = IQ, ``X_3`` = Level (1 for College and 0 for High School).  The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get ``\hat\beta_0 = 25, \hat\beta_1 = 2, \hat\beta_2 = 0.07, \hat\beta_3 = 15``.
   - Which answer is correct, and why?
      - For a fixed value of IQ and Final Grade, high school graduates earn more, on average, than college graduates.
      - For a fixed value of IQ and Final Grade, college graduates earn more, on average, than high school graduates.
   - Predict the salary of a college graduate with IQ of 110 and a Final Grade of 4.0.
2. Suppose we collect data for a group of students in a machine learning class with variables ``X_1 =`` hours studied, ``X_2 =`` grade in statistics class, and ``Y =`` receive a 6 in the machine learning class. We fit a logistic regression and produce estimated coefficients, ``\hat{\beta}_0 = -6``, ``\hat{\beta}_1 = 0.025``, ``\hat{\beta}_2 = 1``.
   - Estimate the probability that a student who studies for 75 hours and had a 4 in the statistics class gets a 6 in the machine learning class.
   - How many hours would the above student need to study to have a 50% chance of getting an 6 in the machine learning class?
3. In this exercise we will derive the loss function implicitly defined by maximum likelihood estimation of the parameters in a classification setting with multiple classes. Remember that the input ``f(x)`` of the softmax function ``s`` is a vector-valued function. Here we assume a linear function ``f`` and write the ``i``th component of this function as ``f_i(x) = \theta_{i0} + \theta_{i1}x_1 + \cdots + \theta_{ip}x_p``. Note that each component ``i`` has now its own parameters ``\theta_{i0}`` to ``\theta_{ip}``. Using matrix multiplication we can also write ``f(x) = \theta x`` where ``\theta = \left(\begin{array}{ccc}\theta_{10} & \cdots & \theta_{1p}\\\vdots & \ddots & \cdots\\\theta_{K0} & \cdots & \theta_{Kp}\end{array}\right)`` is a ``K\times(p+1)`` dimensional matrix and ``x = (1, x_1, x_2, \ldots, x_p)`` is a column vector of length ``p+1``.
    - Write the log-likelihood function for a classification problem with ``K`` classes. *Hint*: to simplify the notation we can use the convention ``s_y(f(x)) = P(y|x)`` to write the conditional probability of class ``y`` given input ``x``. This convention makes sense when the classes are identified by the integers ``1, 2, \ldots, K``; in this case ``s_y(f(x))`` is the ``y``th component of ``s(f(x))``. Otherwise we would could specify a mapping from classes ``C_1, C_2, \ldots, C_K`` to the integers ``1, 2, \ldots, K`` for this convention to make sense.
    - Assume now ``K = 3`` and ``p = 2``. Explicitly write the loss function for the training set ``\mathcal D = ((x_1 = (0, 0), y_1 = C), (x_2 = (3, 0), y_2 = A), (x_3 = (0, 2), y_3 = B))``.
    - Assume ``K = 2`` and ``p = 1`` and set ``\theta_{20} = 0`` and ``\theta_{21} = 0``. Show that we recover standard logistic regression in this case. *Hint*: show that ``s_1(f(x)) = \sigma(f_1(x))`` and ``s_2(f(x)) = 1 - \sigma(f_1(x))``, where ``s`` is the softmax function and ``\sigma(x) = 1/(1 + e^{-x})`` is the logistic function.
    - Show that one can always set ``\theta_{K0}, \theta_{K1}, \ldots, \theta_{Kp}`` to zero. *Hint* Show that the softmax function with the transformed parameters ``\tilde\theta_{ij}=\theta_{ij} - \theta_{Kj}`` has the same value as the softmax function in the original parameters.

## Applied
1. In the multiple linear regression of the weather data set above we used all
   available predictors. We do not know if all of them are relevant. In this exercise our aim is to find models with fewer predictors and quantify the loss in prediction accuracy.
    - Systematically search for the model with at most 2 predictors that has the lowest test rmse. *Hint* write a function `train_and_evaluate` that takes the training and the test data as input as well as an array of two predictors; remember that `data[:, [\"A\", \"B\"]]` returns a sub-dataframe with columns \"A\" and \"B\". This function should fit a `LinearRegressor` on the training set with those two predictors and return the test rmse for the two predictors. To get a list of all pairs of predictors you can use something like `predictors = setdiff(names(train), ["time", "LUZ_wind_peak"]); predictor_pairs = [[p1, p2] for p1 in predictors, p2 in predictors if p1 != p2 && p1 > p2]`
    - How much higher is the test error compared to the fit with all available predictors?
    - How many models did you have to fit to find your result above?
    - How many models would you have to fit to find the best model with at most 5 predictors? *Hint* the function `binomial` may be useful.
2. In this exercise we perform linear classification of the MNIST handwritten digits
   dataset.
   - Load the MNIST data set with `using OpenML; mnist = OpenML.load(554) |> DataFrame; dropmissing!(mnist);`
   - Usually the first 60'000 images are taken as training set, but for this exercise I recommend to use fewer rows, e.g. the first 5000.
   - Scale the input values to the interval [0, 1) with `mnist[:, 1:784] ./= 255`
   - Fit a `MLJLinearModels.MultinomialClassifier(penalty = :none)` to the data. Be patient! This can take a few minutes.
   - Compute the misclassification rate and the confusion matrix on the training set.
   - Use as test data rows 60001 to 70000 and compute the misclassification rate
     and the confusion matrix on this test set.
   - Plot some of the wrongly classified training and test images.
     Are they also difficult for you to classify?
"""

# ╔═╡ 20c5c7bc-664f-4c04-8215-8f3a9a2095c9
begin
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 7f08fcaa-000d-422d-80b4-e58a2f489d74
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─12d5824c-0873-49a8-a5d8-f93c73b633ae
# ╠═94f8e29e-ef91-11eb-1ae9-29bc46fa505a
# ╟─8217895b-b120-4b08-b18f-d921dfdddf10
# ╠═9f84bcc5-e5ab-4935-9076-c19e9bd668e8
# ╠═d14244d4-b54e-42cf-b812-b149d9fa59e0
# ╠═34e527f2-ef80-4cb6-be3a-bee055eca125
# ╠═dbe563b8-2d0d-40b7-8ccf-7efa40f640f2
# ╠═006fc1eb-50d5-4206-8c87-53e873f158f4
# ╟─e4712ebe-f395-418b-abcc-e10ada4b05c2
# ╠═8c9ae8f7-81b2-4d60-a8eb-ded5364fe0cc
# ╠═f4f890b6-0ad4-4155-9321-15d673e15489
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
# ╠═25e1f89f-a5b2-4d5c-a6e8-a990b3bebddb
# ╟─8307e205-3bcd-4e68-914e-621fd8d29e43
# ╠═eeb9db2b-1ba3-4cc4-ac6f-b3249e9f1146
# ╟─48a329d3-19ef-4883-b25f-71b44e824ab6
# ╠═753ec309-1363-485d-a2bd-b9fa100d9058
# ╠═93c39214-1498-4ebd-a7a6-cdb0438da14d
# ╟─fac51c63-f227-49ac-89f9-205bf03e7c08
# ╠═618ef3c7-0fda-4970-88e8-1dac195545de
# ╠═2d25fbb6-dc9b-40ad-bdce-4c952cdad077
# ╠═c9f10ace-3299-45fb-b98d-023a35dd405a
# ╟─2724c9b7-8caa-4ba7-be5c-2a9095281cfd
# ╟─99a371b2-5158-4c42-8f50-329352b6c1f2
# ╠═f10b7cad-eda3-4ec9-99ee-d43ed013a057
# ╠═05354df5-a803-422f-87a3-1c56a34e8a48
# ╟─3d77d753-b247-4ead-a385-7cbbcfc3190b
# ╠═9e61b4c3-1a9f-41a7-9882-25ed797a7b8d
# ╠═c6a59b85-d031-4ad4-9e24-691494d08cde
# ╠═e50b8196-e804-473a-b3b5-e22fdb9d2f45
# ╠═f413ea94-36ca-4afc-8ca8-9a7e88101980
# ╟─2bfa1a57-b171-44c3-b0d7-b8dda48d26d7
# ╟─dbf7fc72-bfd0-4c57-a1a9-fb5881e16e7e
# ╟─db2c6bd4-ee6f-4ba9-b6ec-e7cf94389f93
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
# ╟─26d957aa-36d4-4b90-9b91-2d9d883877ea
# ╟─4f89ceab-297f-4c2c-9029-8d2d7fad084f
# ╟─fd4165dc-c3e3-4c4c-9605-167b5b4416da
# ╟─7738c156-8e1b-4723-9818-fba364822171
# ╟─3336ab15-9e9b-44af-a7d5-1d6472241e62
# ╟─f1a48773-2971-4069-a240-fd1e10aeb1ed
# ╟─0fcfd7d2-6ea3-4c75-bad3-7d0fdd6fde11
# ╟─285c6bfc-5f29-46e0-a2c1-8abbec74501b
# ╟─c98524b5-d6b3-469c-82a1-7d231cc792d6
# ╟─62ad57e5-1366-4635-859b-ccdab2efd3b8
# ╠═29e1d9ff-4375-455a-a69b-8dd0c2cac57d
# ╠═1d1a24c6-c166-49a2-aa21-7acf50b55a66
# ╠═21b66582-3fda-401c-9421-73ae2f455a75
# ╠═32bafa9e-a35e-4f54-9857-d269b47f95c3
# ╟─4e4f4adf-364f-49b9-9391-5050a4c1286a
# ╠═50c035e6-b892-4157-a52f-824578366977
# ╟─ef9489c3-2bff-431b-92c4-f1b9778040cf
# ╠═e7d48a13-b4e6-4633-898c-c13b3e7f68ea
# ╠═8b851c67-0c6e-4081-a8ed-b818c2902c2f
# ╟─a30578dd-aecb-46eb-b947-f009282cf2fc
# ╠═8ed39cdc-e99e-48ff-9973-66df41aa0f78
# ╠═dd463687-b73d-4e70-b2cf-97a56a0ad409
# ╠═57dcadc0-2da2-4521-aeaf-6fd01f4bd82b
# ╠═935adbcd-48ab-4a6f-907c-b04137ca3abe
# ╟─b6689b27-e8a2-44e4-8791-ce237767ee63
# ╠═310999a0-f212-4e69-a4cb-346b3f49f202
# ╠═6384a36d-1dac-4d72-9d7b-84511f20e6ca
# ╠═b4e180b4-5582-4491-89d3-1f730c4f2fbf
# ╟─b9ba1df0-5086-4c0f-a2c9-200c2be27294
# ╠═1bf98d4b-ffbf-4157-9496-38e2c8e92c94
# ╟─6acda59d-cb26-4ec0-8a91-2ec5a71bb2b7
# ╟─058d6dff-d6a5-47db-bba2-756d67b873d4
# ╠═2767c63e-eea4-4227-9d54-293826091e70
# ╟─63e5e978-3357-43db-b379-78e8dec4c224
# ╠═81c55206-bf59-4c4e-ac5e-77a46e31bec7
# ╠═96d0366e-dd40-4214-9fce-ec76b8496958
# ╠═368b2d2c-393a-4b25-9a7c-0404a41e5873
# ╠═d7af774f-5d58-40a4-a5cd-905fdbe460a3
# ╟─6ea40424-22a0-42b9-bfab-8d4903ab8d64
# ╠═f071c985-7be7-454e-8541-28416400882f
# ╟─aa96bbe7-49f4-4244-9c71-8d9b2b3ee065
# ╠═d5b394ac-b243-4825-a5c1-b30146500ef6
# ╠═caa11dd3-577d-4692-b889-3a38d0bf61e0
# ╟─9ec91fbc-b756-4074-a623-1d47925c8239
# ╠═41e5133c-db89-4407-9501-70e869616e9d
# ╠═b8bb7c85-0be8-4a87-96da-4e1b37aea96d
# ╠═d3d7fa67-ca7d-46e1-b705-e30ec9b09f6a
# ╟─8b0451bf-59b0-4e71-be84-549e23b5bfe7
# ╟─20c5c7bc-664f-4c04-8215-8f3a9a2095c9
# ╟─7f08fcaa-000d-422d-80b4-e58a2f489d74
