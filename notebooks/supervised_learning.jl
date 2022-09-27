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

# ╔═╡ f63c04d4-eefe-11eb-3fda-1729ac6de2cb
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ f63c04e8-eefe-11eb-39f6-83b31ebe73e7
using Plots, OpenML, DataFrames, CSV, MLCourse

# ╔═╡ f63c0540-eefe-11eb-01d6-454a05f9182a
using StatsPlots

# ╔═╡ f63c0592-eefe-11eb-0d69-91d4d396c540
# using MLJ makes the functions machine, fit!, fitted_params, predict, etc. available
# MLJLinearModel contains LinearRegressor, LogisticClassifier etc.
using MLJ, MLJLinearModels

# ╔═╡ f63c05e4-eefe-11eb-0848-9b4eb61d920c
using Random

# ╔═╡ 36a44a52-8a29-4f92-8136-2c66a81a1f5f
using Distributions

# ╔═╡ 11e350b2-70bd-437b-acef-af5103a6eb96
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ f63c04dc-eefe-11eb-1e24-1d02a686920a
md"In this notebook we have a first look at some data sets and the MLJ machine learning package. You may want to look at its [documentation](https://alan-turing-institute.github.io/MLJ.jl/dev) or [cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/dev/mlj_cheatsheet/).

# Data Sets for Supervised Learning

## MNIST Handwritten Digit Recognition"


# ╔═╡ f63c04e8-eefe-11eb-1a14-8305504a6f1c
mnist = OpenML.load(554, maxbytes = 10^5) |> DataFrame;

# ╔═╡ f63c04f2-eefe-11eb-2562-e530426e4300
md"With the keyword argument `maxbytes = 10^5`, we load only a small fraction of the dataset. We can load the full dataset by omitting this argument or setting it to `maxbytes = typemax(Int)`. With the semicolon we omit the output of a cell.
"

# ╔═╡ f63c04f2-eefe-11eb-3ceb-ff1e36a2a302
mnist.class # this is a vector of integers between 0 and 9.

# ╔═╡ f63c04fc-eefe-11eb-35b6-5345dda134e7
size(mnist)

# ╔═╡ f63c04fc-eefe-11eb-043c-7fec2fc41913
md"The input of this data set consists of grayscale images of 28x28 pixels.

We can plot different input images with the following code.
"

# ╔═╡ f63c0506-eefe-11eb-1857-e3eaf731c152
plot(Gray.(reshape(Array(mnist[19, 1:end-1]) ./ 255, 28, 28)'))

# ╔═╡ c82fb649-e60d-43fc-b158-444b27b6e9cf
md"Explanation: `data[19, 1:end-1]` gets all pixels of the 19th image.
Next we transform the dataframe row to an `Array`. In OpenML, the inputs are
 8-bit integer; we transform to values between 0 and 1 by dividing by 255.
Then we reshape the vector of 784 entries to a matrix of size 28x28 with `reshape`.
The symbol `'` transposes the matrix (alternatively we could use `PermutedDimsArray`; look it up in the Live docs, if you are interested). `Gray.(...)` applies the `Gray` function elementwise to all the pixels, to reinterpret these numbers as grayscale values."

# ╔═╡ f63c0506-eefe-11eb-399f-455ee06548da
mnist.class[19] # the label corresponding to image 19 is a 6 :)

# ╔═╡ f63c050e-eefe-11eb-1709-0ddce2aee0ed
Markdown.parse("## Spam Detection

We load the (first 50 rows of the ) spam data from a csv file on the harddisk to a DataFrame.
Our goal is to classify emails as spam or \"ham\".
The email texts are preprocessed. Html tags, numbers and punctuation is removed,
all characters are transformed to lowercase and words are reduced to their
[stem](https://en.wikipedia.org/wiki/Word_stem).  If you are interested in the
details of this preprocessing, you find more information about text
preprocessing
[here](https://juliahub.com/docs/TextAnalysis/5Mwet/0.7.3/documents/) and the
preprocessing code in the scripts folder of the MLCourse package
($(normpath(joinpath(dirname(pathof(MLCourse)), "..", "scripts", "preprocess_spam.jl")))).
")

# ╔═╡ f63c051a-eefe-11eb-0db2-c519292f88a2
spam = CSV.read(joinpath(@__DIR__, "..", "data", "spam.csv"), DataFrame, limit = 50)

# ╔═╡ f63c051a-eefe-11eb-16eb-a572fe41aa43
spam[spam.label .== "ham", :] # select only ham

# ╔═╡ f63c0524-eefe-11eb-3abd-63d677b12db9
spam[spam.label .== "spam", :] # select only spam

# ╔═╡ f63c0524-eefe-11eb-3732-258d6989e217
md"## Wind Speed Prediction

We load (the first 5000 rows of) the weather data from a csv file on the harddisk to a DataFrame.
Our goal is to predict the wind speed in Lucerne from 5 hours old measurements.
"

# ╔═╡ f63c052e-eefe-11eb-3a14-e5e8f3d578a8
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"),
                   DataFrame, limit = 5000)

# ╔═╡ 0f475660-ccea-453d-9ee5-96e6d27bc35a
md"The data that we loaded contains already a column with name `LUZ_wind_peak`. However, this is the measurement of the current wind peak. What we would like to do, is to predict the wind peak in 5 hours. We have hourly measurements. We can see this by looking at the `time` column; `2015010113` means the measurement was done in the 13th hour of the first of January 2015. This means that the wind peak value we would like to predict given the measurements in the first row, is contained in the 6th row of the `LUZ_wind_peak` column, etc. Therefore, we take as our response variable `y` all wind peak measurements from hour 6 to the end of the dataframe and as input data all measurements from hour 1 until 5 hours before the end of the dataframe."

# ╔═╡ f63c052e-eefe-11eb-0884-7bd433ce0c5e
y = weather.LUZ_wind_peak[6:end] # from hour 6 until the end we take all wind values

# ╔═╡ f63c0538-eefe-11eb-2a4b-d19efe6b6689
X = weather.LUZ_pressure[1:end-5] # pressure from the first hour until the fifth last hour

# ╔═╡ f63c0538-eefe-11eb-2eca-2d8bf12fae95
histogram2d(X, y, markersize = 3, xlabel = "LUZ_pressure [hPa]",
	        legend = false, ylabel = "LUZ_wind_peak [km/h]", bins = (250, 200))

# ╔═╡ f63c0538-eefe-11eb-2808-0f32a2fa84cf
md"The [StatsPlots package](https://github.com/JuliaPlots/StatsPlots.jl) has some
nice functions to explore data visually. We will use the `@df` macro, which allows
to pass columns as symbols (note that the symbols are not separated by a comma).
Have a look at the [readme of the StatsPlots package](https://github.com/JuliaPlots/StatsPlots.jl) for more examples."

# ╔═╡ f63c054c-eefe-11eb-13db-171084b417a9
@df weather corrplot([:BAS_pressure :LUG_pressure :LUZ_pressure :LUZ_wind_peak],
                     grid = false, fillcolor = cgrad(), size = (700, 600))

# ╔═╡ f63c0560-eefe-11eb-0821-73afecf00357
md"## Other Data Sets

There are many data sets collected by the [OpenML community](https://www.openml.org).
We use them as follows."

# ╔═╡ f63c0560-eefe-11eb-0251-2d0443448e66
OpenML.list_datasets(output_format = DataFrame)

# ╔═╡ f63c056a-eefe-11eb-199e-2b05b1385be4
OpenML.describe_dataset(61)

# ╔═╡ f63c056a-eefe-11eb-218c-afc51fdf1704
iris = OpenML.load(61) |> DataFrame

# ╔═╡ f63c0574-eefe-11eb-2d7c-bd74b7d83f23
md"# Function Family

In the following example we use the function family
    ``\hat y = f_\theta(x) = \theta_0 + \theta_1 x`` from the slides.
You can control the parameters ``\theta_0`` and ``\theta_1`` to find the
function that visually matches the data well. The loss is the mean squared error. This corresponds to the average squared length of the red vertical lines. On the right we track the loss of the currently defined function.
"

# ╔═╡ f63c057e-eefe-11eb-1672-19db9efd0d32
md"θ₀ = $(@bind θ₀ Slider(-1.8:.1:1.8, default = 0, show_value = true))"

# ╔═╡ f63c057e-eefe-11eb-0dd2-3333b4e2d220
md"θ₁ = $(@bind θ₁ Slider(-3:.1:3, default = 0, show_value = true))"

# ╔═╡ f63c0588-eefe-11eb-0687-a7c44f9177a4
let x = [0.62, 0.49, 0.28, 0.54, 0.0, 0.84, 0.29, 0.12, 0.49, 0.37, 0.36, 0.8, 0.63, 0.18, 0.2, 0.43, 0.68, 0.23, 0.72],
    y = [0.0, 0.35, -0.36, 0.31, -0.94, 1.0, -0.54, -1.35, -0.12, -0.45, 0.28, 0.48, 0.57, -0.48, -1.05, 0.32, 0.35, -0.39, 1.07]
    scatter(x, y,
            yrange = (-2, 2), xrange = (-.1, 1.1),
            xlabel = "x", ylabel = "ŷ", label = "data")
    p1 = plot!(x -> θ₀ + θ₁ * x, label = "function", title = "data & function")
	plot_residuals!(x, y, x -> θ₀ + θ₁ * x)
    p2 = scatter([θ₀], [θ₁], label = "current loss", color = :red)
    p2 = contour!(-1.8:.1:1.8, -3:.1:3, (θ₀, θ₁) -> mean(abs2, θ₀ .+ θ₁ * x .- y),
                 xlabel = "θ₀", ylabel = "θ₁", title = "loss", legend = :bottomright,
                 contour_labels = true, levels = collect(2 .^ -3:.5:12),
                 color = cgrad([:green, :red], [0, 10]), cbar = false,
                 )
    plot(p1, p2,
         layout = (1, 2), size = (700, 400))
end

# ╔═╡ f63c0588-eefe-11eb-1b0b-db185fcbdc4e
md"# Linear Regression

## Blackboard Example
In the following we run in code the same example we had on the blackboard."

# ╔═╡ f63c05a6-eefe-11eb-0ef4-df7e458e9ec6
md"For now we use the same data set as in the blackboard example."

# ╔═╡ f63c05a6-eefe-11eb-2927-b1e9dbbe032d
training_data = DataFrame(x = [0., 2., 2.], y = [-1., 4., 3.])

# ╔═╡ f63c05b2-eefe-11eb-03cd-ef71f9055eb3
begin
    scatter(training_data.x, training_data.y,
            xrange = (-1, 3), label = "training data", legend = :topleft)
    plot!(x -> 2x - 1, label = "mean data generator")
end

# ╔═╡ f63c05b2-eefe-11eb-21dc-1f79ae779316
md"Now we define a supervised learning machine. The function family, loss
   function and optimiser are implicitly defined by choosing `LinearRegressor()`.
   All machines for supervised learning have as arguments a method, the input
   and the output."

# ╔═╡ f63c05ba-eefe-11eb-18b5-7522b326ab65
mach = machine(LinearRegressor(),         # model
               select(training_data, :x), # input
               training_data.y);          # output

# ╔═╡ f63c05ba-eefe-11eb-1b7a-21cbd9bbeb37
fit!(mach, verbosity = 0); # fit the machine

# ╔═╡ f63c05c4-eefe-11eb-030e-6f7484849852
fitted_params(mach) # show the result

# ╔═╡ 813238b8-3ce3-4ce8-98f6-cbdcbd1746d6
md"Here, the `intercept` corresponds to $\theta_0$ in the slides and `coefs[1].second` to $\theta_1$. Why do we need to write `coefs[1].second`? The coefficients returned by the `fitted_params` function are, in general, a vector of pairs (`:x => 2.25` is a pair, with first element the name of the data column this coefficient multiplies and the second element the value of the coefficient.). We can also extract these numbers, e.g."

# ╔═╡ 92bf4b9c-4bda-4430-bd0f-d85e09dd5d54
fitted_params(mach).intercept

# ╔═╡ 5911642f-151b-4d28-8109-6feb4d418ddf
fitted_params(mach).coefs[1].second # or fitted_params(mach).coefs[1][2]

# ╔═╡ 25eb24a4-5414-481b-88d5-5ad50d75f8c9
md"We can use the fitted machine and the `predict` function to compute $\hat y = \hat\theta_0 + \hat\theta_1 x$ (the predictions)."

# ╔═╡ bd9a7214-f39e-4ae5-995a-6ec02d613fda
predict(mach) # predictions on the training input x = [0, 2, 2]

# ╔═╡ f0f6ac8f-2e61-4b08-9383-45ff5b02c5b1
predict(mach, DataFrame(x = [0.5, 1.5])) # the second argument is the test input x = [0.5, 1.5]

# ╔═╡ 7336e7b4-a7c2-4409-8d9d-00285cf633fb
md"Next, we define a function to compute the mean squared error for a given machine `mach` and a data set `data`."

# ╔═╡ f63c05c4-eefe-11eb-2d60-f7bf7beb8118
my_mse(mach, data) = mean((predict(mach, select(data, :x)) .- data.y).^2);

# ╔═╡ f63c05ce-eefe-11eb-1ccd-c9b7714f3bd5
my_mse(mach, training_data) # this is the training data

# ╔═╡ 8fe94709-3673-44cc-8702-83bd7e2cad51
md"We can get the same result by squaring the result obtained with the MLJ function `rmse` (root mean squared error)."

# ╔═╡ 27f44b28-f6d6-4e1f-8dbe-6fe7789b9a18
rmse(predict(mach), training_data.y)^2

# ╔═╡ 43b28121-5d07-4400-8710-287de7b978a4
md"For plotting we will define a new function `fitted_linear_func` that extracts the fitted parameters from a machine and returns a linear function with these parameters."

# ╔═╡ f63c05d8-eefe-11eb-2a11-fd8f954bf059
function fitted_linear_func(mach)
    θ̂ = fitted_params(mach)
    θ̂₀ = θ̂.intercept
    θ̂₁ = θ̂.coefs[1].second
    x -> θ̂₀ + θ̂₁ * x # an anonymous function
end;

# ╔═╡ f63c05d8-eefe-11eb-0a2e-970192b02d61
begin
    scatter(training_data.x, training_data.y,
            xrange = (-1, 3), label = "training data", legend = :topleft)
    plot!(x -> 2x - 1, label = "mean data generator")
	plot!(fitted_linear_func(mach), color = :green, label = "fit")
end

# ╔═╡ f63c05e4-eefe-11eb-012a-9bae1d87f2b5
md"Next we actually generate data from the data generator defined in the blackboard example."

# ╔═╡ f63c0592-eefe-11eb-2a76-15de55eff3ad
function data_generator(x, σ)
	y = 2x .- 1 .+ σ * randn(length(x))
	DataFrame(x = x, y = y)
end;

# ╔═╡ ca8e60e6-1d12-4076-8684-9a7c9e5ac509
md"We can use this generator to create a large test set and compute empirically the test loss of the fitted machine."

# ╔═╡ 4987ad0c-d100-4a5d-a725-12f23bcc7aa3
my_mse(mach, data_generator(fill(1.5, 10^4), .5)) # test loss at 1.5 for σ = 0.5

# ╔═╡ 93f529ab-a386-4746-94de-63a536b5e2aa
my_mse(mach, data_generator(randn(10^4), .5)) # test loss of the joint generator

# ╔═╡ 673a773c-2f6b-4676-8631-49ba08ec28a7
md"
## Training and test loss as a function of $n$ and $\sigma$
We will now look at how the training and the test loss depend on the number of samples $n$ and the noise of the data generator $\sigma$.

For reproducibility we fix the seed of the random number generator with the slider below. To do so we load the package `Random`."

# ╔═╡ f63c059c-eefe-11eb-208c-d5781711deb7
md"With the sliders below you can define the noise level of the data generator, the number n of training points and the seed of the (pseudo-)random number generator.

σ = $(@bind σ Slider(.1:.1:2, show_value = true))

n = $(@bind n Slider(2:5:100, default = 52, show_value = true))

seed = $(@bind seed Slider(1:30, show_value = true))
"

# ╔═╡ f63c05ec-eefe-11eb-0a92-215711928d02
begin
    Random.seed!(seed)
    generated_data = data_generator(randn(n), σ)
	generated_test_data = data_generator(-1:.01:3, σ)
	mach2 = machine(LinearRegressor(),
	            select(generated_data, :x),
	            generated_data.y)
	fit!(mach2, verbosity = 0)
end;

# ╔═╡ f63c0600-eefe-11eb-3779-df58afdb52ad
begin
    scatter(generated_data.x, generated_data.y, label = "training data", xrange = (-3.3, 3.3), yrange = (-9, 7), legend = :topleft)
    plot_residuals!(generated_data.x, generated_data.y, fitted_linear_func(mach2))
    plot!(fitted_linear_func(mach2), color = :green, label = "fit")
end

# ╔═╡ f63c0600-eefe-11eb-0023-9d5e5ddd987e
(training_loss = my_mse(mach2, generated_data),
 test_loss = my_mse(mach2, generated_test_data))

# ╔═╡ 0e6efce9-093a-4758-a649-6cff525711a5
md"# Distributions

Use the sliders below to get a feeling for how the distributions depend on ``f(x)``.
In the plots, the triangle indicates the current value of ``f(x)``. Note that ``f(x)`` can be an arbitrarily complicated function of ``x``. What matters is that the output of this function is used to shape the probability distribution.
"

# ╔═╡ fbc70eaa-df15-423a-9885-93a5fa27fbc5
begin
	normalpdf(m) = x -> pdf(Normal(m, 1), x)
	logistic(x) = 1 / (1 + exp(-x))
	softmax(x) = exp.(x)/sum(exp.(x))
end;

# ╔═╡ 4b3b509f-7995-46b3-9130-8d881ee1e0ae
md"## Normal

``f(x)`` = $(@bind ŷₙ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 2adab164-1b4f-4f2f-ade3-dd70544c692e
begin
    plot(normalpdf(ŷₙ), xrange = (-8, 8), label = nothing,
         yrange = (0, .41), xlabel = "y", ylabel = "probability density p(y|x)", size = (350, 250))
    scatter!([ŷₙ], [0], markersize = 8, shape = :utriangle, label = nothing)
end

# ╔═╡ 8060861a-1931-4fae-b6ee-afb94ef2a3d5
md"## Bernoulli

f(x) = $(@bind ŷₛ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 6124e1eb-0aec-4ac9-aee6-383b25865220
let
	p1 = bar([0, 1], [logistic(ŷₛ), logistic(-ŷₛ)],
               xtick = ([0, 1], ["A", "B"]), ylabel = "probability P(class|x)",
               xlabel = "class", ylim = (0, 1), xlim = (-.5, 1.5), label = nothing)
    p2 = plot(logistic, xlim = (-5, 5), ylim = (0, 1),
               xlabel = "f(x)", ylabel = "probability of A", label = nothing)
    scatter!([ŷₛ], [0], markersize = 8, shape = :utriangle, label = nothing)
    vline!([ŷₛ], linestyle = :dash, color = :black)
    hline!([logistic(ŷₛ)], linestyle = :dash, color = :black)
    plot(p1, p2, size = (700, 350), legend = false)
end

# ╔═╡ 5d3efbe5-8937-45c0-894d-d24ece9b2262
md"## Categorical

f₁(x) = $(@bind ŷ₁ Slider(-5:.1:5, show_value = true))

f₂(x) = $(@bind ŷ₂ Slider(-5:.1:5, show_value = true))

f₃(x) = $(@bind ŷ₃ Slider(-5:.1:5, show_value = true))

f₄(x) = $(@bind ŷ₄ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 897fc69d-f1e9-49e9-9a61-25eb6849f7ec
bar(1:4, softmax([ŷ₁, ŷ₂, ŷ₃, ŷ₄]), ylabel = "probability P(class|x)",
    ylim = (0, 1), xlim = (.5, 4.5), xlabel = "class", legend = false,
    xtick = (1:4, ["A", "B", "C", "D"]), size = (350, 250))

# ╔═╡ f63c0616-eefe-11eb-268a-271fdc2ddd98
md"# Logistic Regression (Linear Classification)

## Blackboard Example"


# ╔═╡ f63c0616-eefe-11eb-1cea-dfdaa64e6233
classification_data = DataFrame(x = [0., 2., 3.],
	                            y = categorical(["B", "A", "B"],
									            levels = ["B", "A"],
								                ordered = true))

# ╔═╡ 7dfa42e3-c409-43a7-9952-a64fbad63c7f
md"In the cell above we use the function `categorical` to tell the computer that the strings \"A\" and \"B\" indicate membership in different categories."

# ╔═╡ f63c061e-eefe-11eb-095b-8be221b33d49
mach3 = machine(LogisticClassifier(penalty = :none), # model
                select(classification_data, :x),     # input
                classification_data.y);              # output

# ╔═╡ 9a208086-1fcd-4520-b566-ea295e934d75
fit!(mach3, verbosity = 0);

# ╔═╡ 1bbbb7fe-494b-4692-aaa4-e39599851327
fitted_params(mach3)

# ╔═╡ 38ccf735-5195-4e00-868f-95a895c05985
md"In the following cell we define the log-likelihood loss function `ll` to compute the loss of the optimal parameters (found by the `fit!` function). This is the formula we derived in the slides. Note that the training data is somewhat hidden in this formula: the response of the first data point is a B (see above) at x = 0, therefore its probability is `logistic(-(θ[1] + θ[2]*0)) = logistic(-θ[1])` etc."

# ╔═╡ d0c4804f-c66d-4405-a7dc-1e85974e261f
ll(θ) = log(logistic(-θ[1])) +
        log(logistic(θ[1] + 2θ[2])) +
        log(logistic(-θ[1] - 3θ[2]))

# ╔═╡ b8971599-494a-4fb3-9fa0-fb374fb7d79d
ll([-1.28858, .338548])

# ╔═╡ b8b81c1b-0faf-4ce9-b690-2f6cc9542b0f
md"The likelihood of the optimal parameters is approximately -1.9. We could have obtained the same result using the `MLJ` function `log_loss`. This function computes the negative log-likelihood for each individual data point. To get the total likelihood we need to take the negative of the sum of these values."

# ╔═╡ d034a44b-e331-4929-9053-351e7fe9aa94
-sum(log_loss(predict(mach3), classification_data.y))

# ╔═╡ 0e775dfb-0da4-4536-886c-ada8c176a073
md"For the `LogisticClassifier` the `predict` method returns the conditional probabilities of the classes. Click on the little gray triangle below to toggle the display of the full output."

# ╔═╡ f63c061e-eefe-11eb-3b91-7136b4a16616
p̂ = predict(mach3, DataFrame(x = -1:.5:2))

# ╔═╡ 0c90f5b6-8a3b-41d8-9f51-d7d7c6b06ba0
md"If we want to extract the probability of a given response, we can use the `pdf` function."

# ╔═╡ 5224d406-4e02-424d-9502-a22e0614cb96
pdf.(p̂, "A")

# ╔═╡ 34c49e49-a5e7-48ad-807b-c0624a59a367
md"If we want to get as a response the class with the highest probability, we can use the function `predict_mode`."

# ╔═╡ f63c0628-eefe-11eb-3125-077e533456d9
predict_mode(mach3, DataFrame(x = -1:.5:2))

# ╔═╡ f63c0632-eefe-11eb-3b93-8549af7aaed9
md"""# Exercises

## Conceptual
#### Exercise 1
We have some training data ``((x_1 = 0, y_1 = -1), (x_2 = 2, y_2 = 4),
   (x_3 = 2, y_3 = 3))`` and a family of probability densities
   ``p(y|x) = \frac1{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - \theta_0 - \theta_1 x)^2}{2\sigma^2}\right)``.
   - Write the log-likelihood function of the parameters ``\theta_0`` and ``\theta_1``
     for this data and model.
   - Find the parameters ``\hat\theta_0`` and ``\hat\theta_1`` that maximize the log-likelihood function and compare your result to the solution we found in our blackboard example of linear regression as a loss minimizing machine.
   - Show for general training data ``((x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n))`` that the log-likelihood function is maximized by the same ``\hat\theta`` that minimizes the loss function of linear regression.
#### Exercise 2
   Explain whether each scenario is a classification or regression problem, and
   indicate whether we are most interested in inference/interpretation or prediction.
   Finally, provide ``n`` and ``p``.
   - We collect a set of data on the top 500 firms in the US. For each firm we
     record profit, number of employees, industry and the CEO salary. We are
     interested in understanding which factors affect CEO salary.
   - We are considering launching a new product and wish to know
     whether it will be a success or a failure. We collect data on 20
     similar products that were previously launched. For each product
     we have recorded whether it was a success or failure, price
     charged for the product, marketing budget, competition price,
     and ten other variables.
   - We are interested in predicting the % change in the USD/Euro
     exchange rate in relation to the weekly changes in the world
     stock markets. Hence we collect weekly data for all of 2012. For
     each week we record the % change in the USD/Euro, the %
     change in the US market, the % change in the British market,
     and the % change in the German market.
#### Exercise 3
Take one of the examples of supervised machine learning applications
   discussed in the introductory lecture of this course. Discuss with a colleague
   - the data generating process for X
   - the data generating process for Y|X
   - where the noise comes from
   - a distribution that could be used to model Y|X

## Applied
#### Exercise 4
In this exercise we construct an example, where the response looks noisy, if we consider only some predictors, although the response would be deterministic if we consider all predictors. For this example we assume that the time it takes for a feather to reach the ground when dropping it from one meter is completely determined by the following factors: fluffiness of the feather (``f \in [1, 4]``), shape of the feather (``s = \{\mbox{s, e}\}``, for spherical or elongated), air density (``ρ ∈ [1.1, 1.4]`` kg/m³), wind speed (``w ∈ [0, 5]`` m/s). We assume that time it takes to reach the ground is deterministically given by the function
```math
g(f, s, ρ, w) = ρ + 0.1f^2 + 0.3w + 2χ(s = \mbox{s}) + 0.5χ(s = \mbox{e})
```
where we use the indicator function ``χ(\mbox{true}) = 1`` and ``χ(\mbox{false}) = 0``.
1. Generate an artificial dataset that corresponds to 500 experiments of dropping different feathers under different conditions. *Hints:* you can sample from a uniform distribution over the interval ``[a, b]`` using either `rand(500)*(b-a) .+ a` or `rand(Uniform(a, b), 500)` and you can sample categorical values in the same way as we did it in the first week when sampling column C in exercise 1. To implement the indicator function you can use for example the syntax `(s == :s)` and `(s == :e)`.
2. Create a scatter plot with fluffiness of the feather on the horizontal axis and time to reach the ground on the vertical axis.
3. Argue, why it looks like the time to reach the ground depended probabilistically on the fluffiness, although we used a deterministic function to compute the time to reach the ground. *Optional*: If you feel courageous, use a mathematical argument (marginalization) that uses the fact that $P(t|f, s, \rho, w) = \delta(t - g(f, s, \rho, w))$ is deterministic and shows that $P(t|f)$ is a non-degenerate conditional distribution.
"""

# ╔═╡ 19114da7-441a-4ada-b350-37c65a6211ee
md"""
#### Exercise 5
Change the noise level ``\sigma``, the size of the training data ``n`` and the seed with the sliders of the section "Linear Regression" and observe the training and the test losses. Write down your observations when
- ``n`` is small.
- ``n`` is large.
- Compare training loss and test loss when ``n`` is large for different seed values.
- Compare for large ``n`` the test error to ``\sigma^2`` for different values of ``\sigma``.
"""

# ╔═╡ 20fe6494-2214-48e9-9c05-61a5faf9f91f
md"""
#### Exercise 6
Write a data generator function that samples inputs ``x`` normally distributed
   with mean 2 and standard deviation 3. The response ``y\in\{\mbox{true, false}\}``
   should be sampled from a Bernoulli distribution with rate of ``\mbox{true}``
   equal to ``\sigma(0.5x - 2.7)`` where ``\sigma(x) = 1/(1 + e^{-x})`` is the sigmoid (or logistic) function.
   - Create a training set of size ``n = 20``.
   - Fit the data with logistic regression.
   - Look at the fitted parameters.
   - Predict the probability of class `true` on the training input.  *Hint*: use the `pdf` function.
   - Determine the class with highest predicted probability and compare the result to the labels of the training set.
   - Create a test set of size ``n = 10^4`` where the input is always at ``x = 4``. Estimate the average test error at ``x = 4`` using this test set. Use the negative log-likelihood as error function.
   - Compute the test error at ``x = 4`` using the fitted parameters and compare your result to the previous result. *Hint:* Have a look at the slides for how to compute the test error when the parameters of the generator and the fitted function are known.
   - Rerun your solution with different training sets of size ``n = 20`` and write down your observations.
   - Rerun your solution multiple times with training sets of size ``n = 10^4``. Compare the fitted parameters to the one of the generator and look at the test error. Write down your observations.
"""

# ╔═╡ 0cfbd543-8b9b-406e-b3b4-c6cafbbec212
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ e3aa458a-486c-4d5b-a330-67fb68e6f516
MLCourse.footer()

# ╔═╡ Cell order:
# ╠═f63c04d4-eefe-11eb-3fda-1729ac6de2cb
# ╟─f63c04dc-eefe-11eb-1e24-1d02a686920a
# ╠═f63c04e8-eefe-11eb-39f6-83b31ebe73e7
# ╠═f63c04e8-eefe-11eb-1a14-8305504a6f1c
# ╟─f63c04f2-eefe-11eb-2562-e530426e4300
# ╠═f63c04f2-eefe-11eb-3ceb-ff1e36a2a302
# ╠═f63c04fc-eefe-11eb-35b6-5345dda134e7
# ╟─f63c04fc-eefe-11eb-043c-7fec2fc41913
# ╠═f63c0506-eefe-11eb-1857-e3eaf731c152
# ╟─c82fb649-e60d-43fc-b158-444b27b6e9cf
# ╠═f63c0506-eefe-11eb-399f-455ee06548da
# ╟─f63c050e-eefe-11eb-1709-0ddce2aee0ed
# ╠═f63c051a-eefe-11eb-0db2-c519292f88a2
# ╠═f63c051a-eefe-11eb-16eb-a572fe41aa43
# ╠═f63c0524-eefe-11eb-3abd-63d677b12db9
# ╟─f63c0524-eefe-11eb-3732-258d6989e217
# ╠═f63c052e-eefe-11eb-3a14-e5e8f3d578a8
# ╟─0f475660-ccea-453d-9ee5-96e6d27bc35a
# ╠═f63c052e-eefe-11eb-0884-7bd433ce0c5e
# ╠═f63c0538-eefe-11eb-2a4b-d19efe6b6689
# ╠═f63c0538-eefe-11eb-2eca-2d8bf12fae95
# ╟─f63c0538-eefe-11eb-2808-0f32a2fa84cf
# ╠═f63c0540-eefe-11eb-01d6-454a05f9182a
# ╠═f63c054c-eefe-11eb-13db-171084b417a9
# ╟─f63c0560-eefe-11eb-0821-73afecf00357
# ╠═f63c0560-eefe-11eb-0251-2d0443448e66
# ╠═f63c056a-eefe-11eb-199e-2b05b1385be4
# ╠═f63c056a-eefe-11eb-218c-afc51fdf1704
# ╟─f63c0574-eefe-11eb-2d7c-bd74b7d83f23
# ╟─f63c057e-eefe-11eb-1672-19db9efd0d32
# ╟─f63c057e-eefe-11eb-0dd2-3333b4e2d220
# ╟─f63c0588-eefe-11eb-0687-a7c44f9177a4
# ╟─f63c0588-eefe-11eb-1b0b-db185fcbdc4e
# ╟─f63c05a6-eefe-11eb-0ef4-df7e458e9ec6
# ╠═f63c05a6-eefe-11eb-2927-b1e9dbbe032d
# ╠═f63c05b2-eefe-11eb-03cd-ef71f9055eb3
# ╟─f63c05b2-eefe-11eb-21dc-1f79ae779316
# ╠═f63c0592-eefe-11eb-0d69-91d4d396c540
# ╠═f63c05ba-eefe-11eb-18b5-7522b326ab65
# ╠═f63c05ba-eefe-11eb-1b7a-21cbd9bbeb37
# ╠═f63c05c4-eefe-11eb-030e-6f7484849852
# ╟─813238b8-3ce3-4ce8-98f6-cbdcbd1746d6
# ╠═92bf4b9c-4bda-4430-bd0f-d85e09dd5d54
# ╠═5911642f-151b-4d28-8109-6feb4d418ddf
# ╟─25eb24a4-5414-481b-88d5-5ad50d75f8c9
# ╠═bd9a7214-f39e-4ae5-995a-6ec02d613fda
# ╠═f0f6ac8f-2e61-4b08-9383-45ff5b02c5b1
# ╟─7336e7b4-a7c2-4409-8d9d-00285cf633fb
# ╠═f63c05c4-eefe-11eb-2d60-f7bf7beb8118
# ╠═f63c05ce-eefe-11eb-1ccd-c9b7714f3bd5
# ╟─8fe94709-3673-44cc-8702-83bd7e2cad51
# ╠═27f44b28-f6d6-4e1f-8dbe-6fe7789b9a18
# ╟─43b28121-5d07-4400-8710-287de7b978a4
# ╠═f63c05d8-eefe-11eb-2a11-fd8f954bf059
# ╠═f63c05d8-eefe-11eb-0a2e-970192b02d61
# ╟─f63c05e4-eefe-11eb-012a-9bae1d87f2b5
# ╠═f63c0592-eefe-11eb-2a76-15de55eff3ad
# ╟─ca8e60e6-1d12-4076-8684-9a7c9e5ac509
# ╠═4987ad0c-d100-4a5d-a725-12f23bcc7aa3
# ╠═93f529ab-a386-4746-94de-63a536b5e2aa
# ╟─673a773c-2f6b-4676-8631-49ba08ec28a7
# ╠═f63c05e4-eefe-11eb-0848-9b4eb61d920c
# ╟─f63c059c-eefe-11eb-208c-d5781711deb7
# ╠═f63c05ec-eefe-11eb-0a92-215711928d02
# ╟─f63c0600-eefe-11eb-3779-df58afdb52ad
# ╠═f63c0600-eefe-11eb-0023-9d5e5ddd987e
# ╟─0e6efce9-093a-4758-a649-6cff525711a5
# ╠═36a44a52-8a29-4f92-8136-2c66a81a1f5f
# ╠═fbc70eaa-df15-423a-9885-93a5fa27fbc5
# ╟─4b3b509f-7995-46b3-9130-8d881ee1e0ae
# ╟─2adab164-1b4f-4f2f-ade3-dd70544c692e
# ╟─8060861a-1931-4fae-b6ee-afb94ef2a3d5
# ╟─6124e1eb-0aec-4ac9-aee6-383b25865220
# ╟─5d3efbe5-8937-45c0-894d-d24ece9b2262
# ╟─897fc69d-f1e9-49e9-9a61-25eb6849f7ec
# ╟─f63c0616-eefe-11eb-268a-271fdc2ddd98
# ╠═f63c0616-eefe-11eb-1cea-dfdaa64e6233
# ╟─7dfa42e3-c409-43a7-9952-a64fbad63c7f
# ╠═f63c061e-eefe-11eb-095b-8be221b33d49
# ╠═9a208086-1fcd-4520-b566-ea295e934d75
# ╠═1bbbb7fe-494b-4692-aaa4-e39599851327
# ╟─38ccf735-5195-4e00-868f-95a895c05985
# ╠═d0c4804f-c66d-4405-a7dc-1e85974e261f
# ╠═b8971599-494a-4fb3-9fa0-fb374fb7d79d
# ╟─b8b81c1b-0faf-4ce9-b690-2f6cc9542b0f
# ╠═d034a44b-e331-4929-9053-351e7fe9aa94
# ╟─0e775dfb-0da4-4536-886c-ada8c176a073
# ╠═f63c061e-eefe-11eb-3b91-7136b4a16616
# ╟─0c90f5b6-8a3b-41d8-9f51-d7d7c6b06ba0
# ╠═5224d406-4e02-424d-9502-a22e0614cb96
# ╟─34c49e49-a5e7-48ad-807b-c0624a59a367
# ╠═f63c0628-eefe-11eb-3125-077e533456d9
# ╟─f63c0632-eefe-11eb-3b93-8549af7aaed9
# ╟─19114da7-441a-4ada-b350-37c65a6211ee
# ╟─20fe6494-2214-48e9-9c05-61a5faf9f91f
# ╟─0cfbd543-8b9b-406e-b3b4-c6cafbbec212
# ╟─11e350b2-70bd-437b-acef-af5103a6eb96
# ╟─e3aa458a-486c-4d5b-a330-67fb68e6f516
