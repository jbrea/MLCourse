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

# ╔═╡ f63c04d4-eefe-11eb-3fda-1729ac6de2cb
begin
	using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using PlutoUI
    PlutoUI.TableOfContents()
end


# ╔═╡ f63c04e8-eefe-11eb-39f6-83b31ebe73e7
using Plots, OpenML, DataFrames, CSV

# ╔═╡ f63c0540-eefe-11eb-01d6-454a05f9182a
using StatsPlots

# ╔═╡ f63c0592-eefe-11eb-0d69-91d4d396c540
using MLJ, MLJLinearModels

# ╔═╡ f63c05e4-eefe-11eb-0848-9b4eb61d920c
using Random

# ╔═╡ 36a44a52-8a29-4f92-8136-2c66a81a1f5f
using Distributions

# ╔═╡ 0cfbd543-8b9b-406e-b3b4-c6cafbbec212
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ f63c04dc-eefe-11eb-1e24-1d02a686920a
md"In this notebook we have a first look at some data sets and the [MLJ machine learning package](https://alan-turing-institute.github.io/MLJ.jl).

# Data Sets for Supervised Learning

## MNIST Handwritten Digit Recognition"


# ╔═╡ f63c04e8-eefe-11eb-1a14-8305504a6f1c
mnist = OpenML.load(554) |> DataFrame;

# ╔═╡ f63c04f2-eefe-11eb-2562-e530426e4300
md"With the semicolon we omit the output of a cell.
"

# ╔═╡ f63c04f2-eefe-11eb-3ceb-ff1e36a2a302
mnist.class # labels is a vector of 70'000 integers between 0 and 9.

# ╔═╡ f63c04fc-eefe-11eb-35b6-5345dda134e7
size(mnist)

# ╔═╡ f63c04fc-eefe-11eb-043c-7fec2fc41913
md"The input of this data set consists of 70'000 grayscale images of 28x28 pixels.

We can plot different input images with the following code.
"

# ╔═╡ f63c0506-eefe-11eb-1857-e3eaf731c152
plot(Gray.(reshape(Array(mnist[19, 1:end-1]) ./ 255, 28, 28)'))

# ╔═╡ c82fb649-e60d-43fc-b158-444b27b6e9cf
md"Explanation: `data[19, 1:end-1]` gets all pixels of the 19th image.
Next we transform the dataframe row to an `Array`. In OpenML, the inputs are
 8-bit integer; we transform to values between 0 and 1 by dividing by 255.
Then we reshape the vector of 784 entries to a matrix of size 28x28 with `reshape`.
`'` transposes the matrix (alternatively we could use `PermutedDimsArray`; look it up in the Live docs, if you are interested). `Gray.(...)` applies the `Gray` function elementwise to all the pixels, to reinterpret these numbers as grayscale values."

# ╔═╡ f63c0506-eefe-11eb-399f-455ee06548da
mnist.class[19] # the label corresponding to image 19 is a 6 :)

# ╔═╡ f63c050e-eefe-11eb-1709-0ddce2aee0ed
Markdown.parse("## Spam Detection

We load the spam data from a csv file on the harddisk to a DataFrame.
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
spam = CSV.read(joinpath(@__DIR__, "..", "data", "spam.csv"), DataFrame)

# ╔═╡ f63c051a-eefe-11eb-16eb-a572fe41aa43
spam[spam.label .== "ham", :] # select only ham

# ╔═╡ f63c0524-eefe-11eb-3abd-63d677b12db9
spam[spam.label .== "spam", :] # select only spam

# ╔═╡ f63c0524-eefe-11eb-3732-258d6989e217
md"## Wind Speed Prediction

We load the weather data from a csv file on the harddisk to a DataFrame.
Our goal is to predict the wind speed in Lucerne from 5 hours old measurements.
"

# ╔═╡ f63c052e-eefe-11eb-3a14-e5e8f3d578a8
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), DataFrame)[1:10000,:] # to speed up plotting, we use only the first 10'000 rows of the data set. Change `10000` to `end`, if you want to see all data.

# ╔═╡ f63c052e-eefe-11eb-0884-7bd433ce0c5e
y = weather.LUZ_wind_peak[6:end] # from hour 6 until the end we take all wind values

# ╔═╡ f63c0538-eefe-11eb-2a4b-d19efe6b6689
X = weather.LUZ_pressure[1:end-5] # pressure from the first hour until the fifth last hour

# ╔═╡ f63c0538-eefe-11eb-2eca-2d8bf12fae95
histogram2d(X, y, markersize = 3, xlabel = "LUZ_pressure [hPa]",
	        legend = false, ylabel = "LUZ_wind_peak [km/h]", bins = (250, 200))

# ╔═╡ f63c0538-eefe-11eb-2808-0f32a2fa84cf
md"The [StatsPlots package](https://github.com/JuliaPlots/StatsPlots.jl) has some
nice functions to explore data visually."

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

In the following we run in code the same example we had on the blackboard."

# ╔═╡ f63c05a6-eefe-11eb-0ef4-df7e458e9ec6
md"For now we use the same data set as in the blackboard example."

# ╔═╡ f63c05a6-eefe-11eb-2927-b1e9dbbe032d
training_data = DataFrame(x = [0., 2., 2.], y = [-1., 4., 3.])

# ╔═╡ f63c05b2-eefe-11eb-03cd-ef71f9055eb3
begin
    scatter(training_data.x, training_data.y,
            xrange = (-1, 3), label = "training data",
            legend = :topleft)
    plot!(x -> 2x - 1, label = "mean data generator")
end

# ╔═╡ f63c05b2-eefe-11eb-21dc-1f79ae779316
md"Now we define a supervised learning machine. The function family, loss
   function and optimiser are implicitly defined by choosing `LinearRegressor()`.
   All machines for supervised learning have as arguments a method, the input
   and the output."

# ╔═╡ f63c05ba-eefe-11eb-18b5-7522b326ab65
mach = machine(LinearRegressor(),         # method
               select(training_data, :x), # input
               training_data.y);          # output

# ╔═╡ f63c05ba-eefe-11eb-1b7a-21cbd9bbeb37
fit!(mach); # fit the machine

# ╔═╡ f63c05c4-eefe-11eb-030e-6f7484849852
fitted_params(mach) # show the result

# ╔═╡ 25eb24a4-5414-481b-88d5-5ad50d75f8c9
md"We can use the fitted machine and the `predict` function to compute ŷ (the predictions)."

# ╔═╡ f0f6ac8f-2e61-4b08-9383-45ff5b02c5b1
predict(mach, DataFrame(x = [0.5, 1.5])) # the second argument is the new input.

# ╔═╡ 7336e7b4-a7c2-4409-8d9d-00285cf633fb
md"Next, we define a function to compute the mean squared error loss for a given machine `mach` and a data set `data`."

# ╔═╡ f63c05c4-eefe-11eb-2d60-f7bf7beb8118
L(mach, data) = mean((predict(mach, select(data, :x)) .- data.y).^2);

# ╔═╡ f63c05ce-eefe-11eb-1ccd-c9b7714f3bd5
L(mach, training_data) # this is the training data

# ╔═╡ f63c05d8-eefe-11eb-2a11-fd8f954bf059
function MLCourse.fitted_linear_func(mach)
    θ̂ = fitted_params(mach)
    θ̂₀ = θ̂.intercept
    θ̂₁ = θ̂.coefs[1][2]
    x -> θ̂₀ + θ̂₁ * x
end; # this function extracts the fitted parameters and returns a function with these parameters.

# ╔═╡ f63c05d8-eefe-11eb-0a2e-970192b02d61
plot!(fitted_linear_func(mach), color = :green, label = "fit")

# ╔═╡ f63c05e4-eefe-11eb-012a-9bae1d87f2b5
md"Next we actually generate data from the data generator defined in the blackboard example.

For reproducibility we will fix the seed of the random number generator with the slider below. To do so we load the package `Random` first."

# ╔═╡ f63c059c-eefe-11eb-208c-d5781711deb7
md"With the sliders below you can define the noise level of the data generator, the number n of training points and the seed of the (pseudo-)random number generator.

σ = $(@bind σ Slider(.1:.1:2, show_value = true))

n = $(@bind n Slider(2:5:100, default = 52, show_value = true))

seed = $(@bind seed Slider(1:30, show_value = true))
"

# ╔═╡ f63c0592-eefe-11eb-2a76-15de55eff3ad
data_generator(x) = 2x .- 1 .+ σ * randn(length(x));

# ╔═╡ f63c05ec-eefe-11eb-0a92-215711928d02
begin
    Random.seed!(seed)
    input = randn(n)
    generated_test_data = DataFrame(x = -1:.01:3, y = data_generator(-1:.01:3))
    generated_data = DataFrame(x = input, y = data_generator(input))
end;

# ╔═╡ f63c05f6-eefe-11eb-0682-8d2651409f12
mach2 = machine(LinearRegressor(), select(generated_data, :x), generated_data.y) |> fit!;

# ╔═╡ f63c0600-eefe-11eb-3779-df58afdb52ad
begin
    scatter(generated_data.x, generated_data.y, label = "training data", xrange = (-3.3, 3.3), yrange = (-9, 7), legend = :topleft)
    plot_residuals!(generated_data.x, generated_data.y, fitted_linear_func(mach2))
    plot!(fitted_linear_func(mach2), color = :green, label = "fit")
end

# ╔═╡ f63c0600-eefe-11eb-0023-9d5e5ddd987e
(training_loss = L(mach2, generated_data), test_loss = L(mach2, generated_test_data))

# ╔═╡ 0e6efce9-093a-4758-a649-6cff525711a5
md"# Distributions

Use the sliders below to get a feeling for how the distributions depend on ``f(x)``.
In the plots, the triangle indicates the current value of ``f(x)``.

## Normal

``f(x)`` = $(@bind ŷₙ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ fbc70eaa-df15-423a-9885-93a5fa27fbc5
begin
	normalpdf(m) = x -> pdf(Normal(m, 1), x)
	logistic(x) = 1 / (1 + exp(-x))
	softmax(x) = exp.(x)/sum(exp.(x))
end;

# ╔═╡ 2adab164-1b4f-4f2f-ade3-dd70544c692e
begin
    plot(normalpdf(ŷₙ), xrange = (-8, 8), label = nothing,
         yrange = (0, .41), xlabel = "y", ylabel = "probability density", size = (350, 250))
    scatter!([ŷₙ], [0], markersize = 8, shape = :utriangle, label = nothing)
end

# ╔═╡ 8060861a-1931-4fae-b6ee-afb94ef2a3d5
md"## Bernoulli

f(x) = $(@bind ŷₛ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 6124e1eb-0aec-4ac9-aee6-383b25865220
let
	p1 = bar([0, 1], [logistic(ŷₛ), logistic(-ŷₛ)],
               xtick = ([0, 1], ["A", "B"]), ylabel = "probability",
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
bar(1:4, softmax([ŷ₁, ŷ₂, ŷ₃, ŷ₄]), ylabel = "probability",
    ylim = (0, 1), xlim = (.5, 4.5), xlabel = "class", legend = false,
    xtick = (1:4, ["A", "B", "C", "D"]), size = (350, 250))

# ╔═╡ f63c0616-eefe-11eb-268a-271fdc2ddd98
md"# Logistic Regression (Linear Classification)"


# ╔═╡ f63c0616-eefe-11eb-1cea-dfdaa64e6233
classification_data = DataFrame(x = [0., 1., 1.], y = ["A", "B", "A"])

# ╔═╡ f63c0616-eefe-11eb-1bfa-5b2ef11fcf4d
coerce!(classification_data, :y => Multiclass) # with this we tell the computer to interpret the data in column y as multi-class data (in this case there are only 2 classes).

# ╔═╡ f63c061e-eefe-11eb-095b-8be221b33d49
mach3 = machine(LogisticClassifier(penalty = :none),
                select(classification_data, :x),
                classification_data.y) |> fit!;

# ╔═╡ f63c061e-eefe-11eb-3b91-7136b4a16616
predict(mach3, DataFrame(x = -1:.5:2)) # click on the little gray triangle above to toggle the display of the full output.

# ╔═╡ f63c0628-eefe-11eb-3125-077e533456d9
predict_mode(mach3, DataFrame(x = -1:.5:2))

# ╔═╡ f63c0632-eefe-11eb-3b93-8549af7aaed9
md"""# Exercises

## Conceptual
1. We have some training data ``((x_1, y_1), (x_2, y_2), \ldots,
   (x_n, y_n))`` and a family of probability densities
   ``p(y|x) = \frac1{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - \theta_0 - \theta_1 x)^2}{2\sigma^2}\right)``.
   - Write the log-likelihood function of the parameters ``\theta_0`` and ``\theta_1``
     for this data and model.
   - Find the optimum of this log-likelihood function. Hint: compare your
     likelihood-function with the loss function of our blackboard example of
     linear regression as a loss minimizing machine.
1. Explain whether each scenario is a classification or regression problem, and
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
1. Take one of the examples of supervised machine learning applications
   discussed in the introductory lecture of this course. Discuss with a colleague
   - the data generating process for X
   - the data generating process for Y|X
   - a distribution that could be used to model Y|X

## Applied
1. Change the noise level σ, the size of the training data n and the seed with the sliders of the section "Linear Regression" and observe the training and the test losses. Write down your observations.
1. Write a data generator function that samples inputs ``x`` normally distributed
   with mean 2 and standard deviation 3. The response ``y\in\{\mbox{true, false}\}``
   should be sampled from a Bernoulli distribution with rate of ``\mbox{true}``
   equal to ``\sigma(0.5x - 2.7)``.
   - Create a training set of size ``n = 50``.
   - Fit the data with logistic regression and look at the fitted parameters.
   - Repeat the 2 steps above with a training set of size ``n = 10^4``.
"""

# ╔═╡ e3aa458a-486c-4d5b-a330-67fb68e6f516
MLCourse.footer()

# ╔═╡ Cell order:
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
# ╠═f63c0592-eefe-11eb-0d69-91d4d396c540
# ╟─f63c05a6-eefe-11eb-0ef4-df7e458e9ec6
# ╠═f63c05a6-eefe-11eb-2927-b1e9dbbe032d
# ╠═f63c05b2-eefe-11eb-03cd-ef71f9055eb3
# ╟─f63c05b2-eefe-11eb-21dc-1f79ae779316
# ╠═f63c05ba-eefe-11eb-18b5-7522b326ab65
# ╠═f63c05ba-eefe-11eb-1b7a-21cbd9bbeb37
# ╠═f63c05c4-eefe-11eb-030e-6f7484849852
# ╟─25eb24a4-5414-481b-88d5-5ad50d75f8c9
# ╠═f0f6ac8f-2e61-4b08-9383-45ff5b02c5b1
# ╟─7336e7b4-a7c2-4409-8d9d-00285cf633fb
# ╠═f63c05c4-eefe-11eb-2d60-f7bf7beb8118
# ╠═f63c05ce-eefe-11eb-1ccd-c9b7714f3bd5
# ╠═f63c05d8-eefe-11eb-2a11-fd8f954bf059
# ╠═f63c05d8-eefe-11eb-0a2e-970192b02d61
# ╟─f63c05e4-eefe-11eb-012a-9bae1d87f2b5
# ╠═f63c05e4-eefe-11eb-0848-9b4eb61d920c
# ╠═f63c0592-eefe-11eb-2a76-15de55eff3ad
# ╟─f63c059c-eefe-11eb-208c-d5781711deb7
# ╟─f63c05ec-eefe-11eb-0a92-215711928d02
# ╠═f63c05f6-eefe-11eb-0682-8d2651409f12
# ╟─f63c0600-eefe-11eb-3779-df58afdb52ad
# ╠═f63c0600-eefe-11eb-0023-9d5e5ddd987e
# ╟─0e6efce9-093a-4758-a649-6cff525711a5
# ╠═36a44a52-8a29-4f92-8136-2c66a81a1f5f
# ╟─2adab164-1b4f-4f2f-ade3-dd70544c692e
# ╠═fbc70eaa-df15-423a-9885-93a5fa27fbc5
# ╟─8060861a-1931-4fae-b6ee-afb94ef2a3d5
# ╟─6124e1eb-0aec-4ac9-aee6-383b25865220
# ╟─5d3efbe5-8937-45c0-894d-d24ece9b2262
# ╟─897fc69d-f1e9-49e9-9a61-25eb6849f7ec
# ╟─f63c0616-eefe-11eb-268a-271fdc2ddd98
# ╠═f63c0616-eefe-11eb-1cea-dfdaa64e6233
# ╠═f63c0616-eefe-11eb-1bfa-5b2ef11fcf4d
# ╠═f63c061e-eefe-11eb-095b-8be221b33d49
# ╠═f63c061e-eefe-11eb-3b91-7136b4a16616
# ╠═f63c0628-eefe-11eb-3125-077e533456d9
# ╟─f63c0632-eefe-11eb-3b93-8549af7aaed9
# ╟─0cfbd543-8b9b-406e-b3b4-c6cafbbec212
# ╟─f63c04d4-eefe-11eb-3fda-1729ac6de2cb
# ╟─e3aa458a-486c-4d5b-a330-67fb68e6f516
