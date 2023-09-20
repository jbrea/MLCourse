### A Pluto.jl notebook ###
# v0.19.27

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
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames
import Distributions: Normal
import PlutoPlotly as PP
# Base.download(s::AbstractString) = joinpath(@__DIR__, "..", "data", replace(s, "https://go.epfl.ch/bio322-" => ""))
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ 11e350b2-70bd-437b-acef-af5103a6eb96
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ f63c04dc-eefe-11eb-1e24-1d02a686920a
md"The goal of this week is to

1. Get to know some datasets we will use for supervised learning.
2. Understand linear regression from two equivalent perspectives: as a machine that tunes its own parameters to A) minimize the mean squared error loss or B) maximize the log-likelihood.
3. Translate the blackboard example of linear regression into code and understand the concept of a data generator.
3. Know how to perform (multiple) linear regression on a given dataset.

# 1. Data Sets for Supervised Learning

## MNIST Handwritten Digit Recognition

The MNIST Handwritten Digit dataset consists of 70'000 standardized photos of handwritten digits in the format of 28x28 pixels grayscale images, with corresponding labels
"

# ╔═╡ f63c04e8-eefe-11eb-1a14-8305504a6f1c
mlcode(
"""
using OpenML, DataFrames
mnist = OpenML.load(554, maxbytes = 10^5) |> DataFrame
"""
,
"""
import pandas as pd
import openml

mnist,_,_,_ = openml.datasets.get_dataset(554).get_data(dataset_format="dataframe")
mnist
"""
)

# ╔═╡ f63c04f2-eefe-11eb-2562-e530426e4300
mlstring(md"With the keyword argument `maxbytes = 10^5`, we load only a small fraction of the dataset. We can load the full dataset by omitting this argument or setting it to `maxbytes = typemax(Int)`.

Let us have a look at the column `class` which contains the class labels for all images.
",
""
)

# ╔═╡ f63c04f2-eefe-11eb-3ceb-ff1e36a2a302
mlcode(
"""
mnist.class
"""
,
"""
mnist['class']
"""
,
recompute = false
)

# ╔═╡ f63c04fc-eefe-11eb-35b6-5345dda134e7
mlcode(
"""
size(mnist)
"""
,
"""
mnist.shape
"""
)

# ╔═╡ f63c04fc-eefe-11eb-043c-7fec2fc41913
mlstring(md"Note that we loaded only a small fraction of images, which is why the result of size(mnist) is equal to `(56, 785)`, i.e. 56 images of 28x28=784 pixels plus one column for the class labels.
"
,
"
"
)

# ╔═╡ 601a953d-68f7-4fd0-a793-fef3c72a6821
md"We can plot different input images with the following code."


# ╔═╡ f63c0506-eefe-11eb-1857-e3eaf731c152
mlcode(
"""
using Plots

plot(Gray.(reshape(Array(mnist[19, 1:end-1]) ./ 255, 28, 28)'))
"""
,
"""
import matplotlib.pyplot as plt

plt.imshow(mnist.iloc[18, :-1].values.reshape(28, 28).astype(float)/255,
           cmap='gray')
plt.show()
"""
)

# ╔═╡ c82fb649-e60d-43fc-b158-444b27b6e9cf
mlstring(md"Explanation: `data[19, 1:end-1]` gets all pixels of the 19th image.
Next we transform the dataframe row to an `Array`. In OpenML, the inputs are
 8-bit integer; we transform to values between 0 and 1 by dividing by 255.
Then we reshape the vector of 784 entries to a matrix of size 28x28 with `reshape`.
The symbol `'` transposes the matrix (alternatively we could use `PermutedDimsArray`; look it up in the Live docs, if you are interested). `Gray.(...)` applies the `Gray` function elementwise to all the pixels, to reinterpret these numbers as grayscale values.

Not surprisingly, the class label for the image plotted above is \"6\", as we can confirm below.
"
,
md"
Explanation: `data.iloc[18, :-1]` gets all pixels of the 19th image.
Then we reshape the vector of 784 entries to a matrix of size 28x28 with `reshape`.
Next we transform to values between 0 and 1 by dividing by 255.
`cmap='gray'`interpret these numbers as grayscale values.

Not surprisingly, the class label for the image plotted above is \"6\", as we can confirm below.
"
)

# ╔═╡ f63c0506-eefe-11eb-399f-455ee06548da
mlcode(
"""
mnist.class[19]
"""
,
"""
mnist['class'][18]
"""
)

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
preprocessing code in the [scripts folder of the MLCourse package](https://github.com/jbrea/MLCourse/scripts).
")

# ╔═╡ f63c051a-eefe-11eb-0db2-c519292f88a2
mlcode(
"""
using CSV
spam = CSV.read(download("https://go.epfl.ch/bio322-spam.csv"), DataFrame)
""",
"""
spam = pd.read_csv("https://go.epfl.ch/bio322-spam.csv")
spam
"""
)

# ╔═╡ f63c051a-eefe-11eb-16eb-a572fe41aa43
mlcode(
"""
spam[spam.label .== "ham", :] # only show non-spam email
""",
"""
spam[spam['label'] == 'ham'] # only show non-spam email
"""
)

# ╔═╡ f63c0524-eefe-11eb-3abd-63d677b12db9
mlcode(
"""
spam[spam.label .== "spam", :] # only show spam examples
""",
"""
spam[spam['label'] == 'spam']
"""
)

# ╔═╡ f63c0524-eefe-11eb-3732-258d6989e217
md"## Wind Speed Prediction

We load the weather data from a csv file on the harddisk to a DataFrame.
Our goal is to predict the wind speed in Lucerne from 5 hours old measurements.
"

# ╔═╡ f63c052e-eefe-11eb-3a14-e5e8f3d578a8
mlcode(
"""
using CSV

weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
                   DataFrame)
"""
,
"""
weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv")
weather
"""
,
)

# ╔═╡ 0f475660-ccea-453d-9ee5-96e6d27bc35a
md"The data that we loaded contains already a column with name `LUZ_wind_peak`. However, this is the measurement of the current wind peak. What we would like to do, is to predict the wind peak in 5 hours. We have hourly measurements. We can see this by looking at the `time` column; `2015010113` means the measurement was done in the 13th hour of the first of January 2015. This means that the wind peak value we would like to predict given the measurements in the first row, is contained in the 6th row of the `LUZ_wind_peak` column, etc. Therefore, we take as our response variable `y` all wind peak measurements from hour 6 to the end of the dataframe and as input data all measurements from hour 1 until 5 hours before the end of the dataframe."

# ╔═╡ f63c052e-eefe-11eb-0884-7bd433ce0c5e
mlcode(
"""
y = weather.LUZ_wind_peak[6:end] 
# from hour 6 until the end we take all wind values
"""
,
"""
y = weather["LUZ_wind_peak"][5:].values 
# from hour 6 until the end we take all wind values
y
"""
)

# ╔═╡ f63c0538-eefe-11eb-2a4b-d19efe6b6689
mlcode(
"""
X = weather.LUZ_pressure[1:end-5] 
# pressure from the first hour until the fifth last hour
"""
,
"""
X = weather["LUZ_pressure"][:-5].values 
# pressure from the first hour until the fifth last hour
X
"""
)

# ╔═╡ f63c0538-eefe-11eb-2eca-2d8bf12fae95
mlcode(
"""
histogram2d(X, y, markersize = 3, xlabel = "LUZ_pressure [hPa]",
	        legend = false, ylabel = "LUZ_wind_peak [km/h]", bins = (250, 200))
"""
,
"""
from matplotlib.colors import LogNorm
plt.hist2d(X, y, bins=(50, 50), cmap=plt.cm.jet, norm=LogNorm())
cb = plt.colorbar()
cb.set_label('counts')
plt.xlabel("LUZ_pressure [hPa]")
plt.ylabel("LUZ_wind_peak [km/h]")
plt.show()
"""
)

# ╔═╡ f63c0538-eefe-11eb-2808-0f32a2fa84cf
mlstring(md"The [StatsPlots package](https://github.com/JuliaPlots/StatsPlots.jl) has some
nice functions to explore data visually. We will use the `@df` macro, which allows
to pass columns as symbols (note that the symbols are not separated by a comma).
Have a look at the [readme of the StatsPlots package](https://github.com/JuliaPlots/StatsPlots.jl) for more examples."
,
""
)

# ╔═╡ ce30505e-7488-49f4-9e42-2ec457ca6aa8
mlcode(
"""
using StatsPlots
"""
,
"""
import seaborn as sns
"""
)

# ╔═╡ f63c054c-eefe-11eb-13db-171084b417a9
mlcode(
"""
using StatsPlots

@df weather corrplot([:BAS_pressure :LUG_pressure :LUZ_pressure :LUZ_wind_peak],
                     grid = false, fillcolor = cgrad(), size = (700, 600))
"""
,
"""
g = sns.PairGrid(weather[['BAS_pressure', 'LUG_pressure', 'LUZ_pressure', 'LUZ_wind_peak']])
g.map_lower(sns.regplot, line_kws={'color': 'black'})
g.map_diag(sns.histplot, color = 'darkorange' )
g.map_upper(sns.kdeplot, fill=True,cmap="Reds")
"""
,
recompute = false
)

# ╔═╡ f63c0574-eefe-11eb-2d7c-bd74b7d83f23
md"# 2. Simple Linear Regression
## Supervised Learning as Loss Minimization

In the following example we use the function family
    ``\hat y = f_\theta(x) = \theta_0 + \theta_1 x`` from the slides.
You can control the parameters ``\theta_0`` and ``\theta_1`` to find the
function that visually matches the data well.
"

# ╔═╡ d8a3bb08-df1e-4cf9-a04f-6156d27f0992
# begin
#     if linregsol == "Random Initialization"
#         theta0 = randn()/2
#         theta1 = randn()/2
#     else
#         theta0 = -1.1246030250560215
#         theta1 = 2.468858219596663
#     end
# end;

# ╔═╡ 782a1b8a-d15e-47a3-937d-0c25b136da3d
# @bind linregsol Select(["Random Initialization", "Minimal Loss Solution"])

# ╔═╡ f63c057e-eefe-11eb-1672-19db9efd0d32
md"""θ₀ = $(@bind θ₀ Slider(-1.8:.1:1.8, default = 0, show_value = true))

θ₁ = $(@bind θ₁ Slider(-3:.1:3, default = 0, show_value = true))
"""

# ╔═╡ f63c0588-eefe-11eb-0687-a7c44f9177a4
let x = [0.62, 0.49, 0.28, 0.54, 0.0, 0.84, 0.29, 0.12, 0.49, 0.37, 0.36, 0.8, 0.63, 0.18, 0.2, 0.43, 0.68, 0.23, 0.72],
    y = [0.0, 0.35, -0.36, 0.31, -0.94, 1.0, -0.54, -1.35, -0.12, -0.45, 0.28, 0.48, 0.57, -0.48, -1.05, 0.32, 0.35, -0.39, 1.07]
    scatter(x, y,
            yrange = (-2, 2), xrange = (-.1, 1.1),
            xlabel = "x", ylabel = "ŷ", label = "data")
    p1 = plot!(x -> θ₀ + θ₁ * x, label = "function", c = :blue, title = "data & function")
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

# ╔═╡ d01c2ac5-06c7-4d49-a2d2-20e2c80e3f84
md"**Left:** The data is given by the blue points, the current function by the blue line. The red vertical lines are the residuals: the distance between the function and the data. The mean squared error loss is the average squared length of the red lines.

**Right:** The mean squared error loss as a function of the parameters ``\theta_0`` and ``\theta_1``. The green curves are level lines, along which the loss is constant. The red dot is at the position of the current value of the parameters (given by the slider positions) and indicates the mean squared error.

## Supervised Learning as Likelihood Maximization
"

# ╔═╡ 4acb0dbf-6d10-42f9-adba-4a9358f54b38
# begin
#     if linmllsol == "Random Initialization"
#         theta02 = randn()/2
#         theta12 = randn()/2
#         sigma0 = 2*rand()
#     else
#         theta02 = -1.1246030250560215
#         theta12 = 2.468858219596663
#         sigma0 = 0.29951788038174776
#     end
# end;
# 

# ╔═╡ c0610eee-9d7e-40cf-8b46-5317b0070de9
# @bind linmllsol Select(["Random Initialization", "Maximum Log-Likelihood Solution"])

# ╔═╡ 65c228c1-f6f2-4542-9870-066448b182ee
md"""θ₀ = $(@bind θ₀₂ Slider(-1.8:.1:1.8, default = 0, show_value = true))

θ₁ = $(@bind θ₁₂ Slider(-3:.1:3, default = 0, show_value = true))

σ = $(@bind sigma Slider(.1:.1:2, default = .3, show_value = true))
"""

# ╔═╡ 5ba1c12a-75ce-4d01-aca5-8804d8e747cf
let x = [0.62, 0.49, 0.28, 0.54, 0.0, 0.84, 0.29, 0.12, 0.49, 0.37, 0.36, 0.8, 0.63, 0.18, 0.2, 0.43, 0.68, 0.23, 0.72],
    y = [0.0, 0.35, -0.36, 0.31, -0.94, 1.0, -0.54, -1.35, -0.12, -0.45, 0.28, 0.48, 0.57, -0.48, -1.05, 0.32, 0.35, -0.39, 1.07]
    xgrid = -.1:.05:1.1
    ygrid = -2:.05:2
    M = [pdf(Normal(θ₀₂+θ₁₂*x, sigma), y) for y in ygrid, x in xgrid]
    p1 = heatmap(xgrid, ygrid, M)
    p1 = scatter!(x, y,
                 yrange = (-2, 2), xrange = (-.1, 1.1),
                 xlabel = "x", ylabel = "ŷ", label = "data")
    p2 = scatter([θ₀₂ ], [θ₁₂], label = "current log-likelihood", color = :red)
    p2 = contour!(-1.8:.1:1.8, -3:.1:3, (θ₀, θ₁) -> mean(logpdf.(Normal.(θ₀ .+ θ₁ * x, sigma), y)),
                 xlabel = "θ₀", ylabel = "θ₁", title = "log-likelihood", legend = :bottomright,
                 contour_labels = true,
                 levels = round.(4 .* range(log(minimum(M)), log(maximum(M)), 100)) ./ 4,
                 cbar = false,
                 )
    plot(p1, p2,
         layout = (1, 2), size = (700, 400))
end

# ╔═╡ edb726aa-cca0-492b-bcf4-5ba80a5c1c44
md"**Left:** The data is given by the red points; the current probability density function is shown as a heatmap. The conditional probability of each data point is given by the color at the position of the point. The objective function to maximize is the mean log-likelihood.

**Right:** The mean log-likelihood as a function of the parameters ``\theta_0`` and ``\theta_1`` and ``\sigma``. The green curves are level lines, along which the log-likelihood is constant. The red dot is at the position of the current value of the parameters (given by the slider positions) and indicates the mean log-likelihood.

!!! note
    - The optimal solution in ``\theta_0`` and ``\theta_1`` is at the same value as the minimum of the mean squared error loss function above.
    - The optimal solution in ``\theta_0`` and ``\theta_1`` does not depend on the value of ``\sigma``. Changing ``\sigma`` only changes the \"steepness of the log-likelihood hill\".
"

# ╔═╡ c65b81bd-395f-4461-a73b-3535903cb2d7
md"## Multiple Linear Regression

In multiple linear regression there are multiple predictors ``x_1, x_2, \ldots, x_p``
and the response is ``\hat y = f_{\boldsymbol \beta}(\boldsymbol x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
`` (we use here ``\beta`` instead of ``\theta`` for the parameters).
With ``p = 2`` predictors we can visualize linear regression as the plane
that is closest to the data, in the sense that the vertical residuals (red lines) between the plane and the data are minimized. Use the sliders below to get a feeling for the
parameters. You can also change the viewing angle by clicking and dragging in
the figure."

# ╔═╡ 0f544053-1b7a-48d6-b18b-72092b124305
begin
    Random.seed!(3)
	X = DataFrame(X1 = randn(20), X2 = randn(20))
    f0(X1, X2, β₀, β₁, β₂) = β₀ + β₁ * X1 + β₂ * X2
    f0(β₀, β₁, β₂) = (X1, X2) -> f0(X1, X2, β₀, β₁, β₂)
	data_generator2(X1, X2; β₀, β₁, β₂, σ = 0.8) = f0(X1, X2, β₀, β₁, β₂) + σ * randn()
	y = data_generator2.(X.X1, X.X2, β₀ = .4, β₁ = .5, β₂ = -.6)
end;


# ╔═╡ 51c9ea74-3110-4536-a4af-7cc73b45a4a6
md"β₀ = $(@bind β₀ Slider(-1:.02:1, default = .4, show_value = true))

β₁ = $(@bind β₁ Slider(-1:.02:1, default = .2, show_value = true))

β₂ = $(@bind β₂ Slider(-1:.02:1, default = -.6, show_value = true))
"

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


# ╔═╡ c261ac30-b215-47a2-a22d-6706470b57b4
md"
**Left:** The data is given by the blue dots. The current function by the green plane. The residuals (red lines) are the vertical distance between the plane and the data.

**Right:** The loss function depends on ``\beta_0, \beta_1`` and ``\beta_2``. Because it is difficult to visualize a function with three-dimensional arguments, we show here two projections of the loss function: the level plot on top shows the loss as a function of ``\beta_0`` and ``\beta_1``, for a fixed value of ``\beta_2``, and the level plot at the bottom shows the loss as a function of ``beta_0`` and ``\beta_2``, for a fixed value of ``\beta_1``.

!!! note
    Also for multiple linear regression it is insightful to take the perspective of log-likelihood maximization. It is just a bit difficult to visualize. Can you imagine why?
"

# ╔═╡ f63c0588-eefe-11eb-1b0b-db185fcbdc4e
md"# 3. Blackboard Example
In the following we run in code the same example we had on the blackboard."

# ╔═╡ f63c05a6-eefe-11eb-0ef4-df7e458e9ec6
md"For now we use the same data set as in the blackboard example."

# ╔═╡ f63c05a6-eefe-11eb-2927-b1e9dbbe032d
mlcode(
"""
using DataFrames, Plots
training_data = DataFrame(x = [0., 2., 2.], y = [-1., 4., 3.])
scatter(training_data.x, training_data.y,
        xrange = (-1, 3), label = "training data", legend = :topleft)
plot!(x -> 2x - 1, label = "mean data generator")
"""
,
"""
import pandas as pd
import matplotlib.pyplot as plt

training_data = pd.DataFrame({'x': [0., 2., 2.], 'y': [-1., 4., 3.]})
plt.figure()
plt.scatter(training_data['x'], training_data['y'], label="training data")
plt.plot([-1, 3], [2*(-1) - 1, 2*3 - 1], c='red', label="mean data generator")
plt.legend(loc="upper left")
plt.show()
""",
)

# ╔═╡ f63c05b2-eefe-11eb-21dc-1f79ae779316
md"Now we define a supervised learning machine. The function family, loss
   function and optimiser are implicitly defined by choosing `LinearRegressor()`.
   All machines for supervised learning have as arguments a method, the input
   and the output."

# ╔═╡ f63c05ba-eefe-11eb-18b5-7522b326ab65
mlcode(
"""
# using MLJ makes the functions machine, fit!, fitted_params, predict, ... available
# MLJLinearModel contains LinearRegressor, LogisticClassifier etc.
using MLJ, MLJLinearModels
mach = machine(LinearRegressor(),         # model
               select(training_data, :x), # input
               training_data.y)           # output
fit!(mach, verbosity = 0); # fit the machine
fitted_params(mach) # show the result
"""
,
"""
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()  # Create a linear regression model
X_train = np.array(training_data['x']).reshape(-1, 1)
y_train = np.array(training_data['y']).reshape(-1, 1)
model.fit(X_train,y_train) # Fit the model to the data, with the Input features and the Output variables
"""
)


# ╔═╡ 813238b8-3ce3-4ce8-98f6-cbdcbd1746d6
md"Here, the `intercept` corresponds to $\theta_0$ in the slides and `coefs[1].second` to $\theta_1$. Why do we need to write `coefs[1].second`? The coefficients returned by the `fitted_params` function are, in general, a vector of pairs (`:x => 2.25` is a pair, with first element the name of the data column this coefficient multiplies and the second element the value of the coefficient.). We can also extract these numbers, e.g."

# ╔═╡ 92bf4b9c-4bda-4430-bd0f-d85e09dd5d54
mlcode(
"""
fitted_params(mach).intercept
"""
,
"""
model.intercept_
"""
)

# ╔═╡ 5911642f-151b-4d28-8109-6feb4d418ddf
mlcode(
"""
fitted_params(mach).coefs[1].second # or fitted_params(mach).coefs[1][2]
"""
,
"""
model.coef_
"""
)

# ╔═╡ 25eb24a4-5414-481b-88d5-5ad50d75f8c9
mlstring(
md"We can use the fitted machine and the `predict` function to compute $\hat y = \hat\theta_0 + \hat\theta_1 x$ (the predictions)."
,
""
)

# ╔═╡ bd9a7214-f39e-4ae5-995a-6ec02d613fda
mlcode(
"""
predict(mach) # predictions on the training input x = [0, 2, 2]
"""
,
"""
model.predict(X_train)
"""
)

# ╔═╡ f0f6ac8f-2e61-4b08-9383-45ff5b02c5b1
mlcode(
"""
predict(mach, DataFrame(x = [0.5, 1.5])) # the second argument is the test input x = [0.5, 1.5]
"""
,
"""
model.predict(np.array([0.5, 1.5]).reshape(-1, 1))
"""
)

# ╔═╡ 7336e7b4-a7c2-4409-8d9d-00285cf633fb
md"Next, we define a function to compute the mean squared error for a given machine `mach` and a data set `data`."

# ╔═╡ f63c05c4-eefe-11eb-2d60-f7bf7beb8118
mlcode(
"""
my_mse(mach, data) = mean((predict(mach, select(data, :x)) .- data.y).^2)
my_mse(mach, training_data) # this is the training data
"""
,
"""
def my_mse(model, data):
    x = np.array(data['x']).reshape(-1, 1)
    y_true = np.array(data['y']).reshape(-1, 1)
    y_pred = model.predict(x)
    mse = np.mean((y_pred - y_true) ** 2)
    return mse

mse = my_mse(model, training_data)
mse
"""
)


# ╔═╡ 8fe94709-3673-44cc-8702-83bd7e2cad51
md"We can get the same result by squaring the result obtained with the MLJ function `rmse` (root mean squared error)."

# ╔═╡ 27f44b28-f6d6-4e1f-8dbe-6fe7789b9a18
mlcode(
"""
rmse(predict(mach), training_data.y)^2
"""
,
"""
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_train, model.predict(X_train))
mse
"""
)

# ╔═╡ 43b28121-5d07-4400-8710-287de7b978a4
md"For plotting we will define a new function `fitted_linear_func` that extracts the fitted parameters from a machine and returns a linear function with these parameters."

# ╔═╡ f63c05d8-eefe-11eb-2a11-fd8f954bf059
mlcode(
"""
function fitted_linear_func(mach)
    θ̂ = fitted_params(mach)
    θ̂₀ = θ̂.intercept
    θ̂₁ = θ̂.coefs[1].second
    x -> θ̂₀ + θ̂₁ * x # an anonymous function
end
"""
,
"""
def fitted_linear_func(mach):
    theta_hat = mach.coef_
    theta_hat_0 = mach.intercept_
    theta_hat_1 = theta_hat[0]
    
    def linear_func(x):
        return theta_hat_0 + theta_hat_1 * x
    
    return linear_func
fitted_func = fitted_linear_func(model)
"""
)

# ╔═╡ f63c05d8-eefe-11eb-0a2e-970192b02d61
mlcode(
"""
scatter(training_data.x, training_data.y,
        xrange = (-1, 3), label = "training data", legend = :topleft)
plot!(x -> 2x - 1, label = "mean data generator")
plot!(fitted_linear_func(mach), color = :green, label = "fit")
"""
,
"""
import numpy as np

plt.scatter(training_data['x'], training_data['y'], label="training data") 

x_range = np.linspace(-1, 3, 100)
mean_data_generator = lambda x: 2 * x - 1
plt.plot(x_range, mean_data_generator(x_range), color = "orange", label="mean data generator")

fitted_func = fitted_linear_func(model)
plt.plot(x_range, fitted_func(x_range), color="green", label="fit")

# Set plot properties
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")

# Show the plot
plt.show()
"""
)

# ╔═╡ f63c05e4-eefe-11eb-012a-9bae1d87f2b5
md"
Next we actually generate data from the data generator defined in the blackboard example."

# ╔═╡ f63c0592-eefe-11eb-2a76-15de55eff3ad
mlcode(
"""
function data_generator(x, σ)
	y = 2x .- 1 .+ σ * randn(length(x))
	DataFrame(x = x, y = y)
end
"""
,
"""
def data_generator(x, sigma):
    y = 2 * x - 1 + sigma * np.random.randn(len(x))
    df = pd.DataFrame({'x': x, 'y': y})
    return df

data_generator(np.full((10**4,), 1.5), 0.5)
"""
)

# ╔═╡ ca8e60e6-1d12-4076-8684-9a7c9e5ac509
md"We can use this generator to create a large test set and compute empirically the test loss of the fitted machine."

# ╔═╡ 4987ad0c-d100-4a5d-a725-12f23bcc7aa3
mlcode(
"""
my_mse(mach, data_generator(fill(1.5, 10^4), .5)) # test loss at 1.5 for σ = 0.5
"""
,
"""
my_mse(model, data_generator(np.full((10**4,), 1.5), .5)) # test loss at 1.5 for σ = 0.5
"""
)

# ╔═╡ 93f529ab-a386-4746-94de-63a536b5e2aa
mlcode(
"""
my_mse(mach, data_generator(randn(10^4), .5)) # test loss of the joint generator
"""
,
"""
my_mse(model, data_generator(np.random.randn(10**4), .5)) # test loss of the joint generator
"""
)

# ╔═╡ 673a773c-2f6b-4676-8631-49ba08ec28a7
md"
## Training and test loss as a function of $n$ and $\sigma$
We will now look at how the training and the test loss depend on the number of samples $n$ and the noise of the data generator $\sigma$.
"

# ╔═╡ f63c059c-eefe-11eb-208c-d5781711deb7
md"With the sliders below you can define the noise level of the data generator, the number n of training points and the seed of the (pseudo-)random number generator.

σ = $(@bind σ Slider(.1:.1:2, show_value = true))

n = $(@bind n Slider(2:5:100, default = 52, show_value = true))

seed = $(@bind seed Slider(1:30, show_value = true))
"

# ╔═╡ 8217895b-b120-4b08-b18f-d921dfdddf10
md"# 4. Examples of Linear Regression
## Wind speed prediction with one predictor

In this section we apply linear regression to the weather dataset.
First, we will run a simple linear regression to predict the wind peak in 5 hours based on the current local pressure measurements.
"

# ╔═╡ d14244d4-b54e-42cf-b812-b149d9fa59e0
mlcode(
"""
training_set1 = DataFrame(LUZ_pressure = weather.LUZ_pressure[1:end-5],
                          wind_peak_in5h = weather.LUZ_wind_peak[6:end])
"""
,
"""
training_set1 = pd.DataFrame({
				'LUZ_pressure': weather['LUZ_pressure'][:-5].values,
				'wind_peak_in5h': weather['LUZ_wind_peak'][5:].values
				})
training_set1
"""
,
)

# ╔═╡ 34e527f2-ef80-4cb6-be3a-bee055eca125
mlcode(
"""
m1 = machine(LinearRegressor(),
			 select(training_set1, :LUZ_pressure),
	         training_set1.wind_peak_in5h)
fit!(m1, verbosity = 0)
fitted_params(m1) # Fit the model to the data, with the Input features and the Output variables
"""
,
"""
m1 = LinearRegression()  # Create a linear regression model
X_train = training_set1['LUZ_pressure'].values.reshape(-1, 1)
y_train = training_set1['wind_peak_in5h'].values.reshape(-1, 1)
m1.fit(X_train,y_train) # Fit the model to the data, with the Input features and the Output variables
"""
,
)

# ╔═╡ e4712ebe-f395-418b-abcc-e10ada4b05c2
md"
We find that there is a negative correlation between the pressure in Luzern
and the wind speed 5 hours later in Luzern.
"

# ╔═╡ 4158eaff-884c-4b64-bef9-9f4511cdc5a4
md"The fitted model can be used to make some predictions. Note that only the mean of the conditional distribution is predicted.
"

# ╔═╡ 9b62c374-c26e-4990-8ffc-790928e62e88
mlcode(
"""
predict(m1, (LUZ_pressure = [930., 960., 990.],))
"""
,
"""
m1.predict(np.array([930., 960., 990.]).reshape(-1, 1))
"""
)

# ╔═╡ 0087f48b-7b50-4796-9371-fc87cf33c55d
md"Also with those predictions we see that the expected wind peak decreases with pressure.

We will now use the predict function to plot the prediction on top of the wind peak histogram.
"

# ╔═╡ ded13e94-7c29-430e-8864-373e682b5509
mlcode(
"""
histogram2d(X, y, markersize = 3, xlabel = "LUZ_pressure [hPa]",
	        legend = false, ylabel = "LUZ_wind_peak [km/h]", bins = (250, 200))
plot!(X, predict(m1), linewidth = 3)
"""
,
"""
# Create the 2D histogram plot
plt.hist2d(X, y, bins=(50, 50), cmap=plt.cm.jet, norm=LogNorm())

# Set plot labels and properties
plt.xlabel('LUZ_pressure [hPa]')
plt.ylabel('LUZ_wind_peak [km/h]')
plt.colorbar(label='Counts')

plt.plot(X, m1.predict(X_train), linewidth=3, color='red')

# Show the plot
plt.show()
"""
)


# ╔═╡ 7923a0a8-3033-4dde-91e8-22bf540c9866
md"We use the root-mean-squared-error (`rmse`) = ``\sqrt{\frac1n\sum_{i=1}^n(y_i - \hat y_i)^2}`` to evaluate our training error. To compute the test error, we use hourly data from 2019 to 2020.
"

# ╔═╡ 57f352dc-55ee-4e14-b68d-698938a97d92
mlcode(
"""
rmse(predict(m1), training_set1.wind_peak_in5h)
"""
,
"""
mean_squared_error(m1.predict(X_train), training_set1["wind_peak_in5h"]) ** 0.5
"""
)

# ╔═╡ b0de002f-3f39-4378-8d68-5c4606e488b7
mlcode(
"""
weather_test = CSV.read(download("https://go.epfl.ch/bio322-weather2019-2020.csv"),
                        DataFrame);
test_set1 = DataFrame(LUZ_pressure = weather_test.LUZ_pressure[1:end-5],
                      wind_peak_in5h = weather_test.LUZ_wind_peak[6:end])
rmse(predict(m1, select(test_set1, :LUZ_pressure)), test_set1.wind_peak_in5h)
"""
,
"""
weather_test = pd.read_csv("https://go.epfl.ch/bio322-weather2019-2020.csv")
test_set1 = pd.DataFrame({
				'LUZ_pressure': weather_test['LUZ_pressure'][:-5].values,
				'wind_peak_in5h': weather_test['LUZ_wind_peak'][5:].values
})
mean_squared_error(m1.predict(test_set1["LUZ_pressure"].values.reshape(-1, 1)), test_set1["wind_peak_in5h"]) ** 0.5
"""
)

# ╔═╡ da6462d8-3343-41d8-82dd-48770176d4ba
md"## Wind speed prediction with multiple predictors
"

# ╔═╡ 6eca3452-d47d-4d70-82d1-bedbe91e9820
mlstring(
md"Our weather data set contains multiple measurements.
With `names(weather)` we see all the columns of the the weather data frame.

With `DataFrame(schema(weather))` we get additionally information about the type of data. `MLJ` distinguished between the [scientific type](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types) and the number type of a column. This distinction is useful, because e.g. categorical variables (class 1, class 2, class 3, ...) can be represented as an integer but also count variables (e.g. number of items) can be represented as an integer, but suitability of a given machine learning method depends more on the scientific type than on the number type.
",
md"Our weather data set contains multiple measurements.
With `weather.columns` we see all the names of the columns of the the weather data frame.
Try it in a new cell!

With `weather.dtypes, weather.head(), weather.info()` we get additionally information about the type of data.
"
)

# ╔═╡ 25e1f89f-a5b2-4d5c-a6e8-a990b3bebddb
mlcode(
"""
DataFrame(schema(weather))
"""
,
"""
weather.dtypes
"""
)

# ╔═╡ 8307e205-3bcd-4e68-914e-621fd8d29e43
mlstring(
md"When loading data one should check if the scientific type is correctly detected by the data loader. For the weather data the wind direction is automatically detected as a count variable, because the angle of the wind is measured in integer degrees. But it makes more sense to interpret the angle as a continuous variable. Therefore we use the `MLJ` function `coerce!` to transform all `Count` columns to `Continuous`.

After the transformation to the correct scientific type, we will define a machine and fit it on all measurements, except the response variable and the time variable.
"
,
md"
We define a machine and fit it on all measurements, except the response variable and the time variable.
"
)

# ╔═╡ 753ec309-1363-485d-a2bd-b9fa100d9058
mlcode(
"""
coerce!(weather, Count => Continuous)
m2 = machine(LinearRegressor(),
	         select(weather[1:end-5,:], Not([:LUZ_wind_peak, :time])),
             weather.LUZ_wind_peak[6:end])
fit!(m2, verbosity = 0)
"""
,
"""
m2 = LinearRegression()  # Create a linear regression model
m2.fit(weather.iloc[:-5].drop(['LUZ_wind_peak', 'time'], axis=1),weather['LUZ_wind_peak'][5:].values.reshape(-1, 1))
"""
;
showoutput = false
)

# ╔═╡ fac51c63-f227-49ac-89f9-205bf03e7c08
md"Let us have a look at the fitted parameters. For example, we can conclude that an increase of pressure in Luzern by 1hPa correlates with a decrease of almost 2km/h of the expected wind peaks - everything else being the same. This result is consistent with our expectation that high pressure locally is usually accompanied by good weather without strong winds. On the other hand, an increase of pressure in Genève by 1hPa correlates with an increase of more than 5km/h of the expected wind peaks. This result is also consistent with our expectations, given that stormy west wind weather occurs usually when there is a pressure gradient from south west to north east of Switzerland. Note also that the parameter for pressure in Pully (50 km north east from Genève) is -4.4km/h⋅hPa, i.e. when the pressure in Pully is the same as in Genève these two contributions to the prediction almost compensate each other."

# ╔═╡ 618ef3c7-0fda-4970-88e8-1dac195545de
mlcode(
"""
sort!(DataFrame(predictor = names(select(weather, Not([:LUZ_wind_peak, :time]))),
                value = last.(fitted_params(m2).coefs)), :value)
"""
,
"""
new_df = pd.DataFrame({
	'predictor': weather.columns.difference(['LUZ_wind_peak', 'time']).tolist(),
	'value': m2.coef_[0]
	})
new_df.sort_values(by='value')
"""
)

# ╔═╡ 2d25fbb6-dc9b-40ad-bdce-4c952cdad077
mlcode(
"""
rmse(predict(m2, select(weather[1:end-5,:], Not([:LUZ_wind_peak, :time]))),
     weather.LUZ_wind_peak[6:end])
"""
,
"""
np.sqrt(mean_squared_error(m2.predict(weather.iloc[:-5, :].drop(['LUZ_wind_peak', 'time'], axis=1)), weather['LUZ_wind_peak'][5:]))
"""
)

# ╔═╡ c9f10ace-3299-45fb-b98d-023a35dd405a
mlcode(
"""
rmse(predict(m2, select(weather_test[1:end-5,:], Not([:LUZ_wind_peak, :time]))),
     weather_test.LUZ_wind_peak[6:end])
"""
,
"""
np.sqrt(mean_squared_error(m2.predict(weather_test.iloc[:-5, :].drop(['LUZ_wind_peak', 'time'], axis=1)), weather_test['LUZ_wind_peak'][5:]))
"""
)

# ╔═╡ 2724c9b7-8caa-4ba7-be5c-2a9095281cfd
md"We see that both the training and the test error is lower, when using multiple predictors instead of just one. This means that more than one predictor carries useful information about the wind peak."


# ╔═╡ 08469a60-c96c-4c8a-8c8c-d7f1cb075ae3
begin
function data_generator(x, σ)
	y = 2x .- 1 .+ σ * randn(length(x))
	DataFrame(x = x, y = y)
end
function fitted_linear_func(mach)
    θ̂ = fitted_params(mach)
    θ̂₀ = θ̂.intercept
    θ̂₁ = θ̂.coefs[1].second
    x -> θ̂₀ + θ̂₁ * x # an anonymous function
end
my_mse(mach, data) = mean((predict(mach, select(data, :x)) .- data.y).^2)
end;

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
md"The **training loss** is $(round(my_mse(mach2, generated_data), sigdigits = 3)) and the **test loss** is $(round(my_mse(mach2, generated_test_data), sigdigits = 3)).
"


# ╔═╡ f63c0632-eefe-11eb-3b93-8549af7aaed9
md"""# Exercises

## Conceptual
#### Exercise 1
We have some training data ``((x_1 = 0, y_1 = -1), (x_2 = 2, y_2 = 4),
   (x_3 = 2, y_3 = 3))`` and a family of probability densities
   ``p(y|x) = \frac1{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - \theta_0 - \theta_1 x)^2}{2\sigma^2}\right)``.
   - Write the log-likelihood function of the parameters ``\theta_0`` and ``\theta_1``
     for this data and model.
   - Find the parameters ``\hat\theta_0`` and ``\hat\theta_1`` that maximize the log-likelihood function. *Hint*: You do not need to solve for the optimal parameters analytically; compare the loss function you derived in the previous step to the one we obtained in the lecture in the first blackboard example and use the fact that we know the optimum for the blackboard example.
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
Suppose we generated data with ``y = 3x + 1 + \epsilon`` with ``\mathrm{E}[\epsilon] = 0`` and ``\mathrm{Var}[\epsilon] = \sigma^2``.
We have training data ``\mathcal D = ((0, 1), (2, 9))`` and test data ``\mathcal D_\mathrm{test} = ((0, 0), (3, 20))``.
We define a function family ``f(x) = \theta_0 + \theta_1 x^2`` and loss function ``L(y, \hat y) = |y - \hat y|``.
Which of the following statements is correct?
1. The training loss is minimal for ``\hat\theta_0 = 1`` and ``\hat\theta_1 = 2``.
2. The test loss of ``f(x) = 1 + 2x^2`` at ``x = 0`` for the conditional data generating process is 1.
3. The test loss of ``f(x) = 1 + 2x^2`` for the test set is 1.


#### Exercise 4
Take one of the examples of supervised machine learning applications
   discussed in the introductory lecture of this course. Discuss with a colleague
   - the data generating process for X
   - the data generating process for Y|X
   - where the noise comes from
   - a distribution that could be used to model Y|X

"""

# ╔═╡ 61c67500-c2b5-4610-a916-3b316ba01cce
md"""
## Applied

$(MLCourse.language_selector())` `
"""


# ╔═╡ 1feab086-9464-4fb7-9673-bc0ff30bab81
md"""
#### Exercise 5
In this exercise we construct an example, where the response looks noisy, if we consider only some predictors, although the response would be deterministic if we consider all predictors. For this example we assume that the time it takes for a feather to reach the ground when dropping it from one meter is completely determined by the following factors: fluffiness of the feather (``f \in [1, 4]``), shape of the feather (``s = \{\mbox{s, e}\}``, for spherical or elongated), air density (``ρ ∈ [1.1, 1.4]`` kg/m³), wind speed (``w ∈ [0, 5]`` m/s). We assume that time it takes to reach the ground is deterministically given by the function
```math
g(f, s, ρ, w) = ρ + 0.1f^2 + 0.3w + 2I(s = \mbox{s}) + 0.5I(s = \mbox{e})
```
where we use the indicator function ``I(\mbox{true}) = 1`` and ``I(\mbox{false}) = 0``.
1. Generate an artificial dataset that corresponds to 500 experiments of dropping different feathers under different conditions. *Hints:* $(mlstring(md"you can sample from a uniform distribution over the interval ``[a, b]`` using either `rand(500)*(b-a) .+ a` or `rand(Uniform(a, b), 500)` and you can sample categorical values in the same way as we did it in the first week when sampling column C in exercise 1. To implement the indicator function you can use for example the syntax `(s == :s)` and `(s == :e)`.", ""))
2. Create a scatter plot with fluffiness of the feather on the horizontal axis and time to reach the ground on the vertical axis.
3. Argue, why it looks like the time to reach the ground depended probabilistically on the fluffiness, although we used a deterministic function to compute the time to reach the ground. *Optional*: If you feel courageous, use a mathematical argument (marginalization) that uses the fact that $P(t|f, s, \rho, w) = \delta(t - g(f, s, \rho, w))$ is deterministic and shows that $P(t|f)$ is a non-degenerate conditional distribution.
"""

# ╔═╡ 19114da7-441a-4ada-b350-37c65a6211ee
md"""
#### Exercise 6
Change the noise level ``\sigma``, the size of the training data ``n`` and the seed with the sliders of the section "Linear Regression" and observe the training and the test losses. Write down your observations when
- ``n`` is small.
- ``n`` is large.
- Compare training loss and test loss when ``n`` is large for different seed values.
- Compare for large ``n`` the test error to ``\sigma^2`` for different values of ``\sigma``.
"""


# ╔═╡ 45859648-ef10-4380-8bf8-bca281b958cb
begin
hint1 = mlstring(md"write a function `train_and_evaluate` that takes the training and the test data as input as well as an array of two predictors; remember that `data[:, [\"A\", \"B\"]]` returns a sub-dataframe with columns \"A\" and \"B\". This function should fit a `LinearRegressor` on the training set with those two predictors and return the test rmse for the two predictors. To get a list of all pairs of predictors you can use something like `predictors = setdiff(names(train), [\"time\", \"LUZ_wind_peak\"]); predictor_pairs = [[p1, p2] for p1 in predictors, p2 in predictors if p1 != p2 && p1 > p2]`", "")
md"""
#### Exercise 7


In the multiple linear regression of the weather data set above we used all available predictors. We do not know if all of them are relevant. In this exercise our aim is to find models with fewer predictors and quantify the loss in prediction accuracy.
- Systematically search for the model with at most 2 predictors that has the lowest test rmse. *Hint* $hint1
- How much higher is the test error compared to the fit with all available predictors?
- How many models did you have to fit to find your result above?
- How many models would you have to fit to find the best model with at most 5 predictors? *Hint* the function $(mlstring(md"`binomial`", md"")) may be useful.
"""
end

# ╔═╡ 0cfbd543-8b9b-406e-b3b4-c6cafbbec212
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ e3aa458a-486c-4d5b-a330-67fb68e6f516
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─f63c04dc-eefe-11eb-1e24-1d02a686920a
# ╟─f63c04e8-eefe-11eb-1a14-8305504a6f1c
# ╟─f63c04f2-eefe-11eb-2562-e530426e4300
# ╟─f63c04f2-eefe-11eb-3ceb-ff1e36a2a302
# ╟─f63c04fc-eefe-11eb-35b6-5345dda134e7
# ╟─f63c04fc-eefe-11eb-043c-7fec2fc41913
# ╟─601a953d-68f7-4fd0-a793-fef3c72a6821
# ╟─f63c0506-eefe-11eb-1857-e3eaf731c152
# ╟─c82fb649-e60d-43fc-b158-444b27b6e9cf
# ╟─f63c0506-eefe-11eb-399f-455ee06548da
# ╟─f63c050e-eefe-11eb-1709-0ddce2aee0ed
# ╟─f63c051a-eefe-11eb-0db2-c519292f88a2
# ╟─f63c051a-eefe-11eb-16eb-a572fe41aa43
# ╟─f63c0524-eefe-11eb-3abd-63d677b12db9
# ╟─f63c0524-eefe-11eb-3732-258d6989e217
# ╟─f63c052e-eefe-11eb-3a14-e5e8f3d578a8
# ╟─0f475660-ccea-453d-9ee5-96e6d27bc35a
# ╟─f63c052e-eefe-11eb-0884-7bd433ce0c5e
# ╟─f63c0538-eefe-11eb-2a4b-d19efe6b6689
# ╟─f63c0538-eefe-11eb-2eca-2d8bf12fae95
# ╟─f63c0538-eefe-11eb-2808-0f32a2fa84cf
# ╟─ce30505e-7488-49f4-9e42-2ec457ca6aa8
# ╟─f63c054c-eefe-11eb-13db-171084b417a9
# ╟─f63c0574-eefe-11eb-2d7c-bd74b7d83f23
# ╟─d8a3bb08-df1e-4cf9-a04f-6156d27f0992
# ╟─782a1b8a-d15e-47a3-937d-0c25b136da3d
# ╟─f63c057e-eefe-11eb-1672-19db9efd0d32
# ╟─f63c0588-eefe-11eb-0687-a7c44f9177a4
# ╟─d01c2ac5-06c7-4d49-a2d2-20e2c80e3f84
# ╟─4acb0dbf-6d10-42f9-adba-4a9358f54b38
# ╟─c0610eee-9d7e-40cf-8b46-5317b0070de9
# ╟─65c228c1-f6f2-4542-9870-066448b182ee
# ╟─5ba1c12a-75ce-4d01-aca5-8804d8e747cf
# ╟─edb726aa-cca0-492b-bcf4-5ba80a5c1c44
# ╟─c65b81bd-395f-4461-a73b-3535903cb2d7
# ╟─0f544053-1b7a-48d6-b18b-72092b124305
# ╟─51c9ea74-3110-4536-a4af-7cc73b45a4a6
# ╟─d541a8cd-5aa4-4c2d-bfdf-5e6297bb65a8
# ╟─c261ac30-b215-47a2-a22d-6706470b57b4
# ╟─f63c0588-eefe-11eb-1b0b-db185fcbdc4e
# ╟─f63c05a6-eefe-11eb-0ef4-df7e458e9ec6
# ╟─f63c05a6-eefe-11eb-2927-b1e9dbbe032d
# ╟─f63c05b2-eefe-11eb-21dc-1f79ae779316
# ╟─f63c05ba-eefe-11eb-18b5-7522b326ab65
# ╟─813238b8-3ce3-4ce8-98f6-cbdcbd1746d6
# ╟─25eb24a4-5414-481b-88d5-5ad50d75f8c9
# ╟─bd9a7214-f39e-4ae5-995a-6ec02d613fda
# ╟─f0f6ac8f-2e61-4b08-9383-45ff5b02c5b1
# ╟─7336e7b4-a7c2-4409-8d9d-00285cf633fb
# ╟─f63c05c4-eefe-11eb-2d60-f7bf7beb8118
# ╟─8fe94709-3673-44cc-8702-83bd7e2cad51
# ╟─27f44b28-f6d6-4e1f-8dbe-6fe7789b9a18
# ╟─43b28121-5d07-4400-8710-287de7b978a4
# ╠═f63c05d8-eefe-11eb-2a11-fd8f954bf059
# ╟─f63c05d8-eefe-11eb-0a2e-970192b02d61
# ╟─f63c05e4-eefe-11eb-012a-9bae1d87f2b5
# ╟─f63c0592-eefe-11eb-2a76-15de55eff3ad
# ╟─ca8e60e6-1d12-4076-8684-9a7c9e5ac509
# ╟─4987ad0c-d100-4a5d-a725-12f23bcc7aa3
# ╟─93f529ab-a386-4746-94de-63a536b5e2aa
# ╟─673a773c-2f6b-4676-8631-49ba08ec28a7
# ╟─f63c059c-eefe-11eb-208c-d5781711deb7
# ╟─f63c0600-eefe-11eb-3779-df58afdb52ad
# ╟─f63c0600-eefe-11eb-0023-9d5e5ddd987e
# ╟─8217895b-b120-4b08-b18f-d921dfdddf10
# ╟─d14244d4-b54e-42cf-b812-b149d9fa59e0
# ╟─34e527f2-ef80-4cb6-be3a-bee055eca125
# ╟─e4712ebe-f395-418b-abcc-e10ada4b05c2
# ╟─4158eaff-884c-4b64-bef9-9f4511cdc5a4
# ╟─9b62c374-c26e-4990-8ffc-790928e62e88
# ╟─0087f48b-7b50-4796-9371-fc87cf33c55d
# ╟─ded13e94-7c29-430e-8864-373e682b5509
# ╟─7923a0a8-3033-4dde-91e8-22bf540c9866
# ╟─57f352dc-55ee-4e14-b68d-698938a97d92
# ╟─b0de002f-3f39-4378-8d68-5c4606e488b7
# ╠═da6462d8-3343-41d8-82dd-48770176d4ba
# ╟─6eca3452-d47d-4d70-82d1-bedbe91e9820
# ╟─25e1f89f-a5b2-4d5c-a6e8-a990b3bebddb
# ╟─8307e205-3bcd-4e68-914e-621fd8d29e43
# ╠═753ec309-1363-485d-a2bd-b9fa100d9058
# ╟─fac51c63-f227-49ac-89f9-205bf03e7c08
# ╟─618ef3c7-0fda-4970-88e8-1dac195545de
# ╟─2d25fbb6-dc9b-40ad-bdce-4c952cdad077
# ╟─c9f10ace-3299-45fb-b98d-023a35dd405a
# ╟─2724c9b7-8caa-4ba7-be5c-2a9095281cfd
# ╟─f63c05ec-eefe-11eb-0a92-215711928d02
# ╟─08469a60-c96c-4c8a-8c8c-d7f1cb075ae3
# ╟─f63c0632-eefe-11eb-3b93-8549af7aaed9
# ╟─61c67500-c2b5-4610-a916-3b316ba01cce
# ╟─1feab086-9464-4fb7-9673-bc0ff30bab81
# ╟─19114da7-441a-4ada-b350-37c65a6211ee
# ╟─45859648-ef10-4380-8bf8-bca281b958cb
# ╟─0cfbd543-8b9b-406e-b3b4-c6cafbbec212
# ╟─11e350b2-70bd-437b-acef-af5103a6eb96
# ╟─e3aa458a-486c-4d5b-a330-67fb68e6f516
# ╟─f63c04d4-eefe-11eb-3fda-1729ac6de2cb
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
