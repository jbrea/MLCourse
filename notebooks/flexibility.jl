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

# ╔═╡ 8fa836a6-1133-4a54-b996-a02083fc6bba
begin
using Pkg
Base.redirect_stdio(stderr = devnull, stdout = devnull) do
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
end
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, Statistics, NearestNeighborModels
import MLCourse: fitted_linear_func
import PlutoPlotly as PP
const M = MLCourse.JlMod
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ 12942f34-efb1-11eb-3eb4-c1a38396cfb8
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ cdbcc51d-4cc6-4ab7-b0ef-c6272b5728af
md"The goal of this week is to
1. See a regression and a classification example, where linear methods are not flexible enough.
2. Understand and learn to run polynomial regression and classification.
3. Understand and learn to run k nearest neighbor regression and classification.
4. Understand the error decomposition in regression.
5. Understand the bias-variance decomposition in regression.
"

# ╔═╡ 12942f50-efb1-11eb-01c0-055b6be166e0
md"# 1. When Linear Methods Are Not Flexible Enough

We will use the function ``f(x) = 0.3 \sin(10x) + 0.7 x`` to design two artificial
data sets: one for regression with a single predictor and one for classification
with two predictors.

In this section we will see that standard linear regression or classification is not flexible enough to fit the artifical data sets.
"

# ╔═╡ 12942f5a-efb1-11eb-399a-a1300d636217
mlcode(
"""
using DataFrames, Random

f(x) = .3 * sin(10x) + .7x
# f(x) = .0015*(12x-6)^4 -.035(12x-6)^2 + .5x + .2  # an alternative
σ(x) = 1 / (1 + exp(-x))
function regression_data_generator(; n, σ = 0.1, rng = Random.GLOBAL_RNG)
    x = rand(rng, n)
    DataFrame(x = x, y = f.(x) .+ σ*randn(rng, n))
end
function classification_data_generator(; n, s = 20, rng = Random.GLOBAL_RNG)
    X1 = rand(rng, n)
    X2 = rand(rng, n)
    df = DataFrame(X1 = X1, X2 = X2,
                   y = categorical(σ.(s*(f.(X1) .- X2)) .> rand(rng, n),
                                   levels = [false, true], ordered = true))
end
"""
,
"""
import pandas as pd
import numpy as np
import random

def f(x):
	return 0.3 * np.sin(10*x) + 0.7*x

def sigma(x):
	return 1 / (1 + np.exp(-x))

def regression_data_generator(n, sigma=0.1):
	x = np.random.rand (n)
	y = f(x) + sigma * np.random.randn(n)
	return pd.DataFrame({'x': x, 'y': y})
	#generate regression data

def classification_data_generator(n, s=20):
    X1 = np.random.rand(n)
    X2 = np.random.rand(n)
    y = (sigma(s * (f(X1) - X2)) > np.random.rand(n)).astype(int)
    return pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
	#generate classification data
"""
;
collapse = "Data Generators",
cache = false
)


# ╔═╡ afad9691-c2fc-495c-8291-ce7ff713d052
md"## Linear Regression"

# ╔═╡ aaeb9420-6a71-4da1-8caa-0a0d570822a7
md"There is obviously some structure in the training data that cannot be captured by linear regression."

# ╔═╡ e7e667f3-a4d6-4df5-bffa-f61b73ad0b72
md"## Logistic Regression"

# ╔═╡ 2ae86454-1877-4972-9cf6-24ef9350a296
md"This is an example of multiple logistic regression, where the input data is two dimensional and the class labels are indicated by red and blue dots. The small dots in the background indicate how logistic regression would
would classify test data (with decision threshold 0.5)
and the large points are the training data. Obviously, the linear decision boundary cannot capture the structure in the decision boundary of the data generator."


# ╔═╡ d487fcd9-8b45-4237-ab2c-21f82ddf7f7c
md"# 2. Polynomial Models
## Polynomial Regression

One way to increase the flexibility is to fit a polynomial. This can be achieved by running linear regression on a transformed data set. The transformation consists of computing higher powers of the original input.
"

# ╔═╡ c7427250-1c03-4fbf-9260-6e9df909f5e2
md"In the table above, the `x` and the `y` column contain the original data. The other columns contain powers of the `x` column. We can now run linear regression on this transformed dataset."

# ╔═╡ 4f03750e-1e9b-4db8-810b-3717b642ed75
md"Control the noise level of the data generator with this slider:

σ\_gen = $(@bind σ_gen Slider(.01:.01:.3, default = .1, show_value = true))
"

# ╔═╡ 12942f62-efb1-11eb-3f59-f981bc32f308
begin
MLCourse._eval(:(regression_data = regression_data_generator(n = 50, σ = $σ_gen, rng = $(Random.MersenneTwister(3)))))
regression_data = M.regression_data
end

# ╔═╡ 12942f6e-efb1-11eb-0a49-01a6a2d0196f
begin
    m1 = machine(LinearRegressor(), select(regression_data, :x), regression_data.y)
    fit!(m1, verbosity = 0)
end;

# ╔═╡ 12942f6e-efb1-11eb-35c0-19a7c4d6b44e
begin
    scatter(regression_data.x, regression_data.y, label = "training data")
    plot!(fitted_linear_func(m1), w = 3, xlabel = "X", ylabel = "Y",
          label = "linear regression", legend = :topleft)
end

# ╔═╡ 710d3104-e197-44c1-a10b-de1098d57dd6
md"
And the degree of the polynomial with the following slider:

degree = $(@bind degree Slider(1:17, default = 4, show_value = true))"

# ╔═╡ c50ed135-68d5-43bc-9c26-9c265702a1f0
MLJ.transform(machine(Polynomial(degree = degree)), regression_data)

# ╔═╡ ad50d244-c644-4f61-bd8b-995d0110811d
begin
m3 = machine(Polynomial(degree = degree) |> LinearRegressor(),
             select(regression_data, Not(:y)),
             regression_data.y);
fit!(m3, verbosity = 0)
end;

# ╔═╡ 0272460a-5b9f-4728-a531-2b497b26c512
begin
function compute_loss(mach, data, lossfunc; operation = predict)
    pred = operation(mach, select(data, Not(:y)))
    lossfunc(pred, data.y)
end
rmse(ŷ, y) = sqrt(mean((ŷ .- y).^2))
end;

# ╔═╡ cfcb8f61-af91-40dd-951a-09e8dbf17e30
begin
    regression_test_data = M.regression_data_generator(n = 10^4, σ = σ_gen)
    plosses = hcat([let m = fit!(machine(Polynomial(degree = d) |> LinearRegressor(),
                                         select(regression_data, Not(:y)),
                                         regression_data.y), verbosity = 0)
                        [compute_loss(m, regression_data, rmse),
                         compute_loss(m, regression_test_data, rmse)]
                   end
                   for d in 1:17]...)
end;

# ╔═╡ 6fa9b644-d4a6-4c53-9146-9d978207bfd0
let f = M.f
    scatter(regression_data.x, regression_data.y, label = "training data")
    p5 = plot!(0:.01:1, predict(m3, (x = 0:.01:1,)), w = 3,
		       xlabel = "X", ylabel = "Y",
               label = "$degree-polynomial regression", legend = :topleft)
	plot!(f, color = :green, label = "data generator", w = 2)
    p6 = plot(1:17, plosses[1, :], color = :blue, label = "training loss")
    plot!(1:17, plosses[2, :], color = :red, label = "test loss")
    hline!([minimum(plosses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    hline!([σ_gen], color = :green, style = :dash, label = "irreducible error")
    scatter!([degree], [plosses[1, degree]], color = :blue, label = nothing)
    scatter!([degree], [plosses[2, degree]], color = :red, label = nothing)
    vline!([argmin(plosses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility (degree)", ylabel = "rmse")
    plot(p5, p6, layout = (1, 2), size = (700, 400))
end

# ╔═╡ 8ae5c31d-b272-4974-a47f-04289a6561b5
md"""### Running Polynomial Regression

Here is a code example to run polynomial regression.
$(mlstring(md"Because this is not a commonly used machine learning method, there is no implementation in `MLJ`. Instead we use a custom implementation of `Polynomial`. Don't worry, you do not need to understand the helper functions.", ""))
"""

# ╔═╡ e9b0ea86-a5f5-43fa-aa16-5a0240f298dd
mlcode(
"""
using MLJ, MLJLinearModels, DataFrames

###
### Helper functions
###

polysymbol(x, d) = Symbol(d == 0 ? "" : d == 1 ? "\$x" : "\$x^\$d")
colnames(df::AbstractDataFrame) = names(df)
colnames(d::NamedTuple) = keys(d)
colname(names, predictor::Int) = names[predictor]
function colname(names, predictor::Symbol)
    if predictor ∈ names || string(predictor) ∈ names
        Symbol(predictor)
    else
        error("Predictor \$predictor not found in \$names.")
    end
end
function poly(data, degree, predictors::NTuple{1} = (1,))
    cn = colnames(data)
    col = colname(cn, predictors[1])
    res = DataFrame([getproperty(data, col) .^ k for k in 1:degree],
                    [polysymbol(col, k) for k in 1:degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end
function poly(data, degree, predictors::NTuple{2})
    cn = colnames(data)
    col1 = colname(cn, predictors[1])
    col2 = colname(cn, predictors[2])
    res = DataFrame([getproperty(data, col1) .^ d1 .* getproperty(data, col2) .^ d2
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree],
                    [Symbol(polysymbol(col1, d1), polysymbol(col2, d2))
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end

Base.@kwdef mutable struct Polynomial{T} <: Static
    degree::Int = 3
    predictors::T = (1,)
end
MLJ.transform(p::Polynomial, _, X) = poly(X, p.degree, p.predictors)

###
### Define machine and fit it
###

m3 = machine(Polynomial(degree = 4) |> LinearRegressor(),
             select(regression_data, Not(:y)),
             regression_data.y)
fit!(m3, verbosity = 0)
"""
,
"""
regression_data = regression_data_generator(n = 50)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m3 = PolynomialFeatures(degree=3, include_bias=False)
poly_features = m3.fit_transform(regression_data.drop(['y'], axis=1).values)

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, regression_data["y"])

y_predicted = poly_reg_model.predict(poly_features)
"""
;
showoutput = false
)

# ╔═╡ 75c5471d-926d-4b05-9002-28b14e1dd428
mlstring(
md"""
!!! note "Pipeline"
    To create a machine that transforms the input to a polynomial representation and run linear regression on this transformed representation, we use an `MLJ.Pipeline`.
    Pipelines in MLJ can be written explicitely with `Pipeline(Polynomial(degree = 5), LinearRegressor())` or with the pipe operator `|>`, i.e. `Polynomial(degree = 5) |> LinearRegressor()`.
"""
,
""
)

# ╔═╡ edfb269d-677e-4687-8bff-0aa9ae6e64c3
md"## Polynomial Classification

As in the polynomial regression example above, we can transform the classification dataset to include higher orders of the input."

# ╔═╡ 78424f6f-acd7-431e-b8a9-07ab6321cff3
md"Because we have two-dimensional input, the polynomial contains also mixtures like `X1X_2, X1^2X2` etc."

# ╔═╡ 1dfb4619-a77a-4d56-93d3-94b0ab832de4
md"Change the noise levels with the following slider. `s_gen` is the steepness of the logistic function in the data generating process of the classification task.

s\_gen = $(@bind s_gen Slider(5:50, default = 20, show_value = true))
"


# ╔═╡ 12942f94-efb1-11eb-234e-51492b51e583
begin
MLCourse._eval(:(classification_data = classification_data_generator(n = 400, s = $s_gen, rng = $(MersenneTwister(8)))))
classification_data = M.classification_data
end

# ╔═╡ 12942fa0-efb1-11eb-01c8-6dae80c55fb8
begin
    m2 = machine(LogisticClassifier(penalty = :none),
                 select(classification_data, Not(:y)),
                 classification_data.y)
    fit!(m2, verbosity = 0)
end;

# ╔═╡ 12942f94-efb1-11eb-2c48-a3418b53b886
begin
    xgrid = MLCourse.grid2D(0:.01:1, 0:.02:1, names = (:X1, :X2))
    scatter(xgrid.X1, xgrid.X2, color = coerce(predict_mode(m2, xgrid), Count),
            markersize = 2, label = nothing, markerstrokewidth = 0)
    scatter!(classification_data.X1, classification_data.X2,
		     xlabel = "X₁", ylabel = "X₂",
             color = coerce(classification_data.y, Count), label = nothing)
end

# ╔═╡ 59acced5-16eb-49b8-8cf2-0c43a88d838e
md"degree = $(@bind degree2 Slider(1:17, default = 3, show_value = true))"

# ╔═╡ 5ea1b31d-91e5-4c8f-93d6-5d31816fdbf5
MLJ.transform(machine(Polynomial(degree = degree2, predictors = (:X1, :X2))),
	          classification_data)

# ╔═╡ a782d80e-50d6-4178-ae59-0ee603e1cb02
md"The area under the ROC curve (AUC) of the linear classifier is small, because the decision boundary is not flexible enough and therefore many samples lie on the wrong side of the decision boundary. A polynomial classifier almost approaches the optimal classifier (use the slider above to show the ROC curve for different degrees). Note that the optimal classifier does not have an AUC of 1, because the data generating process is noisy. Change the steepness of the sigmoid of the data generating process with the slider at the top of this notebook to observe the effect it has on the optimal ROC curve!"

# ╔═╡ 16f0d1b3-bd97-407d-9a79-25b0fb05bbeb
begin
    classification_test_data = M.classification_data_generator(n = 10^4)
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
                        p = M.σ.(s_gen * (M.f.(data.X1) .- data.X2))
        mean(log_loss(UnivariateFinite([false, true], p,
			                            augment = true, pool = missing),
			 data.y))
    end
end;

# ╔═╡ 2fa54070-e261-462d-bd63-c225b92fa876
begin
    m4 = machine(Polynomial(degree = degree2, predictors = (:X1, :X2)) |> LogisticClassifier(penalty = :none),
                 select(classification_data, Not(:y)),
                 classification_data.y);
    fit!(m4, verbosity = 0);
end;

# ╔═╡ ed62cb94-8187-4d50-a6f9-6967893dd021
let
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

# ╔═╡ 3fa49ea1-df82-472c-914b-0a67e7412b2c
let
	fprs_l, tprs_l, _ = roc(predict(m2, select(classification_test_data, Not(:y))), classification_test_data.y)
	fprs_p, tprs_p, _ = roc(predict(m4, select(classification_test_data, Not(:y))), classification_test_data.y)
	p = M.σ.(s_gen*(M.f.(classification_test_data.X1) .- classification_test_data.X2))
	fprs_o, tprs_o, _ = roc(UnivariateFinite.(Ref([false, true]), p, augment = true, pool = missing), classification_test_data.y)
	plot(fprs_l, tprs_l, label = "linear classifier",
		legend_position = :bottomright, ylabel = "true positive rate",
	    xlabel = "false positive rate", title = "ROC curve")
	plot!(fprs_p, tprs_p, label = "polynomial classifier degree $degree2")
	plot!(fprs_o, tprs_o, label = "classification based on probability of the data generating process")
end

# ╔═╡ 1b45f462-a2c9-4328-972d-b46873f1bde3
md"### Running Polynomial Classification"

# ╔═╡ 21a5aebb-226e-40e5-806b-63f501016b19
mlcode(
"""
using MLJ, MLJLinearModels
# we use the same custom code for the Polynomial
# as in the example for polynomial regression
m4 = machine(Polynomial(degree = 3, predictors = (:X1, :X2)) |>
                LogisticClassifier(penalty = :none),
             select(classification_data, Not(:y)),
             classification_data.y)
fit!(m4, verbosity = 0)
"""
,
"""
classification_data = classification_data_generator(50)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

m3 = PolynomialFeatures(degree=3, include_bias=False)
poly_features = m3.fit_transform(classification_data.drop(['y'], axis=1).values)

poly_reg_model = LogisticRegression()
poly_reg_model.fit(poly_features, classification_data["y"])

y_predicted = poly_reg_model.predict(poly_features)
"""
;
showoutput = false
)

# ╔═╡ 5c984615-f123-47fc-8330-66694ab1cb9f
md"# 3. K-Nearest-Neighbor Models
## K-Nearest-Neighbor Regression

σ\_gen = $(@bind σ_genk Slider(.01:.01:.3, default = .1, show_value = true))

K = $(@bind K Slider(1:50, show_value = true))
"

# ╔═╡ ef0892e2-6c9d-4cea-ae6e-609b800c6f4c
begin
regression_datak = M.regression_data_generator(n = 50, σ = σ_genk)
regression_test_datak = M.regression_data_generator(n = 10^4, σ = σ_genk)
losses = hcat([let m = fit!(machine(KNNRegressor(K = k),
                                select(regression_datak, Not(:y)),
                                regression_datak.y), verbosity = 0)
               [compute_loss(m, regression_datak, rmse),
                compute_loss(m, regression_test_datak, rmse)]
           end
           for k in 1:50]...);
end;

# ╔═╡ 12942f82-efb1-11eb-2827-df957759b02c
begin
m12 = machine(KNNRegressor(K = K),
	          select(regression_datak, :x),
	          regression_datak.y);
fit!(m12, verbosity = 0);
end;

# ╔═╡ 542327e9-1599-4a14-805f-5d2a3c4eae14
begin
    scatter(regression_datak.x, regression_datak.y, label = "training data")
    p1 = plot!(0:.001:1, predict(m12, (x = 0:.001:1,)), w = 3,
               label = "$K-NN fit", legend = :topleft, xlabel = "x", ylabel = "y")
    plot(1 ./ (1:50), losses[1, :], color = :blue, label = "training loss")
    plot!(1 ./ (1:50), losses[2, :], color = :red, label = "test loss")
    hline!([minimum(losses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    hline!([σ_genk], color = :green, style = :dash, label = "irreducible error")
    scatter!([1 / K], [losses[1, K]], color = :blue, label = nothing)
    scatter!([1 / K], [losses[2, K]], color = :red, label = nothing)
    p2 = vline!([1 / argmin(losses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility 1/K", ylabel = "rmse")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end

# ╔═╡ fc7c5be0-1dce-4b58-b72c-e302d83de2f7
md"### Running K-Nearest-Neighbor Regression"

# ╔═╡ 7a58ca5a-e299-4c26-939c-97e23e8a07a7
mlcode(
"""
using NearestNeighborModels
# WARNING: running KNNRegressor on more than a few predictors
# does not make much sense (curse of dimensionality) and is very slow.

mach = machine(KNNRegressor(K = 4),
	          select(regression_data, :x),
	          regression_data.y)
fit!(mach, verbosity = 0)
"""
,
"""
from sklearn.neighbors import KNeighborsRegressor
# WARNING: running KNNRegressor on more than a few predictors
# does not make much sense (curse of dimensionality) and is very slow.

knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(regression_data.drop(['y'], axis=1).values, regression_data['y'])
y_predicted = knn.predict(regression_data.drop(['y'], axis=1).values)
y_predicted
"""
;
#showoutput = false
)

# ╔═╡ 12942f8c-efb1-11eb-284c-393f6a694818
md"## K-Nearest-Neighbor Classification

s\_gen = $(@bind s_genk Slider(5:50, default = 20, show_value = true))

K = $(@bind Kc Slider(1:2:100, show_value = true))"

# ╔═╡ 0a57f15b-c292-4c64-986d-f046260da66e
begin
    classification_datak = M.classification_data_generator(n = 400, s = s_genk)
    classification_test_datak = M.classification_data_generator(n = 10^4, s = s_genk)
    closses = hcat([let m = fit!(machine(KNNClassifier(K = k),
                                         select(classification_datak, Not(:y)),
                                         classification_datak.y), verbosity = 0)
                        [mean(compute_loss(m, classification_datak, log_loss)),
                         mean(compute_loss(m, classification_test_datak, log_loss))]
                   end
                   for k in 1:2:100]...)
end;

# ╔═╡ 12942fc8-efb1-11eb-3180-dff1921c5bf9
begin
m14 = machine(KNNClassifier(K = Kc),
             select(classification_datak, Not(:y)),
             classification_datak.y);
fit!(m14, verbosity = 0)
end;

# ╔═╡ 12942fc8-efb1-11eb-0c02-0150ef55ae98
begin
    scatter(xgrid.X1, xgrid.X2, color = coerce(predict_mode(m14, xgrid), Count),
            markersize = 2, label = nothing, markerstrokewidth = 0,
            xlabel = "X1", ylabel = "X2")
    p3 = scatter!(classification_datak.X1, classification_datak.X2,
                  color = coerce(classification_datak.y, Count), label = nothing)
    plot(1 ./ (1:2:100), closses[1, :], color = :blue, label = "training loss")
    plot!(1 ./ (1:2:100), closses[2, :], color = :red, label = "test loss")
    hline!([c_irred_error], color = :green, style = :dash, label = "irreducible error")
    hline!([minimum(closses[2, :])], color = :red,
           label = "minimal test loss", style = :dash)
    scatter!([1 / Kc], [closses[1, Kc÷2+1]], color = :blue, label = nothing)
    scatter!([1 / Kc], [closses[2, Kc÷2+1]], color = :red, label = nothing)
    p4 = vline!([1 / argmin(closses[2, :])], color = :red,
                label = nothing, style = :dash, xlabel = "flexibility 1/K", ylabel = "negative loglikelihood")
    plot(p3, p4, layout = (1, 2), size = (700, 400))
end

# ╔═╡ 8b4fd005-2f6b-4aab-85c3-acd39717fb69
md"### Running K-Nearest-Neighbor Classification"


# ╔═╡ 6f371ccd-d92e-44d7-9ab4-23ddcaf13b5f
mlcode(
"""
using NearestNeighborModels
# WARNING: running KNNClassifier on more than a few predictors
# does not make much sense (curse of dimensionality) and is very slow.
mach = machine(KNNClassifier(K = 3),
             select(classification_data, Not(:y)),
             classification_data.y);
fit!(mach, verbosity = 0)

"""
,
"""
from sklearn.neighbors import KNeighborsClassifier
# WARNING: running KNNRegressor on more than a few predictors
# does not make much sense (curse of dimensionality) and is very slow.
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(classification_data.drop(['y'], axis=1).values, classification_data['y'])
y_predicted = knn.predict(classification_data.drop(['y'], axis=1).values)
"""
;
showoutput = false
)



# ╔═╡ cc8ed1de-beab-43e5-979e-e83df23f96ae
md"## Application to Handwritten Digit Recognition (MNIST)

In the following we are fitting a first nearest-neigbor classifier to the MNIST
data set.

WARNING: The following code takes more than 10 minutes to run.
Especially the prediction is slow, because for every test image the closest out
of 60'000 training images has to be found.
"

# ╔═╡ 34db537f-9cd8-4105-9244-0d05b09a9967
mlcode(
"""
using OpenML
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => Continuous)
    df[:, 1:end-1] ./ 255,
    df.class
end

m5 = fit!(machine(KNNClassifier(K = 1),
                  mnist_x[1:60000, :],
                  mnist_y[1:60000]))
test_predictions = predict_mode(m5, mnist_x[60001:70000, :])
mnist_errorrate = mean(test_predictions .!= mnist_y[60001:70000])
"""
,
"""
import openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

mnist,_,_,_ = openml.datasets.get_dataset(554).get_data(dataset_format="dataframe")
X_train, X_test, y_train, y_test = train_test_split(mnist.loc[:, mnist.columns != "class"].values, mnist["class"].values, test_size=1/7, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
test_predictions = knn.predict(X_test)

mnist_errorrate = np.mean(test_predictions != y_test)
mnist_errorrate
"""
;
eval = false
)

# ╔═╡ 1c7ad8e3-dae7-4217-b528-bb0d3d1d5331
md"We find a misclassification rate of approximately 3%. This is clearly better than the approximately 7.8% obtained with Multinomial Logistic Regression.
"

# ╔═╡ 99a371b2-5158-4c42-8f50-329352b6c1f2
md"# 4. Error Decomposition

In the following cells we look at error decomposition discussed in the slides.

The strategy is to
1. Define a function `f(x)` as the mean of a data generator, which is a conditional normal distribution.
2. Define a function `expected_error` that takes an arbitrary function `f̂` as argument and estimates the expected error at a given point `x` by generating a lot (``10^6``) samples from the data generator at this point `x` and computing the mean squared error between the samples and the function `f̂`.
3. Define a test function `f̂` that is different from `f`.
4. Estimate the expected error of `f̂` at `x = 0.1`, compute the reducible error at this point and estimate or compute the irreducible error.
5. Conclude that the sum of reducible and irreducible error is equal to the expected error.
"

# ╔═╡ f10b7cad-eda3-4ec9-99ee-d43ed013a057
mlcode(
"""
f(x) = .3 * sin(10x) + .7x
conditional_generator(x; n = 50, σ = 0.1) = f(x) .+ σ*randn(n)
function expected_error(f̂, x; σ = 0.1)
    mean((conditional_generator(x; n = 10^6, σ) .- f̂(x)).^2)
end

f̂(x) = 0.1 + x
"""
,
"""
def f(x):
    return 0.3 * np.sin(10*x) + 0.7*x

def conditional_generator(x, n=50, sigma=0.1):
    return f(x) + sigma * np.random.randn(n)

def expected_error(f_hat, x, sigma=0.1):
    return np.mean((conditional_generator(x, 10**6, sigma) - f_hat(x))**2)

f_hat = lambda x: 0.1 + x #test function f̂
 
"""
;
showoutput = false,
cache = false
)

# ╔═╡ 3d77d753-b247-4ead-a385-7cbbcfc3190b
md"The expected error of a function `f` at point `x` for our `conditional_generator` can be estimated by computing the mean squared error for many samples obtained from this generator."

# ╔═╡ c6a59b85-d031-4ad4-9e24-691494d08cde
mlcode(
"""
expected_error(f̂, 0.1) # estimated total expected error at 0.1
"""
,
"""
expected_error(f_hat, 0.1) # estimated total expected error at 0.1
"""
)

# ╔═╡ e50b8196-e804-473a-b3b5-e22fdb9d2f45
mlcode(
"""
(f(.1) - f̂(.1))^2 # reducible error
"""
,
"""
(f(0.1) - f_hat(0.1))**2 # reducible error
"""
)

# ╔═╡ f413ea94-36ca-4afc-8ca8-9a7e88101980
mlcode(
"""
expected_error(f, 0.1) # estimated irreducible error
"""
,
"""
expected_error(f, 0.1) # estimated irreducible error
"""
)

# ╔═╡ 2bfa1a57-b171-44c3-b0d7-b8dda48d26d7
md"Instead of using estimates for the irreducible error we could just compute it in this simple example: it is σ\_noise² = 0.1² = 0.01.

Looking at the three results above we can conclude that the estimated expected error is very close to the sum of the reducible error and the estimated (or computed) irreducible error.

The result above holds for `x = 0.1`. In the figure below we can get to the same conclusion for all values of `x` that are shown in the plot and for different levels of the irreducible error. Note that the reducible error - and therefore the expected error - are a change with ``x``, whereas the irreducible error ``=\sigma_\mathrm{noise}^2`` remains constant, by constuction of the data generator."

# ╔═╡ f9970968-cb27-456c-b7ee-70077f9c15ee
md"σ\_noise = $(@bind σ_noise Slider(.01:.01:.5, default = .2, show_value = true))"

# ╔═╡ dbf7fc72-bfd0-4c57-a1a9-fb5881e16e7e
let x = rand(100), grid = 0:.05:1, f = M.f, f̂ = M.f̂
    p1 = scatter(x, vcat(M.conditional_generator.(x; n = 1, σ = σ_noise)...), label = "samples")
    plot!(f, label = "f", w = 2)
    plot!(f̂, label = "f̂", w = 2, ylabel = "y")
    p2 = plot(grid, M.expected_error.(f̂, grid, σ = σ_noise), label = "expected error f̂", w = 3)
    plot!(grid, (f.(grid) .- f̂.(grid)).^2, label = "reducible error", w = 3)
    hline!([σ_noise^2], label = "irreducible error", ylims = (0, .6), w = 3, xlabel = "x", ylabel = "mse")
    plot(p1, p2, layout = (2, 1), legend = :topleft)
end

# ╔═╡ db2c6bd4-ee6f-4ba9-b6ec-e7cf94389f93
md"With machine learning we cannot remove the irreducible error. The irreducible noise is a property of the data generating process. But in cases where accurate prediction is the goal, we should try to minimize the reducible error for all inputs ``x`` we care about."

# ╔═╡ f6093c98-7e89-48ba-95c9-4d1f60a25033
md"# 5. Bias-Variance Decomposition

The error decomposition above holds for any given test function `f̂`. In this section we decompose the error further by considering a function family and looking at multiple test functions `f̂ᵢ` fitted to different training sets.

The will use the same data generator as in the previous section and look at the families of polynomial functions of different degrees.

In the **first plot** below you see many training sets obtained from the same data generator (dots) together with the fits `f̂ᵢ` on each training set (colored curves), the average of the fits (thick red curve) and the mean of the data generator `f` (green thick curve).

With the `x_test` slider, you can choose an input `x` for closer inspection in the second plot. The **second plot** shows the value of the fitted functions `f̂ᵢ(x)` at the test point together with some test data at the test point `x`. The expected error could be estimated by computing the average squared pairwise vertical distances between the predictions and the test points. The expected error can be decomposed into the squared bias (indicated by the arrow between the average fit and the mean of the data generator) the variance of the predictions (indicated as the red arrow, which shows plus-minus one empirical standard deviation) and the irreducible error (indicated by the green arrow, which shows plus-minus one standard deviation of the noise).

The **third plot** shows the squared bias, variance, irreducible noise together with the sum of these three terms and the empirical expected error, which is obtained by computing the average squared pairwise vertical distances between predictions and test points in the second plot. The results in this plot were obtained by averaging over more training sets and larger test sets.
"

# ╔═╡ a669a3b1-87f6-4ac0-8ac9-11509ef0f600
md"degree = $(@bind deg_bvd Slider(1:16, show_value = true))"

# ╔═╡ aad3a522-97b1-4128-abe5-9e9f93f5c448
bvd = let K = 100, Kshow = 10, xgrid = 0:.01:1
    colors = Plots.Colors.distinguishable_colors(Kshow)
    Random.seed!(123)
    x = rand(xgrid, 50)
    y = [vcat(M.conditional_generator.(x, n = 1)...) for _ in 1:K]
    m = [machine(Polynomial(degree = deg_bvd) |> LinearRegressor(), DataFrame(x = x), y[k]) |> m -> fit!(m, verbosity = 0) for k in 1:K]
    pred = [predict(m[k], (x = xgrid,)) for k in 1:K]
    avg = mean(pred)
    vari = var(pred, corrected = false)
    testdata = M.conditional_generator.(xgrid, n = 10^4)
    expected_error = [mean(abs2, pred[i][j] - testdata[j][k]
                           for i in 1:K, k in 1:10^4) for j in 1:length(xgrid)]
    (; K, Kshow, x, y, xgrid, pred, avg, vari, colors, expected_error, testdata)
end;

# ╔═╡ fa8bb711-81ce-44be-98d3-ca87782a858d
md"x\_test = $(@bind x_bvd Slider(sort(bvd.x), show_value = true))"

# ╔═╡ 7e5302a8-8a0d-4c92-8793-4229bc8bc7c6
let colors = bvd.colors
    plot()
    for k in 1:bvd.Kshow
        scatter!(bvd.x, bvd.y[k], c = colors[k], label = "training set $k", alpha = .3)
        plot!(bvd.xgrid, bvd.pred[k], c = colors[k], w = 1, label = "fit $k")
    end
    plot!(M.f, c = :green, w = 2, label = "f")
    plot!(bvd.xgrid, bvd.avg, c = :red,  w = 2, label = "fit average")
    vline!([x_bvd], label = nothing, c = :gray)
    plot!(legend_position = :outertopright,
          ylabel = "y", xlabel = "x",
          yrange = (-.3, 1.1), size = (700, 350))
end

# ╔═╡ 6523395e-33ae-453a-94eb-0a7463e9ea94
let Ktest = 18, dataticks = range(.39, .58, Ktest), fitticks = range(.12, .31, bvd.Kshow),
    f = M.f(x_bvd)
    i = findfirst(==(x_bvd), bvd.xgrid)
    pred = getindex.(bvd.pred[1:bvd.Kshow], i)
    ytest = bvd.testdata[i][1:Ktest]
    estd = std(pred)
    hline([f], c = :green, w = 3, xrange = (.1, .6), label = "f")
    hline!([bvd.avg[i]], c = :red, w = 2, label = "fit average" )
    scatter!(dataticks, ytest, c = :blue, label = "test data")
    scatter!(fitticks, pred, markershape = :utriangle, c = bvd.colors, label = "fit", xticks = ([dataticks; fitticks], [1:Ktest; 1:bvd.K]))
    plot!([.35, .35], [f, bvd.avg[i]], arrow = :both, c = :orange, label = "bias")
    plot!([.485, .485], [f-.2, f+.2], arrow = :both, c = :green, label = "irreducible error")
    plot!([.215, .215], [bvd.avg[i]-2estd, bvd.avg[i]+2estd], arrow = :both, c = :red, label = "irreducible error")
    plot!(legend_position = :outertopright,
          ylabel = "y", xlabel = "fit id                                    test data id",
          yrange = (-.3, 1.1), size = (700, 350))
end

# ╔═╡ 6a89d461-0c68-4dae-9a52-a86d13767674
let
    bias2 = (bvd.avg .- M.f.(bvd.xgrid)).^2
    hline([0.1^2], label = "irreducible error", c = :green, xrange = (0, 1))
    plot!(bvd.xgrid, bias2, label = "bias", c = :orange)
    plot!(bvd.xgrid, bvd.vari, label = "variance", c = :red)
    plot!(bvd.xgrid, bvd.expected_error, label = "empirical expected error", c = :black, xlabel = "x", w = 2, ylabel = "mse")
    plot!(bvd.xgrid, bias2 .+ bvd.vari .+ .1^2, label = "bias² + var + σ²", c = :yellow, linestyle = :dash, yrange = (0, .1), size = (700, 350))
    vline!([x_bvd], label = nothing, c = :gray)
end

# ╔═╡ ae86ee9c-3645-43d2-9a42-79658521c3fb
md"# Exercises

## Conceptual
#### Exercise 1
For each example below, indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.
+ The sample size n is extremely large, and the number of predictors ``p`` is small.
+ The number of predictors ``p`` is extremely large, and the number of observations ``n`` is small.
+ The relationship between the predictors and response is highly non-linear.
+ The variance of the error terms, i.e. ``\sigma^2 = Var(\epsilon)``, is extremely high.
#### Exercise 2
The table below provides a training data set containing six observations, three predictors, and one qualitative response variable. Suppose we wish to use this data set to make a prediction for ``Y`` when ``X_1 = X_2 = X_3 = 0`` using K-nearest neighbors.
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
#### Exercise 3
- Provide a sketch of typical (squared) bias, variance, training error, test error, and irreducible error curves, on a single plot, as we go from less flexible statistical learning methods towards more flexible approaches. The x-axis should represent the amount of flexibility in the method, and the y-axis should represent the values for each curve. There should be five curves. Make sure to label each one.
- Explain why each of the five curves has its particular shape.
#### Exercise 4
Suppose that we take a data set with mutually distinct inputs ``x_i\neq x_j`` for ``i\neq j``, divide it into equally-sized training and test sets, and then try out two different classification procedures. First we use logistic regression and get an error rate of 20% on the training data and 30% on the test data. Next we use 1-nearest neighbors (i.e. ``K = 1``) and get an average error rate (averaged over both test and training data sets) of 18%. Based on these results, which method should we prefer to use for classification of new observations? Why?

## Applied
#### Exercise 5
Apply K-nearest neighbors regression to the weather data set. Use as input all predictors except `:time` and `:LUZ_wind_peak`.
* Compute the training and the test loss for ``K = 5, 10, 20, 50, 100``.
* Which value of the hyper-parameter ``K`` should we prefer to make predictions on new data?
* Should we prefer K-nearest neighbors with optimal ``K`` or multiple linear regression to make predictions on new data? *Hint*: Remember that we found a training error (RMSE) of approximately 8.1 and a test error of 8.9.
"

# ╔═╡ c98a4d29-4dfe-4ca4-a2f1-5342788da6c0
md"
#### Exercise 6
In this exercise we review the error-decomposition and the bias-variance decomposition.
* Write a data generator where the mean of the output depends through the non-linear function ``f(x) = x^2 * \sin(x) + 4 * \tanh(10x)`` on the input and normally distributed noise ``\epsilon`` with mean 0 and standard deviation 1.5.
    * Take the linear function ``\hat f(x) = 2x`` and estimate its reducible error at input point ``x = 0`` and at input point ``x = 2`` in two ways:
        * Using directly ``f``.
        * Using ``10^5`` samples from the data generator. *Hint:* Use the samples to estimate the irreducible error and then use the error decomposition formula to compute the reducible error.
    * Generate ``10^4`` training sets of 100 data points with input ``x`` normally distributed with standard deviation 2 and mean 0 and estimate the bias of linear regression at ``x = 4``  in two ways:
        * Using directly ``f``.
        * Using ``10^4`` samples from the data generator. *Hint:* Use again the samples to estimate the irreducible error and use the bias-variance decomposition formula to compute the bias.
"

# ╔═╡ efda845a-4390-40bc-bdf2-89e555d3b1b2
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 2320f424-7652-4e9f-83ef-fc011b722dcc
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─cdbcc51d-4cc6-4ab7-b0ef-c6272b5728af
# ╟─12942f50-efb1-11eb-01c0-055b6be166e0
# ╟─12942f5a-efb1-11eb-399a-a1300d636217
# ╟─afad9691-c2fc-495c-8291-ce7ff713d052
# ╟─12942f62-efb1-11eb-3f59-f981bc32f308
# ╟─12942f6e-efb1-11eb-35c0-19a7c4d6b44e
# ╟─aaeb9420-6a71-4da1-8caa-0a0d570822a7
# ╟─e7e667f3-a4d6-4df5-bffa-f61b73ad0b72
# ╟─12942f94-efb1-11eb-234e-51492b51e583
# ╟─12942f94-efb1-11eb-2c48-a3418b53b886
# ╟─2ae86454-1877-4972-9cf6-24ef9350a296
# ╟─12942f6e-efb1-11eb-0a49-01a6a2d0196f
# ╟─12942fa0-efb1-11eb-01c8-6dae80c55fb8
# ╟─d487fcd9-8b45-4237-ab2c-21f82ddf7f7c
# ╟─c50ed135-68d5-43bc-9c26-9c265702a1f0
# ╟─c7427250-1c03-4fbf-9260-6e9df909f5e2
# ╟─4f03750e-1e9b-4db8-810b-3717b642ed75
# ╟─710d3104-e197-44c1-a10b-de1098d57dd6
# ╟─6fa9b644-d4a6-4c53-9146-9d978207bfd0
# ╟─ad50d244-c644-4f61-bd8b-995d0110811d
# ╟─0272460a-5b9f-4728-a531-2b497b26c512
# ╟─cfcb8f61-af91-40dd-951a-09e8dbf17e30
# ╟─8ae5c31d-b272-4974-a47f-04289a6561b5
# ╟─e9b0ea86-a5f5-43fa-aa16-5a0240f298dd
# ╟─75c5471d-926d-4b05-9002-28b14e1dd428
# ╟─edfb269d-677e-4687-8bff-0aa9ae6e64c3
# ╟─5ea1b31d-91e5-4c8f-93d6-5d31816fdbf5
# ╟─78424f6f-acd7-431e-b8a9-07ab6321cff3
# ╟─1dfb4619-a77a-4d56-93d3-94b0ab832de4
# ╟─59acced5-16eb-49b8-8cf2-0c43a88d838e
# ╟─ed62cb94-8187-4d50-a6f9-6967893dd021
# ╟─3fa49ea1-df82-472c-914b-0a67e7412b2c
# ╟─a782d80e-50d6-4178-ae59-0ee603e1cb02
# ╟─16f0d1b3-bd97-407d-9a79-25b0fb05bbeb
# ╟─2fa54070-e261-462d-bd63-c225b92fa876
# ╟─1b45f462-a2c9-4328-972d-b46873f1bde3
# ╟─21a5aebb-226e-40e5-806b-63f501016b19
# ╟─5c984615-f123-47fc-8330-66694ab1cb9f
# ╟─542327e9-1599-4a14-805f-5d2a3c4eae14
# ╟─ef0892e2-6c9d-4cea-ae6e-609b800c6f4c
# ╟─12942f82-efb1-11eb-2827-df957759b02c
# ╟─fc7c5be0-1dce-4b58-b72c-e302d83de2f7
# ╟─7a58ca5a-e299-4c26-939c-97e23e8a07a7
# ╟─12942f8c-efb1-11eb-284c-393f6a694818
# ╟─12942fc8-efb1-11eb-0c02-0150ef55ae98
# ╟─0a57f15b-c292-4c64-986d-f046260da66e
# ╟─12942fc8-efb1-11eb-3180-dff1921c5bf9
# ╟─8b4fd005-2f6b-4aab-85c3-acd39717fb69
# ╟─6f371ccd-d92e-44d7-9ab4-23ddcaf13b5f
# ╟─cc8ed1de-beab-43e5-979e-e83df23f96ae
# ╟─34db537f-9cd8-4105-9244-0d05b09a9967
# ╟─1c7ad8e3-dae7-4217-b528-bb0d3d1d5331
# ╟─99a371b2-5158-4c42-8f50-329352b6c1f2
# ╟─f10b7cad-eda3-4ec9-99ee-d43ed013a057
# ╟─3d77d753-b247-4ead-a385-7cbbcfc3190b
# ╟─c6a59b85-d031-4ad4-9e24-691494d08cde
# ╟─e50b8196-e804-473a-b3b5-e22fdb9d2f45
# ╟─f413ea94-36ca-4afc-8ca8-9a7e88101980
# ╟─2bfa1a57-b171-44c3-b0d7-b8dda48d26d7
# ╟─f9970968-cb27-456c-b7ee-70077f9c15ee
# ╟─dbf7fc72-bfd0-4c57-a1a9-fb5881e16e7e
# ╟─db2c6bd4-ee6f-4ba9-b6ec-e7cf94389f93
# ╟─f6093c98-7e89-48ba-95c9-4d1f60a25033
# ╟─aad3a522-97b1-4128-abe5-9e9f93f5c448
# ╟─7e5302a8-8a0d-4c92-8793-4229bc8bc7c6
# ╟─a669a3b1-87f6-4ac0-8ac9-11509ef0f600
# ╟─fa8bb711-81ce-44be-98d3-ca87782a858d
# ╟─6523395e-33ae-453a-94eb-0a7463e9ea94
# ╟─6a89d461-0c68-4dae-9a52-a86d13767674
# ╟─ae86ee9c-3645-43d2-9a42-79658521c3fb
# ╟─c98a4d29-4dfe-4ca4-a2f1-5342788da6c0
# ╟─efda845a-4390-40bc-bdf2-89e555d3b1b2
# ╟─12942f34-efb1-11eb-3eb4-c1a38396cfb8
# ╟─2320f424-7652-4e9f-83ef-fc011b722dcc
# ╟─8fa836a6-1133-4a54-b996-a02083fc6bba
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
