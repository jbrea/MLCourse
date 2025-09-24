### A Pluto.jl notebook ###
# v0.19.46

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
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, Statistics, NearestNeighborModels
import MLCourse: fitted_linear_func
import PlutoPlotly as PP
const M = MLCourse.JlMod 
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
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

s\_gen = $(@bind s_gen Slider(5:30, default = 20, show_value = true))
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
    classification_test_data = M.classification_data_generator(n = 10^4, s = s_gen)
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
	fprs_l, tprs_l, _ = roc_curve(predict(m2, select(classification_test_data, Not(:y))), classification_test_data.y)
	fprs_p, tprs_p, _ = roc_curve(predict(m4, select(classification_test_data, Not(:y))), classification_test_data.y)
	p = M.σ.(s_gen*(M.f.(classification_test_data.X1) .- classification_test_data.X2))
	fprs_o, tprs_o, _ = roc_curve(UnivariateFinite.(Ref([false, true]), p, augment = true, pool = missing, ordered = true), classification_test_data.y)
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
X_train, X_test, y_train, y_test = train_test_split(mnist.loc[:, mnist.columns != "class"].values /255, mnist["class"].values, test_size=1/7, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
test_predictions = knn.predict(X_test)

mnist_errorrate = np.mean(test_predictions != y_test)
mnist_errorrate
"""
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
    plot!([.215, .215], [bvd.avg[i]-2estd, bvd.avg[i]+2estd], arrow = :both, c = :red, label = "variance")
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

# ╔═╡ 78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
md"# 6. Regularization
## Ridge Regression (L2)
"

# ╔═╡ 1009e251-59af-4f1a-9d0a-e96f4b696cad
md"λ₂ = $(@bind λ₂ Slider([0; 10. .^ (-6:.25:3)], show_value = true))"

# ╔═╡ 50ac0b07-ffee-40c3-843e-984b3c628282
l2coefs = M.ridge_regression(M.x, M.y, λ₂)

# ╔═╡ 58746554-ca5a-4e8e-97e5-587a9c2aa44c
let r = λ₂ == 0 ? 6 : norm([l2coefs...]),
    ccol = plot_color(:blue, .3),
	ridge_regression = M.ridge_regression,
    x = M.x, y = M.y
    ls = [0; 10. .^ (-6:.25:3)]
    lbx = -.1; lby = -.1; ubx = .2; uby = .2
    path = hcat([[ridge_regression(x, y, l)...] for l in ls]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y",
	             legend = :topleft)
    plot!(x -> l2coefs.β₀ + x * l2coefs.β₁, w = 3, label = "ridge regression")
    p2 = contour(lbx:.01:ubx, lby:.01:uby, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss", title = "loss with constraints",
		         legend = :bottomright,
                 levels = 100, aspect_ratio = 1, ylims = (lby, uby), xlims = (lbx, ubx))
    plot!(t -> r * sin(t), t -> r * cos(t), 0:.001:2π,
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l2coefs.β₀], [l2coefs.β₁], label = "current fit", markersize = 6, color = :red)
#     logl = [0; 10. .^ (-6:.25:3)]
p3 = plot(log10.(ls), path[1, :], label = "β₀", xlabel = "log₁₀(λ₂)", ylabel = "")
    plot!(log10.(ls), path[2, :], label = "β₁", ylims = (-.05, .2))
    scatter!([log10(λ₂)], [l2coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([log10(λ₂)], [l2coefs.β₁], label = nothing, markersize = 6, color = :red)
    p4 = contour(lbx:.01:ubx, lby:.01:uby, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2) + λ₂ * (β₀^2 + β₁^2),
                 label = "loss", title = "regularized loss",
                 levels = 100, aspect_ratio = 1, ylims = (lby, uby), xlims = (lbx, ubx))
    scatter!([l2coefs.β₀], [l2coefs.β₁], markersize = 6, label = nothing, color = :red)
    plot(p1, p4, p3, p2,
         layout = (2, 2), size = (700, 600), cbar = false)
end

# ╔═╡ e3081073-9ff8-45c5-b98d-ce7932639a1a
md"
**Regularization allows reducing the variance.**

In the figure below, each dot is the result of one dataset, obtained from the data generating process. We see that the coefficients vary a lot for small values of the regularization constant, whereas for large values of the regularization constant, the β₁ coefficient starts to be biased. The test error is smallest for some optimally chosen regularization constant.
"

# ╔═╡ e79b7859-ff6d-4b39-bfff-7d9ab98ba1de
let
    test_x = rand(10^4)
    test_y = .1 * test_x .+ .2*randn(10^4)
    dataset = () -> let x = rand(10)
                    (x, .1 * x .+ .2 * randn(10))
                end

	ridge_regression = M.ridge_regression
    ls = 10 .^ (-6:.25:4)
    logls = log10.(ls)
    n = 500
    markerstyle = (markershape = :circ, markercolor = :white, markeralpha = .5, markerstrokecolor = :darkblue, markersize = 1)
    coeffs = [ridge_regression(dataset()..., l) for l in ls, _ in 1:n]
    p1 = scatter(repeat(logls, outer = n), reshape(first.(coeffs), :);
                 xlabel = "log₁₀(λ₂)", ylabel = "β₀", label = nothing, markerstyle...)
    plot!(logls, mean(first.(coeffs), dims = 2), lw = 3, label = "average")
    plot!(logls, zero(ls), lw = 3, label = "true β₀")
    p2 = scatter(repeat(logls, outer = n), reshape(last.(coeffs), :);
                 xlabel = "log₁₀(λ₂)", ylabel = "β₁", label = nothing, markerstyle...)
    plot!(logls, mean(last.(coeffs), dims = 2), lw = 3, label = "average")
    plot!(logls, zero(ls) .+ .1, lw = 3, label = "true β₁")
    testerrs = [mean(abs2, c.β₀ .+ c.β₁ * test_x - test_y) for c in coeffs]
    p3 = scatter(repeat(logls, outer = n),
                 reshape(testerrs, :); xlabel = "log₁₀(λ₂)", ylabel = "test MSE", label = nothing,
                 markerstyle...)
    plot!(logls, mean(testerrs, dims = 2), label = "average", lw = 3)
    plot!(logls, zero(ls) .+ .2^2, label = "irreducible error", lw = 2, ylims = (.03, .1))
    plot(plot(p1, p2, layout = (2, 1)), p3)
end

# ╔═╡ 64b9cfa0-99f7-439b-b70e-f9266754ff74
md" ## Implementation Details
For the illustrations in this notebook we use some custom code to run ridge regression and the lasso for the simple example of 1-dimensional input. In this example we penalize also the intercept β₀. For ridge regression the solution is
```math
\begin{eqnarray*}
\beta_1 &= \frac{\langle x y \rangle - \frac{\langle x\rangle \langle y\rangle}{1 + \lambda}}{\langle x^2\rangle - \frac{\langle x\rangle^2}{1 + \lambda} + \lambda}\\
\beta_0 &= \frac{\langle y \rangle - \beta_1 \langle x \rangle}{1 + \lambda}
\end{eqnarray*}
```
where ``\langle . \rangle`` denotes the average.

For the lasso, we run a fixed point iteration that starts at the unregularized solution of linear regression and shrinks β₁ and β₀ towards zero until there is not change anymore. You do not need to understand the custom code, but feel free to have a look at it, if you are interested.
"

# ╔═╡ 8bd483cc-f490-11eb-38a1-b342dd2551fd
begin
mlcode(
"""
import Statistics: mean

begin
    function ridge_regression(x, y, λ)
       β₁ = (mean(x .* y) - mean(x) * mean(y)/(1 + λ))/
		    (mean(x.^2) - mean(x)^2/(1 + λ) + λ)
       β₀ = (mean(y) - β₁ * mean(x))/(1 + λ)
       (β₀ = β₀, β₁ = β₁)
    end
    function updateβ₀(x̄, ȳ, β₁, l)
        tmp = ȳ - β₁ * x̄
        abs(tmp) > l ? tmp - sign(tmp) * l : 0.
    end
    function updateβ₁(x̄, ȳ, x2, xy, β₀, l)
        tmp = (x̄ * ȳ - xy - x̄ * sign(β₀) * l)
        abs(tmp) > l ? (tmp - sign(tmp) * l)/(x̄^2 - x2) : 0
    end
    function lasso(x, y, λ)
        x̄ = mean(x)
        ȳ = mean(y)
        x2 = mean(x.^2)
        xy = mean(x .* y)
        β₁ = (x̄ * ȳ - xy)/(x̄^2 - x2)
        β₀ = ȳ - β₁ * x̄
        β₀old, β₁old = zero(β₀), zero(β₁)
        i = 0
        while β₀old != β₀ || β₁old != β₁
            β₀old, β₁old = β₀, β₁
            β₁ = updateβ₁(x̄, ȳ, x2, xy, β₀, λ)
            β₀ = updateβ₀(x̄, ȳ, β₁, λ)
            i += 1
            i == 10^4 && break # prevent infinite loop
        end
       (β₀ = β₀, β₁ = β₁)
    end
end

# custom dataset
using Random
Random.seed!(8)

n = 10
x = rand(n)
y = .1x .+ .2randn(n)
"""
,
"""
import numpy as np

def ridge_regression(x, y, l):
    b1 = (np.mean(x * y) - np.mean(x) * np.mean(y) / (1 + l)) / \
         (np.mean(x**2) - np.mean(x)**2 / (1 + l) + l)
    b0 = (np.mean(y) - b1 * np.mean(x)) / (1 + l)
    return (b0, b1)

def updateb0(x_bar, y_bar, b1, l):
    tmp = y_bar - b1 * x_bar
    return tmp - np.sign(tmp) * l if np.abs(tmp) > l else 0.

def updateb1(x_bar, y_bar, x2, xy, b0, l):
    tmp = (x_bar * y_bar - xy - x_bar * np.sign(b0) * l)
    return (tmp - np.sign(tmp) * l) / (x_bar**2 - x2) if np.abs(tmp) > l else 0.

def lasso(x, y, l):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    x2 = np.mean(x**2)
    xy = np.mean(x * y)
    b1 = (x_bar * y_bar - xy) / (x_bar**2 - x2)
    b0 = y_bar - b1 * x_bar
    b0_old, b1_old = 0., 0.
    while b0_old != b0 or b1_old != b1:
        b0_old, b1_old = b0, b1
        b1 = updateb1(x_bar, y_bar, x2, xy, b0, l)
        b0 = updateb0(x_bar, y_bar, b1, l)
    return (b0, b1)

# custom dataset
n = 10
x = np.random.rand(n)
y = .1 * x + 0.2 * np.random.randn(n)
 
"""
;
showoutput = false,
collapse = "custom code",
cache = false
)
end

# ╔═╡ f43a82e2-1145-426d-8e0e-5363d1c38ccf
md"## Lasso (L1)"

# ╔═╡ ff7cc2bf-2a38-46d2-8d11-529159b08c82
md"λ₁ = $(@bind λ₁ Slider([0; 10. .^ (-8:.25:1)], show_value = true))"

# ╔═╡ 4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
l1coefs = M.lasso(M.x, M.y, λ₁)

# ╔═╡ ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
let r = λ₁ == 0 ? 10 : norm([l1coefs...], 1),
    ccol = plot_color(:blue, .3),
    lasso = M.lasso,
    x = M.x, y = M.y
    ls = [0; 10. .^ (-8:.25:1)]
    lbx = -.1; lby = -.1; ubx = .2; uby = .2
    path = hcat([[lasso(x, y, l)...] for l in ls]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y", legend = :topleft)
    plot!(x -> l1coefs.β₀ + x * l1coefs.β₁, w = 3, label = "lasso")
    p2 = contour(lbx:.01:ubx, lby:.01:uby, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss", title = "loss with constraints",
		         legend = :topleft,
                 levels = 100, aspect_ratio = 1, ylims = (lby, uby), xlims = (lbx, ubx))
    plot!([0, r, 0, -r, 0], [r, 0, -r, 0, r],
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l1coefs.β₀], [l1coefs.β₁], label = "current fit", markersize = 6,
		     color = :red)
    p3 = plot(log10.(ls), path[1, :], label = "β₀", xlabel = "log₁₀(λ₁)", ylabel = "")
    plot!(log10.(ls), path[2, :], label = "β₁", ylims = (-0.05, .2))
    scatter!([log10(λ₁)], [l1coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([log10(λ₁)], [l1coefs.β₁], label = nothing, markersize = 6, color = :red)
    p4 = contour(lbx:.01:ubx, lby:.01:uby, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2)/2 + λ₁ * (abs(β₀) + abs(β₁)),
                 label = "loss", title = "regularized loss",
                 levels = 100, aspect_ratio = 1, ylims = (lby, uby), xlims = (lbx, ubx))
    scatter!([l1coefs.β₀], [l1coefs.β₁], markersize = 6, label = nothing, color = :red)
    plot(p1, p4, p3, p2,
         layout = (2, 2), size = (700, 600), cbar = false)
end


# ╔═╡ 4c3c816c-e901-4931-a27c-632b60291ad7
md"""Instead of using the custom code to compute the ridge regression and the lasso we could have used some $(mlstring("MLJ", "")) functions."""

# ╔═╡ c1033416-334e-4b0e-b81e-6f9137402730
mlcode(
"""
using MLJ, MLJLinearModels, DataFrames

mach = machine(RidgeRegressor(lambda = 3.82, penalize_intercept = true),
	           DataFrame(x = x), y)
fit!(mach, verbosity = 0)
fitted_params(mach)
"""
,
"""
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LassoCV

ridge = Ridge(alpha=3.82, fit_intercept=True)
ridge.fit(x.reshape(-1, 1), y)

("coeff : ", ridge.coef_, "intercept : ", ridge.intercept_)
"""
)

# ╔═╡ 1dae5378-f3eb-4598-a060-445bfd8afe5e
md"You can check with the slider above that we get indeed the same result with our custom method."

# ╔═╡ 0429acfe-d31e-427a-96d9-deddfa2c30f8
mlcode(
"""
mach = machine(LassoRegressor(lambda = .1,
	                          # usually the intercept is not penalized,
	                          # but here we do penalize it.
	                          penalize_intercept = true,
	                          # usually the default optimizer is quite good,
	                          # but here we decrease the tolerance to get
	                          # higher precision.
                              solver = ISTA(tol = 1e-8)),
	           DataFrame(x = x), y)
fit!(mach, verbosity = 0)
fitted_params(mach)
"""
,
"""
lasso = Lasso(alpha=0.1, fit_intercept=True, tol=1e-8)
lasso.fit(x.reshape(-1, 1), y)
("coeff : ", lasso.coef_, "intercept : ", lasso.intercept_)
"""
)

# ╔═╡ 6c87eb35-ddb3-44a3-b4ae-77a371e28960
mlstring(md"There is also the `ElasticNetRegressor` that allows to fit with L1 and L2 penalties of different strengths. Look up the documentation to learn more about it.",
"There is also the `ElasticNet` that allows to fit a Linear regression with combined L1 and L2 priors as regularizer")

# ╔═╡ bb19c718-c401-4c0c-a6ec-5efc83e6588f
md"""
## Polynomial Ridge Regression

The idea of regularization is very powerful and generally applicable. Here we look at regularizing polynomial regression. Remember, that a polynomial of degree 20 is very flexible and may easily overfit the training data? To counter overfitting, we can now simply regularize the polyomial regression; even at degree 20, the fit may rather underfit than overfit, if the regularization constant is large.

$(mlstring(md"In `MLJ`, we can apply ridge regression or the lasso to polynomial regression simply by replacing in the pipeline the `LinearRegressor` with a `RidgeRegressor` or a `LassoRegressor`, for example `mach = Polynomial(degree = 3) |> RidgeRegressor(lambda = 1e-3)`.", ""))
"""


# ╔═╡ 3d50111b-3a08-4a41-96ce-d77a8e37275d
md"degree = $(@bind degree3 Slider(1:20, show_value = true, default = 20))

$(@bind lambda Slider(-14:.1:.5, default = -4))
"

# ╔═╡ a54e3439-69b8-41c8-bfe0-4575795fb9b8
md"λ = $(lambda == -14 ? 0 : 10.0^lambda)"

# ╔═╡ bdbf0dfd-8da5-4e54-89c4-ef4d6b3796ce
let X = select(regression_data, Not(:y)), y = regression_data.y
    mach = fit!(machine(Polynomial(degree = degree3) |> RidgeRegressor(lambda = lambda == -14 ? 0 : 10.0^lambda),
                        X, y), verbosity = 0)
    p1 = scatter(regression_data.x, y, label = "training data", ylims = (-.1, 1.1))
    plot!(M.f, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    pred = predict(mach, (x = grid,))
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
    annotate!([(.28, .6, "reducible error ≈ $(round(mean((pred .- M.f.(grid)).^2), sigdigits = 3))")])
end

# ╔═╡ 8e170a5a-9c46-413e-895d-796e178b69df
md"## Multiple Logistic Ridge Regression on the Spam Data

We load here the preprocessed spam data.
"

# ╔═╡ 8e542a48-ed28-4297-b2e8-d6a755a5fdf9
mlcode(
"""
using CSV
spam_train = CSV.read(download("https://go.epfl.ch/bio322-spam_train.csv"), DataFrame)
spam_train.spam_or_ham = String.(spam_train.spam_or_ham)
coerce!(spam_train, :spam_or_ham => OrderedFactor)
spam_test = CSV.read(download("https://go.epfl.ch/bio322-spam_test.csv"), DataFrame)
spam_test.spam_or_ham = String.(spam_test.spam_or_ham)
coerce!(spam_test, :spam_or_ham => OrderedFactor)
spam_train
"""
,
"""
import pandas as pd

# Read training data from CSV file
spam_train = pd.read_csv("https://go.epfl.ch/bio322-spam_train.csv")
spam_train["spam_or_ham"] = spam_train["spam_or_ham"].astype(str).astype("category")

# Convert column to ordered factor
spam_train["spam_or_ham"] = pd.Categorical(spam_train["spam_or_ham"], ordered=True)

# Read test data from CSV file and Convert column to ordered factor
spam_test = pd.read_csv("https://go.epfl.ch/bio322-spam_test.csv")
spam_test["spam_or_ham"] = spam_test["spam_or_ham"].astype(str).astype("category")
spam_test["spam_or_ham"] = pd.Categorical(spam_test["spam_or_ham"], ordered=True)

spam_train
"""
)

# ╔═╡ c5ef5d4e-200d-46d9-86fa-50af1896a6c3
md"The `LogisticClassifier` and the `MultinomialClassifier` have a `penalty` argument that can be used to enforce an L1 or L2 penalty. Look up the documentation to learn more about it."

# ╔═╡ d956613e-db32-488c-8ebb-fd61dfa31e59
mlcode(
"""
spam_mach = machine(LogisticClassifier(penalty = :l2, lambda = 1e-5),
                    select(spam_train, Not(:spam_or_ham)),
                    spam_train.spam_or_ham)
fit!(spam_mach, verbosity = 0)
confusion_matrix(predict_mode(spam_mach, select(spam_train, Not(:spam_or_ham))),
                 spam_train.spam_or_ham) # on training data
"""
,
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

spam_mach = LogisticRegression(penalty="l2", C=10)
spam_mach.fit(spam_train.drop("spam_or_ham", axis=1), spam_train["spam_or_ham"])

cm = confusion_matrix(spam_train["spam_or_ham"], 	spam_mach.predict(spam_train.drop("spam_or_ham", axis=1)))
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=spam_mach.classes_)
disp.plot()
plt.show()
"""
)


# ╔═╡ ef701511-db7e-4dc0-8d31-ea14471943ab
mlcode(
"""
confusion_matrix(predict_mode(spam_mach, select(spam_test, Not(:spam_or_ham))),
                 spam_test.spam_or_ham) # on test data
"""
,
"""
cm = confusion_matrix(spam_test["spam_or_ham"], 	spam_mach.predict(spam_test.drop("spam_or_ham", axis=1)))
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=spam_mach.classes_)
disp.plot()
plt.show()
"""
)

# ╔═╡ 19bdd76c-4131-422d-983e-1b29cd9edd30
mlstring(md"We see that the test misclassification rate with regularization
is lower than in our original fit without regularization
(notebook \"Generalized Linear Regression\"; 50 false negatives and 34 false
positives). The misclassification rate on the training set is higher. This
indicates that unregularized logistic regression is too flexible for our spam
data set.
",
"
We see that the test misclassification rate with regularization
is lower than in our original fit without regularization
(notebook \"Generalized Linear Regression\"; 56 false negatives and 36 false
positives). The misclassification rate on the training set is higher. This
indicates that unregularized logistic regression is too flexible for our spam
data set.
")

# ╔═╡ 13655a50-fbbb-46c7-bdf7-ed5644646966
md"## The Lasso Path for the Weather Data

For the Lasso it is often interesting to see the fitted parameter values for different regularization values (the Lasso path). In the following we use the package `GLMNet` to do so."

# ╔═╡ 1fa932c1-ce29-40ca-a8dc-e636aa2ecf66
mlcode(
"""
using CSV
weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"), 
                   DataFrame)
"""
,
"""
import pandas as pd

weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv")
weather
"""
,
)

# ╔═╡ ecf80b6a-1946-46fd-b1b4-bcbe91848e3c
mlcode(
"""
import GLMNet: glmnet
weather_input = select(weather, Not(:LUZ_wind_peak))[1:end-5, :]
weather_output = weather.LUZ_wind_peak[6:end]
weather_fits = glmnet(Array(weather_input), weather_output)
"""
,
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path

weather_input = weather.drop("LUZ_wind_peak", axis=1).iloc[:-5, :]
weather_output = weather["LUZ_wind_peak"].iloc[5:]

scaler = StandardScaler().fit(weather_input)
weather_input = scaler.transform(weather_input)

alphas =np.logspace(-6,1,10)
alphas_lasso, coefs_lasso, _ = lasso_path(weather_input, weather_output, alphas=alphas)
alphas_lasso
"""
,
cache_jl_vars = [:weather_fits, :weather_input]
)

# ╔═╡ f6158ed5-bd0d-4781-b9cd-22b271e86ef8
md"In the following figure we can see that the predictor `BER_wind_peak` is the first one to have a non-zero coefficient when we decrease the regularization constant `λ`."

# ╔═╡ 4652a904-5edb-463c-a046-5c5d378f7cca
let fits = M.weather_fits,
    lambda = log.(fits.lambda),
    col_names = names(M.weather_input)
	cols = union((x -> x[1]).(findall(fits.betas .> 0)))
	append!(cols, setdiff(1:length(col_names), cols))
	p = [PP.scatter(x = lambda, y = fits.betas[i, :],
	                name = col_names[i])
        for i in cols]
    PP.PlutoPlot(PP.Plot(p, PP.Layout(xaxis_title = "log(λ)")))
end

# ╔═╡ 83fe757f-a9aa-4a60-8a49-dce5e9fcf56c
# mlstring("",md"Python version :")

# ╔═╡ 35913b1d-86d0-4338-9f7c-41a89651b2dc
mlcode(
"""
""",
"""
df = pd.DataFrame ({"parameters":np.tile(weather.drop("LUZ_wind_peak", axis=1).columns, len(alphas_lasso)),
                    "log_alpha": np.log(np.repeat(alphas_lasso,len(coefs_lasso))),
                    "coefs": np.array(coefs_lasso).T.flatten()})

import plotly.express as px
fig = px.line(df,x="log_alpha", y="coefs", color="parameters")
# fig.show()
"""
,
showinput = false,
eval = false,
showoutput = false
)

# ╔═╡ 6f5b0cd1-ba68-46a5-a348-fed201da4a15
md"Indeed, if we were allowed to use only one predictor, the wind peak in Bern is most informative about the wind peak in Luzern five hours later. The correlation is positive, but there is a lot of noise."

# ╔═╡ c9ed011c-8d36-4926-9ec4-84be3b4878d7
mlcode(
"""
scatter(weather_input.BER_wind_peak, weather_output,
        xlabel = "wind peak in Bern [km/h]",
        ylabel = "wind peak in Luzern 5 hours later [km/h]",
        label = nothing)
"""
,
"""
plt.figure()
plt.scatter(weather["BER_wind_peak"].iloc[:-5], weather_output.values)
plt.xlabel("wind peak in Bern [km/h]")
plt.ylabel("wind peak in Luzern 5 hours later [km/h]")
plt.show()
"""
)

# ╔═╡ 45df70c6-4c5a-419b-af9d-05d276b3759a
md"In the figures below we see that the first few predictors explain most of the variability that is explainable with linear models. In fact, at `log(λ) = 0` we see that less than 10 predictors are sufficient to explain approximately 30% of the variance. Adding more predictors increases the explained variance to less than 40%."

# ╔═╡ 40bb385f-1cbd-4555-a8ab-544a67f33595
mlcode(
"""
lambda = log.(weather_fits.lambda)
p1 = plot(lambda, 100 * weather_fits.dev_ratio, ylabel = "% variance explained")
p2 = plot(lambda, reshape(sum(weather_fits.betas .!= 0, dims = 1), :),
          ylabel = "non-zero parameters",
          xlabel = "log(λ)")
plot(p1, p2, layout = (2, 1), legend = false)
""",
"""
from sklearn.linear_model import LassoCV

# Initialize the LassoCV model with cross-validation
lasso_cv = LassoCV(alphas=alphas, cv=5)

# Fit the LassoCV model to the data
lasso_cv.fit (weather_input, weather_output)

# Get the variance explained by each alpha
variance_explained = 1 - (lasso_cv.mse_path_.mean(axis=1) / np.var(weather_output))

fig, (ax1, ax2) = plt.subplots(2, sharex = True)
ax1.plot(np.log(alphas_lasso), variance_explained)
ax1.set_ylabel("variance explained")
ax2.plot(np.log(alphas_lasso), np.count_nonzero(coefs_lasso.T, axis =1))
ax2.set_ylabel("non zeros parameters")
ax2.set_xlabel("log(alpha)")
plt.show()
"""
)

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
* Use the same kind of preprocessing as in the example of linear regression (or exercise 6) in [2. supervised learning](https://bio322.epfl.ch/notebooks/supervised_learning.html), i.e. load the training and the test set and make sure to shift the predictors and the response by 5 hours.
* Compute the training and the test loss for ``K = 5, 10, 20, 50, 100``.
* Which value of the hyper-parameter ``K`` should we prefer to make predictions on new data?
* Should we prefer K-nearest neighbors with optimal ``K`` or multiple linear regression to make predictions on new data? *Hint*: Remember that we found a training error (RMSE) of approximately 8.1 and a test error of 8.9.
"

# ╔═╡ c98a4d29-4dfe-4ca4-a2f1-5342788da6c0
md"
#### Exercise 6 (optional)
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
# ╟─78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
# ╟─1009e251-59af-4f1a-9d0a-e96f4b696cad
# ╟─50ac0b07-ffee-40c3-843e-984b3c628282
# ╟─58746554-ca5a-4e8e-97e5-587a9c2aa44c
# ╟─e3081073-9ff8-45c5-b98d-ce7932639a1a
# ╟─e79b7859-ff6d-4b39-bfff-7d9ab98ba1de
# ╟─64b9cfa0-99f7-439b-b70e-f9266754ff74
# ╟─8bd483cc-f490-11eb-38a1-b342dd2551fd
# ╟─f43a82e2-1145-426d-8e0e-5363d1c38ccf
# ╟─ff7cc2bf-2a38-46d2-8d11-529159b08c82
# ╟─4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
# ╟─ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
# ╟─4c3c816c-e901-4931-a27c-632b60291ad7
# ╟─c1033416-334e-4b0e-b81e-6f9137402730
# ╟─1dae5378-f3eb-4598-a060-445bfd8afe5e
# ╟─0429acfe-d31e-427a-96d9-deddfa2c30f8
# ╟─6c87eb35-ddb3-44a3-b4ae-77a371e28960
# ╟─bb19c718-c401-4c0c-a6ec-5efc83e6588f
# ╟─3d50111b-3a08-4a41-96ce-d77a8e37275d
# ╟─a54e3439-69b8-41c8-bfe0-4575795fb9b8
# ╟─bdbf0dfd-8da5-4e54-89c4-ef4d6b3796ce
# ╟─8e170a5a-9c46-413e-895d-796e178b69df
# ╟─8e542a48-ed28-4297-b2e8-d6a755a5fdf9
# ╟─c5ef5d4e-200d-46d9-86fa-50af1896a6c3
# ╟─d956613e-db32-488c-8ebb-fd61dfa31e59
# ╟─ef701511-db7e-4dc0-8d31-ea14471943ab
# ╟─19bdd76c-4131-422d-983e-1b29cd9edd30
# ╟─13655a50-fbbb-46c7-bdf7-ed5644646966
# ╟─1fa932c1-ce29-40ca-a8dc-e636aa2ecf66
# ╟─ecf80b6a-1946-46fd-b1b4-bcbe91848e3c
# ╟─f6158ed5-bd0d-4781-b9cd-22b271e86ef8
# ╟─4652a904-5edb-463c-a046-5c5d378f7cca
# ╟─83fe757f-a9aa-4a60-8a49-dce5e9fcf56c
# ╟─35913b1d-86d0-4338-9f7c-41a89651b2dc
# ╟─6f5b0cd1-ba68-46a5-a348-fed201da4a15
# ╟─c9ed011c-8d36-4926-9ec4-84be3b4878d7
# ╟─45df70c6-4c5a-419b-af9d-05d276b3759a
# ╟─40bb385f-1cbd-4555-a8ab-544a67f33595
# ╟─ae86ee9c-3645-43d2-9a42-79658521c3fb
# ╟─c98a4d29-4dfe-4ca4-a2f1-5342788da6c0
# ╟─efda845a-4390-40bc-bdf2-89e555d3b1b2
# ╟─12942f34-efb1-11eb-3eb4-c1a38396cfb8
# ╟─2320f424-7652-4e9f-83ef-fc011b722dcc
# ╟─8fa836a6-1133-4a54-b996-a02083fc6bba
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
