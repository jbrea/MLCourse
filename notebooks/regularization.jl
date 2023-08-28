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

# ╔═╡ 150d58e7-0e73-4b36-836c-d81eef531a9c
begin
using Pkg
Base.redirect_stdio(stderr = devnull, stdout = devnull) do
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
end
using Revise, MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, Statistics
import MLCourse: poly, Polynomial
import PlutoPlotly as PP
MLCourse.CSS_STYLE
end

# ╔═╡ 2e9ce2a9-217e-4910-b6ce-d174f2f2668e
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 7c52d913-65a9-456f-8c54-9ae9c6e4c815
md"The goal of this week is to
1. Unterstand Ridge Regression (L2 regularization)
2. Understand Lasso Regression (L1 regularization)
3. Learn how to apply and tune regularization
4. Learn how to interpret Lasso paths
"

# ╔═╡ 78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
md"# 1. Ridge Regression (L2)
"

# ╔═╡ 1009e251-59af-4f1a-9d0a-e96f4b696cad
md"λ₂ = $(@bind λ₂ Slider(0:.01:5, show_value = true))"

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
const M = MLCourse.JlMod
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
        while β₀old != β₀ || β₁old != β₁
            β₀old, β₁old = β₀, β₁
            β₁ = updateβ₁(x̄, ȳ, x2, xy, β₀, λ)
            β₀ = updateβ₀(x̄, ȳ, β₁, λ)
        end
       (β₀ = β₀, β₁ = β₁)
    end
end

# custom dataset
n = 30
x = rand(n)
y = 2.2x .+ .3 .+ .2randn(n)
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
n = 30
x = np.random.rand(n)
y = 2.2 * x + 0.3 + 0.2 * np.random.randn(n)
 
"""
;
showoutput = false,
collapse = "custom code"
)
end

# ╔═╡ 50ac0b07-ffee-40c3-843e-984b3c628282
l2coefs = M.ridge_regression(M.x, M.y, λ₂)

# ╔═╡ 58746554-ca5a-4e8e-97e5-587a9c2aa44c
let r = λ₂ == 0 ? 6 : norm([l2coefs...]),
    ccol = plot_color(:blue, .3),
	ridge_regression = M.ridge_regression,
    x = M.x, y = M.y
    path = hcat([[ridge_regression(x, y, l)...] for l in 0:.01:5]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y",
	             legend = :topleft)
    plot!(x -> l2coefs.β₀ + x * l2coefs.β₁, w = 3, label = "ridge regression")
    p2 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss", title = "loss with constraints",
		         legend = :bottomright,
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    plot!(t -> r * sin(t), t -> r * cos(t), 0:.001:2π,
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l2coefs.β₀], [l2coefs.β₁], label = "current fit", markersize = 6, color = :red)
    p3 = plot(0:.01:5, path[1, :], label = "β₀", xlabel = "λ₂", ylabel = "")
    plot!(0:.01:5, path[2, :], label = "β₁", ylims = (0, 2.4))
    scatter!([λ₂], [l2coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([λ₂], [l2coefs.β₁], label = nothing, markersize = 6, color = :red)
    p4 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2) + λ₂ * (β₀^2 + β₁^2),
                 label = "loss", title = "regularized loss",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    scatter!([l2coefs.β₀], [l2coefs.β₁], markersize = 6, label = nothing, color = :red)
    plot(p1, p4, p3, p2,
         layout = (2, 2), size = (700, 600), cbar = false)
end

# ╔═╡ f43a82e2-1145-426d-8e0e-5363d1c38ccf
md"# 2. Lasso (L1)"

# ╔═╡ ff7cc2bf-2a38-46d2-8d11-529159b08c82
md"λ₁ = $(@bind λ₁ Slider(0:.01:1, show_value = true))"

# ╔═╡ 4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
l1coefs = M.lasso(M.x, M.y, λ₁)

# ╔═╡ ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
let r = λ₁ == 0 ? 10 : norm([l1coefs...], 1),
    ccol = plot_color(:blue, .3),
    lasso = M.lasso,
    x = M.x, y = M.y
    path = hcat([[lasso(x, y, l)...] for l in 0:.01:1]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y", legend = :topleft)
    plot!(x -> l1coefs.β₀ + x * l1coefs.β₁, w = 3, label = "lasso")
    p2 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss", title = "loss with constraints",
		         legend = :topright,
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    plot!([0, r, 0, -r, 0], [r, 0, -r, 0, r],
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l1coefs.β₀], [l1coefs.β₁], label = "current fit", markersize = 6,
		     color = :red)
    p3 = plot(0:.01:1, path[1, :], label = "β₀", xlabel = "λ₁", ylabel = "")
    plot!(0:.01:1, path[2, :], label = "β₁", ylims = (0, 2.4))
    scatter!([λ₁], [l1coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([λ₁], [l1coefs.β₁], label = nothing, markersize = 6, color = :red)
    p4 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2)/2 + λ₁ * (abs(β₀) + abs(β₁)),
                 label = "loss", title = "regularized loss",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    scatter!([l1coefs.β₀], [l1coefs.β₁], markersize = 6, label = nothing, color = :red)
    plot(p1, p4, p3, p2,
         layout = (2, 2), size = (700, 600), cbar = false)
end


# ╔═╡ ca394b88-06dc-4188-884a-50d7c180aa33
md"# 3. Regularization Examples

## Running the simple examples above
"

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
ridge.coef_
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
lasso.coef_
"""
)

# ╔═╡ 6c87eb35-ddb3-44a3-b4ae-77a371e28960
md"There is also the `ElasticNetRegressor` that allows to fit with L1 and L2 penalties of different strengths. Look up the documentation to learn more about it."

# ╔═╡ bb19c718-c401-4c0c-a6ec-5efc83e6588f
md"""
## Polynomial Ridge Regression

The idea of regularization is very powerful and generally applicable. Here we look at regularizing polynomial regression. Remember, that a polynomial of degree 20 is very flexible and may easily overfit the training data? To counter overfitting, we can now simply regularize the polyomial regression; even at degree 20, the fit may rather underfit than overfit, if the regularization constant is large.

$(mlstring(md"In `MLJ`, we can apply ridge regression or the lasso to polynomial regression simply by replacing in the pipeline the `LinearRegressor` with a `RidgeRegressor` or a `LassoRegressor`, for example `mach = Polynomial(degree = 3) |> RidgeRegressor(lambda = 1e-3)`.", ""))
"""


# ╔═╡ 3d50111b-3a08-4a41-96ce-d77a8e37275d
md"degree = $(@bind degree Slider(1:20, show_value = true, default = 20))

$(@bind lambda Slider(-14:.1:.5, default = -4))
"

# ╔═╡ a54e3439-69b8-41c8-bfe0-4575795fb9b8
md"λ = $(lambda == -14 ? 0 : 10.0^lambda)"

# ╔═╡ 566be5a7-3eae-4c26-ab6d-605dcf08a57d
md"## Tuning Two Hyperparameters: Lambda and Degree

In the cell below we use a `TunedModel` to find with cross validation the best polynomial degree and the best regularization constant for ridge regression. Nore how we can just replace the `LinearRegressor()` by a `RidgeRegressor()` in the usual model for polynomial regression (previously we used `Polynomial() |> LinearRegressor()`)."

# ╔═╡ fe2fe54f-0163-4f5d-9fd1-3d1aa3580875
mlcode(
	"""
using MLJ, MLJLinearModels, DataFrames
import MLCourse: Polynomial
	
model = Polynomial() |> RidgeRegressor()
self_tuning_model = TunedModel(model = model,
                               tuning =  Grid(goal = 500),
                               resampling = CV(nfolds = 5),
                               range = [range(model, :(polynomial.degree),
                                              lower = 1, upper = 20),
                                        range(model, :(ridge_regressor.lambda),
                                              lower = 1e-12, upper = 1e-3,
                                              scale = :log10)],
                               measure = rmse)
f(x) = .3 * sin.(10x) .+ .7x
dataset = let x = rand(50)
    DataFrame(x = x, y = f.(x) .+ .1*randn(50))
end
self_tuning_mach = machine(self_tuning_model, select(dataset, :x), dataset.y)
fit!(self_tuning_mach, verbosity = 0)
report(self_tuning_mach)
"""
,
"""
"""
)

# ╔═╡ 7b4daae4-20d3-4992-94e9-46882e47b840
md"With the report function we can have a look at the best hyper-parameter values (`polynomial.degree`= $(report(M.self_tuning_mach).best_model.polynomial.degree) and
 `ridge_regressor.lambda` = $(report(M.self_tuning_mach).best_model.ridge_regressor.lambda)) found by the self-tuning machine.

The result of the self-tuning machine can be visualized with the `plot` function."

# ╔═╡ f5057d4a-1103-4728-becc-287d93d682ba
mlcode(
"""
using Plots
plot(self_tuning_mach)
"""
,
"""
"""
)

# ╔═╡ adfe4bb8-f3db-4557-b5d8-6efc33f5e321
md"
 In the plot **at the top**, we see the root mean square error estimated with cross-validation for different values of the polynomial degree. We see, for example, that for low polynomial degrees (below degree 4) the self-tuning machine did not find any low errors. For every degree we see multiple blue points, because the self-tuning machine tried for every degree multiple values for the regularization constant.
In the plot **at the right** we see the root mean squared error as for different values of the regularization constant `lambda`. We see, for example, that for high regression values (above ``10^{-6}``) the self-tuning machine did not find any low errors. Again, we see multiple blue points for every value of the regularization constant, because multiple polynomial degrees were tested (the line of blue values at the top of the figure is probably produced by low polynomial degrees). In the plot **at the bottom left** we see with a color-and-size code the loss for different values of polynomial degree and regularization constant lambda. The smaller and darker the circle, the lower the error.

Let us have a look how well the model with the best hyper-parameters fits the data."

# ╔═╡ 596fd0f2-eee0-46ca-a203-e7cbac6f9788
mlcode(
	"""
    p1 = scatter(dataset.x, dataset.y,
                 label = "training data", ylims = (-.1, 1.1))
    plot!(f, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    pred = predict(self_tuning_mach, (x = grid,))
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
	reducible_error_str = string(round(mean((pred .- f.(grid)).^2), sigdigits = 3))
    annotate!([(.28, .6, "reducible error ≈ " * reducible_error_str)])
"""
,
"""
"""
)


# ╔═╡ 13979da4-3d27-4c57-8f7e-f79a5343d46f
Main.PlutoRunner.approx_size(p::Plots.Plot) = try
                sum(p.series_list; init=0) do series
					isnothing(series[:y]) && return 0 # hack to avoid warning
                    length(series[:y])
                end
            catch e
                @warn "Failed to guesstimate plot size" exception=(e,catch_backtrace())
                0
            end

# ╔═╡ b45a9739-0f81-4c0c-a93b-434a5af91490
begin
    f(x) = .3 * sin(10x) + .7x
    function regression_data_generator(; n, seed = 3, rng = MersenneTwister(seed))
        x = rand(rng, n)
        DataFrame(x = x, y = f.(x) .+ .1*randn(rng, n))
    end
    regression_data = regression_data_generator(n = 50)
end;

# ╔═╡ bdbf0dfd-8da5-4e54-89c4-ef4d6b3796ce
let X = select(regression_data, Not(:y)), y = regression_data.y
    mach = fit!(machine(Polynomial(; degree) |> RidgeRegressor(lambda = lambda == -14 ? 0 : 10.0^lambda),
                        X, y), verbosity = 0)
    p1 = scatter(regression_data.x, y, label = "training data", ylims = (-.1, 1.1))
    plot!(f, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    pred = predict(mach, (x = grid,))
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
    annotate!([(.28, .6, "reducible error ≈ $(round(mean((pred .- f.(grid)).^2), sigdigits = 3))")])
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
"""
,
"""
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
"""
)

# ╔═╡ 19bdd76c-4131-422d-983e-1b29cd9edd30
md"We see that the test misclassification rate with regularization
is lower than in our original fit without regularization
(notebook \"Generalized Linear Regression\"; 48 false negatives and 48 false
positives). The misclassification rate on the training set is higher. This
indicates that unregularized logistic regression is too flexible for our spam
data set.
"

# ╔═╡ 13655a50-fbbb-46c7-bdf7-ed5644646966
md"# 4. The Lasso Path for the Weather Data

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
"""
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
"""
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
"""
)

# ╔═╡ 8262b948-6d54-4348-87d1-4c762c74db30
md"# Exercises

## Conceptual

#### Exercise 1.
We review here the two formulations of regularization.
A standard approach to solve constraint optimization
problems makes use of Karush-Kuhn-Tucker (KKT) multipliers. For example, to find
the minimum of function ``f(x)`` under the constraint ``g(x) \leq s`` one can define
the loss function ``L(x, \lambda) = f(x) + \lambda(g(x) - s)`` where
``\lambda\geq0`` is a KKT multiplier.  Minimizing the loss both in ``x`` and
``\lambda`` amounts to solving the equations ``\frac{\partial L}{\partial x} = f'(x) + \lambda g'(x) = 0`` and ``\frac{\partial L}{\partial \lambda} = g(x) - s = 0``,
if the solution is on the boundary of the area defined by the inequality
constraint; otherwise one can find the solution by simply solving the
unconstrained problem, i.e. with ``\lambda = 0``. In this formulation one choses
the size ``s`` of the allowed area and finds ``\lambda`` by solving the equations.

Interestingly, the loss function ``L(x) = f(x) + \lambda
g(x)`` has exactly the same partial derivative in ``x`` as ``L(x, \lambda)`` and
therefore all critical points of ``L(x)`` have corresponding critical
points of ``L(x,\lambda)``. Because of this, regularization is often formulated
as \"adding a regularization term to the cost function\". For example, given the loss
    function of linear regression ``L(\beta) = \frac1n\sum_{i=1}^n(y_i - \beta_0 + \beta_1x_{i1} + \cdots + \beta_p x_{ip})^2`` one can define the
        L1-regularized loss function ``L_\mathrm{L1}(\beta) = L(\beta) + \lambda
\|\beta\|_1`` and choose a value for ``\lambda`` instead of the size ``s`` of the
allowed area.

1. Derive how ``\lambda`` in the second formulation depends on ``s`` in the first formulation for ridge regression with standardized one-dimensional input. *Hint:* use the analytical solution at the top of this notebook, the fact that ``\langle x \rangle = 0`` and ``\langle x^2\rangle = 1`` and note that ``\beta_0^2 + \beta_1^2 = s``, if the solution lies on the boundary.
1. Argue, why choosing ``\lambda = 0`` in the second formulation is equivalent to choosing a sufficiently large ``s`` in the first formulation.
2. Argue, why choosing ``\lambda = \infty`` in the second formulation is equivalent to choosing ``s = 0`` in the first formulation.

#### Exercise 2.
Consider a data set with as many data points as predictors ``n = p``.
Assume ``x_{ii} = 1`` and ``x_{ij} = 0`` for all ``i\neq j`` and arbitrary values
``y_i``. To simplify the problem further we perform regression without an
intercept. We would like to study L1- and L2-regularized multiple linear regression.

1. Write the mean squared error loss once with L1 regularization and once with L2 regularization for this setting and the fomulation of regularization with regularization constant ``\lambda``.
2. Show that in the case of L2 regularization the estimated coefficients take the form ``\hat \beta_j = y_j/(1 + n\lambda)``.
3. Show that in the case of L1 regularization the estimated coefficients take the form ``\hat \beta_j = y_j - \lambda \cdot n/2``, if ``y_j > \lambda\cdot n/2``, ``\hat \beta_j = y_j + \lambda\cdot n/2``, if ``y_j < -\lambda\cdot n/2`` and ``\hat \beta_j = 0`` otherwise.
4. Write a brief summary on how the estimated coefficients ``\hat \beta_j`` are changed relative to the unregularized solution for both kinds of regularization.

## Applied

#### Exercise 3.
Create an artificial dataset with 100 points, 4 predictors ``X_1, X_2, X_3, X_4``
and ``Y = X_1 + \epsilon`` with ``\mathrm{Var}(\epsilon) = 0.1^2``.

You can use the following function to create the dataset
```julia
function data_generator(β; rng = Xoshiro(1), n = 100)
    X = randn(rng, n, 4)
    Y = X * β .+ randn(rng, n) * 0.1
    data = DataFrame(X, :auto)
    data.y = Y
    data
end
data1 = data_generator([1, 0, 0, 0])
```

1. Find with 20-fold cross-validation and the lasso the best model. *Hint:* use a self-tuning machine.
2. Find with 20-fold cross-validation and the ridge regression the best model.
3. Which of the two best models has the lowest test error? *Hint:* use a large test set with ``10^6`` points and a different random seed and the `rmse` loss, e.g.
```julia
test_data1 = data_generator([1, 0, 0, 0], rng = Xoshiro(1234), n = 10^6);
function average_test_error(mach, test_data)
    MLJ.rmse(predict(mach, select(test_data, Not(:y))), test_data.y)
end
```
4. Repeat the above 3 steps on an artificial data set with 100 points and 4 predictors with ``Y = 10X_1 + X_2 + .1 * X_3 + .01 * X_4 + \epsilon``.
"

# ╔═╡ e04c5e8a-15f8-44a8-845d-60acaf795813
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ c48dff95-8028-4e97-8ec6-705ea2b9c72e
MLCourse.FOOTER

# ╔═╡ Cell order:
# ╟─7c52d913-65a9-456f-8c54-9ae9c6e4c815
# ╟─78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
# ╟─1009e251-59af-4f1a-9d0a-e96f4b696cad
# ╟─50ac0b07-ffee-40c3-843e-984b3c628282
# ╟─58746554-ca5a-4e8e-97e5-587a9c2aa44c
# ╟─64b9cfa0-99f7-439b-b70e-f9266754ff74
# ╟─8bd483cc-f490-11eb-38a1-b342dd2551fd
# ╟─f43a82e2-1145-426d-8e0e-5363d1c38ccf
# ╟─ff7cc2bf-2a38-46d2-8d11-529159b08c82
# ╟─4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
# ╟─ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
# ╟─ca394b88-06dc-4188-884a-50d7c180aa33
# ╟─4c3c816c-e901-4931-a27c-632b60291ad7
# ╟─c1033416-334e-4b0e-b81e-6f9137402730
# ╟─1dae5378-f3eb-4598-a060-445bfd8afe5e
# ╟─0429acfe-d31e-427a-96d9-deddfa2c30f8
# ╟─6c87eb35-ddb3-44a3-b4ae-77a371e28960
# ╟─bb19c718-c401-4c0c-a6ec-5efc83e6588f
# ╟─3d50111b-3a08-4a41-96ce-d77a8e37275d
# ╟─a54e3439-69b8-41c8-bfe0-4575795fb9b8
# ╟─bdbf0dfd-8da5-4e54-89c4-ef4d6b3796ce
# ╟─566be5a7-3eae-4c26-ab6d-605dcf08a57d
# ╟─fe2fe54f-0163-4f5d-9fd1-3d1aa3580875
# ╟─7b4daae4-20d3-4992-94e9-46882e47b840
# ╟─f5057d4a-1103-4728-becc-287d93d682ba
# ╟─adfe4bb8-f3db-4557-b5d8-6efc33f5e321
# ╟─596fd0f2-eee0-46ca-a203-e7cbac6f9788
# ╟─13979da4-3d27-4c57-8f7e-f79a5343d46f
# ╟─b45a9739-0f81-4c0c-a93b-434a5af91490
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
# ╟─6f5b0cd1-ba68-46a5-a348-fed201da4a15
# ╟─c9ed011c-8d36-4926-9ec4-84be3b4878d7
# ╟─45df70c6-4c5a-419b-af9d-05d276b3759a
# ╟─40bb385f-1cbd-4555-a8ab-544a67f33595
# ╟─8262b948-6d54-4348-87d1-4c762c74db30
# ╟─e04c5e8a-15f8-44a8-845d-60acaf795813
# ╟─c48dff95-8028-4e97-8ec6-705ea2b9c72e
# ╟─2e9ce2a9-217e-4910-b6ce-d174f2f2668e
# ╟─150d58e7-0e73-4b36-836c-d81eef531a9c
