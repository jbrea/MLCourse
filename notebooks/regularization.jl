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

# ╔═╡ 150d58e7-0e73-4b36-836c-d81eef531a9c
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI, MLJ, MLJLinearModels, Plots, LinearAlgebra, Random,
    DataFrames, CSV
    gr()
    PlutoUI.TableOfContents()
end


# ╔═╡ e04c5e8a-15f8-44a8-845d-60acaf795813
begin
    using MLCourse
    import MLCourse: PolynomialRegressor, poly
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
md"# Ridge Regression (L2 Regularization)
"

# ╔═╡ 9e1e8284-a8c1-47a9-83d0-2d8fbd8ce005
n = 30; x = rand(n); y = 2.2x .+ .3 .+ .2randn(n);

# ╔═╡ 1009e251-59af-4f1a-9d0a-e96f4b696cad
md"λ₂ = $(@bind λ₂ Slider(0:1:100, show_value = true))"

# ╔═╡ 8bd483cc-f490-11eb-38a1-b342dd2551fd
begin
    regmean(x, λ) = 1/(length(x) + λ) * sum(x)
    function ridge_regression(x, y, λ)
       β₁ = (mean(x .* y) - mean(x) * regmean(y, λ))/
		    (mean(x.^2) - mean(x)*regmean(x, λ) + λ/length(y))
       β₀ = regmean(y, λ) - β₁ * regmean(x, λ)
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
        l = λ/length(x)
        β₁ = (x̄ * ȳ - xy)/(x̄^2 - x2)
        β₀ = ȳ - β₁ * x̄
        β₀old, β₁old = zero(β₀), zero(β₁)
        while β₀old != β₀ || β₁old != β₁
            β₀old, β₁old = β₀, β₁
            β₁ = updateβ₁(x̄, ȳ, x2, xy, β₀, l)
            β₀ = updateβ₀(x̄, ȳ, β₁, l)
        end
       (β₀ = β₀, β₁ = β₁)
    end
end;

# ╔═╡ 50ac0b07-ffee-40c3-843e-984b3c628282
l2coefs = ridge_regression(x, y, λ₂)

# ╔═╡ 58746554-ca5a-4e8e-97e5-587a9c2aa44c
let r = λ₂ == 0 ? 6 : norm([l2coefs...]),
    ccol = plot_color(:blue, .3),
    path = hcat([[ridge_regression(x, y, l)...] for l in 0:100]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y")
    plot!(x -> l2coefs.β₀ + x * l2coefs.β₁, w = 3, label = "ridge regression")
    p2 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss", title = "loss with constraints",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    plot!(t -> r * sin(t), t -> r * cos(t), 0:.001:2π,
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l2coefs.β₀], [l2coefs.β₁], label = "current fit", markersize = 6, color = :red)
    p3 = plot(0:100, path[1, :], label = "β₀", xlabel = "λ₂", ylabel = "")
    plot!(0:100, path[2, :], label = "β₁", ylims = (0, 2.4))
    scatter!([λ₂], [l2coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([λ₂], [l2coefs.β₁], label = nothing, markersize = 6, color = :red)
    p4 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> sum((β₀ .+ β₁*x .- y).^2) + λ₂ * (β₀^2 + β₁^2),
                 label = "loss", title = "regularized loss",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    scatter!([l2coefs.β₀], [l2coefs.β₁], markersize = 6, label = nothing, color = :red)
    plot(p1, p4, p3, p2,
         layout = (2, 2), size = (700, 600), cbar = false, legend = :right)
end

# ╔═╡ f43a82e2-1145-426d-8e0e-5363d1c38ccf
md"# Lasso (L1 Regression)"

# ╔═╡ ff7cc2bf-2a38-46d2-8d11-529159b08c82
md"λ₁ = $(@bind λ₁ Slider(0:1:30, show_value = true))"

# ╔═╡ 4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
l1coefs = lasso(x, y, λ₁)

# ╔═╡ ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
let r = λ₁ == 0 ? 10 : norm([l1coefs...], 1),
    ccol = plot_color(:blue, .3),
    path = hcat([[lasso(x, y, l)...] for l in 0:30]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y")
    plot!(x -> l1coefs.β₀ + x * l1coefs.β₁, w = 3, label = "lasso")
    p2 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss", title = "loss with constraints",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    plot!([0, r, 0, -r, 0], [r, 0, -r, 0, r],
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l1coefs.β₀], [l1coefs.β₁], label = nothing, markersize = 6)
    p3 = plot(0:30, path[1, :], label = "β₀", xlabel = "λ₁", ylabel = "")
    plot!(0:30, path[2, :], label = "β₁", ylims = (0, 2.4))
    scatter!([λ₁], [l1coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([λ₁], [l1coefs.β₁], label = nothing, markersize = 6, color = :red)
    p4 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> sum((β₀ .+ β₁*x .- y).^2)/2 + λ₁ * (abs(β₀) + abs(β₁)),
                 label = "loss", title = "regularized loss",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    scatter!([l1coefs.β₀], [l1coefs.β₁], markersize = 6, label = nothing, color = :red)
    plot(p1, p4, p3, p2,
         layout = (2, 2), size = (700, 600), cbar = false, legend = :right)
end


# ╔═╡ ca394b88-06dc-4188-884a-50d7c180aa33
md"# Regularization Examples

## Polynomial Ridge Regression"

# ╔═╡ b45a9739-0f81-4c0c-a93b-434a5af91490
begin
    f(x) = .3 * sin(10x) + .7x
    function regression_data_generator(; n, seed = 3, rng = MersenneTwister(seed))
        x = rand(rng, n)
        DataFrame(x = x, y = f.(x) .+ .1*randn(rng, n))
    end
    regression_data = regression_data_generator(n = 50)
end;

# ╔═╡ 3d50111b-3a08-4a41-96ce-d77a8e37275d
md"degree = $(@bind degree Slider(0:20, show_value = true, default = 20))

$(@bind lambda Slider(-14:.1:.5, default = -4))
"

# ╔═╡ a54e3439-69b8-41c8-bfe0-4575795fb9b8
md"λ = $(lambda == -14 ? 0 : 10.0^lambda)"

# ╔═╡ bdbf0dfd-8da5-4e54-89c4-ef4d6b3796ce
let X = select(poly(regression_data, degree), Not(:y)), y = regression_data.y
    mach = fit!(machine(RidgeRegressor(lambda = lambda == -14 ? 0 : 10.0^lambda),
                        X, y), verbosity = 0)
    p1 = scatter(regression_data.x, y, label = "training data", ylims = (-.1, 1.1))
    plot!(f, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    pred = predict(mach, poly((x = grid,), degree))
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
    annotate!([(.28, .6, "reducible error ≈ $(round(mean((pred .- f.(grid)).^2), sigdigits = 3))")])
end

# ╔═╡ fe2fe54f-0163-4f5d-9fd1-3d1aa3580875
begin
    model = PolynomialRegressor(regressor = RidgeRegressor())
    self_tuning_model = TunedModel(model = model,
                                   tuning =  Grid(goal = 500),
                                   resampling = CV(nfolds = 5),
                                   range = [range(model, :degree,
                                                  lower = 1, upper = 20),
                                            range(model, :(regressor.lambda),
                                                  lower = 1e-12, upper = 1e-3,
                                                  scale = :log)],
                                   measure = rmse)
    self_tuning_mach = machine(self_tuning_model,
                               select(regression_data, :x),
                               regression_data.y) |> fit!
end;

# ╔═╡ f5057d4a-1103-4728-becc-287d93d682ba
plot(self_tuning_mach)

# ╔═╡ bd54bfcd-f682-4b74-8a44-35463d421491
report(self_tuning_mach)

# ╔═╡ 596fd0f2-eee0-46ca-a203-e7cbac6f9788
let
    p1 = scatter(regression_data.x, regression_data.y,
                 label = "training data", ylims = (-.1, 1.1))
    plot!(f, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    pred = predict(self_tuning_mach, (x = grid,))
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
    annotate!([(.28, .6, "reducible error ≈ $(round(mean((pred .- f.(grid)).^2), sigdigits = 3))")])
end


# ╔═╡ 8e170a5a-9c46-413e-895d-796e178b69df
md"## Multiple Logistic Ridge Regression on the Spam Data

We load here the preprocessed spam data.
If the following csv files cannot be imported, please check that they were
correctly generated in the \"Generalized Linear Regression\" notebook.
"

# ╔═╡ 8e542a48-ed28-4297-b2e8-d6a755a5fdf9
begin
    spam_train = CSV.read(joinpath(dirname(pathof(MLCourse)), "..", "data",
                                   "spam_preprocessed.csv"), DataFrame)
    coerce!(spam_train, :spam_or_ham => Binary)
    spam_test = CSV.read(joinpath(dirname(pathof(MLCourse)), "..", "data",
                                  "spam_preprocessed_test.csv"), DataFrame)
    coerce!(spam_test, :spam_or_ham => Binary)
end;

# ╔═╡ 552f14fc-06c7-4c2a-9515-e64f28828b70
begin
    spam_fit = fit!(machine(LogisticClassifier(penalty = :l2, lambda = 1e-2),
                            select(spam_train, Not(:spam_or_ham)),
                            spam_train.spam_or_ham))
    confusion_matrix(predict_mode(spam_fit, select(spam_train, Not(:spam_or_ham))),
                     spam_train.spam_or_ham)
end

# ╔═╡ ef701511-db7e-4dc0-8d31-ea14471943ab
confusion_matrix(predict_mode(spam_fit, select(spam_test, Not(:spam_or_ham))),
                 spam_test.spam_or_ham)

# ╔═╡ 19bdd76c-4131-422d-983e-1b29cd9edd30
md"We see that the test misclassification rate with regularization
is much lower than in our original fit without regularization
(notebook \"Generalized Linear Regression\"). The misclassification rate
on the training set is higher. This indicates that unregularized logistic
regression is too flexible for our spam data set.
"

# ╔═╡ 13655a50-fbbb-46c7-bdf7-ed5644646966
md"## The Lasso Path for the Weather Data"

# ╔═╡ 1fa932c1-ce29-40ca-a8dc-e636aa2ecf66
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), DataFrame);

# ╔═╡ 470dc7f4-04a9-4253-8125-9112778021eb
import Lasso

# ╔═╡ ecf80b6a-1946-46fd-b1b4-bcbe91848e3c
begin
    weather_input = select(weather, Not(:LUZ_wind_peak))[1:end-5, :]
    weather_output = weather.LUZ_wind_peak[6:end]
    weather_fits = Lasso.fit(Lasso.LassoPath, Array(weather_input), weather_output)
end

# ╔═╡ 4652a904-5edb-463c-a046-5c5d378f7cca
let lambda = log.(weather_fits.λ),
    col_names = names(weather_input)
    plotly()
    p = plot()
    for i in 1:size(weather_fits.coefs, 1)
        plot!(lambda, weather_fits.coefs[i, :], label = col_names[i])
    end
    plot!(legend = :outertopright, xlabel = "log(λ)", size = (700, 400))
    gr()
    p
end

# ╔═╡ 40bb385f-1cbd-4555-a8ab-544a67f33595
let lambda = log.(weather_fits.λ)
    p1 = plot(lambda, 100 * weather_fits.pct_dev, ylabel = "% variance explained")
    p2 = plot(lambda, reshape(sum(weather_fits.coefs .!= 0, dims = 1), :),
              ylabel = "non-zero parameters",
              xlabel = "log(λ)")
    plot(p1, p2, layout = (2, 1), legend = false)
end

# ╔═╡ c9ed011c-8d36-4926-9ec4-84be3b4878d7
scatter(weather_input.BER_wind_peak, weather_output)

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
the size ``s`` of the allowed area and find ``\lambda`` by solving the equations.

Interestingly, the loss function ``L(x) = f(x) + \lambda
g(x)`` has exactly the same partial derivative in ``x`` as ``L(x, \lambda)`` and
therefore all critical points of ``L(x)`` have corresponding critical
points of ``L(x,\lambda)``. Because of this, regularization is often formulated
as \"adding a regularization term to the cost function\". For example, given loss
    function of linear regression ``L(\beta) = \frac1n\sum_{i=1}^n(y_i - \beta_0 + \beta_1x_{i1} + \cdots + \beta_p x_{ip})^2`` one can define the
        L1-regularized loss function ``L_\mathrm{L1}(\beta) = L(\beta) + \lambda
\|\beta\|_1`` and choose a value for ``\lambda`` instead of the size ``s`` of the
allowed area.

1. Argue, why choosing ``\lambda = 0`` in the second formulation is equivalent to choosing ``s = \infty`` in the first formulation.
2. Argue, why choosing ``\lambda = \infty`` in the second formulation is equivalent to choosing ``s = 0`` in the first formulation.

#### Exercise 2.
Consider a data set with as many data points as predictors ``n = p``.
Assume ``x_{ii} = 1`` and ``x_{ij} = 0`` for all ``i\neq j`` and arbitrary values
``y_i``. To simplify the problem further we perform regression without an
intercept. We would like to study L1- and L2-regularized multiple linear regression.

1. Write the mean squared error loss once with L1 regularization and once with L2 regularization for this setting and the fomulation of regularization with regularization constant ``\lambda``.
2. Show that in the case of L2 regularization the estimated coefficients take the form ``\hat \beta_j = y_j/(1 + \lambda)``.
3. Show that in the case of L1 regularization the estimated coefficients take the form ``\hat \beta_j = y_j - \lambda/2``, if ``y_j > \lambda/2``, ``\hat \beta_j = y_j + \lambda/2``, if ``y_j < -\lambda/2`` and ``\hat \beta_j = 0`` otherwise.
4. Write a brief summary on how the estimated coefficients ``\hat \beta_j`` are changed relative to the unregularized solution for both kinds of regularization.

## Applied

#### Exercise 1.
Create an artificial dataset with 10 points, 4 predictors ``X_1, X_2, X_3, X_4``
and ``Y = X_1 + \epsilon`` with ``\mathrm{Var}(\epsilon) = 0.1^2``.

1. Find with cross-validation and the lasso the best model.
2. Find with cross-validation and the ridge regression the best model.
3. Which of the two best models has the lowest reducible error?
4. Repeat the above 3 steps on an artificial data set with 10 points and 4 predictors with ``Y = X_1 + .1 * X_2 + .01 * X3 + .001 * X_4 + \epsilon``.
"

# ╔═╡ 912ffa99-9e9c-4be9-9bf4-3e477de17ee4


# ╔═╡ Cell order:
# ╟─78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
# ╠═9e1e8284-a8c1-47a9-83d0-2d8fbd8ce005
# ╟─1009e251-59af-4f1a-9d0a-e96f4b696cad
# ╠═50ac0b07-ffee-40c3-843e-984b3c628282
# ╟─8bd483cc-f490-11eb-38a1-b342dd2551fd
# ╟─58746554-ca5a-4e8e-97e5-587a9c2aa44c
# ╟─f43a82e2-1145-426d-8e0e-5363d1c38ccf
# ╟─ff7cc2bf-2a38-46d2-8d11-529159b08c82
# ╠═4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
# ╟─ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
# ╟─ca394b88-06dc-4188-884a-50d7c180aa33
# ╠═b45a9739-0f81-4c0c-a93b-434a5af91490
# ╟─3d50111b-3a08-4a41-96ce-d77a8e37275d
# ╟─a54e3439-69b8-41c8-bfe0-4575795fb9b8
# ╟─bdbf0dfd-8da5-4e54-89c4-ef4d6b3796ce
# ╠═fe2fe54f-0163-4f5d-9fd1-3d1aa3580875
# ╠═f5057d4a-1103-4728-becc-287d93d682ba
# ╠═bd54bfcd-f682-4b74-8a44-35463d421491
# ╠═596fd0f2-eee0-46ca-a203-e7cbac6f9788
# ╟─8e170a5a-9c46-413e-895d-796e178b69df
# ╠═8e542a48-ed28-4297-b2e8-d6a755a5fdf9
# ╠═552f14fc-06c7-4c2a-9515-e64f28828b70
# ╠═ef701511-db7e-4dc0-8d31-ea14471943ab
# ╟─19bdd76c-4131-422d-983e-1b29cd9edd30
# ╟─13655a50-fbbb-46c7-bdf7-ed5644646966
# ╠═1fa932c1-ce29-40ca-a8dc-e636aa2ecf66
# ╠═470dc7f4-04a9-4253-8125-9112778021eb
# ╠═ecf80b6a-1946-46fd-b1b4-bcbe91848e3c
# ╟─4652a904-5edb-463c-a046-5c5d378f7cca
# ╟─40bb385f-1cbd-4555-a8ab-544a67f33595
# ╠═c9ed011c-8d36-4926-9ec4-84be3b4878d7
# ╟─8262b948-6d54-4348-87d1-4c762c74db30
# ╠═912ffa99-9e9c-4be9-9bf4-3e477de17ee4
# ╟─e04c5e8a-15f8-44a8-845d-60acaf795813
# ╟─150d58e7-0e73-4b36-836c-d81eef531a9c
