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

# ╔═╡ 7aa547f8-25d4-488d-9fc3-f633f7f03f57
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLJ, MLJLinearModels, Plots, LinearAlgebra, Zygote, Flux, Random, DataFrames, Distributions, MLCourse
    import MLCourse: poly
end

# ╔═╡ e03882f9-843e-4552-90b1-c47b6cbba19b
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 9f18f610-79e2-403a-a792-fe2dafd2054a
md"# Gradient Descent

## Gradient Descent for Linear Regression
"

# ╔═╡ fcbac567-36db-40c5-9436-6315c0caa40a
function gradient_descent(f, x, η, T; callback = x -> nothing)
    for t in 1:T
		∇f = gradient(f, x)[1] # compute the gradient
        x .-= η * ∇f # update parameters in direction of -∇f
        callback(x) # the callback will be used to save intermediate values
    end
    x
end

# ╔═╡ 49e744dc-33b7-455a-9cf7-4e8e12f11152
X, y = make_regression(40, 1, rng = 16); # make_regression is an MLJ data generator

# ╔═╡ dcc76840-6552-4a86-8268-1291ab8ea0d3
begin
    lin_reg_loss(β₀, β₁, x, y) = mean((y .- β₀  .- β₁ * x).^2)
    lin_reg_loss(β₀, β₁) = lin_reg_loss(β₀, β₁, X.x1, y)
    lin_reg_loss(β) = lin_reg_loss(β[1], β[2])
end

# ╔═╡ 3ca1e8f3-d69f-454f-b917-4bbe2dcfce01
md"β₀⁽⁰⁾ = $(@bind b0 Slider(-3.:.1:3., show_value = true, default = 0.))

β₁⁽⁰⁾ = $(@bind b1 Slider(-3.:.1:3., show_value = true, default = -3.))

η = $(@bind η Slider(.01:.01:2, default = .05, show_value = true))

step t = $(@bind t Slider(0:100, show_value = true))
"

# ╔═╡ b1fc14bb-1fd2-4739-a761-7a605fd4559b
begin
    params = [b0, b1] # initial parameters (defined by the sliders above)
	lin_reg_path = [copy(params)] # store the initial value in a path variable
    gradient_descent(lin_reg_loss, params, η, 100, # η is defined by the slider
                     callback = x -> push!(lin_reg_path, copy(x)))
end

# ╔═╡ bbc0b514-4789-44d1-8d90-9fc325d9ad6b
let
    p1 = scatter(X.x1, y, xlim = (-2.5, 2.5), ylim = (-2, 2),
                 legend = :bottomright,
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> lin_reg_path[t+1]' * [1, x], c = :red, w = 2, label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3,
                 lin_reg_loss, cbar = false, aspect_ratio = 1)
    scatter!(first.(lin_reg_path), last.(lin_reg_path), markersize = 1, c = :red, label = nothing)
    scatter!([lin_reg_path[t+1][1]], [lin_reg_path[t+1][2]], label = nothing,
             markersize = 4, c = :red, xlabel = "β₀", ylabel = "β₁")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 07f3f800-6e06-416c-bfdd-e4ece8565a42
md"
!!! tip
	Use the sliders above to see how gradient descent proceeds.
"

# ╔═╡ fd0dca4d-d2db-4c6a-b218-cdacc49d2946
md"In the following we want to keep track of the learning progress in many different examples. For this we define a `Tracker` object that does exactly the same as what we did for the `lin_reg_path` in the cell above."

# ╔═╡ 2fe2049c-e9db-4a1e-a927-f147f137b3c4
# In this cell we define a callable object.
# This is somewhat advanced, you do not need to understand everything
begin 
	struct Tracker{T} # structure definition with parametric type
		path::T
	end
	Tracker(x::T) where T = Tracker{Vector{T}}([copy(x)]) # constructor
	(t::Tracker)(x) = push!(t.path, copy(x))        # making the object callable
end

# ╔═╡ 396d2cc3-b7de-4c1b-ad56-1bed63852b18
md"With this new `Tracker` we can write the gradient descent procedure above as
```julia
    params = [b0, b1] # initial parameters
	tracker = Tracker(params) # initialize the tracker
    gradient_descent(lin_reg_loss, params, η, 100, callback = tracker)
    lin_reg_path = tracker.path
```
"

# ╔═╡ eb9e2b3f-dc4b-49a2-a611-b45f2918adcf
md"## Gradient Descent for Logistic Regression"

# ╔═╡ 402bb576-8945-403e-a9a6-fd5bfb8016bc
begin
    X2, y2 = make_regression(50, 1, binary = true, rng = 161)
	y2 = coerce(y2, MLJ.Continuous) .- 1
    σ(x) = 1/(1 + exp(-x))
    log_reg_loss(β) = log_reg_loss(β[1], β[2])
    function log_reg_loss(β₀, β₁)
        p = σ.(β₀  .+ β₁ * X2.x1) # probability of positive class for all inputs
        -mean(@. y2 * log(p) + (1-y2) * log(1 - p)) # negative log-likelihood
    end
end

# ╔═╡ cd5f079f-a06d-4d55-9666-e2b05ddf8989
md"β₀⁽⁰⁾ = $(@bind b02 Slider(-3.:.1:3., show_value = true, default = -2.))

β₁⁽⁰⁾ = $(@bind b12 Slider(-3.:.1:3., show_value = true, default = 2.))

η = $(@bind η2 Slider(.1:.1:2, show_value = true))

step t = $(@bind t2 Slider(0:100, show_value = true))
"

# ╔═╡ 93699313-5e42-48a7-abc6-ad146e5bdcdd
begin
    params2 = [b02, b12]
    tracker2 = Tracker(params2)
    gradient_descent(log_reg_loss, params2, η2, 100, callback = tracker2)
end

# ╔═╡ 33bdc912-e46a-4310-9184-733be7871768
let path = tracker2.path
    p1 = scatter(X2.x1, y2, xlim = (-3.5, 3.5), ylim = (-.1, 1.1),
                 legend = :bottomleft, marker_style = :vline,
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> σ(path[t2+1]' * [1, x]), c = :red, w = 2, label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, log_reg_loss, cbar = false, aspect_ratio = 1)
    scatter!(first.(path), last.(path), markersize = 1, c = :red, label = nothing)
    scatter!([path[t2+1][1]], [path[t2+1][2]], label = nothing,
             markersize = 4, c = :red, xlabel = "β₀", ylabel = "β₁")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 979feb42-c328-44b5-b519-8e8cda474140
md"## Gradient Descent to Minimize a Complicated Function"

# ╔═╡ c448238c-f712-4af3-aedb-cec3a2c1a73e
begin
	Random.seed!(5)
    f(x, θ) = θ[1] * sin.(θ[2] * x .+ θ[3]) .+ θ[4] * sin.(θ[5] * x .+ θ[6])
    f(x) = 0.3 * sin(2x) + .7 * sin(3.5x + 1)
    x3 = randn(50)
    y3 = f.(x3)
end

# ╔═╡ 6e38a3f6-a592-4dc1-a6c4-d0f0050c9399
mse(θ) = mean((y3 .- f(x3, θ)).^2)

# ╔═╡ 4b3ac87f-39f6-4e6c-b7b3-536925e9b112
md"step t = $(@bind t3 Slider(1:10^4, show_value = true))

seed = $(@bind special_seed Slider(1:10, default = 3, show_value = true))"

# ╔═╡ 9a61332f-cdc2-4129-b7b5-5ab54ba387a3
begin
	Random.seed!(special_seed) # defined by the slider below
    params3 = .1 * randn(6)
    tracker3 = Tracker(params3)
    gradient_descent(mse, params3, .1, 10^4, callback = tracker3)
end

# ╔═╡ 01221937-b6d9-4dac-be90-1b8a1c5e9d87
let path = tracker3.path
    p1 = plot(f, label = "orginal function", xlabel = "x", ylabel = "y",
              ylim = (-1.3, 1.2), xlim = (-3, 3))
    plot!(x -> f(x, path[t3]), label = "current fit")
	scatter!(x3, y3, label = "training data")
    th = round.(path[t3], digits = 2)
    annotate!([(0, -1.1, text("f̂(x) = $(th[1]) * sin($(th[2])x + $(th[3])) + $(th[4]) * sin($(th[5])x + $(th[6]))", pointsize = 7))])
    losses = mse.(path)
    p2 = plot(0:10^4, losses, label = "learning curve", c = :black, yscale = :log10)
    scatter!([t3], [losses[t3]], label = "current loss", xlabel = "t", ylabel = "loss")
    p3 = contour(-4:.1:4, -4:.1:4, (x2, x5) -> mse([path[t3][1]; x2; path[t3][3:4]; x5; path[t3][6]]), xlabel = "θ₂", ylabel = "θ₅", title = "loss")
	scatter!([path[t3][2]], [path[t3][5]])
    plot(p1, plot(p2, p3, layout = (2, 1)), layout = (1, 2), size = (700, 400))
end

# ╔═╡ f5e27275-8751-4e80-9888-c3d22d8e80e3
md"*Optional* If you want to know more about the magic of automatic differentiation and how julia computes derivates of (almost) arbitrary code: have a look at this [didactic introduction to automatic differentiation](https://github.com/MikeInnes/diff-zoo) or this [video with code examples in python](https://www.youtube.com/watch?v=wG_nF1awSSY&t)."

# ╔═╡ ca1301fd-8978-44d6-bfae-e912f939d7a8
md"# Stochastic Gradient Descent

In each step of stochastic gradient descent (SGD) the gradient is computed only on a (stochastically selected) subset of the data. As long as the selected subset of the data is somewhat representative of the full dataset the gradient will point more or less in the same direction as the full gradient computed on the full dataset. The advantage of computing the gradient only on a subset of the data is that it takes much less time to compute the gradient on a small dataset than on a large dataset. In the figure below you see the niveau lines of the full loss in black and the niveau lines of the loss computed on a green, purple, yellow and a blue subset. The gradient computed on, say, the yellow subset is always perpendicular to the yellow niveau lines but it may not be perpendicular to the niveau lines of the full loss. When selecting in each step another subset we observe a jittered trajectory around the full gradient descent trajectory.
"

# ╔═╡ 75528011-05d9-47dc-a37b-e6bb6be52c25
md"η = $(@bind η_st Slider(.01:.01:.2, default = .1, show_value = true))

step t = $(@bind t5 Slider(0:100, show_value = true))
"

# ╔═╡ e821fb15-0bd3-4fa7-93ea-692bf05097b5
begin
    seed = 123
    Random.seed!(seed)
    function lin_reg_loss_st(β)
        # With Zygote.ignore the batch selection is not considered in the gradient
        # computation, i.e. the gradient gets computed as if xb and yb were the
        # full data.
        xb, yb = Zygote.ignore() do
            batch = rand(1:4)                 # select batch
            idxs = (batch-1)*10 + 1:batch*10  # compute indices for this batch
            X.x1[idxs], y[idxs]
        end
        lin_reg_loss(β[1], β[2], xb, yb)
    end
    params_st = [b0, b1]
    tracker_st = Tracker(params_st)
    gradient_descent(lin_reg_loss_st, params_st, η_st, 100, callback = tracker_st)
end;

# ╔═╡ 7541c203-f0dc-4445-9d2a-4cf16b7e912a
let path = tracker_st.path,
    lin_reg_loss_b(i) = (β₀, β₁) -> lin_reg_loss(β₀, β₁, X.x1[(i-1)*10+1:i*10],
                                                 y[(i-1)*10+1:i*10])
    Random.seed!(seed)
    batches = rand(1:4, 101)
    b = batches[t5+1]
    colors = [:green, :blue, :orange, :purple]
    ma = fill(.2, 40)
    ma[(b-1)*10 + 1:b*10] .= 1
    p1 = scatter(X.x1, y, xlim = (-2.5, 2.5), ylim = (-2, 2),
                 legend = false, ma = ma,
                 c = vcat([fill(c, 10)
                           for c in colors]...),
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> path[t5+1]' * [1, x], c = :red, w = 2,
          label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, lin_reg_loss, cbar = false,
                 xlabel = "β₀", ylabel = "β₁",
                 linestyle = :dash, c = :black, aspect_ratio = 1)
    for i in 1:4
        contour!(-3:.1:3, -3:.1:3, lin_reg_loss_b(i), cbar = false, w = 2,
                 linestyle = :dash, c = colors[i], alpha = b == i ? 1 : .2)
    end
    plot!(first.(lin_reg_path), last.(lin_reg_path),
          c = :black, w = 3, label = "GD")
    plot!(first.(path), last.(path),
          c = :red, w = 1.5, label = "SGD")
    ps = path[t5+1]
    scatter!([ps[1]], [ps[2]], label = nothing, c = :red)
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 913cf5ee-ca1e-4063-bd34-6cccd0cc548b
md"# Improved Versions of (S)GD

There are many tricks to improve over standard (stochastic) gradient descent.
One popular idea is to use [momentum](https://distill.pub/2017/momentum/).
We do not discuss these ideas further here, but you should know that `ADAM()` and variants like `ADAMW()` are particularly popular (and successful) improvements of standard (S)GD. These methods usually require no (or very little) tuning of the learning rate."

# ╔═╡ 2739fb52-fb1b-46d6-9708-e24bfdc459e2
function advanced_gradient_descent(f, x, optimizer, T; callback = x -> nothing)
    for t in 1:T
        ∇f = gradient(f, x)[1] # compute ∇f
		Flux.update!(optimizer, x, ∇f) # apply the changes to x
        callback(x) # the callback will be used to save intermediate values
    end
    x
end

# ╔═╡ eb289254-7167-4183-a4d0-52f68be66b04
begin
	Random.seed!(1234)
    params4 = .1 * randn(6)
	tracker4 = Tracker(params4)
	opt = ADAMW()
	advanced_gradient_descent(mse, params4, opt, 10^4, callback = tracker4)
end

# ╔═╡ 7b57c3f0-ef5a-4dd7-946f-72c8dde2ae8f
md"t = $(@bind t6 Slider(1:10^4, show_value = true))"

# ╔═╡ 166472c5-c0f4-4261-a476-4c9b0f82abd6
let special_path = tracker3.path, special_path2 = tracker4.path
    p1 = plot(f, label = "orginal function", xlabel = "x", ylabel = "y",
              ylim = (-1.3, 1.2))
    plot!(x -> f(x, special_path2[t6]), label = "current fit")
    th = round.(special_path2[t6], digits = 2)
    annotate!([(0, -1.1, text("f̂(x) = $(th[1]) * sin($(th[2])x + $(th[3])) + $(th[4]) * sin($(th[5])x + $(th[6]))", pointsize = 7))])
    losses2 = mse.(special_path2)
    losses = mse.(special_path)
    p2 = plot(0:10^4, losses2, label = "learning curve", c = :black, yscale = :log10)
    plot!(0:10^4, losses, label = "GD learning curve", c = :black, linestyle = :dot)
    scatter!([t6], [losses2[t6]], label = "current loss", xlabel = "t", ylabel = "loss", legend = :bottomleft)
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ 08a9418f-786e-4992-b1a5-04cf9060f8fe
md"# Early Stopping as Regularization

In early stopping we start (stochastic) gradient descent with smart parameter values, keep track of training and validation loss throughout gradient descent and stop gradient descent, when the validation loss reached the smallest value. The effect of early stopping is similar to regularization: the parameter values found at early stopping have usually a smaller norm than the parameter values with the lowest training error."

# ╔═╡ dc57d700-2a82-4ab0-9bd2-6ce622cb0fa5
begin
    g(x) = .3 * sin(10x) + .7x
    function regression_data_generator(; n, seed = 3, rng = MersenneTwister(seed))
        x = range(0, 1, length = n)
        DataFrame(x = x, y = g.(x) .+ .1*randn(rng, n))
    end
    regression_data = regression_data_generator(n = 10, seed = 10)
	regression_valid = regression_data_generator(n = 50, seed = 123)
end;

# ╔═╡ c72dc506-fb1d-42ee-83cf-cc49753ecd4f
begin
    h_training = Array(select(poly(regression_data, 12), Not(:y)))
    h_valid = Array(select(poly(regression_valid, 12), Not(:y)))
    poly_regression_loss(θ, x, y) = mean((y .- θ[1] .- x * θ[2:end]).^2)
    target = regression_data.y
    poly_regression_loss(θ) = poly_regression_loss(θ, h_training, target)
    poly_params = 1e-3 * randn(13)
    tracker5 = Tracker(poly_params)
	poly_opt = ADAMW()
	advanced_gradient_descent(poly_regression_loss, poly_params, poly_opt, 3*10^4, callback = tracker5)
end

# ╔═╡ 075f35cf-4271-4676-b9f3-e2bcf610c2d1
md"step t = $(@bind t4 Slider(0:10^3:3*10^4, show_value = true))"

# ╔═╡ 0d431c00-9eef-4ce4-9542-9571728d1501
let poly_path = tracker5.path
    p1 = scatter(regression_data.x, regression_data.y,
                 label = "training data", ylims = (-.1, 1.1))
    plot!(g, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    θ = poly_path[t4 + 1]
    pred =  Array(poly((x = grid,), 12)) * θ[2:end] .+ θ[1]
    plot!(grid, pred, xlabel = "input", ylabel = "output",
          label = "fit", w = 3, c = :red, legend = :topleft)
    losses = poly_regression_loss.(poly_path, Ref(h_training), Ref(regression_data.y))
    losses_v = poly_regression_loss.(poly_path, Ref(h_valid), Ref(regression_valid.y))
    p2 = plot(0:length(poly_path)-1, losses, yscale = :log10,
              c = :blue, label = "training loss")
    scatter!([t4], [losses[t4 + 1]], c = :blue, label = nothing)
    plot!(0:length(poly_path)-1, losses_v, c = :red, label = "validation loss")
    scatter!([t4], [losses_v[t4 + 1]], c = :red, label = nothing,
		     xlabel = "t", ylabel = "loss")
    vmin, idx = findmin(losses_v)
    vline!([idx], c = :red, linestyle = :dash, label = nothing)
    hline!([vmin], c = :red, linestyle = :dash, label = nothing)
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end



# ╔═╡ e0cc188c-9c8f-47f1-b1fe-afc2a578973d
md"# Exercise

Assume the noise in a linear regression setting comes from a Laplace
distribution, i.e. the conditional probability density of the response is given by
``p(Y = y | X = x, \beta) = \frac1{2s}\exp(-\frac{|y - x^T\beta|}{s})``.
For simplicity we assume throughout this exercise that the intercept ``\beta_0 = 0``
and does not need to be fitted.

(a) Generate a training set of 100 points with the following data generator.
Notice that the noise follows a Laplace distribution instead of a normal distribution.
For once, we do not use a `DataFrame` here but represent the input explicitly as a matrix and the full dataset as a `NamedTuple`. If you run `data = data_generator()`, you can access the input matrix as `data.x` and the output vector as `data.y`.
```julia
function data_generator(; n = 100, β = [1., 2., 3.])
    x = randn(n, 3)
    y = x * β .+ rand(Laplace(0, 0.3), n)
    (x = x, y = y)
end
```

(b) Calculate with paper and pencil the negative log-likelihood loss. Apply transformations to the negative log-likelihood function to obtain a good loss function for gradient descent based on the practical considerations in the slides.
The solution you should find is
```math
\tilde L = \frac1{n} \sum_{i=1}^n |y_i - x_i^T\beta|.
```

(c) Code a function to compute the loss on the training set for a given
parameter vector. *Hint:* use matrix multiplication, e.g. `data.x * β`.

(d) Perform gradient descent on the training set. Plot the learning curve to see
whether gradient descent has converged. If you see large fluctuations at the end
of training, decrease the learning rate. If the learning curve is not flat at
the end, increase the maximal number of steps. To see well the loss towards the end of gradient descent it is advisable to use log-scale for the y-axis (`yscale = :log10`).

(e) Estimate the coefficients with the standard linear regression.
Hint: do not forget that we fit without intercept (use `fit_intercept = false` in the `LinearRegressor`).

(f) Compare which method (d) or (e) found parameters closer to the one of our data generating process `[1, 2, 3]` and explain your finding.
"

# ╔═╡ cb9f858a-f60a-11eb-3f0e-a9b68cf33921
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 8459f86e-bce7-4839-9c51-57335ac6353c
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─e03882f9-843e-4552-90b1-c47b6cbba19b
# ╠═7aa547f8-25d4-488d-9fc3-f633f7f03f57
# ╟─9f18f610-79e2-403a-a792-fe2dafd2054a
# ╠═fcbac567-36db-40c5-9436-6315c0caa40a
# ╠═49e744dc-33b7-455a-9cf7-4e8e12f11152
# ╠═dcc76840-6552-4a86-8268-1291ab8ea0d3
# ╟─3ca1e8f3-d69f-454f-b917-4bbe2dcfce01
# ╠═b1fc14bb-1fd2-4739-a761-7a605fd4559b
# ╟─bbc0b514-4789-44d1-8d90-9fc325d9ad6b
# ╟─07f3f800-6e06-416c-bfdd-e4ece8565a42
# ╟─fd0dca4d-d2db-4c6a-b218-cdacc49d2946
# ╠═2fe2049c-e9db-4a1e-a927-f147f137b3c4
# ╟─396d2cc3-b7de-4c1b-ad56-1bed63852b18
# ╟─eb9e2b3f-dc4b-49a2-a611-b45f2918adcf
# ╠═402bb576-8945-403e-a9a6-fd5bfb8016bc
# ╟─cd5f079f-a06d-4d55-9666-e2b05ddf8989
# ╠═93699313-5e42-48a7-abc6-ad146e5bdcdd
# ╟─33bdc912-e46a-4310-9184-733be7871768
# ╟─979feb42-c328-44b5-b519-8e8cda474140
# ╠═c448238c-f712-4af3-aedb-cec3a2c1a73e
# ╠═6e38a3f6-a592-4dc1-a6c4-d0f0050c9399
# ╠═9a61332f-cdc2-4129-b7b5-5ab54ba387a3
# ╟─4b3ac87f-39f6-4e6c-b7b3-536925e9b112
# ╟─01221937-b6d9-4dac-be90-1b8a1c5e9d87
# ╟─f5e27275-8751-4e80-9888-c3d22d8e80e3
# ╟─ca1301fd-8978-44d6-bfae-e912f939d7a8
# ╠═e821fb15-0bd3-4fa7-93ea-692bf05097b5
# ╟─75528011-05d9-47dc-a37b-e6bb6be52c25
# ╟─7541c203-f0dc-4445-9d2a-4cf16b7e912a
# ╟─913cf5ee-ca1e-4063-bd34-6cccd0cc548b
# ╠═2739fb52-fb1b-46d6-9708-e24bfdc459e2
# ╠═eb289254-7167-4183-a4d0-52f68be66b04
# ╟─7b57c3f0-ef5a-4dd7-946f-72c8dde2ae8f
# ╟─166472c5-c0f4-4261-a476-4c9b0f82abd6
# ╟─08a9418f-786e-4992-b1a5-04cf9060f8fe
# ╠═dc57d700-2a82-4ab0-9bd2-6ce622cb0fa5
# ╠═c72dc506-fb1d-42ee-83cf-cc49753ecd4f
# ╟─075f35cf-4271-4676-b9f3-e2bcf610c2d1
# ╟─0d431c00-9eef-4ce4-9542-9571728d1501
# ╟─e0cc188c-9c8f-47f1-b1fe-afc2a578973d
# ╟─cb9f858a-f60a-11eb-3f0e-a9b68cf33921
# ╟─8459f86e-bce7-4839-9c51-57335ac6353c
