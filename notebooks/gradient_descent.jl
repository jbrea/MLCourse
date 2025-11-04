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

# ╔═╡ 7aa547f8-25d4-488d-9fc3-f633f7f03f57
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, Flux, Zygote
import PlutoPlotly as PP
const M = MLCourse.JlMod
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ e03882f9-843e-4552-90b1-c47b6cbba19b
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 06479d01-5c03-4b26-9d1c-2ee71032c3ae
md"The goal of this week is to
1. understand gradient descent.
2. understand stochatic gradient descent.
3. know improved versions of (stochastic) gradient descent.
4. understand, why early stopping gradient descent has a similar effect as regularization.
5. Understand how we can solve the XOR-Problem with feature engineering and gradient descent
"

# ╔═╡ 9f18f610-79e2-403a-a792-fe2dafd2054a
md"""# 1. Gradient Descent

Gradient descent is beautifully simple. Given a loss function and an initial parameter value, it moves the parameter values iteratively in the opposite direction of the vector of partial derivatives. Thanks to powerful automatic differentiation packages, we do not need to calculate the partial derivatives ourselves, but they are computed for basically any loss function.

## Gradient Descent for Linear Regression

In this section we write the `gradient_descent` function. To avoid having to compute partial derivatives manually, we use $(mlstring(md"the `gradient` funtion defined in the `Zygote` package.", md"`torch` (PyTorch). To use `torch`, all data and parameters need to be `torch.tensors`. The gradient computation is somewhat hidden in `torch`: whenever the `backward()` function is called on a `torch.tensor`, the partial derivatives for all tensors with argument `requires_grad = True` are computed."))

For later examples we only need to write a new loss function.
Here we define the loss function for linear regression and run it to illustrate gradient descent. Note that, usually, linear regression is not solved with gradient descent, because there are other (specialized) methods that find the minimum of the loss function more efficiently.
"""

# ╔═╡ fcbac567-36db-40c5-9436-6315c0caa40a
mlcode(
"""
###
### Gradient Descent
###

using Zygote
import Statistics: mean

function gradient_descent(loss, x; η, T, callback = x -> nothing)
    for t in 1:T
		∇loss = gradient(loss, x)[1] # compute the gradient
        x .-= η * ∇loss # update parameters in direction of -∇loss
        callback(x) # the callback can be used to save intermediate values
    end
    x
end

###
### Linear regression loss function
###

lin_reg_loss(β₀, β₁, x, y) = mean((y .- β₀  .- β₁ * x).^2)
function lin_reg_loss_function(X, y) # returns loss for given dataset
    β -> lin_reg_loss(β[1], β[2], X, y)
end

###
### Running gradient descent for linear regression
###

X = randn(40)
y = .1 .- .4 * X .+ .1 * randn(40)
loss = lin_reg_loss_function(X, y) # loss for this dataset

params = [.1, .2] # initial parameters
gradient_descent(loss, params, η = .1, T = 100)
"""
,
"""
import numpy as np
import torch

###
### Gradient Descent
###

def gradient_descent(loss, x, eta, T):
	for t in range(T):
		loss(x).backward() # compute gradient
		x.data -= eta * x.grad.data # update parameters
		x.grad.data.zero_() # reset gradient to zero for next iteration
	return x

###
### Linear Regression Loss Function
###

def lin_reg_loss(X, y):
	def loss(beta): 
		return torch.mean((y - beta[0] - beta[1]*X)**2)
	return loss

###
### Running Gradient Descent of Linear Regression
###

X = torch.tensor(np.random.randn(40))                   # input as a torch tensor
y = .1 - .4 * X + .1*torch.tensor(np.random.randn(40))  # output
loss = lin_reg_loss(X, y)                               # loss function

# Initial parameter values. As we want to compute the partial derivatives with respect to these parameters, we set `require_grad = True`.
beta = torch.tensor(np.array([.1, .2]), requires_grad = True)

gradient_descent(loss, beta, .1, 100)
"""
,
showoutput = false,
cache = false
)

# ╔═╡ e65c80e2-c951-4ff0-aeff-2cdddac26479
md"Below you can change the initial conditions and the learning rate of gradient descent. Note that gradient descent becomes unstable, when the learning rate is too large and it does not converge to the minimum, when the learning rate is too small.

The level lines in the **figure on the right** illustrate the loss function. The black dots are the positions in parameter space along which gradient descent traveled from initial point to final point. With the `step t` slider you can move along the gradient descent path."

# ╔═╡ 3ca1e8f3-d69f-454f-b917-4bbe2dcfce01
md"β₀⁽⁰⁾ = $(@bind b0 Slider(-3.:.1:3., show_value = true, default = 0.))

β₁⁽⁰⁾ = $(@bind b1 Slider(-3.:.1:3., show_value = true, default = -3.))

η = $(@bind η Slider(.01:.01:2, default = .05, show_value = true))

step t = $(@bind t Slider(0:100, show_value = true))
"

# ╔═╡ 2fe2049c-e9db-4a1e-a927-f147f137b3c4
begin
struct Tracker{T} # structure definition with parametric type
    path::T
end
Tracker(x::T) where T = Tracker{Vector{T}}([copy(x)]) # constructor
(t::Tracker)(x) = push!(t.path, copy(x))              # making object callable
end

# ╔═╡ b1fc14bb-1fd2-4739-a761-7a605fd4559b
begin
    params = [b0, b1] # initial parameters (defined by the sliders above)
    tracker = Tracker(params)
    M.gradient_descent(M.loss, params; η, T = 100, # η is defined by the slider
                       callback = tracker)
    lin_reg_path = tracker.path
end;

# ╔═╡ bbc0b514-4789-44d1-8d90-9fc325d9ad6b
let X = M.X, y = M.y,
    lin_reg_loss = (x, y) -> M.loss([x, y])
    p1 = scatter(X, y, xlim = (-2.5, 2.5), ylim = (-2, 2),
                 legend = :bottomright,
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> lin_reg_path[t+1]' * [1, x], c = :red, w = 2, label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, title = "loss function",
                 lin_reg_loss, cbar = false, aspect_ratio = 1)
    scatter!(first.(lin_reg_path), last.(lin_reg_path), markersize = 1, c = :red, label = nothing)
    scatter!([lin_reg_path[t+1][1]], [lin_reg_path[t+1][2]], label = nothing,
             markersize = 4, c = :red, xlabel = "β₀", ylabel = "β₁")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end


# ╔═╡ eb9e2b3f-dc4b-49a2-a611-b45f2918adcf
md"## Gradient Descent for Logistic Regression

Here we define the negative log-likelihood loss function for logistic regression.
As soon as we have this, we can use again gradient descent to find the solution of logistic regression. Note, that also for logistic regression, there are alternative specialized methods that find the optimum more efficiently usually.
"

# ╔═╡ 402bb576-8945-403e-a9a6-fd5bfb8016bc
mlcode(
"""
###
### Logistic regression loss function
###

σ(x) = 1/(1 + exp(-x))
function log_reg_loss(β₀, β₁, x, y)
    p = σ.(β₀  .+ β₁ * x) # probability of positive class for all inputs
    -mean(@. y * log(p) + (1-y) * log(1 - p)) # negative log-likelihood
end
function log_reg_loss_function(X, y)
    β -> log_reg_loss(β[1], β[2], X, y)
end

###
### Running gradient descent for logistic regression
###

X2 = randn(500)
y2 = σ.(-.1 .+ 1.7*X2) .> rand(500)
logloss = log_reg_loss_function(X2, y2)
params = [.1, .2] # initial parameters
gradient_descent(logloss, params, η = .1, T = 100)
"""
,
"""
###
### Logistic regression loss function
###

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def log_reg_loss(X, y):
	def loss(beta):
		p = sigmoid(beta[0] + beta[1]*X)
 		# negative log-likelihood. Note ~y negates the boolean tensor y
		return -torch.mean(y * torch.log(p) + ~y * torch.log(1 - p))
	return loss

###
### Running gradient descent for Logistic regression
###

X2 = torch.tensor(np.random.randn(500))
y2 = sigmoid(-.1 + 1.7*X2) > torch.tensor(np.random.rand(500))
loss2 = log_reg_loss(X2, y2)

beta2 = torch.tensor(np.array([.1, .2]), requires_grad = True)

gradient_descent(loss2, beta2, .1, 100)
"""
,
showoutput = false,
cache = false
)


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
    M.gradient_descent(M.logloss, params2, η = η2, T = 100, callback = tracker2)
end;

# ╔═╡ 33bdc912-e46a-4310-9184-733be7871768
let path = tracker2.path, X2 = M.X2, y2 = M.y2
    log_reg_loss = (x, y) -> M.logloss([x, y])
    p1 = scatter(X2, y2, xlim = (-3.5, 3.5), ylim = (-.1, 1.1),
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
md"## Gradient Descent to Minimize a Complicated Function

Although there are specialized methods to find the optima for many specific settings, the nice thing about gradient descent is its general applicability. In this section we use it to minimize a complicated function.

We construct the complicated function in the following way:
1. We sample some input points ``X`` from a normal distribution and use a noise-free conditional data generator with ``Y = 0.3 * \sin(2X) + 0.7 * \sin(3.5X + 1)``.
2. We define the loss function
```math
\ell(\theta) = \frac1n\sum_{i=1}^n\big(y_i - \theta_1\sin(\theta_2x_i + \theta_3) + \theta_4\sin(\theta_5x_i + \theta_6)\big)^2
```

We know that the optimal value is found when the parameters match the conditional data generator, for example when ``\theta_1 = 0.3, \theta_2 = 2, \theta_3 = 0, \theta_4 = 0.7, \theta_5 = 3.5, \theta_6 = 1``.
But the loss function has some symmetries. Another solution would be ``\theta_1 = 0.7, \theta_2 = 3.5, \theta_3 = 1, \theta_4 = 0.3, \theta_5 = 2, \theta_6 = 0``, where the first term of the conditional data generator is matched by the second term of the fitted function. And there are even more solutions, because ``a\sin(b) = -a\sin(-b)``.

Which solution does gradient descent find?
"

# ╔═╡ 4b3ac87f-39f6-4e6c-b7b3-536925e9b112
md"step t = $(@bind t3 Slider(1:10^4, show_value = true))

seed = $(@bind special_seed Slider(1:20, default = 3, show_value = true))"

# ╔═╡ cecca28a-23e7-467f-b1b3-5d6b292c298b
md"The figure on the **top right** shows the learning curve, i.e. the loss versus gradient descent steps. Visualizing the loss itself is difficult, because it depends on 6 parameters. In the figure on the **bottom right** the loss is shown as a function of ``\theta_2`` and ``\theta_5``. Note, that in contrast to most other loss figures you have seen so far, the level lines change during training, because the level lines depend on the values of the other parameters.

The important observation to make with this loss function is, that it is non-convex, i.e. there are multiple minima. This is an important qualitative difference to the loss functions of linear or logistic regression, where the loss function is convex and a single minimum exists. Here, in the case of multiple minima, the solution of gradient descent depends on the initialization.
"

# ╔═╡ f5e27275-8751-4e80-9888-c3d22d8e80e3
md"*Optional* If you want to know more about the magic of automatic differentiation and how julia computes derivates of (almost) arbitrary code: have a look at this [didactic introduction to automatic differentiation](https://github.com/MikeInnes/diff-zoo) or this [video with code examples in python](https://www.youtube.com/watch?v=wG_nF1awSSY&t)."

# ╔═╡ c448238c-f712-4af3-aedb-cec3a2c1a73e
begin
	Random.seed!(5)
    f(x, θ) = θ[1] * sin.(θ[2] * x .+ θ[3]) .+ θ[4] * sin.(θ[5] * x .+ θ[6])
    f(x) = 0.3 * sin(2x) + .7 * sin(3.5x + 1)
    x3 = randn(50)
    y3 = f.(x3)
end;

# ╔═╡ 6e38a3f6-a592-4dc1-a6c4-d0f0050c9399
mse(θ) = mean((y3 .- f(x3, θ)).^2);

# ╔═╡ 9a61332f-cdc2-4129-b7b5-5ab54ba387a3
begin
	Random.seed!(special_seed) # defined by the slider below
    params3 = .1 * randn(6)
    tracker3 = Tracker(params3)
    M.gradient_descent(mse, params3, η = .1, T = 10^4, callback = tracker3)
end;

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
	scatter!([path[t3][2]], [path[t3][5]], label = "current loss")
    plot(p1, plot(p2, p3, layout = (2, 1)), layout = (1, 2), size = (700, 400))
end

# ╔═╡ ca1301fd-8978-44d6-bfae-e912f939d7a8
md"# 2. Stochastic Gradient Descent

In each step of stochastic gradient descent (SGD) the gradient is computed only on a (stochastically selected) subset of the data. As long as the selected subset of the data is somewhat representative of the full dataset the gradient will point more or less in the same direction as the full gradient computed on the full dataset. The advantage of computing the gradient only on a subset of the data is that it takes much less time to compute the gradient on a small dataset than on a large dataset. In the figure below you see the level lines of the full loss in black and the level lines of the loss computed on a green, purple, yellow and a blue subset. The gradient computed on, say, the yellow subset is always perpendicular to the yellow level lines but it may not be perpendicular to the level lines of the full loss. When selecting in each step another subset we observe a jittered trajectory around the full gradient descent trajectory.
"

# ╔═╡ e821fb15-0bd3-4fa7-93ea-692bf05097b5
mlcode(
"""
function lin_reg_loss_stochastic(X, y)
    # With Zygote.ignore the batch selection is not considered in the gradient
    # computation, i.e. the gradient gets computed as if xb and yb were the
    # full data.
    function(β)
        xb, yb = Zygote.ignore() do
            batch = rand(1:4)                 # select batch
            idxs = (batch-1)*10 + 1:batch*10  # compute indices for this batch
            X[idxs], y[idxs]
        end
        lin_reg_loss(β[1], β[2], xb, yb)
    end
end
"""
,
"""
def lin_reg_loss_stochastic(X, y):
	def loss(beta):
		batch = np.random.randint(1, 4)            # select batch
		idxs = np.arange((batch-1)*10, batch*10)   # compute indices for this batch
		return torch.mean((y[idxs] - beta[0] - beta[1]*X[idxs])**2)
	return loss

loss = lin_reg_loss_stochastic(X, y)              # loss function
beta = torch.tensor(np.array([.1, .2]), requires_grad = True)

gradient_descent(loss, beta, .1, 100)
"""
,
showoutput = false,
cache = false
)

# ╔═╡ 75528011-05d9-47dc-a37b-e6bb6be52c25
md"η = $(@bind η_st Slider(.01:.01:.2, default = .05, show_value = true))

step t = $(@bind t5 Slider(0:100, show_value = true))
"

# ╔═╡ fa706e4b-eaa9-4be2-b4f5-1906931c8ef6
begin
    seed = 123
    Random.seed!(seed)
    params_st = [b0, b1]
    tracker_st = Tracker(params_st)
    M.gradient_descent(M.lin_reg_loss_stochastic(M.X, M.y), params_st, η = η_st, T = 100, callback = tracker_st)
end;

# ╔═╡ 7541c203-f0dc-4445-9d2a-4cf16b7e912a
let path = tracker_st.path,
    X = M.X, y = M.y
    lin_reg_loss_b(i) = (β₀, β₁) -> M.lin_reg_loss(β₀, β₁, X[(i-1)*10+1:i*10],
                                                   y[(i-1)*10+1:i*10])
    Random.seed!(seed)
    batches = rand(1:4, 101)
    b = batches[t5+1]
    colors = [:green, :blue, :orange, :purple]
    ma = fill(.2, 40)
    ma[(b-1)*10 + 1:b*10] .= 1
    p1 = scatter(X, y, xlim = (-2.5, 2.5), ylim = (-2, 2),
                 legend = false, ma = ma,
                 c = vcat([fill(c, 10)
                           for c in colors]...),
                 xlabel = "x", ylabel = "y", label = "training data")
    plot!(x -> path[t5+1]' * [1, x], c = :red, w = 2,
          label = "current fit")
    p2 = contour(-3:.1:3, -3:.1:3, (β₀, β₁) -> M.lin_reg_loss(β₀, β₁, X, y), cbar = false,
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


# ╔═╡ 7b6e950a-b3db-4a73-b4ef-06bc215365c2
md"In the training curve below, we plot the loss for different numbers of data points used for the gradient computation. If a given data point is used multiple times, we count it multiple times: for one step of gradient descent, we use all 40 data points, whereas for one step of stochastic gradient descent with batch size 10, we use only 10 points.
- We see that stochastic gradient descent (SGD) decreases the training loss faster for the same learning rate η.
- For stochastic gradient descent we can plot either the true training loss of the full data set (red), or the training loss evaluated on the given batch (green, thin), or a moving average of the loss evaluated on batches (green, thick). In practice one does not compute the loss on the full dataset in each step of SGD, because this could be costly for a large training set. As the moving average of the loss evaluated on individual batches follows the true training loss quite closely, it is a useful and fast-to-compute approximation of the true training loss.
"

# ╔═╡ 07f57518-aa01-4979-8ad3-9bd052aee1fd
let X = M.X, y = M.y, sgd_path = tracker_st.path
	lin_reg_loss_b(i) = (β₀, β₁) -> M.lin_reg_loss(β₀, β₁, X[(i-1)*10+1:i*10],
                                                   y[(i-1)*10+1:i*10])
	Random.seed!(seed)
    batches = rand(1:4, 101)
	f = plot(length(y)*eachindex(lin_reg_path),
		     M.loss.(lin_reg_path), xlabel = "number of data points used for gradient computation", ylabel = "training loss", yscale = :log10, label = "gradient descent η = $η", lw = 3)
	loss_per_batch = [lin_reg_loss_b(batches[i])(x[1], x[2]) for (i, x) in pairs(sgd_path)]
	plot!(length(y)/4*eachindex(sgd_path),
	      loss_per_batch, label = "SGD (loss evaluated on batches) η = $η_st", c = :green)
	plot!(length(y)/4*(3:length(sgd_path)-3),
	      [mean(loss_per_batch[i-3+1:i+3]) for i in 3:length(sgd_path)-3], label = "SGD (moving average of loss evaluated on batches) η = $η_st", c = :green, lw = 4)
	plot!(length(y)/4*eachindex(sgd_path),
	         M.loss.(sgd_path), label = "SGD (loss evaluated on all data) η = $η_st", lw = 2, c = :red)
	xlims!(0, 1500)
	f
end

# ╔═╡ 913cf5ee-ca1e-4063-bd34-6cccd0cc548b
md"# 3. Improved Versions of (S)GD

There are many tricks to improve over standard (stochastic) gradient descent.
One popular idea is to use [momentum](https://distill.pub/2017/momentum/).
We do not discuss these ideas further here, but you should know that `ADAM()` and variants like `ADAMW()` are particularly popular (and successful) improvements of standard (S)GD. These methods usually require no (or very little) tuning of the learning rate."

# ╔═╡ 2739fb52-fb1b-46d6-9708-e24bfdc459e2
mlcode(
"""
using Flux

function advanced_gradient_descent(loss, x; T, optimizer = ADAMW(),
                                         callback = x -> nothing)
    for t in 1:T
        ∇loss = gradient(loss, x)[1]       # compute ∇f
		Flux.update!(optimizer, x, ∇loss)  # apply the changes to x
        callback(x) # the callback will be used to save intermediate values
    end
    x
end

callback = let learning_curve = Float64[]
	x -> push!(learning_curve, loss(x))
end
params = [0., 0.]
advanced_gradient_descent(loss, params; T = 1000, callback)
(params, callback.learning_curve)
"""
,
"""
def advanced_gradient_descent(loss, x, optimizer, T):
	learning_curve = []
	for t in range(T):
		optimizer.zero_grad()            # set old partial derivatives to zero
		l = loss(x)                      # compute loss
		learning_curve.append(l.item())  # append loss to learning curve
		l.backward()                     # compute new partial derivatives
		optimizer.step()                 # update parameters
	return x, learning_curve

beta = torch.tensor(np.array([0., 0.]), requires_grad = True)
optimizer = torch.optim.Adam((beta,)) # or torch.optim.AdamW((beta,))
advanced_gradient_descent(loss, beta, optimizer, 1000)
"""
,
showoutput = false,
cache = false
)

# ╔═╡ 7b57c3f0-ef5a-4dd7-946f-72c8dde2ae8f
md"t = $(@bind t6 Slider(1:10^4, show_value = true))"

# ╔═╡ eb289254-7167-4183-a4d0-52f68be66b04
begin
	Random.seed!(1234)
    params4 = .1 * randn(6)
	tracker4 = Tracker(params4)
	M.advanced_gradient_descent(mse, params4, T = 10^4, callback = tracker4)
end;

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
md"# 4. Early Stopping as Regularization

In early stopping we start (stochastic) gradient descent with smart parameter values, keep track of training and validation loss throughout gradient descent and stop gradient descent, when the validation loss reached the smallest value. The effect of early stopping is similar to regularization: the parameter values found at early stopping have usually a smaller norm than the parameter values with the lowest training error."

# ╔═╡ 075f35cf-4271-4676-b9f3-e2bcf610c2d1
md"step t = $(@bind t4 Slider(0:10^3:3*10^4, show_value = true))"

# ╔═╡ a10b3fa1-358b-4a94-b927-5e0f71edd1f3
md"# 5. The XOR-Problem and Vector Features

In this section we look at a version of the famous xor problem. It is a binary classification problem with 2-dimensional input. It is called xor problem, because the class label is true if and only if the ``X_1`` and ``X_2`` coordinate of a point have different sign, which is reminiscent of the [logical exclusive or operation](https://en.wikipedia.org/wiki/Exclusive_or). The decision boundary is non-linear. By taking the scalar product with 4 different 2-dimensional vectors and setting to 0 all scalar product that would be negative, we find a 4-dimensional feature representation for which linear logistic regression can classify each point correctly."

# ╔═╡ b66c7efc-8fd8-4152-9b0f-9332f760b51b
mlcode(
"""
using DataFrames, MLJ, MLJLinearModels

function xor_generator(; n = 200)
	x = 2 * rand(n, 2) .- 1
	DataFrame(X1 = x[:, 1], X2 = x[:, 2],
		      y = coerce((x[:, 1] .> 0) .⊻ (x[:, 2] .> 0), Binary))
end
xor_data = xor_generator()
"""
,
"""
import pandas as pd

def xor_generator(n=200):
    x = 2 * np.random.rand(n, 2) - 1
    df = pd.DataFrame({'X1': x[:, 0], 'X2': x[:, 1], 'y': np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)})
    return df

xor_data = xor_generator()
xor_data
"""
,
cache = true
)

# ╔═╡ 8cd98650-7f52-40d9-b92d-ce564e57cfa7
mlcode("""
using Plots

scatter(xor_data.X1, xor_data.X2, c = int.(xor_data.y) .+ 1,
		label = nothing, xlabel = "X₁", ylabel = "X₂")
hline!([0], label = nothing, c = :black)
vline!([0], label = nothing, c = :black, size = (400, 300))
""",
"""
import matplotlib.pyplot as plt

plt.scatter(xor_data['X1'], xor_data['X2'], c=np.array(xor_data['y'], dtype=int)+1, label=None, cmap ='coolwarm')
plt.axhline(y=0, color='black', linestyle='-', label=None)
plt.axvline(x=0, color='black', linestyle='-', label=None)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.show()
"""
)

# ╔═╡ 99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
mlcode(
"""
lin_mach = machine(LogisticClassifier(penalty = :none),
                   select(xor_data, Not(:y)),
                   xor_data.y)
fit!(lin_mach, verbosity = 0)
lin_pred = predict_mode(lin_mach, select(xor_data, Not(:y)))
(training_misclassification_rate = mean(lin_pred .!= xor_data.y),)
"""
,
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lin_mach = LogisticRegression(penalty=None)
lin_mach.fit(xor_data[['X1', 'X2']], xor_data['y'])
lin_pred = lin_mach.predict(xor_data[['X1', 'X2']])
training_misclassification_rate = np.mean(xor_data['y'] != lin_pred)

("training_misclassification_rate",training_misclassification_rate,)
"""
)

# ╔═╡ 1c13dd1a-83ce-47e7-98b5-b6a0d8d2426f
md"Now we transform the data to the vector features discussed in the slides."

# ╔═╡ 3da4ce70-d4ab-4d9b-841d-d1f3fb385b81
mlcode(
"""
xor_data_h = DataFrame(H1 = max.(0, xor_data.X1 + xor_data.X2),
	                   H2 = max.(0, xor_data.X1 - xor_data.X2),
	                   H3 = max.(0, -xor_data.X1 + xor_data.X2),
	                   H4 = max.(0, -xor_data.X1 - xor_data.X2),
                       y = xor_data.y)
"""
,
"""
xor_data_h = pd.DataFrame({
		'H1': np.maximum(0, xor_data['X1'] + xor_data['X2']), 
		'H2': np.maximum(0, xor_data['X1'] - xor_data['X2']), 
		'H3': np.maximum(0, -xor_data['X1'] + xor_data['X2']), 
		'H4': np.maximum(0, -xor_data['X1'] - xor_data['X2']), 
		'y': xor_data['y']})

xor_data_h
"""
)

# ╔═╡ d73c777a-f993-4ef8-a3f5-8f6717920436
md"... and we find that the training misclassification rate becomes 0 with this feature representation."

# ╔═╡ 60689196-ca58-4937-8999-4e6f6cd7bdbc
mlcode(
"""
lin_mach_h = machine(LogisticClassifier(penalty = :none),
                     select(xor_data_h, Not(:y)),
                     xor_data_h.y)
fit!(lin_mach_h, verbosity = 0)
lin_pred_h = predict_mode(lin_mach_h, select(xor_data_h, Not(:y)))
(training_misclassification_rate_h = mean(lin_pred_h .!= xor_data_h.y),)
"""
,
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lin_mach = LogisticRegression(penalty=None)
lin_mach.fit(xor_data_h.drop(["y"], axis=1), xor_data_h['y'])
lin_pred = lin_mach.predict(xor_data_h.drop(["y"], axis=1))
training_misclassification_rate = np.mean(xor_data_h['y'] != lin_pred)

("training_misclassification_rate", training_misclassification_rate,)
"""
)



# ╔═╡ c95ad296-5191-4f72-a7d7-87ddebc43a65
md"## Solving the XOR Problem with Gradient Descent and Learned Feature Vectors

In the following example we define a loss function that learns the features (weights of the hidden neurons) and the \"regression coefficients\" (output weights) with gradient descent.
"

# ╔═╡ a908176b-319a-445d-9f33-6271c2ef6149
md"First, we define a neural network with one hidden layer of 4 relu-neurons and the negative log-likelihood loss function."

# ╔═╡ a2539e77-943e-464c-9ad3-c8542eab36ab
mlcode(
"""
import Statistics: mean

relu(x) = max(0, x)
f(x, w₁, w₂, w₃, w₄, β) = β[1] * relu.(x * w₁) .+
                          β[2] * relu.(x * w₂) .+
                          β[3] * relu.(x * w₃) .+
                          β[4] * relu.(x * w₄) .+
                          β[5]
f(x, θ) = f(x, θ[1:2], θ[3:4], θ[5:6], θ[7:8], θ[9:13])
logp(x) = ifelse(x < -40, x, -log(1 + exp(-x))) # for numerical stability
function log_reg_loss_function(X, y)
    function(θ)
        tmp = f(X, θ)
        -mean(@. y * logp(tmp) + (1 - y) * logp(-tmp))
    end
end
"""
,
"""
import numpy as np

def relu(x):
    return np.maximum(0, x)

def f(x, w1, w2, w3, w4, beta):
    return beta[0] * relu(x * w1) + beta[1] * relu(x * w2) + beta[2] * relu(x * w3) + beta[3] * relu(x * w4) + beta[4]

def f(x, θ) :
	return f(x, θ[1:2], θ[3:4], θ[5:6], θ[7:8], θ[9:13])

def logp(x):
    return np.where(x < -40, x, -np.log(1 + np.exp(-x)))

def log_reg_loss_function(X, y, theta):
    def loss_function(theta):
        tmp = f(X, theta)
        return -np.mean(y * logp(tmp) + (1 - y) * logp(-tmp))
    return loss_function
""",
showoutput = false,
cache = false
)

# ╔═╡ 235c6588-8210-4b8e-812c-537a72ce950f
md"Now we can run gradient descent. The initial parameters θ₀ depend on a pseudo-random number generator whose seed can be set in the dropdown menu below."

# ╔═╡ 258a3459-3fef-4655-830c-3bdf11eb282d
md"seed = $(@bind(seed2, Select([string(i) for i in 1:20])))

t = $(@bind t7 Slider(1:10^2))
"

# ╔═╡ 9aa58d2b-783a-4b74-ae36-f2ff0fb0be3f
md"The green arrows in the figure above are the weights w₁, …, w₄ of the four (hidden) neurons. The coloring of the dots is based on classification at decision threshold 0.5, i.e. if the activation of the output neuron is above 0.5 for a given input point, this point is classified as red (green, otherwise). Note that the initial configuration of the green vectors depends on the initialization of gradient descent and therefore on the seed. For some seeds the optimal solution to the xor-problem is not found by gradient descent."

# ╔═╡ 8c54bfb2-54c5-4777-ad53-97926da97f7e
Markdown.parse("
## Avoiding Local Minima with Overparametrization
The following code fits 10 times with different initial guesses for the
parameters an MLP with 4 hidden neurons and 10 times an MLP with 10 hidden
neurons to our xor data. We see that training and test accuracy is better for
the larger network. The reason is that gradient descent for the the larger
network does not get stuck in suboptimal solutions.

$(MLCourse.embed_figure("overparametrized.png"))
")

# ╔═╡ fde9639d-5f41-4037-ab7b-d3dbb09e8d3d
begin
    function xor_generator(; n = 200)
        x = 2 * rand(n, 2) .- 1
        DataFrame(X1 = x[:, 1], X2 = x[:, 2],
                  y = (x[:, 1] .> 0) .⊻ (x[:, 2] .> 0))
    end
    xor_data = xor_generator()
    xor_input = Array(select(xor_data, Not(:y)))
    xor_target = xor_data.y
end;

# ╔═╡ 9661f274-180f-4e97-90ce-8beb7d1d69bc
mlcode(
"""
    using MLJFlux

    function xor_generator(; n = 200)
        x = 2 * rand(n, 2) .- 1
        DataFrame(X1 = x[:, 1], X2 = x[:, 2],
                  y = (x[:, 1] .> 0) .⊻ (x[:, 2] .> 0))
    end

    function fit_xor(n_hidden)
        x = select(xor_data, Not(:y))
        y = coerce(xor_data.y, Binary)
        builder = MLJFlux.Short(n_hidden = n_hidden,
                                dropout = 0,
                                σ = relu)
        mach = machine(NeuralNetworkClassifier(builder = builder,
                                               batch_size = 32,
                                               epochs = 10^4),
                       x, y) |> fit!
        xor_test = xor_generator(n = 10^4)
        x_test = select(xor_test, Not(:y))
        y_test = coerce(xor_test.y, Binary)
        DataFrame(n_hidden = n_hidden,
                  training_accuracy = mean(predict_mode(mach, x) .== y),
                  test_accuracy = mean(predict_mode(mach, x_test) .== y_test))
    end

    xor_results = vcat([fit_xor(n_hidden) for n_hidden in [4, 10], _ in 1:10]...)

    @df xor_results dotplot(:n_hidden, :training_accuracy,
                            xlabel = "number of hidden units",
                            ylabel = "accuracy",
                            label = "training",
                            yrange = (.5, 1.01),
                            aspect_ratio = 20,
                            legend = :bottomright,
                            xticks = [4, 10])
    @df xor_results dotplot!(:n_hidden, :test_accuracy, label = "test")
"""
,
nothing
,
eval = false,
collapse = "Implementation details"
)


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

# ╔═╡ 4dabcbed-9c35-4226-a78f-e7afa5115d92
begin
    h_training = Array(select(MLCourse.poly(regression_data, 12), Not(:y)))
    h_valid = Array(select(MLCourse.poly(regression_valid, 12), Not(:y)))
    poly_regression_loss(θ, x, y) = mean((y .- θ[1] .- x * θ[2:end]).^2)
    target = regression_data.y
    poly_regression_loss(θ) = poly_regression_loss(θ, h_training, target)
    poly_params = 1e-3 * randn(13)
    tracker5 = Tracker(poly_params)
	M.advanced_gradient_descent(poly_regression_loss, poly_params, T = 3*10^4, callback = tracker5)
end;

# ╔═╡ 0d431c00-9eef-4ce4-9542-9571728d1501
let poly_path = tracker5.path
    p1 = scatter(regression_data.x, regression_data.y,
                 label = "training data", ylims = (-.1, 1.1))
    plot!(g, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    θ = poly_path[t4 + 1]
    pred =  Array(MLCourse.poly((x = grid,), 12)) * θ[2:end] .+ θ[1]
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
md"""# Exercises
## Conceptual
#### Exercise 1

Assume the noise in a linear regression setting comes from a Laplace
distribution, i.e. the conditional probability density of the response is given by
``p(Y = y | X = x, \beta) = \frac1{2s}\exp(-\frac{|y - x^T\beta|}{s})``.
For simplicity we assume throughout this exercise that the intercept ``\beta_0 = 0``
and does not need to be fitted.

(a) Calculate with paper and pencil the negative log-likelihood loss. Apply transformations to the negative log-likelihood function to obtain a good loss function for gradient descent based on the practical considerations in the slides.
The solution you should find is
```math
\tilde L = \frac1{n} \sum_{i=1}^n |y_i - x_i^T\beta|.
```

(b) Assume the training data consists of ``x_1 = (1, 0, 0), y_1 = 4`` and ``x_2 = (1, 0, 2), y_2 = -1`` and you start with ``\beta_1^{(0)} = \beta_2^{(0)} = \beta_3^{(0)} = 0``. Compute with paper and pencil two steps of gradient descent on the negative log-likelihood loss with learning rate ``\eta  = \frac1{10}``.

(c) Generate a training set of 100 points with the following data generator.
Notice that the noise follows a Laplace distribution instead of a normal distribution.
$(mlstring(md"For once, we do not use a `DataFrame` here but represent the input explicitly as a matrix and the full dataset as a `NamedTuple`. If you run `data = data_generator()`, you can access the input matrix as `data.x` and the output vector as `data.y`.",""))
"""

# ╔═╡ 8f3a30c3-0d7b-4028-a8cd-185fb1858b40
mlcode("""
import Distributions: Laplace

function data_generator(; n = 100, β = [1., 2., 3.])
    x = randn(n, 3)
    y = x * β .+ rand(Laplace(0, 0.3), n)
    (x = x, y = y)
end
""",
"""
import numpy as np

def data_generator(n=100, beta=[1., 2., 3.]):
    x = np.random.randn(n, 3)
    y = np.dot(x, beta) + np.random.laplace(0, 0.3, n)
    return {'x': x, 'y': y}
""")


# ╔═╡ c0285107-0131-404f-9617-229f1c88f091
begin
pyhint = md"""`torch.matmul(data["x"], beta)`. For conversion from `numpy` to `torch` there are multiple options, like `torch.tensor(numpy_array)` or `torch.from_numpy(numpy_array)`. For conversion back one can use `torch_tensor.detach().numpy()`"""
md"""
(d) Code a function to compute the loss on the training set for a given
parameter vector. *Hint:* use matrix multiplication, e.g. $(mlstring(md"`data.x * β`", pyhint)).

(e) Perform gradient descent on the training set. Plot the learning curve to see
whether gradient descent has converged. If you see large fluctuations at the end
of training, decrease the learning rate. If the learning curve is not flat at
the end, increase the maximal number of steps. To see well the loss towards the end of gradient descent it is advisable to use log-scale for the y-axis.

(f) Estimate the coefficients with the standard linear regression.
Hint: do not forget that we fit without intercept.

(g) Compare which method (e) or (f) found parameters closer to the one of our data generating process `[1, 2, 3]` and explain your finding.
"""
end

# ╔═╡ b6af3881-04fb-483a-aa6b-35b9b4574980
md"""
#### Exercise 2
In the \"Vector Features\" section we said that the xor problem has a non-linear decision boundary in the original ``X_1, X_2`` coordinates, but a linear decision boundary in the ``H_1, H_2, H_3, H_4`` coordinates, with ``H_i = \max(0, w_i^TX)`` and we use the vector features ``w_1 = [1, 1]``, ``w_2 = [1, -1]``, ``w_3 = [-1, 1]`` and ``w_4 = [-1, -1]``. Here we prove that ``-H_1 + H_2 + H_3 - H_4 = 0`` defines indeed a linear decision boundary that solves the xor-problem.
* Show that ``-H_1 + H_2 + H_3 - H_4  < 0`` for all points with ``X_1 > 0`` and ``X_2 > 0``.
* Show that ``-H_1 + H_2 + H_3 - H_4  < 0`` for all points with ``X_1 < 0`` and ``X_2 < 0``.
* Show that ``-H_1 + H_2 + H_3 - H_4  > 0`` for all other points.

## Applied
#### Exercise 3
You are given the following artificial dataset. The data is not linearly separable, meaning that there is no linear decision boundary that would perfectly seperate the blue from the red data points.
    * Find a 2-dimensional feature representation ``H_1 = f_1(X_1, X_2), H_2 = f_2(X_1, X_2)``, such that the transformed data is linearly separable.
    * Generate a similar plot as the one below for the transformed data to visualize that the decision boundary has become linear for the transformed data.
"""

# ╔═╡ 9a0e0bf7-44e6-4385-ac46-9a6a8e4245bb
mlcode(
"""
cldata = DataFrame(2*rand(400, 2).-1, :auto)
cldata.y = cldata.x1.^2 .> cldata.x2
cldata
"""
,
"""
import pandas as pd

cldata = pd.DataFrame(2*np.random.rand(400, 2)-1, columns=['x1', 'x2'])
cldata['y'] = cldata['x1']**2 > cldata['x2']
cldata
"""
)

# ╔═╡ 6f4427a0-18eb-4637-a19a-ec9aa7b6fda8
mlcode(
"""
using Plots

scatter(cldata.x1, cldata.x2,
        c = Int.(cldata.y) .+ 1, legend = false,
        xlabel = "X₁", ylabel = "X₂")
"""
,
"""
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(cldata["x1"], cldata["x2"],
            c=cldata["y"].astype(int) + 1,
            label=None, cmap ='coolwarm')
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.show()
"""
)

# ╔═╡ a81db4a1-e561-4796-896c-26133a8efc60
md"#### Exercise 4 (optional)
In Exercise 5 of \"Generalized Linear Regression\" we fitted the bike sharing data using only `:temp` and `:humidity` as predictors. The quality of the fit was not good. Here we try to improve the fit by including more predictors. Many predictors can be treated as categorical, e.g. even the `:hour`, which is actually an ordered, periodic integer, can be treated as categorical to give the linear model a lot of flexibility. Try out different transformations of the input until you find a linear Poisson model that fits the data clearly better than what we had in the previous Exercise. You can measure quality of fit by looking at the same plot as in the previous exercise or by using cross-validation.
"

# ╔═╡ cb9f858a-f60a-11eb-3f0e-a9b68cf33921
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 8459f86e-bce7-4839-9c51-57335ac6353c
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

# ╔═╡ 49c2b3a1-50c0-4fb9-a06b-2de7be216702
begin
    Random.seed!(Meta.parse(seed2))
    θ₀ = .1 * randn(13)
    tracker_xor = Tracker(θ₀)
    MLCourse.advanced_gradient_descent(M.log_reg_loss_function(xor_input, xor_target),
                              θ₀, T = 2*10^4,
                              callback = tracker_xor)
    idxs = min.(max.(1:100, floor.(Int, (2e4^(1/100)).^(1:100))), 10^5)
    losses = M.log_reg_loss_function(xor_input, xor_target).(tracker_xor.path[idxs])
end;

# ╔═╡ 16d0808e-4094-46ff-8f92-1ed21aa6191b
let idx = idxs[t7], path = tracker_xor.path
    θ = path[idx]
    prediction = σ.(M.f(xor_input, θ)) .> .5
    p1 = scatter(xor_data.X1, xor_data.X2, c = prediction .+ 1,
                 xlim = (-1.5, 1.5), ylim = (-1.5, 1.5),
                 xlabel = "X1", ylabel = "X2")
    for i in 1:4
        plot!([0, θ[i*2 - 1]], [0, θ[i*2]], c = :green, w = 3, arrow = true)
    end
    hline!([0], c = :black)
    vline!([0], c = :black)
    p2 = plot(idxs, losses, xscale = :log10, xlabel = "gradient descent step t",
              ylabel = "loss", title = "learning curve", yrange = (-.03, 1))
    scatter!([idxs[t7]], [losses[t7]], c = :red)
    plot(p1, p2, layout = (1, 2), size = (700, 400), legend = false)
end

# ╔═╡ Cell order:
# ╟─e03882f9-843e-4552-90b1-c47b6cbba19b
# ╟─06479d01-5c03-4b26-9d1c-2ee71032c3ae
# ╟─9f18f610-79e2-403a-a792-fe2dafd2054a
# ╟─fcbac567-36db-40c5-9436-6315c0caa40a
# ╟─e65c80e2-c951-4ff0-aeff-2cdddac26479
# ╟─3ca1e8f3-d69f-454f-b917-4bbe2dcfce01
# ╟─bbc0b514-4789-44d1-8d90-9fc325d9ad6b
# ╟─b1fc14bb-1fd2-4739-a761-7a605fd4559b
# ╟─2fe2049c-e9db-4a1e-a927-f147f137b3c4
# ╟─eb9e2b3f-dc4b-49a2-a611-b45f2918adcf
# ╟─402bb576-8945-403e-a9a6-fd5bfb8016bc
# ╟─cd5f079f-a06d-4d55-9666-e2b05ddf8989
# ╟─93699313-5e42-48a7-abc6-ad146e5bdcdd
# ╟─33bdc912-e46a-4310-9184-733be7871768
# ╟─979feb42-c328-44b5-b519-8e8cda474140
# ╟─4b3ac87f-39f6-4e6c-b7b3-536925e9b112
# ╟─01221937-b6d9-4dac-be90-1b8a1c5e9d87
# ╟─cecca28a-23e7-467f-b1b3-5d6b292c298b
# ╟─f5e27275-8751-4e80-9888-c3d22d8e80e3
# ╟─c448238c-f712-4af3-aedb-cec3a2c1a73e
# ╟─9a61332f-cdc2-4129-b7b5-5ab54ba387a3
# ╟─6e38a3f6-a592-4dc1-a6c4-d0f0050c9399
# ╟─ca1301fd-8978-44d6-bfae-e912f939d7a8
# ╟─e821fb15-0bd3-4fa7-93ea-692bf05097b5
# ╟─fa706e4b-eaa9-4be2-b4f5-1906931c8ef6
# ╟─75528011-05d9-47dc-a37b-e6bb6be52c25
# ╟─7541c203-f0dc-4445-9d2a-4cf16b7e912a
# ╟─7b6e950a-b3db-4a73-b4ef-06bc215365c2
# ╟─07f57518-aa01-4979-8ad3-9bd052aee1fd
# ╟─913cf5ee-ca1e-4063-bd34-6cccd0cc548b
# ╟─2739fb52-fb1b-46d6-9708-e24bfdc459e2
# ╟─7b57c3f0-ef5a-4dd7-946f-72c8dde2ae8f
# ╟─166472c5-c0f4-4261-a476-4c9b0f82abd6
# ╟─eb289254-7167-4183-a4d0-52f68be66b04
# ╟─08a9418f-786e-4992-b1a5-04cf9060f8fe
# ╟─075f35cf-4271-4676-b9f3-e2bcf610c2d1
# ╟─0d431c00-9eef-4ce4-9542-9571728d1501
# ╟─a10b3fa1-358b-4a94-b927-5e0f71edd1f3
# ╟─b66c7efc-8fd8-4152-9b0f-9332f760b51b
# ╟─8cd98650-7f52-40d9-b92d-ce564e57cfa7
# ╟─99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
# ╟─1c13dd1a-83ce-47e7-98b5-b6a0d8d2426f
# ╟─3da4ce70-d4ab-4d9b-841d-d1f3fb385b81
# ╟─d73c777a-f993-4ef8-a3f5-8f6717920436
# ╟─60689196-ca58-4937-8999-4e6f6cd7bdbc
# ╟─c95ad296-5191-4f72-a7d7-87ddebc43a65
# ╟─a908176b-319a-445d-9f33-6271c2ef6149
# ╟─a2539e77-943e-464c-9ad3-c8542eab36ab
# ╟─235c6588-8210-4b8e-812c-537a72ce950f
# ╟─258a3459-3fef-4655-830c-3bdf11eb282d
# ╟─16d0808e-4094-46ff-8f92-1ed21aa6191b
# ╟─9aa58d2b-783a-4b74-ae36-f2ff0fb0be3f
# ╟─8c54bfb2-54c5-4777-ad53-97926da97f7e
# ╟─fde9639d-5f41-4037-ab7b-d3dbb09e8d3d
# ╟─9661f274-180f-4e97-90ce-8beb7d1d69bc
# ╟─dc57d700-2a82-4ab0-9bd2-6ce622cb0fa5
# ╟─4dabcbed-9c35-4226-a78f-e7afa5115d92
# ╟─e0cc188c-9c8f-47f1-b1fe-afc2a578973d
# ╟─8f3a30c3-0d7b-4028-a8cd-185fb1858b40
# ╟─c0285107-0131-404f-9617-229f1c88f091
# ╟─b6af3881-04fb-483a-aa6b-35b9b4574980
# ╟─9a0e0bf7-44e6-4385-ac46-9a6a8e4245bb
# ╟─6f4427a0-18eb-4637-a19a-ec9aa7b6fda8
# ╟─a81db4a1-e561-4796-896c-26133a8efc60
# ╟─cb9f858a-f60a-11eb-3f0e-a9b68cf33921
# ╟─8459f86e-bce7-4839-9c51-57335ac6353c
# ╟─7aa547f8-25d4-488d-9fc3-f633f7f03f57
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
# ╟─49c2b3a1-50c0-4fb9-a06b-2de7be216702
