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

In each step of stochastic gradient descent (SGD) the gradient is computed only on a (stochastically selected) subset of the data. As long as the selected subset of the data is somewhat representative of the full dataset the gradient will point more or less in the same direction as the full gradient computed on the full dataset. The advantage of computing the gradient only on a subset of the data is that it takes much less time to compute the gradient on a small dataset than on a large dataset. In the figure below you see the niveau lines of the full loss in black and the niveau lines of the loss computed on a green, purple, yellow and a blue subset. The gradient computed on, say, the yellow subset is always perpendicular to the yellow niveau lines but it may not be perpendicular to the niveau lines of the full loss. When selecting in each step another subset we observe a jittered trajectory around the full gradient descent trajectory.
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
md"η = $(@bind η_st Slider(.01:.01:.2, default = .1, show_value = true))

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
md"""# Exercise

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
pyhint = md"""`torch.matmul(data["x"], beta)`. For conversion from `numpy` to `torch` there are multiple options, like `torch.tensor(numpy_array)` or `torch.from_numpy(numpy_array)`. For conversion back one can use `torch_tensor.detach().numpy()` or `torch_tensor.data.numpy()`"""
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

# ╔═╡ cb9f858a-f60a-11eb-3f0e-a9b68cf33921
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 8459f86e-bce7-4839-9c51-57335ac6353c
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

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
# ╟─913cf5ee-ca1e-4063-bd34-6cccd0cc548b
# ╟─2739fb52-fb1b-46d6-9708-e24bfdc459e2
# ╟─7b57c3f0-ef5a-4dd7-946f-72c8dde2ae8f
# ╟─166472c5-c0f4-4261-a476-4c9b0f82abd6
# ╟─eb289254-7167-4183-a4d0-52f68be66b04
# ╟─08a9418f-786e-4992-b1a5-04cf9060f8fe
# ╟─075f35cf-4271-4676-b9f3-e2bcf610c2d1
# ╟─0d431c00-9eef-4ce4-9542-9571728d1501
# ╟─dc57d700-2a82-4ab0-9bd2-6ce622cb0fa5
# ╟─4dabcbed-9c35-4226-a78f-e7afa5115d92
# ╟─e0cc188c-9c8f-47f1-b1fe-afc2a578973d
# ╟─8f3a30c3-0d7b-4028-a8cd-185fb1858b40
# ╟─c0285107-0131-404f-9617-229f1c88f091
# ╟─cb9f858a-f60a-11eb-3f0e-a9b68cf33921
# ╟─8459f86e-bce7-4839-9c51-57335ac6353c
# ╟─7aa547f8-25d4-488d-9fc3-f633f7f03f57
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
