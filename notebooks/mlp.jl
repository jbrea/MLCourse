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

# ╔═╡ 87f59dc7-5149-4eb6-9d81-440ee8cecd72
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, Flux
import PlutoPlotly as PP
import MLCourse: heaviside
const M = MLCourse.JlMod
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ 83e2c454-042f-11ec-32f7-d9b38eeb7769
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ c1cbcbf4-f9a9-4763-add8-c5ed7cbd2889
md"The goal of this week is to
1. understand, how the XOR problem can be solved without feature engineering.
2. understand, how neurons and neural networks can be written in code.
3. learn how to run regression with neural networks.
4. learn how to run classification with neural networks.
"

# ╔═╡ c95ad296-5191-4f72-a7d7-87ddebc43a65
md"# 1. Solving the XOR Problem with Learned Feature Vectors

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
md"seed = $(@bind(seed, Select([string(i) for i in 1:20])))

t = $(@bind t Slider(1:10^2))
"

# ╔═╡ 9aa58d2b-783a-4b74-ae36-f2ff0fb0be3f
md"The green arrows in the figure above are the weights w₁, …, w₄ of the four (hidden) neurons. The coloring of the dots is based on classification at decision threshold 0.5, i.e. if the activation of the output neuron is above 0.5 for a given input point, this point is classified as red (green, otherwise). Note that the initial configuration of the green vectors depends on the initialization of gradient descent and therefore on the seed. For some seeds the optimal solution to the xor-problem is not found by gradient descent."

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

# ╔═╡ 5c2154ef-2a35-4d91-959d-f72100049894
md"# 2. Multilayer Perceptrons

Below you see three equivalent ways to implement a neural network. It is recommendable to read and try to understand this code, to get a feeling for the different ways, how a neural network can be implemented. Later we will use dedicated deep learning packages instead of this custom code, however. The deep learning packages usually work with the matrix version.

1. In the `expanded` form all parameters and activation functions are explicitely written out in \"one line\". With more neurons or hidden layers, this form quickly becomes annoying to write.
2. In the `neuron` form we see clearly that the neural network is composed of multiple simple neurons with different input weights and biases.
3. The `matrix` form is the most succinct form and most commonly used in modern implementations. In this form the biases of each layer are organized in a vector and the input weights of all neurons in the same layer are organized in a matrix.
"


# ╔═╡ 71dabb22-816b-4aa2-9a29-f2894989b2ea
mlcode(
"""
# 1. expanded version
function neural_net_expanded(x,
                             w₁₀¹, w₁₁¹, w₂₀¹, w₂₁¹, w₃₀¹, w₃₁¹, w₄₀¹, w₄₁¹, g¹,
                             w₁₀², w₁₁², w₁₂², w₁₃², w₁₄², g²)
    g²(w₁₀² +
       w₁₁² * g¹(w₁₀¹ + w₁₁¹ * x) +
       w₁₂² * g¹(w₂₀¹ + w₂₁¹ * x) +
       w₁₃² * g¹(w₃₀¹ + w₃₁¹ * x) +
       w₁₄² * g¹(w₄₀¹ + w₄₁¹ * x))
end

# 2. neuron version
neuron(x, b, w, g) = g.(b .+ w' * x)
function neural_net_neurons(x,
                            w₁₀¹, w₁₁¹, w₂₀¹, w₂₁¹, w₃₀¹, w₃₁¹, w₄₀¹, w₄₁¹, g¹,
                            w₁₀², w₁₁², w₁₂², w₁₃², w₁₄², g²)
    second_layer_input = [neuron(x, w₁₀¹, w₁₁¹, g¹),
                          neuron(x, w₂₀¹, w₂₁¹, g¹),
                          neuron(x, w₃₀¹, w₃₁¹, g¹),
                          neuron(x, w₄₀¹, w₄₁¹, g¹)]
    neuron(second_layer_input, w₁₀², [w₁₁², w₁₂², w₁₃², w₁₄²], g²)
end

# 3. matrix version
function neural_net_matrix(x, b¹, w¹, g¹, b², w², g²)
    g².(b² .+ w² * g¹.(b¹ .+ w¹ * x))
end
function neural_net_matrix(x,
                           w₁₀¹, w₁₁¹, w₂₀¹, w₂₁¹, w₃₀¹, w₃₁¹, w₄₀¹, w₄₁¹, g¹,
                           w₁₀², w₁₁², w₁₂², w₁₃², w₁₄², g²)
    b¹ = [w₁₀¹, w₂₀¹, w₃₀¹, w₄₀¹]
    w¹ = [w₁₁¹
          w₂₁¹
          w₃₁¹
          w₄₁¹]
    b² = [w₁₀²]
    w² = [w₁₁² w₁₂² w₁₃² w₁₄²]
    neural_net_matrix(x, b¹, w¹, g¹, b², w², g²)
end
"""
,
"""
import numpy as np

# 1. expanded version
def neural_net_expanded(x, w10_1, w11_1, w20_1, w21_1, w30_1, w31_1, w40_1, w41_1,g1, w10_2, w11_2, w12_2, w13_2, w14_2, g2):

    return g2(w11_2 * g1(w10_1 + w11_1 * x) 
			+ w12_2 * g1(w20_1 + w21_1 * x)
			+ w13_2 * g1(w30_1 + w31_1 * x)
			+ w14_2 * g1(w40_1 + w41_1 * x)
			+ w10_2)

# 2. neuron version
def neuron(x, b, w, g):
    return g(b + np.dot(w.T, x))

def neural_net_neurons(x, w10_1, w11_1, w20_1, w21_1, w30_1, w31_1, w40_1, w41_1, g1, w10_2, w11_2, w12_2, w13_2, w14_2, g2):

  second_layer_input = [neuron(x, w10_1, np.array([w11_1]), g1),
							neuron(x, w20_1, np.array([w21_1]), g1),
							neuron(x, w30_1, np.array([w31_1]), g1),
							neuron(x, w40_1, np.array([w41_1]), g1)]

  return neuron(second_layer_input, 
					w10_2, 
					np.array([w11_2, w12_2, w13_2, w14_2]), g2)

# 3. matrix version
def neural_net_m(x, b1, w1, g1, b2, w2, g2):
    return g2(b2 + np.dot(w2.T, g1(b1 + np.dot(w1.T, x))))

def neural_net_matrix(x, w10_1, w11_1, w20_1, w21_1, w30_1, w31_1, w40_1, w41_1, g1, w10_2, w11_2, w12_2, w13_2, w14_2, g2):

    b1 = np.array([w10_1, w20_1, w30_1, w40_1])
    w1 = np.array([w11_1,w21_1, w31_1,w41_1])
    b2 = w10_2
    w2 = np.array([w11_2, w12_2, w13_2, w14_2])

    return neural_net_m(x, b1, w1, g1, b2, w2, g2)
"""
,
cache = false
)


# ╔═╡ 653ea4e8-47b2-4d9f-bc5c-583468d6e1ae
md"## 1D to 1D
**First Layer** activation function ``g^{(1)}`` = $(@bind g1 Select(string.([relu, selu, sigmoid, tanh, identity, heaviside])))

``w_{10}^{(1)}`` = $(@bind w101 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{11}^{(1)}`` = $(@bind w111 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{20}^{(1)}`` = $(@bind w201 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{21}^{(1)}`` = $(@bind w211 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{30}^{(1)}`` = $(@bind w301 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{31}^{(1)}`` = $(@bind w311 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{40}^{(1)}`` = $(@bind w401 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{41}^{(1)}`` = $(@bind w411 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

**Second Layer** activation function ``g^{(2)}`` = $(@bind g2 Select(string.([identity, relu, selu, sigmoid, tanh, heaviside])))

``w_{10}^{(2)}`` = $(@bind w102 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{11}^{(2)}`` = $(@bind w112 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{12}^{(2)}`` = $(@bind w122 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{13}^{(2)}`` = $(@bind w132 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{14}^{(2)}`` = $(@bind w142 Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
"

# ╔═╡ 1171bcb8-a2af-49f3-9a7e-2e7cac34f9f4
let neuron = M.neuron
    x = -5:.01:5
    _g1 = getproperty(@__MODULE__, Symbol(g1))
    _g2 = getproperty(@__MODULE__, Symbol(g2))
    y = M.neural_net_expanded.(x, w101, w111, w201, w211, w301, w311, w401, w411, _g1, w102, w112, w122, w132, w142, _g2)
    py = plot(x, y, xlabel = "x", ylabel = "y")
    p1 = plot(x, neuron.(x, w101, w111, _g1), xlabel = "x", ylabel = "hidden neuron 1")
    p2 = plot(x, neuron.(x, w201, w211, _g1), xlabel = "x", ylabel = "hidden neuron 2")
    p3 = plot(x, neuron.(x, w301, w311, _g1), xlabel = "x", ylabel = "hidden neuron 3")
    p4 = plot(x, neuron.(x, w401, w411, _g1), xlabel = "x", ylabel = "hidden neuron 4")
    plot(plot(p1, p2, p3, p4, layout = (1, 4)), py, layout = (2, 1), legend = false)
end


# ╔═╡ 1c70c3c4-bded-44ec-ac40-c71268b7307e
md"## 2D to 1D
**First Layer** activation function ``g^{(1)}`` = $(@bind g1_2d Select(string.([relu, selu, sigmoid, tanh, identity, heaviside]), default = string(:heaviside)))

``w_{10}^{(1)}`` = $(@bind w101_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{11}^{(1)}`` = $(@bind w111_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{12}^{(1)}`` = $(@bind w121_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{20}^{(1)}`` = $(@bind w201_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{21}^{(1)}`` = $(@bind w211_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{22}^{(1)}`` = $(@bind w221_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{30}^{(1)}`` = $(@bind w301_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{31}^{(1)}`` = $(@bind w311_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{32}^{(1)}`` = $(@bind w321_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{40}^{(1)}`` = $(@bind w401_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{41}^{(1)}`` = $(@bind w411_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{42}^{(1)}`` = $(@bind w421_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

**Second Layer** activation function ``g^{(2)}`` = $(@bind g2_2d Select(string.([identity, relu, selu, sigmoid, tanh, heaviside])))

``w_{10}^{(2)}`` = $(@bind w102_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{11}^{(2)}`` = $(@bind w112_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{12}^{(2)}`` = $(@bind w122_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))

``w_{13}^{(2)}`` = $(@bind w132_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
``w_{14}^{(2)}`` = $(@bind w142_2d Slider(-3.:.1:3., default = floor(60*(rand()-.5))/10, show_value = true))
"

# ╔═╡ 15cd82d3-c255-4eb5-9f0d-1e662c488027
md"If you want to tinker more with simple neural networks there is [this nice web-application](http://playground.tensorflow.org/)."

# ╔═╡ 9e6ad99f-8cd4-4372-a4a2-f280e49e21a3
begin
    function tocol(y)
        ma, mi = maximum(y), minimum(y)
        palette(:jet1, 101)[min.(101, max.(1, reshape(floor.(Int, (y .+ 6) * 5) .+ 1, :)))]
    end
end;

# ╔═╡ 66bc40f3-7710-4892-ae9f-79b9c850b0ea
let x = -5:.1:5, neuron = M.neuron
    grid = MLCourse.grid2D(x, x, output_format = DataFrame)
    X = Array(Array(grid)')
    _g1 = getproperty(@__MODULE__, Symbol(g1_2d))
    _g2 = getproperty(@__MODULE__, Symbol(g2_2d))
    b1 = [w101_2d, w201_2d, w301_2d, w401_2d]
    w1 = [w111_2d w121_2d
          w211_2d w221_2d
          w311_2d w321_2d
          w411_2d w421_2d]
    b2 = [w102_2d]
    w2 = [w112_2d w122_2d w132_2d w142_2d]
    y = M.neural_net_matrix(X, b1, w1, _g1, b2, w2, _g2)
    p1 = scatter(grid.X, grid.Y, c = tocol(neuron(X, b1[1], w1[1, :], _g1)), title = "hidden 1")
    p2 = scatter(grid.X, grid.Y, c = tocol(neuron(X, b1[2], w1[2, :], _g1)), title = "hidden 2")
    p3 = scatter(grid.X, grid.Y, c = tocol(neuron(X, b1[3], w1[3, :], _g1)), title = "hidden 3")
    p4 = scatter(grid.X, grid.Y, c = tocol(neuron(X, b1[4], w1[4, :], _g1)), title = "hidden 4")
    py = scatter(grid.X, grid.Y, c = tocol(y), title = "y",)
    plot(plot(p1, p2, p3, p4, layout = (1, 4)),
         py, layout = (2, 1), markersize = 2, markerstrokewidth = 0,
                 xlabel = "x₁", ylabel = "x₂", legend = false, aspect_ratio = 1)
end

# ╔═╡ 9d3e7643-34fd-40ee-b442-9d2a434f30e0
md"""# 3. Regression with MLPs

## Constructing Multilayer Neural Networks

There is no need to use custom code to construct neural networks, because there are many good software packages that allow to construct networks very easily.

In the first example we construct a neural network with 3-dimensional input, 30 hidden relu neurons and 2-dimensional output. We can confirm that the weight matrix of the first layers is of shape (30, 3) and there are 30 biases.
"""

# ╔═╡ 4bbae793-059d-4741-a761-c38be74b274c
mlcode(
"""
using Flux

nn = Chain(Dense(3, 30, relu), # input to hidden layer (with relu non-linearity)
           Dense(30, 2))       # hidden layer to output (without non-linearity)

(w¹ = nn.layers[1].weight, b¹ = nn.layers[1].bias)
"""
,
"""
import torch

class NeuralNetExample1(torch.nn.Module):
    def __init__(self):
        super(NeuralNetExample1, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2)
        )

    def forward(self, x):
        return self.layers(x)

nn = NeuralNetExample1()
w1, b1 = map(lambda parameter: parameter.data.numpy(), list(nn.layers[0].parameters()))
print(f'{w1=}\n{b1=}')
"""
)

# ╔═╡ 114201bf-d841-496f-8ca5-2fbd43726995
md"""
Note that neural networks are often by default generated with single precision (float32) parameters. We will usually want to make sure that the input also has single precision and use for example type promotion like $(mlstring(md"`Float32.(X)`", "")), if the input `X` is not given in single precision.

We can also confirm that the network computes what we expect it to compute.
We generate a 3-dimensional input and compute the output of the neural network in two ways:
"""

# ╔═╡ 6944c24e-67e7-47ba-aea6-5f75f58dae5a
mlcode(
"""
x = rand(Float32, 3)
nn(x) # apply neural network to input x
"""
,
"""
x = torch.rand(3)
nn(x).data.numpy()
"""
)

# ╔═╡ 03b2c815-65ef-4c74-a673-21830dbcb40e
mlcode(
"""
w¹ = nn.layers[1].weight
b¹ = nn.layers[1].bias
w² = nn.layers[2].weight
b² = nn.layers[2].bias
b² + w² * relu.(b¹ + w¹ * x)
"""
,
"""
w1, b1, w2, b2 = map(lambda parameter: parameter.data.numpy(), list(nn.layers.parameters())) 
b2 + np.dot(w2, relu(b1 + np.dot(w1, x)))
"""
)

# ╔═╡ 631d3ab5-817a-4ccf-88fb-950f85fc6be3
md"Creating arbitrarily deep and wide neural networks is straightforward.
In the following example we create a neural network with 20-100-50-100-10 neurons in input-hidden1-hidden2-hidden3-output layers with different non-linearities.
"

# ╔═╡ bf6bf8f3-dbc9-496a-ac5b-1aaf6e7824e1
mlcode(
"""
nn2 = Chain(Dense(20, 100, relu),
            Dense(100, 50, tanh),
            Dense(50, 100, Flux.σ),
            Dense(100, 10))
nn2(rand(Float32, 20))
"""
,
"""
class NeuralNetExample2(torch.nn.Module):
    def __init__(self):
        super(NeuralNetExample2, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(20, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.layers(x)

nn2 = NeuralNetExample2()
nn2(torch.rand(20)).data.numpy()
"""
)

# ╔═╡ a869d11a-2124-4d2b-8517-5cca4382309e
md"""
## Fitting the Weather Data

Let us find out if with an MLP we can get better predictions than with multiple
linear regression on the weather data set.

!!! note

    We standardize the input and the output to mean zero and standard deviation one.
    We do this, because the usual initializations
    of neural networks work best with standardized input.

    To learn more about the usual initializations you can have a look at this [blog post](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79) (optional).

!!! note

    Our builder constructs a neural network with one hidden layer of 128 relu neurons.
    In the construction we specify that we would like to train with 50% dropout
    to reduce the risk of overfitting.

"""

# ╔═╡ 6393aae7-a8ac-4dff-b188-5e682f6ab1f9
mlstring(
md"""
Some explanations for the code below:
1. The neural network used for regression in the `NeuralNetworkRegressor` is specified with the argument `builder`. Here we use the `MLJFlux.Short` builder, to construct a neural network with one hidden layer of 128 relu-neurons that are trained with dropout rate 0.5
2. The `TransformedTargetModel` allows to fit to standardized output data. When making predictions, the machine automatically applies the inverse of the standardization.
"""
,
"""
"""
)


# ╔═╡ a5568dfb-4467-4dca-b1c7-714bbfcbd6e6
mlcode(
"""
# preparing the data
using CSV, DataFrames, MLJFlux

weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
                   DataFrame)
coerce!(weather, Count => Continuous)
weather_test = CSV.read(download("https://go.epfl.ch/bio322-weather2019-2020.csv"),
                        DataFrame)
coerce!(weather_test, Count => Continuous)
weather_x = select(weather, Not([:LUZ_wind_peak, :time]))[1:end-5, :]
weather_y = weather.LUZ_wind_peak[6:end]
weather_test_x = select(weather_test, Not([:LUZ_wind_peak, :time]))[1:end-5, :]
weather_test_y = weather_test.LUZ_wind_peak[6:end]

# setting up the model
using Random, MLJ, MLJFlux, Flux

Random.seed!(31)
nn = NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 128,
                                                    dropout = .5,
                                                    σ = relu),
                            optimiser = ADAMW(),
                            batch_size = 128,
                            epochs = 150)
model = TransformedTargetModel(Standardizer() |> (x -> Float32.(x)) |> nn,
                               transformer = Standardizer())
mach = machine(model, weather_x, weather_y)

# fitting and evaluating the model
fit!(mach, verbosity = 0)
(training_error = rmse(predict(mach, weather_x), weather_y),
 test_error = rmse(predict(mach, weather_test_x), weather_test_y))
"""
,
"""
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import random
import tqdm
from sklearn.metrics import mean_squared_error

# Reproducibility
seed_num = 66
torch.use_deterministic_algorithms(True)
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(seed_num)

# Load the data
weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv")
weather_test = pd.read_csv("https://go.epfl.ch/bio322-weather2019-2020.csv")

# Standardisers
weather_x_std = StandardScaler().fit(weather['LUZ_pressure'][:-5].values.reshape(-1, 1))
weather_y_std = StandardScaler().fit(weather['LUZ_wind_peak'][5:].values.reshape(-1, 1))

# Standardise the data
weather_x, weather_y, weather_test_x, weather_test_y = map(
    lambda data, std_mach: std_mach.transform(data),
    [
        weather['LUZ_pressure'][:-5].values.reshape(-1, 1),
        weather['LUZ_wind_peak'][5:].values.reshape(-1, 1),
        weather_test['LUZ_pressure'][:-5].values.reshape(-1, 1),
        weather_test['LUZ_wind_peak'][5:].values.reshape(-1, 1),
    ],
    [weather_x_std, weather_y_std, weather_x_std, weather_y_std],
)

# Convert data to pytorch tensors
weather_x, weather_y, weather_test_x, weather_test_y = map(
    lambda array: torch.tensor(array, dtype=torch.float32),
    [weather_x, weather_y, weather_test_x, weather_test_y]
)

# Pytorch training data loader, for using batches
training_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(weather_x, weather_y),
    batch_size = 128,
    shuffle = True,
    worker_init_fn = seed_worker, # for reproducibility
    generator = g # for reproducibility
)

# Setting up the model
class NeuralNetExample3(torch.nn.Module):
    def __init__(self):
        super(NeuralNetExample3, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.layers(x)

batch_size = 128
epochs = 150

model = NeuralNetExample3()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

# Training
model.train()
for epoch in tqdm.tqdm(range(epochs)):
    for data, target in training_dataloader:
        optimizer.zero_grad()
        pred = model(data)
        loss = torch.sqrt(loss_func(pred, target))
        loss.backward()
        optimizer.step()

# Evaluation
with torch.no_grad():
    model.eval()
    train_preds = weather_y_std.inverse_transform(model(weather_x).detach().numpy())
    train_gt = weather_y_std.inverse_transform(weather_y.detach().numpy())
    train_loss = np.sqrt(mean_squared_error(train_preds, train_gt))

    test_preds = weather_y_std.inverse_transform(model(weather_test_x).detach().numpy())
    test_gt = weather_y_std.inverse_transform(weather_test_y.detach().numpy())
    test_loss = np.sqrt(mean_squared_error(test_preds, test_gt))
    print(f'{train_loss=}, {test_loss=}')
"""
,
cache = true
)


# ╔═╡ 47024142-aabb-40d8-8336-5f8aeb2aa623
md"""
Both errors are lower than what we found with multiple linear regression
(training error ≈ 8.09, test_error ≈ 8.91). However, comparing the predictions
to the ground truth we see that a lot of the variability is still uncaptured
by the model. If the model would prefectly predict the test data, all points in the figure below would lie on the red diagonal. This is not the case and means most likely that the irreducible error is pretty high in this dataset.
"""

# ╔═╡ 0a48241c-2be0-46cf-80ae-dc86abefad6f
mlcode(
"""
using Plots

scatter(predict(mach, weather_test_x), weather_test_y,
        xlabel = "predicted", ylabel = "data")
plot!(identity, legend = false)
"""
,
"""
"""
,
showoutput = true,
)


# ╔═╡ 0ba71a2a-f2f9-4fc7-aa81-416799e79e57
md"""## Parametrizing Variances of Log-Normal Distributions with MLPs

A key insight is that neural networks and gradient descent are very flexible tools to fit any function or probability distribution/density. To illustrate this, we use a neural network with two outputs to parametertrize the mean *and* the standard deviation of a conditional normal distribution. This is considerably more flexible than standard linear regression, where the mean is a linear function of the input and the standard deviation is fixed. With this neural network, both the mean and the standard deviation can be non-linear functions of the input.

The fun thing is that we do not need to code up much to get this running.
Basically, all we need is
1. a loss function,
2. a neural network,
3. run stochastic gradient descent.
"""

# ╔═╡ 3d6ae9f2-105a-4c7f-bf2f-c42afe2683cf
mlcode(
"""
function loss(x, y)
    output = nn(x)
    m = output[1, :]
    s = softplus.(output[2, :])
    mean((m .- y) .^ 2 ./ (2 * s) .+ 1/2 * log.(s))
end

"""
,
"""
"""
)

# ╔═╡ 67d386dc-18c8-4dbe-8e13-cfc8541e8b6c
md"""
This function computes the two output dimensions of the neural network.
The first output dimension is our mean. The second output dimension is transformed
to positive values with the `softplus` function ``x \mapsto \log(\exp(x) + 1)``
to enode the variance of the normal distribution. In the log-likelihood loss we
drop the constant term ``\frac12\log2\pi``.
"""

# ╔═╡ 6701f32c-015e-45f5-a59b-4d5bf9412f99
mlcode(
"""
# Standardization of input and output
using CSV, MLJ

weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
                   DataFrame)
input_standardizer = machine(Standardizer(), weather.LUZ_pressure[1:end-5])
fit!(input_standardizer, verbosity = 0)
weather_input = Float32.(MLJ.transform(input_standardizer,
                                       weather.LUZ_pressure[1:end-5]))
output_standardizer = machine(Standardizer(), log.(weather.LUZ_wind_peak[6:end]))
fit!(output_standardizer, verbosity = 0)
weather_output = Float32.(MLJ.transform(output_standardizer,
                                        log.(weather.LUZ_wind_peak[6:end])))

# Neural network
using Flux

nn = Chain(Dense(1, 50, tanh), Dense(50, 2)) # note that we define 2 output dims

# Training with stochastic gradient descent
opt = ADAMW()
p = Flux.params(nn) # these are the parameters to be adapted.
data = Flux.DataLoader((weather_input', weather_output), batchsize = 32)
for epoch in 1:20 # run for 20 epochs
    for batch in data # in each epoch we run through the whole dataset
        l, ∇ = Flux.withgradient(p) do
            loss(batch...)
        end
        Flux.update!(opt, p, ∇)
    end
end

# Plotting
grid = Float32.(collect(-5:.1:3.5))
pred = nn(grid')
m = pred[1, :]
s = sqrt.(softplus.(pred[2, :]))
u_x(x) = inverse_transform(input_standardizer, x) # function to "un-standardize"
u_y(y) = exp.(inverse_transform(output_standardizer, y))
histogram2d(u_x.(weather_input), u_y.(weather_output), bins = (250, 200),
            markersize = 3, xlabel = "LUZ_pressure [hPa]",
            ylabel = "LUZ_wind_peak [km/h]", label = nothing, cbar = false)
plot!(u_x.(grid), u_y.(m), c = :red, w = 3, label = "mean")
plot!(u_x.(grid), u_y.(m .+ s), c = :red, linestyle = :dash, w = 3,
      label = "± standard deviation")
plot!(u_x.(grid), u_y.(m .- s), c = :red, linestyle = :dash, w = 3,
      label = nothing)
"""
,
"""
"""
,
collapse = "Code for training and plotting",
showoutput = false
)

# ╔═╡ 23051a87-a4b2-49c2-984b-305f3d16a9f7
mlcode("plot!()", "", showinput = false)

# ╔═╡ 6bf6598b-590e-4794-b8de-b48cc3fd2c84
md"
We see that the neural network found a slightly non-linear relationship between
input and average output and the variance of the noise is smaller for high
pressure values than for low pressure values. Note that the dashed lines are not symmetric around the mean, because we fitted a log-normal distribution; the dashed lines would be symmetric around the mean if we plotted the logarithm of the wind peak on the y-axis.
"

# ╔═╡ 4bd63efb-0aef-41cd-a9b7-9fa78db107a0
md"""
## Fitting a Polynomial in 1000 Input Dimensions
The following artificial data set has ``n=10^5`` points of ``p = 1000`` dimensions.
The response ``y = x_7^2 - 1`` depends only on the seventh input dimension.
Although neural networks are not specifically designed to fit polynomials and this is a somewhat silly and artificial example,
this demonstrates that with only 50 hidden relu-neurons an accurate fit
can be found in this high-dimensional noisy dataset.
"""

# ╔═╡ ec0e41bb-0b13-4cb7-abfe-d52bc9e2389d
mlcode(
"""
# dataset
x = DataFrame(randn(Float32, 10^5, 1000), :auto)
y = x.x7 .^2 .- 1

# fitting
model = NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 50,
                                                       dropout = 0,
                                                       σ = relu),
                               batch_size = 128,
                               epochs = 10^2)
mach = machine(model, x, y)
fit!(mach, verbosity = 0)

# plotting
grid = -3:.1f0:3
x_test = DataFrame(randn(Float32, length(grid), 1000), :auto)
x_test.x7 = grid
plot(grid, grid.^2 .- 1, label = \"ground truth\", w = 3, legend = :bottomright)
plot!(grid, predict(mach, x_test),
      label = \"fit\", xlabel = \"x₇\", ylabel = \"y\", w = 3)
"""
,
"""
import matplotlib.pyplot as plt

# dataset
x = np.random.randn(10**5, 1000)
y = (x[:, 6] ** 2 - 1).reshape(-1, 1)

x, y = map(lambda x_y: torch.tensor(x_y, dtype=torch.float32), [x, y])

batch_size = 128
epochs = 10**2

train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y),
    batch_size = 128,
    shuffle = True,
    worker_init_fn = seed_worker, # for reproducibility
    generator = g # for reproducibility
)

# fitting
class NeuralNetExample4(torch.nn.Module):
    def __init__(self):
        super(NeuralNetExample4, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1000, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )
    def forward(self, x):
        return self.layers(x)


model = NeuralNetExample4()
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

model.train()
for epoch in tqdm.tqdm(range(epochs)):
    for data, target in train_dataloader:
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

grid = np.arange(-3.0, 3.0, 0.1)
x_test = np.random.randn(grid.shape[0], 1000)
x_test[:, 6] = grid
x_test = torch.tensor(x_test, dtype=torch.float32)

plt.plot(grid, (grid ** 2 - 1).reshape(-1, 1), label="ground truth")
plt.plot(grid, model(x_test).detach().numpy(), label="model prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
"""
)

# ╔═╡ a7657bf1-7036-4093-a6b8-0b3f8fb30d29
M.eval(:(using MLJFlux)) # hack to have @builder work in the next cell

# ╔═╡ 14a8eacb-352e-4c3b-8908-2a6f77ffc6fe
md"# 4. Classification with MLPs

## Fitting MNIST

The difference between regression and classification with neural networks is exactly the same as between linear and logistic regression: the only thing that changes is the negative log-likelihood loss function. In the case of regression it is based on the normal distribution and leads to the mean squared error applied to an output layer without any non-linearity, whereas for classification it is based on the categorical distribution and leads to the negative log-likelihood function applied to a softmax output layer.
"

# ╔═╡ 429b5335-c468-4f95-8351-f3652926d023
mlstring(
md"""
When using MLJ, we do not actually need to think about the softmax output layer.
Using the `NeuralNetworkClassifier` takes care of this automatically.

In the code snippet below we do not use the `MLJFlux.Short` builder, but use the generic `MLJFlux.@builder` where we can feed in any `Chain`. Note that the arguments `n_in` and `n_out` indicate that the builder should determine automatically the number of input and output neurons. In this case, we could use `MLJFlux.Short(n_hidden = 100, dropout = 0, σ = relu)` to get exactly the same result, but `MLJFlux.@builder` allows to build arbitrary neural networks, whereas `MLJFlux.Short` limits to neural networks with a single hidden layer.
"""
,
"""
"""
)

# ╔═╡ 5f8a6035-059d-4288-8007-e9176ed5c297
mlcode(
"""
using OpenML, DataFrames, MLJ, MLJFlux

# data loading
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    Float32.(df[:, 1:end-1] ./ 255),
    df.class
end

# fitting
model = NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                                          Dense(100, n_out))),
                         batch_size = 32,
                         epochs = 20)
mnist_mach = machine(model, mnist_x[1:60000, :], mnist_y[1:60000])
fit!(mnist_mach, verbosity = 0)
mean(predict_mode(mnist_mach, mnist_x[60001:70000, :]) .!= mnist_y[60001:70000])
"""
,
"""
import openml
from sklearn.preprocessing import OneHotEncoder

# data loading
mnist,_,_,_ = openml.datasets.get_dataset(554).get_data(dataset_format="dataframe")

# pre-processing
x = (mnist.iloc[:, :-1].values.astype(float)/255) # normalising
y = mnist.iloc[:, -1].values.astype(int)

one_hot_enc = OneHotEncoder()
y_ohe = one_hot_enc.fit_transform(y.reshape(-1, 1)) # one hot encoding
y_ohe = np.array(y_ohe.todense()) # fit_transform outputs a sparse matrix but it is easier to work wiith an array

x_train, y_train, x_test, y_test = x[:60000], y_ohe[:60000], x[60000:], y_ohe[60000:] # train-test split

# converting to pytorch tensors
x_train, y_train, x_test, y_test = map(
    lambda data: torch.tensor(data, dtype=torch.float32), 
    [x_train, y_train, x_test, y_test],
)

# Pytorch data loader, for using batches
train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size = batch_size,
    shuffle = True,
)

class MNISTLinearNN(torch.nn.Module):
    def __init__(self):
        super(MNISTLinearNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
        )
    def forward(self, x):
        return self.layers(x)

batch_size = 32
epochs = 20
model = MNISTLinearNN()
loss_func = torch.nn.CrossEntropyLoss() # used for classification
optimizer = torch.optim.AdamW(model.parameters())

model.train()
for epoch in tqdm.tqdm(range(epochs)):
    for data, target in train_dataloader:
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    model.eval()
    test_preds = one_hot_enc.inverse_transform(model(x_test).detach().numpy())
    test_gt = y[60000:].reshape(-1, 1)
    print(f'\nMissclassifictation rate: {np.mean((test_preds != test_gt))}')
"""
)

# ╔═╡ 4266ce96-d65a-42f9-bd44-ab6cbf7e92ea
md"
The result is a misclassification rate between 2% and 3%, which is better
than our first nearest-neigbor result. Additionally, prediction of the class
for new images is much faster than with nearest neighbor approaches, because the
information about the training data is stored in the weights of the neural network
and no explicit comparison with all training images is required.
"

# ╔═╡ 2f034d19-1c00-4a10-a880-99436ab00957
md"# Exercises

## Conceptual
#### Exercise 1
To get a feeling for the kind of functions of one predictor can be fitted with neural networks, we will draw ``y`` as a function of ``x`` for some values of the weights. It may be helpful to sketch this neural network with the input neuron, the hidden neurons and the output neuron and label the connections with the weights.
* Draw in the same figure ``a_1^{(1)} = g(w_{10}^{(1)} + w_{11}^{(1)} x)``, ``a_2^{(1)}=g(w_{20}^{(1)} + w_{21}^{(1)} x)`` and ``\bar y = w_0^{(2)} + w_1^{(2)}a_1^{(1)} +  w_2^{(2)}a_2^{(1)}`` as a function of ``x``. Use ``w_{10}^{(1)} = 0``, ``w_{11}^{(1)} = 1``, ``w_{20}^{(1)} = - 2``, ``w_{21}^{(1)} = 2``,  ``w_0^{(2)} = 1``, ``w_1^{(2)} = 2``, ``w_2^{(2)} = 1`` and use the rectified linear activation function ``g = \mbox{relu}``. At which ``x``-values does the slope change? Give the answer in terms of the weights.
* Draw a similar graph for ``w_{11}^{(1)} < 0`` and ``w_{10}^{(1)}/w_{11}^{(1)}<w_{20}^{(1)}/w_{21}^{(1)}``, e.g. with ``w_{10}^{(1)} = 0``, ``w_{11}^{(1)} = -1``, ``w_{20}^{(1)} = 2``, ``w_{21}^{(1)} = 2``,  ``w_0^{(2)} = 1``, ``w_1^{(2)} = 2``, ``w_2^{(2)} = 1``.
* How does the graph above change, if ``w_1^{(2)} = -2`` and ``w_2^{(2)} = -1``?
* Let us assume we add more neurons to the same hidden layer, i.e. we have ``a_1^{(1)}, \ldots, a_{d^{(1)}}^{(1)}`` activations in the first layer. How would the graph look differently in this case? Draw a sketch and describe in one sentence the qualitative difference.
* Let us assume we add instead more hidden layers with relu-activations. Would the graph of this neural network look qualitatively different from the ones we have drawn so far in this exercise?
* (optional) Show that a neural network with one hidden layer of 3 relu-neurons can perfectly fit any continuous piece-wise linear function of the form ``y = \left\{\begin{array}{ll} a_1 + b_1 x & x < c_1 \\ a_2 + b_2 x & c_1 \leq x < c_2 \\ a_3 + b_3 x & c_2 \leq x \end{array}\right. \hspace{1cm}  ``
                with ``c_1 =  \frac{a_1 - a_2}{b_2 - b_1} < c_2 = \frac{a_2 - a_3}{b_3 - b_2}``. Express ``a_1, a_2, a_3`` and ``b_1, b_2, b_3`` in terms of the network weights. There are multiple solutions; find one of them.
#### Exercise 2
Consider a neural network with two hidden layers: ``p = 4`` input units, ``2`` units in the first hidden layer, ``3`` units in the second hidden layer, and a single output.
- Draw the network with the input neurons, the hidden neurons, the output neuron and all connections.
- How many parameters are there?
- Assume the output of this network is the mean of a conditional normal distribution. Write the negative log-likelihood loss using matrix notation.
#### Exercise 3
Let us assume you want to predict the proportion ``y`` of citizens voting for one of two parties based on 113 different features that depend e.g. on the voting history, income distribution or demographics (see e.g. [here](https://www.ozy.com/news-and-politics/the-forecast-the-methodology-behind-our-2020-election-model/379778/) for an actual example). Because the proportion is a number between 0 and 1, it is reasonable to model it as a sample from a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) ``p(y|\alpha, \beta) = \frac{y^{\alpha - 1}(1-y)^{\beta - 1}}{B(\alpha, \beta)}``, where ``\alpha>0`` and ``\beta>0`` are shape parameters and ``B`` is the beta function. Let us assume you want to model the dependence of the shape parameters on ``x`` with a neural network.
* How many input units should this neural network have?
* How many output units should this neural network have?
* What kind of activation function would you choose for the output layer?
* Which loss function would you use to estimate the network parameters?
* What is the advantage of using a beta distribution instead of trying to predict directly the expected proportion with a neural network, which has a single sigmoid output unit?
#### Exercise 4
(Optional) Grant Sanderson has some beautiful [videos about neural networks](https://www.3blue1brown.com/topics/neural-networks). Have a look at them, if you are interested.

## Applied
#### Exercise 5
In this exercise our goal is to find a good machine learning model to predict the fat content of a meat sample on the basis of its near infrared absorbance spectrum. We use the Tecator data set `OpenML.describe_dataset(505)`. The first 100 columns of this data set contain measurements of near infrared absorbance at different frequencies for different pieces of meat. *Hint:* you can select all these columns based on name with `select(data, r\"absorbance\")` (The \"r\" in this command stands for Regex and it means that all columns with name containing the word \"absorbance\" should be selected). The column `:fat` contains the fat content of each piece of meat. You can either use a validation set approach with the first 172 data points for training and (cross-)validation and the rest of the data points as a test set or you can take a nested cross-validation approach (for the neural network this may take some time to run). Take our recipe for supervised learning (last slide of the presentation on \"Model Assessment and Hyperparameter Tuning\") as a guideline.
- Have a look at the raw data by e.g. checking if there are missing values and looking at a correlation plot.
- Fit some multiple linear regression models (with e.g. regularization constants tuned with cross-valdiation), compute the `rmse` of your best model on the test set and create a scatter plot that shows the actual fat content of the test data point versus the predicted fat content.
- Fit some neural network model (with 2 hyper-paramters of your choice tuned by cross-validation; warning: it may take quite some time to fit if you use a high number for `nfolds`), compute the `rmse` of your best model on the test set and create a scatter plot that shows the actual fat content of the test data point versus the predicted fat content.
    * *Hint 1:* standardization of input and output matters.
    * *Hint 2:* If you want to tune neural network parameters in a pipeline you can access them in the `range` function, for example, as `:(neural_network_regressor.builder.dropout)`.
"


# ╔═╡ 7c6951e8-726a-4181-b15f-b629a2835d03
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 825c40e9-de99-4829-aad0-0c5c901df5e9
begin
	struct Tracker{T} # structure definition with parametric type
		path::T
	end
	Tracker(x::T) where T = Tracker{Vector{T}}([copy(x)]) # constructor
	(t::Tracker)(x) = push!(t.path, copy(x))        # making the object callable

function MLCourse.embed_figure(name)
    "![](data:img/png; base64,
         $(open(MLCourse.base64encode,
                joinpath(Pkg.devdir(), "MLCourse", "notebooks", "figures", name)))))"
end
end;

# ╔═╡ 49c2b3a1-50c0-4fb9-a06b-2de7be216702
begin
    Random.seed!(Meta.parse(seed))
    θ₀ = .1 * randn(13)
    tracker = Tracker(θ₀)
    MLCourse.advanced_gradient_descent(M.log_reg_loss_function(xor_input, xor_target),
                              θ₀, T = 2*10^4,
                              callback = tracker)
    idxs = min.(max.(1:100, floor.(Int, (2e4^(1/100)).^(1:100))), 10^5)
    losses = M.log_reg_loss_function(xor_input, xor_target).(tracker.path[idxs])
end;

# ╔═╡ 16d0808e-4094-46ff-8f92-1ed21aa6191b
let idx = idxs[t], path = tracker.path
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
    scatter!([idxs[t]], [losses[t]], c = :red)
    plot(p1, p2, layout = (1, 2), size = (700, 400), legend = false)
end

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

# ╔═╡ 8c72c4b5-452d-4ba3-903b-866cac1c799d
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─c1cbcbf4-f9a9-4763-add8-c5ed7cbd2889
# ╟─c95ad296-5191-4f72-a7d7-87ddebc43a65
# ╟─a908176b-319a-445d-9f33-6271c2ef6149
# ╟─a2539e77-943e-464c-9ad3-c8542eab36ab
# ╟─235c6588-8210-4b8e-812c-537a72ce950f
# ╟─258a3459-3fef-4655-830c-3bdf11eb282d
# ╟─16d0808e-4094-46ff-8f92-1ed21aa6191b
# ╟─9aa58d2b-783a-4b74-ae36-f2ff0fb0be3f
# ╟─fde9639d-5f41-4037-ab7b-d3dbb09e8d3d
# ╟─49c2b3a1-50c0-4fb9-a06b-2de7be216702
# ╟─8c54bfb2-54c5-4777-ad53-97926da97f7e
# ╟─9661f274-180f-4e97-90ce-8beb7d1d69bc
# ╟─5c2154ef-2a35-4d91-959d-f72100049894
# ╟─71dabb22-816b-4aa2-9a29-f2894989b2ea
# ╟─653ea4e8-47b2-4d9f-bc5c-583468d6e1ae
# ╟─1171bcb8-a2af-49f3-9a7e-2e7cac34f9f4
# ╟─1c70c3c4-bded-44ec-ac40-c71268b7307e
# ╟─66bc40f3-7710-4892-ae9f-79b9c850b0ea
# ╟─15cd82d3-c255-4eb5-9f0d-1e662c488027
# ╟─9e6ad99f-8cd4-4372-a4a2-f280e49e21a3
# ╟─9d3e7643-34fd-40ee-b442-9d2a434f30e0
# ╠═4bbae793-059d-4741-a761-c38be74b274c
# ╟─114201bf-d841-496f-8ca5-2fbd43726995
# ╠═6944c24e-67e7-47ba-aea6-5f75f58dae5a
# ╠═03b2c815-65ef-4c74-a673-21830dbcb40e
# ╟─631d3ab5-817a-4ccf-88fb-950f85fc6be3
# ╟─bf6bf8f3-dbc9-496a-ac5b-1aaf6e7824e1
# ╟─a869d11a-2124-4d2b-8517-5cca4382309e
# ╟─6393aae7-a8ac-4dff-b188-5e682f6ab1f9
# ╠═a5568dfb-4467-4dca-b1c7-714bbfcbd6e6
# ╟─47024142-aabb-40d8-8336-5f8aeb2aa623
# ╟─0a48241c-2be0-46cf-80ae-dc86abefad6f
# ╟─0ba71a2a-f2f9-4fc7-aa81-416799e79e57
# ╟─3d6ae9f2-105a-4c7f-bf2f-c42afe2683cf
# ╟─67d386dc-18c8-4dbe-8e13-cfc8541e8b6c
# ╟─6701f32c-015e-45f5-a59b-4d5bf9412f99
# ╟─23051a87-a4b2-49c2-984b-305f3d16a9f7
# ╟─6bf6598b-590e-4794-b8de-b48cc3fd2c84
# ╟─4bd63efb-0aef-41cd-a9b7-9fa78db107a0
# ╟─ec0e41bb-0b13-4cb7-abfe-d52bc9e2389d
# ╟─a7657bf1-7036-4093-a6b8-0b3f8fb30d29
# ╟─14a8eacb-352e-4c3b-8908-2a6f77ffc6fe
# ╟─429b5335-c468-4f95-8351-f3652926d023
# ╟─5f8a6035-059d-4288-8007-e9176ed5c297
# ╟─4266ce96-d65a-42f9-bd44-ab6cbf7e92ea
# ╟─2f034d19-1c00-4a10-a880-99436ab00957
# ╟─7c6951e8-726a-4181-b15f-b629a2835d03
# ╟─825c40e9-de99-4829-aad0-0c5c901df5e9
# ╟─83e2c454-042f-11ec-32f7-d9b38eeb7769
# ╟─8c72c4b5-452d-4ba3-903b-866cac1c799d
# ╟─87f59dc7-5149-4eb6-9d81-440ee8cecd72
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
