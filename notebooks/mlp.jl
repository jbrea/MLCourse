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

# ╔═╡ 87f59dc7-5149-4eb6-9d81-440ee8cecd72
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using PlutoUI, Plots, MLJ, DataFrames, Random, CSV, Flux, Distributions,
          StatsPlots, MLJFlux, OpenML, Random
    gr()
    PlutoUI.TableOfContents()
end

# ╔═╡ 83e2c454-042f-11ec-32f7-d9b38eeb7769
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ c95ad296-5191-4f72-a7d7-87ddebc43a65
md"# Solving the XOR Problem with Learned Feature Vectors"

# ╔═╡ fde9639d-5f41-4037-ab7b-d3dbb09e8d3d
function xor_generator(; n = 200)
	x = 2 * rand(n, 2) .- 1
	DataFrame(X1 = x[:, 1], X2 = x[:, 2],
		      y = (x[:, 1] .> 0) .⊻ (x[:, 2] .> 0))
end

# ╔═╡ ea7fc5b6-79cc-4cc5-820e-5c55da83207e
begin
    xor_data = xor_generator()
    xor_input = Array(select(xor_data, Not(:y)))
    xor_target = xor_data.y
end;

# ╔═╡ a2539e77-943e-464c-9ad3-c8542eab36ab
begin
    f(x, w₁, w₂, w₃, w₄, β) = β[1] * relu.(x * w₁) .+
                              β[2] * relu.(x * w₂) .+
                              β[3] * relu.(x * w₃) .+
                              β[4] * relu.(x * w₄) .+
                              β[5]
    f(x, θ) = f(x, θ[1:2], θ[3:4], θ[5:6], θ[7:8], θ[9:14])
    function log_reg_loss(θ)
        p = σ.(f(xor_input, θ))
        -mean(@. xor_target * log(p) + (1 - xor_target) * log(1 - p))
    end
end

# ╔═╡ 258a3459-3fef-4655-830c-3bdf11eb282d
md"seed = $(@bind(seed, Select([string(i) for i in 1:20])))

t = $(@bind t Slider(1:10^2))
"

# ╔═╡ 49c2b3a1-50c0-4fb9-a06b-2de7be216702
begin
    Random.seed!(Meta.parse(seed))
    θ₀ = .1 * randn(14)
    path = [copy(θ₀)]
    MLCourse.gradient_descent(log_reg_loss, θ₀, .05, 10^4,
                              callback = x -> push!(path, copy(x)))
end;

# ╔═╡ b650ddb4-5854-4862-ba26-0bf92f77c3df
begin
    idxs = min.(max.(1:100, floor.(Int, 1.097.^(1:100))), 10^4)
    losses = log_reg_loss.(path[idxs])
end;

# ╔═╡ 16d0808e-4094-46ff-8f92-1ed21aa6191b
let idx = idxs[t]
    θ = path[idx]
    prediction = σ.(f(xor_input, θ)) .> .5
    p1 = scatter(xor_data.X1, xor_data.X2, c = prediction .+ 1,
                 xlim = (-1.5, 1.5), ylim = (-1.5, 1.5),
                 xlabel = "X1", ylabel = "X2")
    for i in 1:4
        plot!([0, θ[i*2 - 1]], [0, θ[i*2]], c = :green, w = 3, arrow = true)
    end
    hline!([0], c = :black)
    vline!([0], c = :black)
    p2 = plot(idxs, losses, ylim = (0, 1), xscale = :log10, xlabel = "t",
              ylabel = "loss")
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

Because it takes a bit of time to run the code, it is included here in text form.
Copy-paste it to a cell to run it.
```julia
begin
    using MLJFlux
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
                            xlabel = \"number of hidden units\",
                            ylabel = \"accuracy\",
                            label = \"training\",
                            yrange = (.5, 1.01),
                            aspect_ratio = 20,
                            legend = :bottomright,
                            xticks = [4, 10])
    @df xor_results dotplot!(:n_hidden, :test_accuracy, label = \"test\")
end
```
$(MLCourse.embed_figure("overparametrized.png"))
")

# ╔═╡ 5c2154ef-2a35-4d91-959d-f72100049894
md"# Multilayer Perceptrons

Below you see three equivalent ways to implement a neural network:
1. In the `expanded` form all parameters and activation functions are explicitely written out in \"one line\". With more neurons or hidden layers, this form quickly becomes annoying to write.
2. In the `neuron` form we see clearly that the neural network is composed of multiple simple neurons with different input weights and biases.
3. The `matrix` form is the most succinct form and most commonly used in modern implementations. In this form the biases of each layer are organized in a vector and the input weights of all neurons in the same layer are organized in a matrix.
"


# ╔═╡ 71dabb22-816b-4aa2-9a29-f2894989b2ea
begin
    function neural_net_expanded(x,
                                 w₁₀¹, w₁₁¹, w₂₀¹, w₂₁¹, w₃₀¹, w₃₁¹, w₄₀¹, w₄₁¹, g¹,
                                 w₁₀², w₁₁², w₁₂², w₁₃², w₁₄², g²)
        g²(w₁₀² +
           w₁₁² * g¹(w₁₀¹ + w₁₁¹ * x) +
           w₁₂² * g¹(w₂₀¹ + w₂₁¹ * x) +
           w₁₃² * g¹(w₃₀¹ + w₃₁¹ * x) +
           w₁₄² * g¹(w₄₀¹ + w₄₁¹ * x))
    end
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
    function neural_net_matrix(x, b¹, w¹, g¹, b², w², g²)
        g².(b² .+ w² * g¹.(b¹ .+ w¹ * x))
    end
end;


# ╔═╡ 9e6ad99f-8cd4-4372-a4a2-f280e49e21a3
begin
    heaviside(x) = x > 0 ? 1. : 0.;
    function tocol(y)
        ma, mi = maximum(y), minimum(y)
        palette(:jet1, 101)[min.(101, max.(1, reshape(floor.(Int, (y .+ 6) * 5) .+ 1, :)))]
    end
end;

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
let
    x = -5:.01:5
    _g1 = getproperty(@__MODULE__, Symbol(g1))
    _g2 = getproperty(@__MODULE__, Symbol(g2))
    y = neural_net_expanded.(x, w101, w111, w201, w211, w301, w311, w401, w411, _g1, w102, w112, w122, w132, w142, _g2)
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

# ╔═╡ 66bc40f3-7710-4892-ae9f-79b9c850b0ea
let x = -5:.1:5
    grid = MLCourse.grid(x, x, output_format = DataFrame)
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
    y = neural_net_matrix(X, b1, w1, _g1, b2, w2, _g2)
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
Markdown.parse("# Regression with MLPs

## Fitting the Weather Data

Let us find out if with an MLP we can get better predictions than with multiple
linear regression on the weather data set.

The following code takes roughly 2 minutes to run.

```julia
weather = CSV.read(joinpath(@__DIR__, \"..\", \"data\", \"weather2015-2018.csv\"),
                   DataFrame)
weather_test = CSV.read(joinpath(@__DIR__, \"..\", \"data\", \"weather2019-2020.csv\"),
                        DataFrame)
weather_x = float.(select(weather, Not([:LUZ_wind_peak, :time]))[1:end-5, :])
weather_y = weather.LUZ_wind_peak[6:end]
weather_test_x = float.(select(weather_test, Not([:LUZ_wind_peak, :time]))[1:end-5, :])
weather_test_y = weather_test.LUZ_wind_peak[6:end]

Random.seed!(31)
mach = machine(@pipeline(Standardizer(),
                         NeuralNetworkRegressor(
                             builder = MLJFlux.Short(n_hidden = 128,
                                                     dropout = .5,
                                                     σ = relu),
                             optimiser = ADAMW(),
                             batch_size = 128,
                             epochs = 150),
                         target = Standardizer()),
               weather_x, weather_y) |> x -> fit!(x, verbosity = 2)
```

!!! note

    We use here an `MLJ.@pipeline` that standardizes with a `Standardizer`
    the input first to mean zero and standard deviation one. The same standardization
    is applied to the target values. We do this, because the standard initializations
    of neural networks work best with standardized input and output.

!!! note

    Our builder constructs a neural network with one hidden layer of 128 relu neurons.
    In the construction we specify that we would like to train with 50% dropout
    to reduce the risk of overfitting.

Let us evalute the learned machine.

```julia
(training_error = rmse(predict(mach, weather_x), weather_y),
 test_error = rmse(predict(mach, weather_test_x), weather_test_y))
```
We find a training error of 7.20 and a test error of 8.66.
Both errors are lower than what we found with multiple linear regression
(training error ≈ 8.09, test_error ≈ 8.91). However, comparing the predictions
to the ground truth we see that a lot of the variability is still uncaptured
by the model.
```julia
scatter(predict(mach, weather_test_x), weather_test_y,
        xlabel = \"predicted\", ylabel = \"data\")
plot!(identity, legend = false)
```
$(MLCourse.embed_figure("weather_mlp.png"))
")

# ╔═╡ 0ba71a2a-f2f9-4fc7-aa81-416799e79e57
Markdown.parse("## Parametrizing Variances of Log-Normal Distributions with MLPs

For this example we do not use the abstractions provided by MLJ. Instead we use
directly Flux.

Let us load the data.
```julia
weather = CSV.read(joinpath(@__DIR__, \"..\", \"data\", \"weather2015-2018.csv\"),
                   DataFrame)
standardize(x) = (x .- mean(x)) ./ std(x)
weather_input = Float32.(standardize(weather.LUZ_pressure[1:end-5])')
weather_output = Float32.(standardize(log.(weather.LUZ_wind_peak[6:end])))
```

We define a simple neural network with one hidden layer of 50 tanh neurons.
```julia
nn = Chain(Dense(1, 50, tanh), Dense(50, 2))
```

Let us now define the negative log-likelihood loss function.
```julia
function loss(x, y)
    output = nn(x)
    m = output[1, :]
    s = softplus.(output[2, :])
    mean((m .- y) .^ 2 ./ (2 * s) .+ log.(s))
end
```
This function computes the two output dimensions of the neural network.
The first output dimension is our mean. The second output dimension is transformed
to positive values with the `softplus` function ``x \\mapsto \\log(\\exp(x) + 1)``
to enode the variance of the normal distribution. In the log-likelihood loss we
drop the constant term ``\\frac12\\log2\\pi``.

Now we train the neural network for 50 epochs.
```julia
opt = ADAMW()
p = Flux.params(nn) # these are the parameters to be adapted.
data = Flux.DataLoader((weather_input, weather_output), batchsize = 32)
for _ in 1:50
    Flux.Optimise.train!(loss, p, data, opt)
end
```

Let us now plot the result.
```julia
grid = collect(-5:.1:3.5)'
pred = nn(grid)
m = pred[1, :]
s = sqrt.(softplus.(pred[2, :]))
histogram2d(weather_input', weather_output, bins = (250, 200),
            markersize = 3, xlabel = \"LUZ_pressure [hPa]\",
            ylabel = \"LUZ_wind_peak [km/h]\", label = nothing, cbar = false)
plot!(grid', m, c = :red, w = 3, label = \"mean\")
plot!(grid', m .+ s, c = :red, linestyle = :dash, w = 3, label = \"± standard deviation\")
plot!(grid', m .- s, c = :red, linestyle = :dash, w = 3, label = nothing)
```

$(MLCourse.embed_figure("weather_mlp_normal.png"))

We see that the neural network found a slightly non-linear relationship between
input and average output and the variance of the noise is smaller for high
pressure values than for low pressure values.
")

# ╔═╡ 4bd63efb-0aef-41cd-a9b7-9fa78db107a0
Markdown.parse("
## Fitting a Polynomial in 1000 Input Dimensions
The following artificial data set has ``n=10^5`` points of ``p = 1000`` dimensions.
The response ``y = x_7^2 - 1`` depends only on the seventh input dimension.
Although neural networks are not specifically designed to fit polynomials,
this example demonstrates that with only 50 hidden relu-neurons an accurate fit
can be found.

WARNING: it takes more than 10 minutes to run this code.
```julia
let
    x = DataFrame(randn(10^5, 1000), :auto)
    y = x.x7 .^2 .- 1
    mach = fit!(machine(NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 50,
                                                                       dropout = 0,
                                                                       σ = relu),
                                               batch_size = 128,
                                               epochs = 10^2),
                        x, y), verbosity = 2)
    grid = -3:.1:3
    x_test = DataFrame(randn(length(grid), 1000), :auto)
    x_test.x7 = grid
    plot(grid, grid.^2 .- 1, label = \"ground truth\", w = 3, legend = :bottomright)
    plot!(grid, predict(mach, x_test),
          label = \"fit\", xlabel = \"x₇\", ylabel = \"y\", w = 3)
end
```

$(MLCourse.embed_figure("poly.png"))
")

# ╔═╡ 14a8eacb-352e-4c3b-8908-2a6f77ffc6fe
md"# Classification with MLPs

## Fitting MNIST

The following code to fit the MNIST data with a multilayer perceptron takes a
few minutes to run.

```julia
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    Float32.(df[:, 1:end-1] ./ 255),
    df.class
end
mnist_mach = machine(NeuralNetworkClassifier(
                         builder = MLJFlux.@builder(Chain(Dense(n_in, 100, relu),
                                                          Dense(100, n_out))),
                         batch_size = 32,
                         epochs = 20),
                      mnist_x[1:60000, :],
                      mnist_y[1:60000])
fit!(mnist_mach, verbosity = 2)
mean(predict_mode(mnist_mach, mnist_x[60001:70000, :]) .!= mnist_y[60001:70000])
```
The result is a misclassification rate of approximately 2.2%, which is better
than our first nearest-neigbor result. Additionally, prediction of the class
for new images is much faster than with nearest neighbor approaches, because the
information about the training data is stored in the weights of the neural network
and no explicit comparison with all training images is required.
"

# ╔═╡ 2f034d19-1c00-4a10-a880-99436ab00957
md"# Exercises

## Conceptual
1. To get a feeling for the kind of functions that can be fitted with neural networks, we will draw ``y`` as a function of ``x`` for some values of the weights. It may be helpful to sketch this neural network with the input neuron, the hidden neurons and the output neuron and label the connections with the weights.
    * Draw in the same figure ``a_1^{(1)} = g(w_{10}^{(1)} + w_{11}^{(1)} x)``, ``a_2^{(1)}=g(w_{20}^{(1)} + w_{21}^{(1)} x)`` and ``\bar y = w_0^{(2)} + w_1^{(2)}a_1^{(1)} +  w_2^{(2)}a_2^{(1)}`` as a function of ``x``. Use ``w_{10}^{(1)} = 0``, ``w_{11}^{(1)} = 1``, ``w_{20}^{(1)} = - 2``, ``w_{21}^{(1)} = 2``,  ``w_0^{(2)} = 1``, ``w_1^{(2)} = 2``, ``w_2^{(2)} = 1`` and use the rectified linear activation function ``g = \mbox{relu}``. At which ``x``-values does the slope change? Give the answer in terms of the weights.
    * Draw a similar graph for ``w_{11}^{(1)} < 0`` and ``w_{10}^{(1)}/w_{11}^{(1)}<w_{20}^{(1)}/w_{21}^{(1)}``, e.g. with ``w_{10}^{(1)} = 0``, ``w_{11}^{(1)} = -1``, ``w_{20}^{(1)} = 2``, ``w_{21}^{(1)} = 2``,  ``w_0^{(2)} = 1``, ``w_1^{(2)} = 2``, ``w_2^{(2)} = 1``.
    * How does the graph above change, if ``w_1^{(2)}`` and ``w_2^{(2)}`` become negative?
    * Let us assume we add more neurons to the same hidden layer, i.e. we have ``a_1^{(1)}, \ldots, a_{d^{(1)}}^{(1)}`` activations in the first layer. How would the graph look differently in this case? Draw a sketch and describe in one sentence the qualitative difference.
    * Let us assume we add instead more hidden layers with relu-activations. Would the graph of this neural network look qualitatively different from the ones we have drawn so far in this exercise?
    * Show that a neural network with one hidden layer of 3 relu-neurons can perfectly fit any continuous piece-wise linear function of the form ``y = \left\{\begin{array}{ll} a_1 + b_1 x & x < c_1 \\ a_2 + b_2 x & c_1 \leq x < c_2 \\ a_3 + b_3 x & c_2 \leq x \end{array}\right. \hspace{1cm}  ``
                    with ``c_1 =  \frac{a_1 - a_2}{b_2 - b_1} < c_2 = \frac{a_2 - a_3}{b_3 - b_2}``. Express ``a_1, a_2, a_3`` and ``b_1, b_2, b_3`` in terms of the network weights. There are multiple solutions; find one of them.
1. Consider a neural network with two hidden layers: ``p = 4`` input units, ``2`` units in the first hidden layer, ``3`` units in the second hidden layer, and a single output.
    - Draw a picture of the network.
    - How many parameters are there?
    - Assume the output of this network is the mean of a conditional normal distribution. Write the negative log-likelihood loss using matrix notation.
1. Derive the loss function we used in section \"Parametrizing Variances of Log-Normal Distributions with MLPs\" above.

## Applied
1. In this exercise our goal is to find a good machine learning model to classify images of Zalando's articles. You can load a description of the so-called Fashion-MNIST data set with `OpenML.describe_dataset(40996)` and load the data set with `OpenML.load(40996)`. Take our recipe for supervised learning (last slide of the presentation on \"Model Assessment and Hyperparameter Tuning\") as a guideline. Hints: cleaning is not necessary, but plotting some examples is advisable; linear classification is a good starting point for a first benchmark, but you should also explore other models and tune their hyper-parameters; if you are impatient: choose smaller training sets.
2. In this exercise our goal is to find a good machine learning model to predict the fat content of a meat sample on the basis of its near infrared absorbance spectrum. We use the Tecator data set `OpenML.describe_dataset(505)`.
"

# ╔═╡ 8c72c4b5-452d-4ba3-903b-866cac1c799d
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─c95ad296-5191-4f72-a7d7-87ddebc43a65
# ╠═fde9639d-5f41-4037-ab7b-d3dbb09e8d3d
# ╠═ea7fc5b6-79cc-4cc5-820e-5c55da83207e
# ╠═a2539e77-943e-464c-9ad3-c8542eab36ab
# ╠═49c2b3a1-50c0-4fb9-a06b-2de7be216702
# ╟─258a3459-3fef-4655-830c-3bdf11eb282d
# ╟─16d0808e-4094-46ff-8f92-1ed21aa6191b
# ╟─b650ddb4-5854-4862-ba26-0bf92f77c3df
# ╟─8c54bfb2-54c5-4777-ad53-97926da97f7e
# ╟─5c2154ef-2a35-4d91-959d-f72100049894
# ╠═71dabb22-816b-4aa2-9a29-f2894989b2ea
# ╟─653ea4e8-47b2-4d9f-bc5c-583468d6e1ae
# ╟─1171bcb8-a2af-49f3-9a7e-2e7cac34f9f4
# ╟─1c70c3c4-bded-44ec-ac40-c71268b7307e
# ╟─66bc40f3-7710-4892-ae9f-79b9c850b0ea
# ╟─9e6ad99f-8cd4-4372-a4a2-f280e49e21a3
# ╟─9d3e7643-34fd-40ee-b442-9d2a434f30e0
# ╟─0ba71a2a-f2f9-4fc7-aa81-416799e79e57
# ╟─4bd63efb-0aef-41cd-a9b7-9fa78db107a0
# ╟─14a8eacb-352e-4c3b-8908-2a6f77ffc6fe
# ╟─2f034d19-1c00-4a10-a880-99436ab00957
# ╟─83e2c454-042f-11ec-32f7-d9b38eeb7769
# ╟─87f59dc7-5149-4eb6-9d81-440ee8cecd72
# ╟─8c72c4b5-452d-4ba3-903b-866cac1c799d
