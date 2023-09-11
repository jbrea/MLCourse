### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ a022be72-b0b0-43b7-8f39-c02a53875ff6
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, Flux, MLJXGBoostInterface, MLJDecisionTreeInterface, MLJLIBSVMInterface
import PlutoPlotly as PP
const M = MLCourse.JlMod
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ 64bac944-8008-498d-a89b-b9f8ee54aa98
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 184c3e36-a3d0-4554-a5e4-88996510f329
md"The goal of this week is to
1. Understand how neural networks can be used in creative ways to solve image and text classification problems.
2. Unterstand tree-based methods.
3. See an example of a support vector classifier.
4. Learn how to use Random Forests and Gradient Boosting.
"

# ╔═╡ 7cfd274d-75d8-4cd9-b38b-4176466c9e26
md"""# 1. Other Neural Network Architectures

## Fitting MNIST with a Convolutional Neural Networks

In the following we fit a convolutional neural network to the MNIST data set.

"""


# ╔═╡ 995fb5d1-4083-491f-aed7-f1cc127eb897
mlcode(
"""
using MLJ, Plots, MLJFlux, Flux, OpenML, DataFrames

mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    Float32.(df[:, 1:end-1] ./ 255), # scaling the input
    df.class
end

images = coerce(PermutedDimsArray(reshape(Array(mnist_x), :, 28, 28), (3, 2, 1)),
                GrayImage);
plot(images[1])
"""
,
"""
"""
)

# ╔═╡ 2068970a-4aa7-4580-a137-3e38460ba0a2
md"
The `front` consists of two convolutional layers. The volume of the second
convolutional layer is flattened and taken as input to a single dense layer.
"

# ╔═╡ eeecfe16-631f-4eb0-a7d6-581d8b49aa9c
mlcode(
"""
builder = MLJFlux.@builder(
              begin
                  front = Chain(Conv((8, 8), n_channels => 16, relu),
                                Conv((4, 4), 16 => 32, relu),
                                Flux.flatten)
                  d = first(Flux.outputsize(front, (n_in..., n_channels, 1)))
                  Chain(front, Dense(d, n_out))
              end)
m = machine(ImageClassifier(builder = builder,
                            batch_size = 32,
                            epochs = 5),
            images, mnist_y)
fit!(m, rows = 1:60000, verbosity = 0)

# error rate
mean(predict_mode(m, images[60001:70000]) .!= mnist_y[60001:70000])

"""
,
"""
"""
)

# ╔═╡ 69ac3eec-bc07-414b-b66c-1e9cd45821fe
md"""
The result is a misclassification rate of 1.4% which is again a strong improvement
over the MLP result. Note that we achieved this value without tuning of the hyper-parameters. Well-tuned convolutional networks achieve performances below 1% error.

If you are interested in learning more about convolutional neural networks I can highly recommend to have a look at one of the excellent tutorials on the internet, like [this](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1), [this](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) or [this](https://distill.pub/2017/feature-visualization/) one.
"""

# ╔═╡ dd12ea16-f087-4507-a571-c95bf38aa976
md"## Language Detection with a Recurrent Neural Network

The following code (adapted from [here](https://github.com/FluxML/model-zoo/blob/master/text/lang-detection/model.jl)) uses wikipedia articles to detect the language of a sentence
by simply reading each character of the sentence with a recurrent neural network
and applying a classifier to the last hidden state.
"

# ╔═╡ c76ab462-333f-4852-b0bf-2d7d603c34ab
mlcode(
"""
using Unicode, Random
using Flux: onehot, onehotbatch, logitcrossentropy, reset!, throttle

function load_corpora(; langs = ["es", "en", "da", "it", "fr"])
    corpora = Dict()
    for lang in langs
        url = "https://go.epfl.ch/bio322-" * lang * ".txt"
        corpus = split(read(download(url), String), \".\")
        corpus = strip.(Unicode.normalize.(corpus, casefold=true, stripmark=true))
        corpus = filter(!isempty, corpus)
        corpora[lang] = corpus
    end
    corpora
end
function get_processed_data(corpora; test_len = 100)
    vocabulary = ['a':'z'; '0':'9'; ' '; '\\n'; '_']
    langs = collect(keys(corpora))
    # See which chars will be represented as \"unknown\"
    unknown_chars = unique(filter(x -> x ∉ vocabulary,
                                  join(vcat(values(corpora)...))))
    dataset = [(onehotbatch(s, vocabulary, '_'), onehot(l, langs))
               for l in langs for s in corpora[l]] |> shuffle
    train, test = dataset[1:end-test_len], dataset[end-test_len+1:end]
    return train, test, unknown_chars
end

Random.seed!(171)

corpora = load_corpora()
train_data, test_data, unknown_chars = get_processed_data(corpora)
"""
,
"""
"""
,
showoutput = false,
collapse = "Data Loading and Preprocessing"
)

# ╔═╡ c769c35c-ea59-4d4a-8451-a9343fc75d88
md"We downloaded a few pages from wikipedia in 5 different languages, applied some normalization and split the text into sentences. Here is the result:
"

# ╔═╡ cee63cef-6ce3-4738-941d-632b2d43f647
mlcode(
"""
corpora
"""
,
"""
"""
,
showinput = false
)

# ╔═╡ 736ffb3d-64a7-43f6-b781-66cf2540bb42
md"These raw sentences are encoded on a character level into a one-hot code with the use of a vocabulary. In our case, the vocabulary consists of characters 'a' to 'z', numbers '0' to '9', white space ' ' and newline '\n' and the placeholder '_' for all characters that are not in this set. Our vocabulary consists of 39 elements. Therefore, a sentence of ``K`` characters is represented as a matrix of size ``39\times K``, as you can see in the following examples, where we encode, first, one word and, second, a full sentence:
"

# ╔═╡ 789b6c31-ae1d-49b2-8fac-365e6517a3cc
mlcode(
"""
examples = let vocabulary = ['a':'z'; '0':'9'; ' '; '\n'; '_']
Dict(corpora["fr"][1] => onehotbatch(corpora["fr"][1], vocabulary, '_'),
corpora["fr"][23] => onehotbatch(corpora["fr"][23], vocabulary, '_'))
end
"""
,
"""
"""
,
showinput = false
)

# ╔═╡ 75aed643-0185-4fe4-abdf-02634865404e
md"""
Note that text can also be encoded in other ways. For example, one can encode sentences on the word level instead of the character level. The text encoding process is usually called "tokenization" and is an important step when working with text. So for example [here](https://www.freecodecamp.org/news/evolution-of-tokenization/) for advantages and disadvantages of different tokenizers (optional).

Now we actually build the model.
It consists of a recurrent neural network (scanner) that takes sentences of different lenghts as input and a classifier that takes the last state of the scanner as input.
"""

# ╔═╡ b9ca66df-f416-40a6-95d2-5012a73a62ff
mlcode(
"""
function build_model(; vocabulary_len, langs_len, N)
    # The scanner is the recurrent neural network; it uses a special kind of
    # neuron in the recurrent layer, called LSTM.
    scanner = Chain(Dense(vocabulary_len, N, relu), LSTM(N, N))
    # The classifier takes the last element of the sequence and computes the
    # log-probability of each language for the given sequence using a standard
    # multi-layer perceptron (MLP) with one hidden layer.
    classifier = Chain(Dense(N, N, relu), Dense(N, langs_len))
    return scanner, classifier
end

function model(x, scanner, classifier)
    # scan all characters of sentence x
    state = mapslices(scanner, x, dims = 1)[:, end]
    reset!(scanner)
    # take the last state of the neural network
    # to compute the probability of each language.
    classifier(state)
end

scanner, classifier = build_model(vocabulary_len = 39, langs_len = 5, N = 64)
"""
,
"""
"""
,
cache = false
)

# ╔═╡ e71921f8-e98d-46b3-b35d-0119e614f6c5
md"Below you see the result of applying the scanner to two sentences in the training set of different lenghts. The plotted scanner activity is the output of the 64 neurons of the recurrent network as characters are fed-in one by one. Because the scanner is a recurrent neural network, the activity of those neurons depends on the already shown characters and the current one. Therefore, the last activity state contains information about the full sequence and therefore can be used for language classification.
"

# ╔═╡ 4bc9a437-d9a9-4543-ae3a-f41c208da351
mlcode(
"""
let ks = collect(keys(examples))
states = mapslices(scanner, examples[ks[1]], dims = 1)
p1 = heatmap(states, title = "word \\"wikipedia\\"", ylabel = "scanner activity", xlabel = "character number")
states = mapslices(scanner, examples[ks[2]], dims = 1)
p2 = heatmap(states, title = "example sentence", ylabel = "scanner activity", xlabel = "character number")
plot(p1, p2, layout = (1, 2), size = (700, 400))
end
"""
,
"""
"""
,
showinput = false,
)


# ╔═╡ c51b2449-b1b5-4e91-a86b-0cd1e645b16b
md"The important point to realize here is that also this complicated example follows actually a simple pattern:
1. Encode input data in a useful way. Here: character-level tokenization
2. Parametrize the conditional probability distribution. Here: recurrent neural network with scanner and classifier.
3. Define the loss function: Here: negative log-likelihood loss.
4. Run stochastic gradient descent with early stopping.

Points 3 and 4 is what we do in the next code block.
"

# ╔═╡ 06b27480-1dea-456a-94b3-9eef1a5b1b71
mlcode(
"""
using Flux, Random

Random.seed!(123)

scanner, classifier = build_model(vocabulary_len = 39, langs_len = 5, N = 64)

# we use the negative loglikelihood loss (here called logitcrossentropy)
loss(x, y) = logitcrossentropy(model(x, scanner, classifier), y)
function evaluate(scanner, classifier, dataset)
    () -> begin
        tmp = [(model(x, scanner, classifier), y) for (x, y) in dataset]
        (loss = mean(logitcrossentropy(ŷ, y) for (ŷ, y) in tmp),
         accuracy = mean(argmax(ŷ) == findfirst(y) for (ŷ, y) in tmp))
    end
end
traineval = evaluate(scanner, classifier, train_data)
testeval = evaluate(scanner, classifier, test_data)

opt = ADAM(1e-3)
ps = Flux.params(scanner, classifier)
training_curve = [traineval()]
validation_curve = [testeval()]

for epoch in 1:20
    Flux.train!(loss, ps, train_data, opt)
    push!(training_curve, traineval())
    push!(validation_curve, testeval())
end

"""
,
"""
"""
,
eval = true,
collapse = "Code for Training"
)

# ╔═╡ feeb8006-abef-4b46-a4da-0cab8a025601
mlcode(
"""
p1 = plot(first.(training_curve), label = "training",
          title = "loss", ylabel = "loss", xlabel = "epoch")
plot!(first.(validation_curve), label = "validation")
p2 = plot(last.(training_curve), label = "training",
          title = "accuracy", ylabel = "accuracy", xlabel = "epoch")
plot!(last.(validation_curve), label = "validation")
plot(p1, p2, layout = (1, 2), size = (700, 400))
"""
,
"""
"""
,
showinput = false
)

# ╔═╡ d0891e35-3aaa-4cbe-884b-27a5a2980526
md"""
Training has not yet finished, as we can see by looking at the learning curves, but we reach already a validation accuracy of more than 60%, which is much bettern than the chance level of 20% (randomly guessing one of five languages). Note, however, that an accuracy of 100% is probably not reachable on this dataset, as it consists also single words that are the same in different languages (like "wikipedia").
"""

# ╔═╡ e68603b3-6f95-4c55-aded-14eaa7615176
md"# 2. Tree-Based Methods

Despite the popularity of neural networks, tree-based methods are still widely used, for good reasons: they are fast to fit, and therefore fast to tune, and often lead to very good performance.

## Decision Trees

Decision trees split the input space into different regions.

### Example: Decision Tree for Regression
Here we show the decision tree for a 1D regression task once in its if-else form and once as a plot. In the if-else form, we see that the input space is always split along `Feature 1` (there is no other features in a 1D task ;)).
"

# ╔═╡ 4b4714f7-49bf-4191-a59a-79fa1c826519
begin
function data_generator()
    x = 8*rand(40) .- 4
    y = sin.(x) .+ .3*randn(40)
    DataFrame(x = x, y = y)
end
training_set = data_generator()
tree = machine(DecisionTreeRegressor(min_samples_leaf = 4),
               select(training_set, Not(:y)),
               training_set.y)
fit!(tree, verbosity = 0)
MLJDecisionTreeInterface.DecisionTree.print_tree(fitted_params(tree).raw_tree, 5)
end

# ╔═╡ 16fef29a-753f-44b0-89af-6f5a0195e704
let grid = -4:.01:4
    scatter(training_set.x, training_set.y, label = "training set",
            markersize = 3, markerstrokewidth = 0)
    plot!(grid, sin.(grid), lw = 3, c = :green,
          label = "generator without noise")
    plot!(grid, predict(tree, DataFrame(x = grid)), lw = 3, c = :blue,
          label = "tree fit", legend = :topleft, xlabel = "x", ylabel = "y")
end

# ╔═╡ a0133675-567d-4d91-bd5e-b6ed8edf567e
md"""
### Decision Tree for Classification

In the following example a decision tree splits a two-dimensional input into different regions. The output is the probability of a green cross in that region.

"""

# ╔═╡ f2c245cb-ebd0-4a88-809e-d028413f4ec9
begin
function data_generator2()
    x1 = 8*rand(200) .- 4
    x2 = 4*rand(200) .- 2
    y = sin.(x1) .> x2 + .3*randn(200)
    DataFrame(x1 = x1, x2 = x2, y = categorical(y))
end
training_set2 = data_generator2()
tree2 = machine(DecisionTreeClassifier(min_samples_leaf = 4),
               select(training_set2, Not(:y)),
               training_set2.y)
fit!(tree2, verbosity = 0)
MLJDecisionTreeInterface.DecisionTree.print_tree(fitted_params(tree2).raw_tree, 5)
end


# ╔═╡ 747f893e-8867-49c6-9f0d-538b25d0a116
let grid = MLCourse.grid2D(-4:.125:4, -2:.125:2, names = (:x1, :x2), output_format = DataFrame)
    grid2 = -4:.01:4
    plot(grid2, sin.(grid2), lw = 3, c = :black, size = (700, 600),
          label = "decision boundary of generator")
    pred = pdf.(predict(tree2, grid), true)
    vals = sort(unique(pred))
    dict = Dict(v => i for (i, v) in pairs(vals))
    for (i, v) in pairs(vals)
        scatter!([0], [0], c = i, label = "P(Y = true|X) = $(v)", markersize = 2, markerstrokewidth = 0)
    end
    scatter!(grid.x1, grid.x2, c = getindex.(Ref(dict), pred), legend = :outerbottom, xlabel = "X₁", ylabel = "X₂", markersize = 2, markerstrokewidth = 0, label = nothing)
    scatter!(training_set2.x1, training_set2.x2, label = "training set",
            markersize = 4, markershape = :x, c = Int.(int.(training_set2.y) .+ 1),
            markerstrokewidth = 2)
end

# ╔═╡ 038a847e-f994-42ce-a13f-ba95d036f011
md"## Ensembles

If we would have access to multiple training sets we could fit a tree to each training set and average the resulting trees to get a fit that is quite close to the mean of the generator.
"

# ╔═╡ e9552f59-3b05-4f25-a5fd-e794798b331d
let grid = -4:.01:4, m = zeros(length(grid))
    plot()
	Random.seed!(8)
    for _ in 1:20
        d = data_generator()
        scatter!(d.x, d.y, label = nothing, markersize = 3, markerstrokewidth = 0)
        mach = machine(DecisionTreeRegressor(min_samples_leaf = 3),
                       select(d, Not(:y)), d.y)
        fit!(mach, verbosity = 0)
        p = predict(mach, DataFrame(x = grid))
        m .+= p
        plot!(grid, p, label = nothing)
    end
    m ./= 20
    plot!(grid, sin.(grid), lw = 3, c = :green, ylim = (-2, 2),
		  label = "generator without noise", xlabel = "x", ylabel = "y")
    plot!(grid, m, lw = 3, c = :red, label = "mean of the tree fits", size = (700, 400))
end

# ╔═╡ f8eb425f-8279-40d2-bc87-acb0902dcb67
md"## Bagging (Bootstrap Aggregation)

If we have only a single training set we can artifially generate multiple training sets with the bootstrap method, i.e. each artificial training set is obtained by sampling with replacement ``n`` points from the true training set, and average the trees fitted to each bootstrap training set. Bagging is used in Random Forests.
"

# ╔═╡ c014cb7c-e11c-46ea-a2d9-81d26cf29341
let grid = -4:.01:4, m = zeros(length(grid))
    plot()
	Random.seed!(8)
    for _ in 1:50
        idxs = rand(1:nrows(training_set), nrows(training_set))
        d = training_set[idxs, :]
        scatter!(d.x, d.y, label = nothing, markersize = 3, markerstrokewidth = 0)
        mach = machine(DecisionTreeRegressor(min_samples_leaf = 3),
                       select(d, Not(:y)), d.y)
        fit!(mach, verbosity = 0)
        p = predict(mach, DataFrame(x = grid))
        m .+= p
        plot!(grid, p, label = nothing, size = (700, 400))
    end
    m ./= 50
    plot!(grid, sin.(grid), lw = 3, c = :green, ylim = (-2, 2),
		  label = "mean of the generator")
    plot!(grid, m, lw = 3, c = :red, label = "mean of the tree fits")
    mach = machine(DecisionTreeRegressor(min_samples_leaf = 3),
                   select(training_set, Not(:y)), training_set.y)
    fit!(mach, verbosity = 0)
    p = predict(mach, DataFrame(x = grid))
    plot!(grid, p, lw = 3, c = :blue, label = "single tree fit", xlabel = "x", ylabel = "y")
end

# ╔═╡ 9fd69edd-9d2d-4341-ac34-efe9b4c86f68
md"# 3. Support Vector Machines

In this section we fit a support vector machine to some binary classification task, where the data is rather well, but non-linearly, separated. It is charateristic of support vector machines to find a flexible decision boundary that matches well the decision boundary that humans would intuitively draw in such situations.
"

# ╔═╡ 2d9cf0a4-6ba7-438b-8e8f-4a911a340f06
begin
Random.seed!(23); X, y = make_blobs(centers = 3); # generate some random data
y[y .== 3] .= 1; # we want only two classes here
m5 = machine(SVC(), X, y);
fit!(m5, verbosity = 0);
let grid = MLCourse.grid2D(-5:.5:11.5, -9:.5:12, names = (:x1, :x2))
	pred = predict(m5, grid)
	scatter(grid.x1, grid.x2, c = Int.(int.(pred)),
		    markersize = 2, markerstrokewidth = 0, legend = false)
	scatter!(X.x1, X.x2, c = Int.(int.(y)), xlabel = "x₁", ylabel = "x₂")
end
end

# ╔═╡ 93c9127c-935c-4e8a-8da9-d817d2c24a6b
md"# 4. Using Random Forests and Gradient Boosting

In this section we will analyse the cars data set and try to predict the miles per gallon a car drives based on different predictors. We will use a one-hot encoding for all categorical predictors.

You should observe the following:
1. It is fast to tune a gradient boosting machine in the most important hyper-parameters learning rate, number of rounds and maximal tree depth.
2. A tuned gradient boosting machine is clearly better than a linear regressor.
3. Even without any tuning, a random forest regressor with 500 trees reaches almost the same performance as the the tuned gradient boosting machine.
4. It takes time to tune a neural network regressor that works well.

## Linear Fit

We start with loading the data and fitting a linear model as a baseline.
"

# ╔═╡ 987264db-aca7-41c8-aa3a-b9437fcbd7cd
mlcode(
"""
using OpenML, DataFrames, MLJ, MLJLinearModels

summarize(x) = (avg_rmse = x.measurement[], rmse_per_fold = x.per_fold[])

cars = dropmissing(DataFrame(OpenML.load(455)))

preprocessor = OneHotEncoder(drop_last = true)

m1 = machine(preprocessor |> LinearRegressor(),
             select(cars, Not([:name, :mpg])),
             cars.mpg)
evaluate!(m1, measure = rmse, verbosity = 0) |> summarize
"""
,
"""
"""
)

# ╔═╡ 3538a710-e6c7-48fb-872b-8936df6c95b7
md"## Gradient Boosting"

# ╔═╡ ba5563e7-844e-46b9-a4fd-c2d277d3ed5c
mlcode(
"""
using MLJXGBoostInterface

xgb = preprocessor |> XGBoostRegressor()
m2 = machine(TunedModel(model = xgb,
                        resampling = CV(nfolds = 4),
                        measure = rmse,
                        tuning = Grid(goal = 25),
                        range = [range(xgb, :(xg_boost_regressor.eta),
                                       lower = 1e-2, upper = .1, scale = :log),
                                 range(xgb, :(xg_boost_regressor.num_round),
                                       lower = 50, upper = 500),
                                 range(xgb, :(xg_boost_regressor.max_depth),
                                       lower = 2, upper = 6)]),
             select(cars, Not([:name, :mpg])),
             cars.mpg)
fit!(m2, verbosity = 0)
evaluate!(machine(report(m2).best_model,
          select(cars, Not([:name, :mpg])),
          cars.mpg),
	      measure = rmse, verbosity = 0) |> summarize
"""
,
"""
"""
)

# ╔═╡ ebb2a943-d434-48aa-8561-00e280b285f0
md"The gradient boosting machine is quick to tune and leads to a considerably better prediction than the linear model."

# ╔═╡ 6bedf584-2744-46df-baeb-f31f1b0c2ded
mlcode(
"""
linpred = predict(m1)
xgbpred = predict(m2)
scatter(cars.mpg, linpred,
        xlabel = "actual miles per gallon",
        ylabel = "predicted miles per gallon",
        label = "linear")
scatter!(cars.mpg, xgbpred, label = "XGBoost")
plot!(identity, w = 2, label = nothing)
"""
,
"""
"""
)

# ╔═╡ 56b1d5da-eb6d-4a80-92ac-1de73dc76e9f
md"It looks like the non-linear gradient boosting machine makes better predictions, in particular, for very small and very large examples.

## Random Forest"

# ╔═╡ df56c35c-713a-494c-b59f-3f54644bac71
mlcode(
"""
using MLJDecisionTreeInterface

rf = preprocessor |> RandomForestRegressor(n_trees = 500)
m3 = machine(rf,
             select(cars, Not([:name, :mpg])),
             cars.mpg)
evaluate!(m3, measure = rmse, verbosity = 0) |> summarize
"""
,
"""
"""
)

# ╔═╡ 1d26dfb9-44b9-4878-9d1c-d8611fc98d7c
md"We see that a random forest with 500 trees leads to very good predictions, even without tuning!

## Support Vector Machine"

# ╔═╡ e4f0b6fc-4f19-429e-a920-c086b70ae386
mlcode(
"""
using MLJLIBSVMInterface

svm = preprocessor |> NuSVR()
m4 = machine(svm,
             select(cars, Not([:name, :mpg])),
             cars.mpg)
evaluate!(m4, measure = rmse, verbosity = 0) |> summarize
"""
,
"""
"""
)


# ╔═╡ c98e416b-c319-409e-8ee2-e88c3e7df175
md"
Support vector machines can be used for regression, but are probably more often used for classification problems. Without further tuning, support vector regression does not lead to great results on our data set.

## Neural Network
"

# ╔═╡ 3db123ac-1d42-471f-94f1-61d1e43a7f5f
mlcode(
"""
using MLJFlux, Flux, Random

Random.seed!(12)

m4 = machine(preprocessor |> (x -> Float32.(x)) |>
             NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 64,
	                                                        dropout = 0.1,
	                                                        σ = relu),
                                    epochs = 5e3, batch_size = 32),
             select(cars, Not([:name, :mpg])),
             cars.mpg)
fit!(m4, verbosity = 0)
evaluate!(m4, measure = rmse, verbosity = 0) |> summarize
"""
,
"""
"""
)

# ╔═╡ d57e6a40-2b84-47cb-a916-7b427f2c6db0
md"A neural network could be made to work equally well on this data set, but tuning the hyper-parameters of the neural network would take quite a bit of time.

If you are interested in a more in-depth comparison of tree-based methods and neural networks, you can have a look at the article [Why do tree-based models still outperform deep learning on typical tabular data?](https://openreview.net/forum?id=Fp7__phQszn)"

# ╔═╡ 9c30b5d2-6ad4-4d4c-9013-57d2f7cd2d0e
# Old exercises
# ## Conceptual
# #### Exercise 1
# Here below is an image (with padding 1 already applied). We would like to process it with a convolutional network with one convolution layer with two ``3 \\times 3`` filters (depicted below the image), stride 1 and relu non-linearity.
# - Determine the width, height and depth of the volume after the convolutional layer.
# - Compute the output of the convolutional layer assuming the two biases to be zero.
# $(MLCourse.embed_figure("conv_exercise.png"))
# #### Exercise 2
# Given a volume of width ``n``, height ``n`` and depth ``c`` and a convolutional layer with ``k`` filters of size ``f\\times f\\times c`` with stride ``s`` and padding ``p``.
# - Convince yourself that this convolution layer has an output volume of size ``\\left( \\left\\lfloor\\frac{n + 2p - f}{s}\\right\\rfloor + 1\\right)\\times \\left( \\left\\lfloor\\frac{n + 2p - f}{s}\\right\\rfloor + 1\\right)\\times k\\, .``
# - Flux knows the padding option `pad = SamePad()`. It means that the output volume should have the same x and y dimension as the input volume. In this exercise we compute the padding needed this option.  What padding ``p`` do you have to choose for a given ``n`` and ``f`` such that the input and output volumes have the same width and depth for stride ``s=1``. Check you result for the special case of ``n=4`` and ``f=3``.
# - Compute the number of weights and biases of this convolutional layer.
# - Compute the number of weights and biases for a dense layer (standard MLP layer) that has the same number of activations as this convolutional layer. Is the number of parameters larger for the dense layer or the convolutional layer?
# #### Exercise 3
# You are given a dataset with RGB color images with ``100\\times100`` pixels that you would like to classify into 20 different classes. Compute for the following neural network architectures the number of parameters (weights and biases).
# - a multilayer perceptron with one hidden layer of 100 neurons.
# - a convolutional network with 10 filters of size ``3\\times3\\times3`` stride 1 padding 1 followed by a dense hidden layer of 10 neurons.
# - a convolution network with a first convolutional layer with 20 filters of size ``5\\times5\\times3`` padding 2 stride 5, a second convolutional layer with 10 filters of size ``3\\times3\\times20``, stride 2 padding 1, followed by a dense hidden layer of 10 neurons.

# ╔═╡ 01a467a5-7389-44d1-984d-244dfb1ea39f
begin
    function MLCourse.embed_figure(name)
        "![](data:img/png; base64,
             $(open(MLCourse.base64encode,
                    joinpath(Pkg.devdir(), "MLCourse", "notebooks", "figures", name))))"
    end
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 2bbf9876-0676-11ec-3985-73f4dcaea02f
Markdown.parse("# Exercises
## Conceptual
#### Exercise 1
Here below is an image (with padding 1 already applied; padding 1 means that one row/column of zeros is added to the top and bottom/left and right of the original image.). We would like to process it with a convolutional network with one convolution layer with two ``3 \\times 3`` filters (depicted below the image), stride 1 and relu non-linearity (stride 1 means that the filter moves 1 position between each application, as in the formula in the slides).
- Determine the width, height and depth of the volume after the convolutional layer.
- Compute the output of the convolutional layer assuming the two biases to be zero.
$(MLCourse.embed_figure("conv_exercise.png"))

#### Exercise 2
* Sketch the tree corresponding to the partition of the predictor space illustrated in the left-hand panel of the figure below. The num- bers inside the boxes indicate the mean of Y within each region.
* Create a diagram similar to the left-hand panel of the figure below, using the tree illustrated in the right-hand panel. You should divide up the predictor space into the correct regions, and indicate the mean for each region.
$(MLCourse.embed_figure("trees3.png"))

## Applied
#### Exercise 3
In this exercise our goal is to find a good machine learning model to classify images of Zalando's articles. You can load a description of the so-called Fashion-MNIST data set with `OpenML.describe_dataset(40996)` and load the data set with `OpenML.load(40996)`. Take our recipe for supervised learning (last slide of the presentation on \"Model Assessment and Hyperparameter Tuning\") as a guideline. Hints: cleaning is not necessary, but plotting some examples is advisable; linear classification is a good starting point for a first benchmark, but you should also explore other models like random forests (`RandomForestClassifier`), multilayer perceptrons (`NeuralNetworkClassifier`) and convolutional neural networks (`ImageClassifer`) and play manually a bit with their hyper-parameters (proper tuning with `TunedModel` may be too time-intensive). To reduce the computation time, we will only use the first 5000 images of the full dataset for training and we will not do cross-validation here but instead use samples 5001 to 10000 as a validation set to estimate the classification accuracy. *Hint:* you can use the resampling strategy `Holdout` for model tuning and evaluation.
#### (optional) Exercise 4
Use a recurrent neural network to classify emails in our spam dataset.
")

# ╔═╡ 3bd0ed27-7d3b-4036-87a4-2dd0e2874b58
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─184c3e36-a3d0-4554-a5e4-88996510f329
# ╟─7cfd274d-75d8-4cd9-b38b-4176466c9e26
# ╟─995fb5d1-4083-491f-aed7-f1cc127eb897
# ╟─2068970a-4aa7-4580-a137-3e38460ba0a2
# ╟─eeecfe16-631f-4eb0-a7d6-581d8b49aa9c
# ╟─69ac3eec-bc07-414b-b66c-1e9cd45821fe
# ╟─dd12ea16-f087-4507-a571-c95bf38aa976
# ╟─c76ab462-333f-4852-b0bf-2d7d603c34ab
# ╟─c769c35c-ea59-4d4a-8451-a9343fc75d88
# ╟─cee63cef-6ce3-4738-941d-632b2d43f647
# ╟─736ffb3d-64a7-43f6-b781-66cf2540bb42
# ╟─789b6c31-ae1d-49b2-8fac-365e6517a3cc
# ╟─75aed643-0185-4fe4-abdf-02634865404e
# ╟─b9ca66df-f416-40a6-95d2-5012a73a62ff
# ╟─e71921f8-e98d-46b3-b35d-0119e614f6c5
# ╟─4bc9a437-d9a9-4543-ae3a-f41c208da351
# ╟─c51b2449-b1b5-4e91-a86b-0cd1e645b16b
# ╟─06b27480-1dea-456a-94b3-9eef1a5b1b71
# ╟─feeb8006-abef-4b46-a4da-0cab8a025601
# ╟─d0891e35-3aaa-4cbe-884b-27a5a2980526
# ╟─e68603b3-6f95-4c55-aded-14eaa7615176
# ╟─4b4714f7-49bf-4191-a59a-79fa1c826519
# ╟─16fef29a-753f-44b0-89af-6f5a0195e704
# ╟─a0133675-567d-4d91-bd5e-b6ed8edf567e
# ╟─f2c245cb-ebd0-4a88-809e-d028413f4ec9
# ╟─747f893e-8867-49c6-9f0d-538b25d0a116
# ╟─038a847e-f994-42ce-a13f-ba95d036f011
# ╟─e9552f59-3b05-4f25-a5fd-e794798b331d
# ╟─f8eb425f-8279-40d2-bc87-acb0902dcb67
# ╟─c014cb7c-e11c-46ea-a2d9-81d26cf29341
# ╟─9fd69edd-9d2d-4341-ac34-efe9b4c86f68
# ╟─2d9cf0a4-6ba7-438b-8e8f-4a911a340f06
# ╟─93c9127c-935c-4e8a-8da9-d817d2c24a6b
# ╟─987264db-aca7-41c8-aa3a-b9437fcbd7cd
# ╟─3538a710-e6c7-48fb-872b-8936df6c95b7
# ╟─ba5563e7-844e-46b9-a4fd-c2d277d3ed5c
# ╟─ebb2a943-d434-48aa-8561-00e280b285f0
# ╟─6bedf584-2744-46df-baeb-f31f1b0c2ded
# ╟─56b1d5da-eb6d-4a80-92ac-1de73dc76e9f
# ╟─df56c35c-713a-494c-b59f-3f54644bac71
# ╟─1d26dfb9-44b9-4878-9d1c-d8611fc98d7c
# ╟─e4f0b6fc-4f19-429e-a920-c086b70ae386
# ╟─c98e416b-c319-409e-8ee2-e88c3e7df175
# ╟─3db123ac-1d42-471f-94f1-61d1e43a7f5f
# ╟─d57e6a40-2b84-47cb-a916-7b427f2c6db0
# ╟─9c30b5d2-6ad4-4d4c-9013-57d2f7cd2d0e
# ╟─2bbf9876-0676-11ec-3985-73f4dcaea02f
# ╟─01a467a5-7389-44d1-984d-244dfb1ea39f
# ╟─64bac944-8008-498d-a89b-b9f8ee54aa98
# ╟─3bd0ed27-7d3b-4036-87a4-2dd0e2874b58
# ╟─a022be72-b0b0-43b7-8f39-c02a53875ff6
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
