### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a022be72-b0b0-43b7-8f39-c02a53875ff6
begin
using Pkg
Base.redirect_stdio(stderr = devnull, stdout = devnull) do
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
end
using Revise, MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, Flux, MLJXGBoostInterface, MLJDecisionTreeInterface, MLJLIBSVMInterface
import PlutoPlotly as PP
const M = MLCourse.JlMod
MLCourse.CSS_STYLE
end

# ╔═╡ 01a467a5-7389-44d1-984d-244dfb1ea39f
begin
    function MLCourse.embed_figure(name)
        "![](data:img/png; base64,
             $(open(MLCourse.base64encode,
                    joinpath(Pkg.devdir(), "MLCourse", "notebooks", "figures", name))))"
    end
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 64bac944-8008-498d-a89b-b9f8ee54aa98
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 7cfd274d-75d8-4cd9-b38b-4176466c9e26
Markdown.parse("# Fitting MNIST with a Convolutional Neural Networks

In the following we fit a convolutional neural network to the MNIST data set.

Running this example takes more than 10 minutes.

We load the data
```julia
using MLJ, Plots, MLJFlux, Flux, OpenML, DataFrames
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    Float32.(df[:, 1:end-1] ./ 255), # scaling the input
    df.class
end
```
and transform the input to an image representation
```julia
images = coerce(PermutedDimsArray(reshape(Array(mnist_x), :, 28, 28), (3, 2, 1)),
                GrayImage);
plot(images[1])
```
$(MLCourse.embed_figure("mnist_example1.png"))

Now we define our network builder
```julia
builder = MLJFlux.@builder(
              begin
                  front = Chain(Conv((8, 8), n_channels => 16, relu),
                                Conv((4, 4), 16 => 32, relu),
                                Flux.flatten)
                  d = first(Flux.outputsize(front, (n_in..., n_channels, 1)))
                  Chain(front, Dense(d, n_out))
              end)
```
The `front` consists of two convolutional layers. The volume of the second
convolutional layer is flattened and taken as input to a single dense layer.
```julia
m = machine(ImageClassifier(builder = builder,
                            batch_size = 32,
                            epochs = 5),
            images, mnist_y)
fit!(m, rows = 1:60000, verbosity = 2)
```
Finally, let us compute the test error rate
```julia
mean(predict_mode(m, images[60001:70000]) .!= mnist_y[60001:70000])
```
The result is a misclassification rate of 1.4% which is again a strong improvement
over the MLP result. Note that we achieved this value without tuning of the hyper-parameters. Well-tuned convolutional networks achieve performances below 1% error.

If you are interested in learning more about convolutional neural networks I can highly recommend to have a look at one of the excellent tutorials on the internet, like [this](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1), [this](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) or [this](https://distill.pub/2017/feature-visualization/) one.
")


# ╔═╡ dd12ea16-f087-4507-a571-c95bf38aa976
Markdown.parse("# Language Detection with a Recurrent Neural Network

The following code (adapted from [here](https://github.com/FluxML/model-zoo/blob/master/text/lang-detection/model.jl)) uses wikipedia articles to detect the language of a sentence
by simply reading each character of the sentence with a recurrent neural network
and applying a classifier to the last hidden state.

Here are some example sentences in English, French, Italian, Spanish and Danish
after preprocessing by the `get_processed_data` function below. The preprocessing replaces most special characters by a single special character (we use the special character ˌ below). Ultimately, the recurrent neural network gets these sentences in one-hot encoded form; for better readability for humans we print them here in the standard form:

- \"jewish philosophy refers to the conjunction between serious study of philosophy and jewish theology\"
-  \"en 756ˌ abderraman i ˌun omeya superviviente del exterminio de la familia califal destronada por los abbasiesˌ fue acogido por sus partidarios en alˌandalus y se impuso como emir\"
- \"ˌ167ˌ\\ncritics argue that wikipediaˌs open nature and a lack of proper sources for most of the information makes it unreliable\"
- \"tambien existia una clase media formada por comerciantesˌ pequenos propietariosˌ funcionarios y profesionales liberalesˌ que en general toleraba el franquismo pero no comulgaba totalmente con sus ideas\"
- \"wikipedia ˌpronunciaˌ vedi sottoˌ e unˌenciclopediaonline a contenuto liberoˌ collaborativaˌ multilingue e gratuitaˌ nata nel 2001ˌ sostenuta e ospitata dalla wikimedia foundationˌ unˌorganizzazione non a scopo di lucrostatunitense\"
- \"dˌautre partˌ des jalons sont poses pour une integration de toutes les economies nationales dans un grand espace economique domine par lˌallemagneˌ65ˌ\"
- \"ˌ158ˌˌ en cuanto al sistema monetarioˌ al dinero carolingio sucedio tras la independizacion del condado el dinero de vellonˌ con un 75ˌˌ menos de plata\"
- \"es responsable ante las cortes generales\"
- \"ˌ102ˌ\\ndopo pochi giorni il marito mori\"
- \"ˌ116ˌ\\na kittel ˌyiddishˌ ˌˌˌˌˌˌ a white kneeˌlength overgarmentˌ is worn by prayer leaders and some observant traditional jews on the high holidays\"
- \"afin dˌassurer la perennite des liens vers ces sourcesˌ internet archive a repare en 2018 pres de 9 millions de liens brisesˌ37ˌ\"
- \"de hechoˌ 18 de los 25 mayores hoteles del mundo se encuentran en ella\"

The code in this section is somewhat more advanced than in the rest of this course.
You do not need to understand everything. But if you are interested in learning
about recurrent neural networks, this may give you an idea for how to implement
them.

WARNING: This code takes more than 30 minutes to run.

```julia
using Flux
using Flux: onehot, onehotbatch, logitcrossentropy, reset!, throttle
using Statistics: mean
using Random
using Unicode

Base.@kwdef mutable struct Args
    lr::Float64 = 1e-3     # learning rate
    N::Int = 32            # Number of perceptrons in hidden layer
    test_len::Int = 100    # length of test data
    langs_len::Int = 0     # Number of different languages in Corpora
    alphabet_len::Int = 0  # Total number of characters possible, in corpora
    throttle::Int = 120    # throttle timeout
end

function get_processed_data(args)
    corpora = Dict()

    dir = joinpath(@__DIR__, \"..\", \"data\", \"corpus\")
    for file in readdir(dir)
        lang = Symbol(match(r\"(.*)\\.txt\", file).captures[1])
        corpus = split(String(read(joinpath(dir, file))), \".\")
        corpus = strip.(Unicode.normalize.(corpus, casefold=true, stripmark=true))
        corpus = filter(!isempty, corpus)
        corpora[lang] = corpus
    end

    langs = collect(keys(corpora))
    args.langs_len = length(langs)
    alphabet = ['a':'z'; '0':'9'; ' '; '\\n'; '_']
    args.alphabet_len = length(alphabet)

    # See which chars will be represented as \"unknown\"
    unique(filter(x -> x ∉ alphabet, join(vcat(values(corpora)...))))

    dataset = [(onehotbatch(s, alphabet, '_'), onehot(l, langs))
               for l in langs for s in corpora[l]] |> shuffle

    train, test = dataset[1:end-args.test_len], dataset[end-args.test_len+1:end]
    return train, test
end

function build_model(args)
    # The scanner is the recurrent neural network; it uses a special kind of
    # neuron in the recurrent layer, called LSTM.
    scanner = Chain(Dense(args.alphabet_len, args.N, relu), LSTM(args.N, args.N))
    # The encoder takes the last element of the sequence and computes the
    # log-probability of each language for the given sequence using a standard
    # multi-layer perceptron (MLP) with one hidden layer.
    encoder = Chain(Dense(args.N, args.N, relu), Dense(args.N, args.langs_len))
    return scanner, encoder
end

function model(x, scanner, encoder)
    # scan all characters of sentence x
    state = mapslices(scanner, x, dims = 1)[:, end]
    reset!(scanner)
    # take the last state of the neural network
    # to compute the probability of each language.
    encoder(state)
end

args = Args()
train_data, test_data = get_processed_data(args)

@info(\"Constructing Model...\")
scanner, encoder = build_model(args)

# we use the negative loglikelihood loss (here called logitcrossentropy)
loss(x, y) = logitcrossentropy(model(x, scanner, encoder), y)
function evaluate(scanner, encoder, dataset)
    () -> begin
        tmp = [(model(x, scanner, encoder), y) for (x, y) in dataset]
        (loss = mean(logitcrossentropy(ŷ, y) for (ŷ, y) in tmp),
         accuracy = mean(argmax(ŷ) == findfirst(y) for (ŷ, y) in tmp))
    end
end
traineval = evaluate(scanner, encoder, train_data)
testeval = evaluate(scanner, encoder, test_data)

opt = ADAM(args.lr)
ps = params(scanner, encoder)
evalcb = throttle(() -> @show(traineval(), testeval()), args.throttle)
@info(\"Training...\")
for epoch in 1:100
    @show epoch
    Flux.train!(loss, ps, train_data, opt, cb = evalcb)
end
```
The result is around 60% accuracy on the test set (which is much better than the
chance level of 20%). With longer training, more neurons in the scanner and the
encoder and potentially well chosen regularization, one could improve this
result a lot.
")

# ╔═╡ e68603b3-6f95-4c55-aded-14eaa7615176
md"# Tree-Based Methods

## Decision Trees

"

# ╔═╡ 4b4714f7-49bf-4191-a59a-79fa1c826519
function data_generator()
    x = 8*rand(40) .- 4
    y = sin.(x) .+ .3*randn(40)
    DataFrame(x = x, y = y)
end

# ╔═╡ c5873c6f-53e8-45b6-931d-56088f0b60b1
training_set = data_generator();

# ╔═╡ f355f66c-a242-4e15-bb09-9e240948e1ed
tree = machine(DecisionTreeRegressor(min_samples_leaf = 4),
               select(training_set, Not(:y)),
               training_set.y);

# ╔═╡ b894f127-89a6-4903-9578-a6ae60e9bae7
fit!(tree, verbosity = 0);

# ╔═╡ ba27162a-7e83-4235-87c2-14d6f9b0550c
MLJDecisionTreeInterface.DecisionTree.print_tree(fitted_params(tree).tree)

# ╔═╡ 16fef29a-753f-44b0-89af-6f5a0195e704
let grid = -4:.01:4
    scatter(training_set.x, training_set.y, label = "training set",
            markersize = 3, markerstrokewidth = 0)
    plot!(grid, sin.(grid), lw = 3, c = :green,
          label = "generator without noise")
    plot!(grid, predict(tree, DataFrame(x = grid)), lw = 3, c = :blue,
          label = "tree fit", legend = :topleft, xlabel = "x", ylabel = "y")
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
    plot!(grid, m, lw = 3, c = :red, label = "mean of the tree fits")
end

# ╔═╡ f8eb425f-8279-40d2-bc87-acb0902dcb67
md"## Bagging (Bootstrap Aggregation)

If we have only a single training set we can artifially generate multiple training sets with the bootstrap method, i.e. each artificial training set is obtained by sampling with replacement ``n`` points from the true training set, and average the trees fitted to each bootstrap training set.
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
        plot!(grid, p, label = nothing)
    end
    m ./= 50
    plot!(grid, sin.(grid), lw = 3, c = :green, ylim = (-2, 2),
		  label = "mean of the generator")
    plot!(grid, m, lw = 3, c = :red, label = "mean of the tree fits")
    mach = machine(DecisionTreeRegressor(min_samples_leaf = 3),
                   select(training_set, Not(:y)), training_set.y)
    fit!(mach, verbosity = 0)
    p = predict(mach, DataFrame(x = grid))
    plot!(grid, p, lw = 3, c = :blue, label = "tree fit", xlabel = "x", ylabel = "y")
end

# ╔═╡ 93c9127c-935c-4e8a-8da9-d817d2c24a6b
md"## Random Forests and Gradient Boosting

In this section we will analyse the cars data set and try to predict the miles per gallon a car drives based on different predictors. We will use a one-hot encoding for all categorical predictors.

You should observe the following:
1. It is fast to tune a gradient boosting machine in the most important hyper-parameters learning rate `:eta`, number of rounds `:num_round` and maximal tree depth `:max_depth`.
2. A tuned gradient boosting machine is clearly better than a linear regressor.
3. Even without any tuning, a random forest regressor with 500 trees reaches almost the same performance as the the tuned gradient boosting machine.
4. It takes time to tune a neural network regressor that works well.
"

# ╔═╡ 987264db-aca7-41c8-aa3a-b9437fcbd7cd
cars = dropmissing(DataFrame(OpenML.load(455)));

# ╔═╡ 5be3a1e7-de72-49e6-8170-fc5aac38e3bb
cars_input = MLJ.transform(fit!(machine(OneHotEncoder(drop_last = true),
	                                    select(cars, Not([:name, :mpg]))),
								 verbosity = 0),
	                       select(cars, Not([:name, :mpg])))

# ╔═╡ ef75fad9-41ca-4960-8c86-3f71c9581582
m1 = machine(LinearRegressor(), cars_input, cars.mpg);

# ╔═╡ b75ff61a-9deb-4ec4-a2c0-3b8231b4d109
fit!(m1, verbosity = 0);

# ╔═╡ 89a6c574-aa3d-4eec-8bf3-50c0d64bee40
evaluate!(m1, measure = rmse)

# ╔═╡ ba5563e7-844e-46b9-a4fd-c2d277d3ed5c
begin
	xgb = XGBoostRegressor()
    m2 = machine(TunedModel(model = xgb,
                            resampling = CV(nfolds = 4),
		                    measure = rmse,
                            tuning = Grid(goal = 25),
                            range = [range(xgb, :eta,
                                           lower = 1e-2, upper = .1, scale = :log),
                                     range(xgb, :num_round, lower = 50, upper = 500),
                                     range(xgb, :max_depth, lower = 2, upper = 6)]),
                  cars_input, cars.mpg)
	fit!(m2, verbosity = 0)
end;

# ╔═╡ 56b1d5da-eb6d-4a80-92ac-1de73dc76e9f
evaluate!(machine(report(m2).best_model, cars_input, cars.mpg),
	      measure = rmse, verbosity = 0)

# ╔═╡ df56c35c-713a-494c-b59f-3f54644bac71
m3 = machine(RandomForestRegressor(n_trees = 500),
	         cars_input, cars.mpg);

# ╔═╡ 2942f37c-b848-49c9-a599-0deb5040045c
evaluate!(m3, measure = rmse, verbosity = 0)

# ╔═╡ c4b21c4a-f8ec-468e-b86b-7e4c3e0203df
md"The following code takes a moment to run. It is certainly not the best neural network regressor, but it would take quite a bit of time to tune it and find one that outperforms gradient boosting.
```julia
using MLJFlux
m4 = machine(NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 64,
	                                                        dropout = 0.1,
	                                                        σ = MLJFlux.Flux.relu),
                                    epochs = 5e3, batch_size = 20),
	         cars_input, cars.mpg);
evaluate!(m4, measure = rmse)
```
The result is a cross-validation estimate of the test error of approximately 3.13,
which is better than linear regression, but worse than the results we found quickly with the tree based approaches.
"

# ╔═╡ 9fd69edd-9d2d-4341-ac34-efe9b4c86f68
md"# Support Vector Machines

In this section we fit a support vector machine to some binary classification task, where the data is rather well, but non-linearly, separated. It is charateristic of support vector machines to find a flexible decision boundary that matches well the decision boundary that humans would intuitively draw in such situations.
"

# ╔═╡ 2d9cf0a4-6ba7-438b-8e8f-4a911a340f06
Random.seed!(23); X, y = make_blobs(centers = 3); # generate some random data

# ╔═╡ 87a48194-fc87-414f-8d44-8bf00a87b3ec
y[y .== 3] .= 1; # we want only two classes here

# ╔═╡ 3a7fa61a-69e0-4082-9bb2-13df40089bb2
m5 = machine(SVC(), X, y);

# ╔═╡ 4729c124-c2f8-442d-8f2f-f8e75bee7e40
fit!(m5, verbosity = 0);

# ╔═╡ a48c53e8-99f7-4b54-9a80-0be243b4cddb
let grid = MLCourse.grid(-5:.5:11.5, -9:.5:12, names = (:x1, :x2))
	pred = predict(m5, grid)
	scatter(grid.x1, grid.x2, c = Int.(int.(pred)),
		    markersize = 2, markerstrokewidth = 0, legend = false)
	scatter!(X.x1, X.x2, c = Int.(int.(y)), xlabel = "x₁", ylabel = "x₂")
end

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

# ╔═╡ Cell order:
# ╟─7cfd274d-75d8-4cd9-b38b-4176466c9e26
# ╟─dd12ea16-f087-4507-a571-c95bf38aa976
# ╟─e68603b3-6f95-4c55-aded-14eaa7615176
# ╠═4b4714f7-49bf-4191-a59a-79fa1c826519
# ╠═c5873c6f-53e8-45b6-931d-56088f0b60b1
# ╠═f355f66c-a242-4e15-bb09-9e240948e1ed
# ╠═b894f127-89a6-4903-9578-a6ae60e9bae7
# ╠═ba27162a-7e83-4235-87c2-14d6f9b0550c
# ╟─16fef29a-753f-44b0-89af-6f5a0195e704
# ╟─038a847e-f994-42ce-a13f-ba95d036f011
# ╟─e9552f59-3b05-4f25-a5fd-e794798b331d
# ╟─f8eb425f-8279-40d2-bc87-acb0902dcb67
# ╟─c014cb7c-e11c-46ea-a2d9-81d26cf29341
# ╟─93c9127c-935c-4e8a-8da9-d817d2c24a6b
# ╠═a022be72-b0b0-43b7-8f39-c02a53875ff6
# ╠═987264db-aca7-41c8-aa3a-b9437fcbd7cd
# ╠═5be3a1e7-de72-49e6-8170-fc5aac38e3bb
# ╠═ef75fad9-41ca-4960-8c86-3f71c9581582
# ╠═b75ff61a-9deb-4ec4-a2c0-3b8231b4d109
# ╠═89a6c574-aa3d-4eec-8bf3-50c0d64bee40
# ╠═ba5563e7-844e-46b9-a4fd-c2d277d3ed5c
# ╠═56b1d5da-eb6d-4a80-92ac-1de73dc76e9f
# ╠═df56c35c-713a-494c-b59f-3f54644bac71
# ╠═2942f37c-b848-49c9-a599-0deb5040045c
# ╟─c4b21c4a-f8ec-468e-b86b-7e4c3e0203df
# ╟─9fd69edd-9d2d-4341-ac34-efe9b4c86f68
# ╠═cf267836-0ac5-494d-9b5e-12eb314ac170
# ╠═2d9cf0a4-6ba7-438b-8e8f-4a911a340f06
# ╠═87a48194-fc87-414f-8d44-8bf00a87b3ec
# ╠═3a7fa61a-69e0-4082-9bb2-13df40089bb2
# ╠═4729c124-c2f8-442d-8f2f-f8e75bee7e40
# ╠═a48c53e8-99f7-4b54-9a80-0be243b4cddb
# ╟─9c30b5d2-6ad4-4d4c-9013-57d2f7cd2d0e
# ╟─2bbf9876-0676-11ec-3985-73f4dcaea02f
# ╟─01a467a5-7389-44d1-984d-244dfb1ea39f
# ╟─64bac944-8008-498d-a89b-b9f8ee54aa98
# ╟─3bd0ed27-7d3b-4036-87a4-2dd0e2874b58
