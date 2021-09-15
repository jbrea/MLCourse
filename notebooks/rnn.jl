### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ baabe9c8-055e-46f2-a495-17ccc3bfced1
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 27127d83-ee72-4cc1-8772-d85130fb5fa4
begin
	using Flux
	dump(Dense(12, 12))
end

# ╔═╡ 2f376d84-0460-4f19-b6a3-b9d17d452823
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 9c575c1a-9c9f-4c27-9a53-717e5eca2dab
Markdown.parse("# Language Detection with a Recurrent Neural Network

The following code (adapted from [here](https://github.com/FluxML/model-zoo/blob/master/text/lang-detection/model.jl)) uses wikipedia articles to detect the language of a sentence
by simply reading each character of the sentence with a recurrent neural network
and applying a classifier to the last hidden state.

WARNING: This code takes more than 30 minutes to run.

```julia
using Flux
using Flux: onehot, onehotbatch, logitcrossentropy, reset!, throttle
using Statistics: mean
using Random
using Unicode

Base.@kwdef mutable struct Args
    lr::Float64 = 1e-3     # learning rate
    N::Int = 15            # Number of perceptrons in hidden layer
    test_len::Int = 100    # length of test data
    langs_len::Int = 0     # Number of different languages in Corpora
    alphabet_len::Int = 0  # Total number of characters possible, in corpora
    throttle::Int = 10     # throttle timeout
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

    dataset = [(onehotbatch(s, alphabet, '_'), onehot(l, langs)) for l in langs for s in corpora[l]] |> shuffle

    train, test = dataset[1:end-args.test_len], dataset[end-args.test_len+1:end]
    return train, test
end

function build_model(args)
    scanner = Chain(Dense(args.alphabet_len, args.N, σ), LSTM(args.N, args.N))
    encoder = Dense(args.N, args.langs_len)
    return scanner, encoder
end

function model(x, scanner, encoder)
    state = mapslices(scanner, x, dims = 1)[:, end] # scan all characters of sentence x
    reset!(scanner)
    encoder(state) # take the last state of the neural network to compute the probabilities of each language.
end

args = Args()
train_data, test_data = get_processed_data(args)

@info(\"Constructing Model...\")
scanner, encoder = build_model(args)

loss(x, y) = logitcrossentropy(model(x, scanner, encoder), y)
testloss() = mean(loss(t...) for t in test_data)
accuracy = () -> mean([argmax(model(first(x), scanner, encoder)) == findfirst(last(x)) for x in test_data])

opt = ADAM(args.lr)
ps = params(scanner, encoder)
evalcb = () -> @show testloss() accuracy()
@info(\"Training...\")
for _ in 1:200
    Flux.train!(loss, ps, train_data, opt, cb = throttle(evalcb, args.throttle))
end
```
")


# ╔═╡ d1258b22-0680-11ec-1a56-1bde845491c0
md"# Exercise
1. Adapt the example above to write a character-level spam detector for our
    spam data set.
"

# ╔═╡ Cell order:
# ╟─9c575c1a-9c9f-4c27-9a53-717e5eca2dab
# ╠═27127d83-ee72-4cc1-8772-d85130fb5fa4
# ╟─d1258b22-0680-11ec-1a56-1bde845491c0
# ╟─2f376d84-0460-4f19-b6a3-b9d17d452823
# ╟─baabe9c8-055e-46f2-a495-17ccc3bfced1
