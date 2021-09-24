using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import Pluto, MLCourse

session = Pluto.ServerSession()
session.options.server.port = 40404
session.options.security.require_secret_for_access = false

# for file in last.(MLCourse.NOTEBOOKS)
#     Base.include(Module(), joinpath(@__DIR__, "..", "notebooks", file))
# end
using CSV, DataFrames, MLJ, MLJLinearModels, NearestNeighborModels,
      Random, Distributions, MLJGLMInterface, Plots, StatsPlots, Statistics,
      LinearAlgebra

weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), DataFrame)
@df weather corrplot([:BAS_pressure :LUG_pressure :LUZ_pressure :LUZ_wind_peak],
                     grid = false, fillcolor = cgrad(), size = (700, 600))

X = float.(select(weather, Not(:LUZ_wind_peak)))[1:1000, :]
y = weather.LUZ_wind_peak[1:1000]

m = fit!(machine(MLJLinearModels.LinearRegressor(), X, y))
predict(m, X)

m = fit!(machine(MLJGLMInterface.LinearRegressor(), X, y))
predict(m, X)
predict_mean(m, X)

pdf(Normal(0., 1.), 2.)
pdf(Bernoulli(.3), 1)
pdf(Categorical([.2, .3, .5]), [1, 0, 0])

for backend in (gr(), plotly())
    scatter(rand(10), rand(10))
    plot(sin)
    contour(0:10, 0:10, (x, y) -> tanh(x + y))
end

spamdata = CSV.read(joinpath(@__DIR__, "..", "data", "spam.csv"), DataFrame)
dropmissing!(spamdata)
import TextAnalysis: Corpus, StringDocument, DocumentTermMatrix, lexicon,
                     update_lexicon!, tf
crps = Corpus(StringDocument.(spamdata.text[1:200]))
update_lexicon!(crps)
lexicon(crps)
small_lex = Dict(k => lexicon(crps)[k]
                 for k in findall(x -> 80 <= last(x) <= 10^2, lexicon(crps)))
m = DocumentTermMatrix(crps, small_lex)
spam_or_ham = coerce(String.(spamdata.label[1:200]), Binary)
normalized_word_counts = float.(DataFrame(tf(m), :auto))
m3 = fit!(machine(LinearBinaryClassifier(),
                  normalized_word_counts,
                  spam_or_ham));
confusion_matrix(predict_mode(m3, normalized_word_counts), spam_or_ham)
function losses(machine, input, response)
    (logikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = auc(predict(machine, input), response))
end;
losses(m3, normalized_word_counts, spam_or_ham)

begin
    f(x) = .3 * sin(10x) + .7x
    # f(x) = .0015*(12x-6)^4 -.035(12x-6)^2 + .5x + .2  # an alternative
	σ(x) = 1 / (1 + exp(-x))
    function regression_data_generator(; n, rng = Random.GLOBAL_RNG)
        x = rand(rng, n)
        DataFrame(x = x, y = f.(x) .+ .1*randn(rng, n))
    end
    function classification_data_generator(; n, rng = Random.GLOBAL_RNG)
        X1 = rand(rng, n)
        X2 = rand(rng, n)
        df = DataFrame(X1 = X1, X2 = X2,
                       y = σ.(20(f.(X1) .- X2)) .> rand(rng, n))
        coerce!(df, :y => Multiclass)
    end
end

classification_data = classification_data_generator(n = 400, rng = MersenneTwister(8))

polysymbol(x, d) = Symbol(d == 0 ? "" : d == 1 ? "$x" : "$x^$d")
function poly2(data, degree)
    res = DataFrame([data.X1 .^ d1 .* data.X2 .^ d2
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree],
                    [Symbol(polysymbol("x₁", d1), polysymbol("x₂", d2))
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end;

m4 = machine(LogisticClassifier(penalty = :none),
             select(poly2(classification_data, 4), Not(:y)),
             classification_data.y) |> fit!;

m14 = machine(KNNClassifier(K = 4),
             select(classification_data, Not(:y)),
             classification_data.y) |> fit!;

path = tempname()
original = joinpath(@__DIR__, "..", "index.jl")
# so that we don't overwrite the file:
Pluto.readwrite(original, path)

@info "Loading notebook"
nb = Pluto.load_notebook(Pluto.tamepath(path));
session.notebooks[nb.notebook_id] = nb;

@info "Running notebook"
Pluto.update_save_run!(session, nb, nb.cells; run_async=false, prerender_text=true)

# nice! we ran the notebook, so we already precompiled a lot

@info "Starting HTTP server"
# next, we'll run the HTTP server which needs a bit of nasty code
t = @async Pluto.run(session)

sleep(15)
download("http://localhost:40404/")

# this is async because it blocks for some reason
@async Base.throwto(t, InterruptException())
sleep(15) # i am pulling these numbers out of thin air

@info "Warmup done"
