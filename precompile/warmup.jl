using Pkg, Logging
Pkg.activate(joinpath(@__DIR__, ".."))

using Pluto, MLCourse, MLJ, MLJLinearModels, StatsPlots, DataFrames,
      Distributions, Random, Flux, Zygote, ReinforcementLearning
import MLCourse: poly

X, y = make_regression()

mach = fit!(machine(LinearRegressor(), X, y), verbosity = 0)

predict(mach)

Random.seed!(23)
Random.seed!(Random.TaskLocalRNG(), 12)
rand(Random.TaskLocalRNG())

plot(rand(10), rand(10))
scatter(rand(10), rand(10))
plot(xlim = (0, 1), ylim = (0, 1), size = (600, 400),
     bg = :red, framestyle = :none, legend = false)

MLCourse.gradient_descent(x -> sum(abs2, x), rand(4), .1, 2)

df = DataFrame(a = rand(10), b = rand(10))
@df df scatter(:a, :b)

rand(Bernoulli(.4))

g(x) = .3 * sin(10x) + .7x
function regression_data_generator(; n, seed = 3, rng = MersenneTwister(seed))
    x = range(0, 1, length = n)
    DataFrame(x = x, y = g.(x) .+ .1*randn(rng, n))
end
regression_data = regression_data_generator(n = 10, seed = 8)
regression_valid = regression_data_generator(n = 50, seed = 123)
struct Tracker{T} # structure definition with parametric type
    path::T
end
Tracker(x::T) where T = Tracker{Vector{T}}([copy(x)]) # constructor
(t::Tracker)(x) = push!(t.path, copy(x))        # making the object callable
function advanced_gradient_descent(f, x, optimizer, T; callback = x -> nothing)
    for t in 1:T
        ∇f = gradient(f, x)[1] # compute ∇f
		Flux.update!(optimizer, x, ∇f) # apply the changes to x
        callback(x) # the callback will be used to save intermediate values
    end
    x
end
h_training = Array(select(poly(regression_data, 12), Not(:y)))
h_valid = Array(select(poly(regression_valid, 12), Not(:y)))
poly_regression_loss(θ, x, y) = mean((y .- θ[1] .- x * θ[2:end]).^2)
target = regression_data.y
poly_regression_loss(θ) = poly_regression_loss(θ, h_training, target)
poly_params = 1e-3 * randn(13)
tracker5 = Tracker(poly_params)
poly_opt = ADAMW()
advanced_gradient_descent(poly_regression_loss, poly_params, poly_opt, 2, callback = tracker5)

mnist = OpenML.load(554, maxbytes = 10^4) |> DataFrame;
plot(Gray.(reshape(Array(mnist[1, 1:end-1]) ./ 255, 28, 28)'))

with_logger(SimpleLogger(Logging.Warn)) do

session = Pluto.ServerSession()
session.options.server.port = 40404
session.options.server.launch_browser = false
session.options.security.require_secret_for_access = false

path = tempname()
original = joinpath(@__DIR__, "..", "index.jl")
# so that we don't overwrite the file:
Pluto.readwrite(original, path)

# @info "Loading notebook"
nb = Pluto.load_notebook(Pluto.tamepath(path));
session.notebooks[nb.notebook_id] = nb;

# @info "Running notebook"
Pluto.update_save_run!(session, nb, nb.cells; run_async=false, prerender_text=true)

# nice! we ran the notebook, so we already precompiled a lot

# @info "Starting HTTP server"
# next, we'll run the HTTP server which needs a bit of nasty code
t = @async Pluto.run(session)

sleep(5)
# download("http://localhost:40404/")

# this is async because it blocks for some reason
@async Base.throwto(t, InterruptException())
sleep(2) # i am pulling these numbers out of thin air

end
@info "Warmup done"
