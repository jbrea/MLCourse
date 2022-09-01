using Pkg, Logging
Pkg.activate(joinpath(@__DIR__, ".."))

using Pluto, MLCourse, MLJ, MLJLinearModels, StatsPlots, DataFrames,
      Distributions, Random, Flux, Zygote

X, y = make_regression()

mach = fit!(machine(LinearRegressor(), X, y), verbosity = 0)

predict(mach)

Random.seed!(23)
rand(Random.TaskLocalRNG())

plot(rand(10), rand(10))
scatter(rand(10), rand(10))
plot(xlim = (0, 1), ylim = (0, 1), size = (600, 400),
     bg = :red, framestyle = :none, legend = false)

MLCourse.gradient_descent(x -> sum(abs2, x), rand(4), .1, 2)

df = DataFrame(a = rand(10), b = rand(10))
@df df scatter(:a, :b)

rand(Bernoulli(.4))

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
