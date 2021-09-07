module MLCourse

import Pkg

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(MLCourse))), xs...))

include("notebooks.jl")
include_dependency("../Project.toml")
const _VERSION = VersionNumber(Pkg.TOML.parsefile(project_relative_path("Project.toml"))["version"])

using Zygote, Plots, MLJ, MLJLinearModels, MLJGLMInterface, Markdown, DataFrames
export plot_residuals!, fitted_linear_func, grid

function plot_residuals!(x, y, f; kwargs...)
    for (xi, yi) in zip(x, y)
        plot!(fill(xi, 2), [yi, f(xi)]; color = :red, linewidth = 2,
              label = nothing, kwargs...)

    end
end
function plot_residuals!(x1, x2, y, f; kwargs...)
    for (x1i, x2i, yi) in zip(x1, x2, y)
        plot!(fill(x1i, 2), fill(x2i, 2), [yi, f(x1i, x2i)];
              color = :red, linewidth = 2, label = nothing, kwargs...)
    end
end

_slope(::MLJLinearModels.LinearRegressor, θ̂) = θ̂.coefs[1][2]
_slope(::MLJGLMInterface.LinearRegressor, θ̂) = θ̂.coef[1]

function fitted_linear_func(mach)
    θ̂ = fitted_params(mach)
    θ̂₀ = θ̂.intercept
    θ̂₁ = _slope(mach.model, θ̂)
    x -> θ̂₀ + θ̂₁ * x
end;

function gradient_descent(f, x₀, η, T; callback = x -> nothing)
    x = copy(x₀) # use copy to not overwrite the input
    for t in 1:T
        x .-= η * gradient(f, x)[1] # update parameters in direction of -∇f
        callback(x) # the callback will be used to save intermediate values
    end
    x
end

_grid(x, y) = (repeat(x, length(y)), repeat(y, inner = length(x)))
function grid(x, y; names = (:X, :Y), output_format = NamedTuple)
    t = NamedTuple{names}(_grid(x, y))
    output_format(t)
end

polysymbol(x, d) = Symbol(d == 0 ? "" : d == 1 ? "$x" : "$x^$d")
function poly(data, degree)
    res = DataFrame([data.x .^ k for k in 1:degree],
                    [polysymbol("x", k) for k in 1:degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end
function poly2(data, degree)
    res = DataFrame([data.X1 .^ d1 .* data.X2 .^ d2
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree],
                    [Symbol(polysymbol("x₁", d1), polysymbol("x₂", d2))
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end

Base.@kwdef mutable struct PolynomialRegressor{T <: Deterministic} <: Deterministic
    degree::Int = 3
    regressor::T = LinearRegressor()
end
function MLJ.MLJBase.fit(model::PolynomialRegressor, verbosity, X, y)
    Xpoly = poly(X, model.degree)
    MLJ.MLJBase.fit(model.regressor, verbosity, Xpoly, y)
end
function MLJ.MLJBase.predict(model::PolynomialRegressor, fitresult, Xnew)
    Xpoly = poly(Xnew, model.degree)
    MLJ.MLJBase.predict(model.regressor, fitresult, Xpoly)
end


function start()
    sysimg = project_relative_path("precompile", "mlcourse.so")
    root = project_relative_path()
    exe = joinpath(Sys.BINDIR, "julia")
    script = :(using Pkg;
               Pkg.activate($root);
               using Pluto;
               Pluto.run(notebook = $(joinpath(root, "notebooks", "welcome.jl"))))
    if isfile(sysimg)
        run(`$exe -J$sysimg -e $script`)
    else
        run(`$exe -e $script`)
    end
end

function update()
    @info "Performing an automatic update while keeping local changes.
    If this fails, please run manually `git pull` in the directory
    `$(project_relative_path())`."
    current_dir = pwd()
    cd(project_relative_path())
    if !isempty(readlines(`git diff --stat`))
        run(`git add -u`)
        run(`git commit -m "automatic commit of local changes"`)
    end
    run(`git pull origin main -s recursive -X patience -X ours -X ignore-all-space --no-edit`)
    cd(current_dir)
    Pkg.activate(project_relative_path())
    Pkg.instantiate()
end

function create_sysimage()
    exe = joinpath(Sys.BINDIR, "julia")
    run(`$exe $(project_relative_path("precompile", "precompile.jl"))`)
end

if isfile(project_relative_path("precompile", "mlcourse.so"))
    @warn "You may have to create a new system image with this update."
end
@info """\n
    Welcome to the Machine Learning Course v$(_VERSION)!

    Get started with:

    julia> MLCourse.start()

    Or create a new system image for faster loading of the notebooks.
    This might take several minutes (~30 minutes on my laptop).
    During this process you may be asked to download data sets (answer: yes :))
    and a browser window will open, which you can close again. Do not interrupt
    the process. If all goes well it shows at the end for several minutes:
    `[ Info: PackageCompiler: creating system image object file, this might take a while...`

    julia> MLCourse.create_sysimage() # Be patient!
\n"""


end # module
