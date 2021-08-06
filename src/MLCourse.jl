module MLCourse

import Pkg

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(MLCourse))), xs...))

include("notebooks.jl")
include_dependency("../Project.toml")
const _VERSION = VersionNumber(Pkg.TOML.parsefile(project_relative_path("Project.toml"))["version"])

using RDatasets, Zygote, Plots, MLJ, MLJLinearModels, MLJGLMInterface,
      MLJScikitLearnInterface, Markdown
export description, gradient_descent, plot_residuals!, fitted_linear_func, grid

struct RDatasetsDescription
    content::String
end
function description(package_name::AbstractString, dataset_name::AbstractString)
    RDatasetsDescription(read(joinpath(pkgdir(RDatasets), "doc",
                                       package_name, "$dataset_name.html"), String))
end
function Base.show(io::IO, mime::MIME"text/plain", d::RDatasetsDescription)
    nohtml = replace(d.content, Regex("<[^>]*>") => "")
    s = replace(nohtml, Regex("\n\n+") => "\n\n")
    show(io, mime, Docs.Text(s))
end
function Base.show(io::IO, mime::MIME"text/html", d::RDatasetsDescription)
    show(io, mime, HTML(d.content))
end

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

function gradient_descent(f, x0;
        learning_rate = .1, maxiters = 10^4, callback = x -> nothing)
    x = copy(x0)
    for i in 1:maxiters
        x .-= learning_rate * gradient(f, x)[1]
        callback(x)
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
