module MLCourse

import Pkg
using Requires, PrecompilePlutoCourse
using Markdown, Base64

include("notebooks.jl")

function __init__()
ext = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"
PrecompilePlutoCourse.configure(@__MODULE__,
    start_notebook = pkgdir(@__MODULE__, "index.jl"),
    sysimage_path = pkgdir(@__MODULE__, "precompile", "mlcourse.$ext"),
    warmup_file = pkgdir(@__MODULE__, "precompile", "warmup.jl"),
    packages = ["Pluto", "Images", "PlotlyBase", "PlutoPlotly", "StatsBase", "ScientificTypes", "MLJLinearModels", "DataFrames", "Plots", "StatsPlots", "Distributions", "Flux", "Zygote", "ReinforcementLearning"],
    kwargs = (cpu_target="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)",)
)

@require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
using Zygote
"""
    gradient_descent(f, x₀, η, T; callback = x -> nothing)

Run gradient descent on `f` with initial condition `x₀` and learning rate `η`
for `T` steps. The function `callback` is executed after every update of the
of the parameters and takes the current value of the parameters as input.
"""
function gradient_descent(f, x₀, η, T; callback = x -> nothing)
    x = copy(x₀) # use copy to not overwrite the input
    for t in 1:T
        x .-= η * gradient(f, x)[1] # update parameters in direction of -∇f
        callback(x) # the callback will be used to save intermediate values
    end
    x
end
end # Zygote
@require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
import Plots: plot!, scatter, scatter!, annotate!
export plot_residuals!, biplot
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
"""
    biplot(pca; pc = (1, 2), score_style = :text, loading = :all)

Show a biplot for a fitted machine `pca` and components `pc` (by default the
first two components are shown. If `score_style` is `:text` the index of each
data point (row in the input data of a `PCA` machine) is plotted at the
coordinates that indicate the scores of that data point. If `score_style` is an
integer the scores are plotted as points of size `score_style`. If `loadings` is
`:all`, all loadings are shown. If `loadings` is an integer ``n``, the ``n``
largest loadings along the PCs defined by `pc` are shown. If `loadings` is a
vector or integers, the loadings for the input dimensions given in this list of
integers is shown.

### Example
```
julia> using MLJ, MLJMultivariateStatsInterface, DataFrames, Plots
julia> mach = fit!(machine(PCA(), DataFrame(randn(10, 4) * randn(4, 6), :auto)))
julia> biplot(mach)
julia> biplot(mach, loadings = 3)
julia> biplot(mach, pc = (1, 3), score_style = 4, loadings = [3, 4])
```
"""
function biplot(m;
                pc = (1, 2),
                score_style = :text,
                loadings = :all)
    scores = MLJ.transform(m, m.data[1])
    p = scatter(getproperty(scores, Symbol(:x, pc[1])),
                getproperty(scores, Symbol(:x, pc[2])),
                label = nothing, aspect_ratio = 1, size = (600, 600),
                xlabel = "PC$(pc[1])", ylabel = "PC$(pc[2])",
                framestyle = :axis, markeralpha = score_style == :text ? 0 : 1,
                c = :gray,
                txt = score_style == :text ? Plots.text.(1:nrows(scores), 8, :gray) :
                                             nothing,
                markersize = score_style == :text ? 0 :
                             isa(score_style, Int) ? score_style : 3)
    plot!(p[1], inset = (1, Plots.bbox(0, 0, 1, 1)),
          right_margin = 10Plots.mm, top_margin = 10Plots.mm)
    p2 = p[1].plt.subplots[end]
    plot!(p2, aspect_ratio = 1, mirror = true, legend = false)
    params = fitted_params(m)
    ϕ = if hasproperty(params, :pca)
        params.pca.projection
    else
        params.projection
    end
    n = colnames(m.data[1])
    idxs = if loadings == :all
        1:length(n)
    elseif isa(loadings, Int)
        l = reshape(sum(abs2.(ϕ[:, [pc...]]), dims = 2), :)
        sortperm(l, rev = true)[1:loadings]
    else
        loadings
    end
    for i in idxs
        plot!(p2, [0, .9*ϕ[i, pc[1]]], [0, .9*ϕ[i, pc[2]]],
              c = :red, arrow = true)
        annotate!(p2, [(ϕ[i, pc[1]], ϕ[i, pc[2]],
                        (n[i], :red, :center, 8))])
    end
    scatter!(p2, 1.1*ϕ[:, pc[1]], 1.1*ϕ[:, pc[2]],
             markersize = 0, markeralpha = 0) # dummy for sensible xlims and xlims
    plot!(p2,
          background_color_inside = Plots.RGBA{Float64}(0, 0, 0, 0),
          tickfontcolor = Plots.RGB(1, 0, 0))
    p
end
end # Plots
@require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" begin
using DataFrames
polysymbol(x, d) = Symbol(d == 0 ? "" : d == 1 ? "$x" : "$x^$d")

colnames(df::AbstractDataFrame) = names(df)
colnames(d::NamedTuple) = keys(d)
colname(names, predictor::Int) = names[predictor]
colname(names, predictor::Symbol) = predictor ∈ names || string(predictor) ∈ names ? Symbol(predictor) : error("Predictor $predictor not found in $names.")
function poly(data, degree, predictors::NTuple{1} = (1,))
    cn = colnames(data)
    col = colname(cn, predictors[1])
    res = DataFrame([getproperty(data, col) .^ k for k in 1:degree],
                    [polysymbol(col, k) for k in 1:degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end
function poly(data, degree, predictors::NTuple{2})
    cn = colnames(data)
    col1 = colname(cn, predictors[1])
    col2 = colname(cn, predictors[2])
    res = DataFrame([getproperty(data, col1) .^ d1 .* getproperty(data, col2) .^ d2
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree],
                    [Symbol(polysymbol(col1, d1), polysymbol(col2, d2))
                     for d1 in 0:degree, d2 in 0:degree if 0 < d1 + d2 ≤ degree])
    if hasproperty(data, :y)
        res.y = data.y
    end
    res
end
poly(_, _, _) = error("Polynomials in more than 2 predictors are not implemented.")
end # DataFrames
@require MLJ="add582a8-e3ab-11e8-2d5e-e98b27df1bc7" begin
using MLJ
Base.@kwdef mutable struct Polynomial{T} <: Static
    degree::Int = 3
    predictors::T = (1,)
end
MLJ.transform(p::Polynomial, _, X) = poly(X, p.degree, p.predictors)
export Polynomial
PolynomialRegressor(; kwargs...) = error("`PolynomialRegressor(degree, regressor) is deprecated. Use e.g. `@pipeline(Polynomial(degree), LinearRegressor())` instead.")
end # MLJ
end # __init__

function fitted_linear_func(mach)
    θ̂ = fitted_params(mach)
    θ̂₀ = θ̂.intercept
    θ̂₁ = θ̂.coefs[1][2]
    x -> θ̂₀ + θ̂₁ * x
end;

_grid(x, y) = (repeat(x, length(y)), repeat(y, inner = length(x)))
function grid(x, y; names = (:X, :Y), output_format = NamedTuple)
    t = NamedTuple{names}(_grid(x, y))
    output_format(t)
end

function embed_figure(name)
    "![](data:img/png; base64,
         $(open(base64encode,
                pkgdir(@__MODULE__, "notebooks", "figures", name))))"
end

heaviside(x::T) where T = ifelse(x > 0, one(T), zero(T))

end # module
