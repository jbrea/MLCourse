### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 7aa547f8-25d4-488d-9fc3-f633f7f03f57
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI, MLJ, MLJLinearModels, Plots, LinearAlgebra
    plotly()
    PlutoUI.TableOfContents()
end

# ╔═╡ cb9f858a-f60a-11eb-3f0e-a9b68cf33921
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ Cell order:
# ╟─7aa547f8-25d4-488d-9fc3-f633f7f03f57
# ╟─cb9f858a-f60a-11eb-3f0e-a9b68cf33921
