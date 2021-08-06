### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 150d58e7-0e73-4b36-836c-d81eef531a9c
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI, MLJ, MLJLinearModels, Plots, LinearAlgebra
    gr()
    PlutoUI.TableOfContents()
end


# ╔═╡ e04c5e8a-15f8-44a8-845d-60acaf795813
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
md"# Ridge Regression (L2 Regularization)
"

# ╔═╡ 9e1e8284-a8c1-47a9-83d0-2d8fbd8ce005
n = 30; x = rand(n); y = 2.2x .+ .3 .+ .2randn(n);

# ╔═╡ 1009e251-59af-4f1a-9d0a-e96f4b696cad
md"λ₂ = $(@bind λ₂ Slider(0:2:100, show_value = true))"

# ╔═╡ 8bd483cc-f490-11eb-38a1-b342dd2551fd
begin
    regmean(x, λ) = 1/(length(x) + λ) * sum(x)
    function ridge_regression(x, y, λ)
       β₁ = (mean(x .* y) - mean(x) * regmean(y, λ))/
		    (mean(x.^2) - mean(x)*regmean(x, λ) + λ/length(y))
       β₀ = regmean(y, λ) - β₁ * regmean(x, λ)
       (β₀ = β₀, β₁ = β₁)
    end
    function updateβ₀(x̄, ȳ, β₁, l)
        tmp = ȳ - β₁ * x̄
        abs(tmp) > l ? tmp - sign(tmp) * l : 0.
    end
    function updateβ₁(x̄, ȳ, x2, xy, β₀, l)
        tmp = (x̄ * ȳ - xy - x̄ * sign(β₀) * l)
        abs(tmp) > l ? (tmp - sign(tmp) * l)/(x̄^2 - x2) : 0
    end
    function lasso(x, y, λ)
        x̄ = mean(x)
        ȳ = mean(y)
        x2 = mean(x.^2)
        xy = mean(x .* y)
        l = λ/length(x)
        β₁ = (x̄ * ȳ - xy)/(x̄^2 - x2)
        β₀ = ȳ - β₁ * x̄
        β₀old, β₁old = zero(β₀), zero(β₁)
        while β₀old != β₀ || β₁old != β₁
            β₀old, β₁old = β₀, β₁
            β₁ = updateβ₁(x̄, ȳ, x2, xy, β₀, l)
            β₀ = updateβ₀(x̄, ȳ, β₁, l)
        end
       (β₀ = β₀, β₁ = β₁)
    end
end;

# ╔═╡ 50ac0b07-ffee-40c3-843e-984b3c628282
l2coefs = ridge_regression(x, y, λ₂)

# ╔═╡ 58746554-ca5a-4e8e-97e5-587a9c2aa44c
let r = λ₂ == 0 ? 6 : norm([l2coefs...]),
    ccol = plot_color(:blue, .3),
    path = hcat([[ridge_regression(x, y, l)...] for l in 0:100]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y")
    plot!(x -> l2coefs.β₀ + x * l2coefs.β₁, w = 3, label = "ridge regression")
    p2 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    plot!(t -> r * sin(t), t -> r * cos(t), 0:.001:2π,
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l2coefs.β₀], [l2coefs.β₁], label = "current fit", markersize = 6, color = :red)
    p3 = plot(0:100, path[1, :], label = "β₀", xlabel = "λ₂", ylabel = "parameters")
    plot!(0:100, path[2, :], label = "β₁", ylims = (0, 2.4))
    scatter!([λ₂], [l2coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([λ₂], [l2coefs.β₁], label = nothing, markersize = 6, color = :red)
    plot(plot(p1, p2, layout = (1, 2)), p3,
         layout = (2, 1), size = (700, 600), cbar = false, legend = :right)
end

# ╔═╡ f43a82e2-1145-426d-8e0e-5363d1c38ccf
md"# Lasso (L1 Regression)"

# ╔═╡ ff7cc2bf-2a38-46d2-8d11-529159b08c82
md"λ₁ = $(@bind λ₁ Slider(0:1:30, show_value = true))"

# ╔═╡ 4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
l1coefs = lasso(x, y, λ₁)

# ╔═╡ ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
let r = λ₁ == 0 ? 10 : norm([l1coefs...], 1),
    ccol = plot_color(:blue, .3),
    path = hcat([[lasso(x, y, l)...] for l in 0:30]...)
    p1 = scatter(x, y, label = "data", xlabel = "x", ylabel = "y")
    plot!(x -> l1coefs.β₀ + x * l1coefs.β₁, w = 3, label = "lasso")
    p2 = contour(-1:.1:3, -1:.1:3, (β₀, β₁) -> mean((β₀ .+ β₁*x .- y).^2),
                 label = "loss",
                 levels = 100, aspect_ratio = 1, ylims = (-1, 3), xlims = (-1, 3))
    plot!([0, r, 0, -r, 0], [r, 0, -r, 0, r],
          fill = (0, ccol), label = "constraint", color = ccol)
    plot!(path[1, :], path[2, :], label = "path", color = :blue, w = 3)
    scatter!([l1coefs.β₀], [l1coefs.β₁], label = nothing, markersize = 6)
    p3 = plot(0:30, path[1, :], label = "β₀", xlabel = "λ₁", ylabel = "parameters")
    plot!(0:30, path[2, :], label = "β₁", ylims = (0, 2.4))
    scatter!([λ₁], [l1coefs.β₀], label = nothing, markersize = 6, color = :red)
    scatter!([λ₁], [l1coefs.β₁], label = nothing, markersize = 6, color = :red)
    plot(plot(p1, p2, layout = (1, 2)), p3,
         layout = (2, 1), size = (700, 600), cbar = false, legend = :right)
end


# ╔═╡ Cell order:
# ╟─78bdd11d-b6f9-4ba6-8b2e-6189c4005bf1
# ╠═9e1e8284-a8c1-47a9-83d0-2d8fbd8ce005
# ╟─1009e251-59af-4f1a-9d0a-e96f4b696cad
# ╠═50ac0b07-ffee-40c3-843e-984b3c628282
# ╟─8bd483cc-f490-11eb-38a1-b342dd2551fd
# ╟─58746554-ca5a-4e8e-97e5-587a9c2aa44c
# ╟─f43a82e2-1145-426d-8e0e-5363d1c38ccf
# ╟─ff7cc2bf-2a38-46d2-8d11-529159b08c82
# ╠═4841f9ba-f3d2-4c65-9225-bc8d0c0a9478
# ╟─ed2b7969-79cd-43c8-bcdb-34dab89c2cb0
# ╟─e04c5e8a-15f8-44a8-845d-60acaf795813
# ╟─150d58e7-0e73-4b36-836c-d81eef531a9c
