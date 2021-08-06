### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 736856ce-f490-11eb-3349-057c86edfe7e
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI, MLJ, MLJLinearModels, Plots, DataFrames, Random
    gr()
    PlutoUI.TableOfContents()
end

# ╔═╡ c693088f-7f80-4cdd-b9b5-65a50da732ac
begin
    using MLCourse
    import MLCourse: poly, poly2
    MLCourse.list_notebooks(@__FILE__)
end


# ╔═╡ 7eb6a060-f948-4d85-881a-4909e74c15bd
md"# Validation Set Approach"

# ╔═╡ 7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
begin
    f(x) = 0.3 + 2x - 0.8x^2 - 0.4x^3
    function data_generator(; n = 150, seed = 12)
        rng = MersenneTwister(seed)
        x = randn(rng, n)
        DataFrame(x = x, y = f.(x) .+ randn(rng, n))
    end
    function data_split(data;
                        idx_train = 1:50,
                        idx_valid = 51:100,
                        idx_test = 101:150)
        (train = data[idx_train, :],
         valid = data[idx_valid, :],
         test = data[idx_test, :])
    end
end;

# ╔═╡ fa9b4b9e-4e97-4e5c-865d-ad3ef288e4cf
data1 = data_split(data_generator(seed = 5))

# ╔═╡ 751880ec-1a82-4142-b875-177d436bbc72
begin
    preprocessor(degree) = X -> MLCourse.poly(select(X, Not(:y)), degree)
    function fit_and_evaluate(model, data, degree)
        p = preprocessor(degree)
        mach = fit!(machine(model, p(data.train), data.train.y))
        mean((predict(mach, p(data.valid)) .- data.valid.y).^2)
    end
end

# ╔═╡ 2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
losses1 = [fit_and_evaluate(LinearRegressor(), data1, degree) for degree in 1:10]

# ╔═╡ 91eecd2b-af18-4a63-9684-28950e604d1a
let i = argmin(losses1)
    plot(1:10, losses1, label = nothing)
    scatter!([i], [losses1[i]], label = nothing)
end

# ╔═╡ ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
losses = [fit_and_evaluate(LinearRegressor(),
                           data_split(data_generator(seed = seed)),
                           degree)
          for degree in 1:10, seed in 1:20]

# ╔═╡ 26362233-b006-423d-8fb5-7cd9150405b4
let i = argmin(losses, dims = 1)
    plot(losses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), losses[i], label = nothing)
end

# ╔═╡ 46bffe01-1a7c-405b-b3ec-bfe7570b8a3c
md"# Cross-Validation"

# ╔═╡ Cell order:
# ╟─7eb6a060-f948-4d85-881a-4909e74c15bd
# ╠═7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
# ╠═fa9b4b9e-4e97-4e5c-865d-ad3ef288e4cf
# ╠═751880ec-1a82-4142-b875-177d436bbc72
# ╠═2e02ac0d-c4d0-47ba-be57-445adeb6ab8b
# ╠═91eecd2b-af18-4a63-9684-28950e604d1a
# ╠═ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
# ╟─26362233-b006-423d-8fb5-7cd9150405b4
# ╟─46bffe01-1a7c-405b-b3ec-bfe7570b8a3c
# ╟─c693088f-7f80-4cdd-b9b5-65a50da732ac
# ╟─736856ce-f490-11eb-3349-057c86edfe7e
