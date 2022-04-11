### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 34beb7d6-fba9-11eb-15b8-e34afccc9f88
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using CSV, DataFrames, Distributions, Plots, MLJ, MLJLinearModels, Random, OpenML
end

# ╔═╡ 4e30f307-7b07-4cb7-a34e-ada7e2f6946d
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ e688a9de-2dba-4fab-b4e6-9803c5361a62
begin
	using MLCourse
	MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
md"# Transformations of the Input

## Splines"

# ╔═╡ 026de033-b9b2-4165-80e7-8603a386381c
function spline_features(data, degree, knots)
	q = length(knots) + degree
	names = [Symbol("H", i) for i in 1:q]      # create the feature names
	features = [data[:, 1] .^ d for d in 1:degree] # compute the first features
	append!(features, [max.(0, (data[:, 1] .- k).^degree)
		               for k in knots])        # remaining features
	DataFrame(features, names)
end

# ╔═╡ 69b9c8ed-acf1-43bf-8478-3a9bc57bc36a
md"We will use an OpenML dataset that was collected in 1985 in the US to determine the factors influencing the wage of different individuals. Here we will just fit the wage as a function of the age, i.e. we will transform the age into a spline representation and perform linear regression on that. Wages are given in USD/hour (see `OpenML.describe_dataset(534)` for more details)." 

# ╔═╡ 6532d7c4-2385-401a-9e99-8dc86655687f
wage = OpenML.load(534) |> DataFrame

# ╔═╡ 28cba8c7-c6d9-4a62-92cc-4856353a1969
md"We can use now the `spline_features` function to compute spline features of the `AGE` predictor."

# ╔═╡ 933cf81e-39e7-428c-a270-c638e6ead6fe
spline_features(DataFrame(x = wage.AGE), 3, [30, 40, 60])

# ╔═╡ 75a2d9fa-2edd-4506-bb10-ee8e9c780626
md"Similarly to polynomial regression, we will use now the `spline_features` function to compute spline features as a first step in a pipeline and then perform linear regression on these spline features."

# ╔═╡ 001c88b3-c0db-4ad4-9c0f-04a2ea7b090d
md"spline degree = $(@bind spline_degree Slider(1:5, show_value = true))

knot 1 = $(@bind knot1 Slider(10:50, default = 30, show_value = true))

knot 2 = $(@bind knot2 Slider(20:60, default = 40, show_value = true))

knot 3 = $(@bind knot3 Slider(30:70, default = 50, show_value = true))

knot 4 = $(@bind knot4 Slider(40:80, default = 60, show_value = true))"

# ╔═╡ cc65c9c0-cd8d-432c-b697-8f90a7b831f9
spline_mod = Pipeline(x -> spline_features(x,
	                                       spline_degree,
				                           [knot1, knot2, knot3, knot4]),
			          LinearRegressor());

# ╔═╡ d35acc45-7404-4b47-98fe-6e13db8b657e
spline_fit = machine(spline_mod, select(wage, :AGE), wage.WAGE);

# ╔═╡ 836afa27-9e57-4717-9850-dbd1a470ee17
fit!(spline_fit, verbosity = 0);

# ╔═╡ 0815ba19-5c7d-40d6-b948-8edccfb5c386
begin grid = minimum(wage.AGE):maximum(wage.AGE)
	scatter(wage.AGE, wage.WAGE)
	plot!(grid, predict(spline_fit, DataFrame(x = grid)), w = 3)
	vline!([knot1, knot2, knot3, knot4], linestyle = :dash, legend = false)
end

# ╔═╡ 5774b82f-a09e-46f8-a458-0f85fc4c53d8
fitted_params(spline_fit)

# ╔═╡ 004958cf-0abb-4d8e-90b1-5966412cba91
md"## Categorical Predictors"

# ╔═╡ b43365c6-384e-4fde-95f3-b2ad1ebc421a
cdata = DataFrame(gender = ["male", "male", "female", "female", "female"],
                  treatment = [1, 2, 2, 1, 3])

# ╔═╡ a91c6298-98c4-4dc6-9450-279da1c34146
coerce!(cdata, :gender => Multiclass, :treatment => Multiclass)

# ╔═╡ 2755577e-8b4c-4cdc-902e-fe661aa91731
MLJ.transform(fit!(machine(OneHotEncoder(), cdata)), cdata)

# ╔═╡ 81fc52cd-e67b-4188-a028-79abcc77cb92
MLJ.transform(fit!(machine(OneHotEncoder(drop_last = true), cdata)), cdata)

# ╔═╡ e928e4f7-0ebc-4db5-b83e-0c5d22c0ff3c
md"We will now apply this transformation to fit the wage data with linear regression."

# ╔═╡ 1bf42b12-5fd1-4b4c-94c6-14a52317b15f
begin
	preprocessor = machine(OneHotEncoder(drop_last = true), select(wage, Not(:WAGE)))
	fit!(preprocessor, verbosity = 0)
	wage_input = MLJ.transform(preprocessor, select(wage, Not(:WAGE)))
end

# ╔═╡ 0a22b68c-1811-4de1-b3f8-735ee50299d2
wage_mach = machine(LinearRegressor(), wage_input, wage.WAGE) |> fit!;

# ╔═╡ 6323a352-a5f3-4ba7-93bd-903fb1bf22a0
fitted_params(wage_mach)

# ╔═╡ 85c9650d-037a-4e9f-a15c-464ad479ff44
md"From the fitted parameters we can see, for example, that years of education correlate positively with wage, women earn on average ≈2 USD per hour less than men (keeping all other factors fixed) or persons in a management position earn ≈4 USD per hour more than those in a sales position (keeping all other factors fixed)."

# ╔═╡ a10b3fa1-358b-4a94-b927-5e0f71edd1f3
md"## Vector Features

In this section we look at a version of the famous xor problem. It is a binary classification problem with 2-dimensional input. It is called xor problem, because the class label is true if and only if the ``X_1`` and ``X_2`` coordinate of a point have different sign, which is reminiscent of the [logical exclusive or operation](https://en.wikipedia.org/wiki/Exclusive_or). The decision boundary is non-linear. By taking the scalar product with 4 different 2-dimensional vectors and setting to 0 all scalar product that would be negative, we find a 4-dimensional feature representation for which linear logistic regression can classify each point correctly."

# ╔═╡ b66c7efc-8fd8-4152-9b0f-9332f760b51b
function xor_generator(; n = 200)
	x = 2 * rand(n, 2) .- 1
	DataFrame(X1 = x[:, 1], X2 = x[:, 2],
		      y = coerce((x[:, 1] .> 0) .⊻ (x[:, 2] .> 0), Binary))
end

# ╔═╡ ebe38042-5160-469b-8c07-484b7c019063
xor_data = xor_generator()

# ╔═╡ fa7cf2ae-9777-4ba9-9fe0-df545b94c5d8
begin
	scatter(xor_data.X1, xor_data.X2, c = int.(xor_data.y) .+ 1,
            label = nothing, xlabel = "X₁", ylabel = "X₂")
	hline!([0], label = nothing, c = :black)
	vline!([0], label = nothing, c = :black, size = (400, 300))
end

# ╔═╡ 99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
begin
	lin_mach = fit!(machine(LogisticClassifier(penalty = :none),
			                select(xor_data, Not(:y)),
			                xor_data.y))
	lin_pred = predict_mode(lin_mach, select(xor_data, Not(:y)))
	(training_misclassification_rate = mean(lin_pred .!= xor_data.y),)
end

# ╔═╡ aa002263-4dc8-4238-ab27-a9094947600c
md"## Data Cleaning

### Dealing with Missing Data

In the following artififial dataset the age and the gender is missing for some of the subjects.
"

# ╔═╡ c7dab115-b990-4760-85d0-4214d33ada5d
datam = DataFrame(age = [12, missing, 41, missing, 38, 35], gender = categorical([missing, "female", "male", "male", "male", "female"]))

# ╔═╡ e88f9e54-f86c-4d3a-80d7-1a9c5006476a
md"Removing rows with missing values is a safe way to deal with such datasets."

# ╔═╡ 89cbf454-8d1b-4c82-ae39-0672a225e7dd
dropmissing(datam)

# ╔═╡ 315f536b-4f10-4471-8e22-18c385ca80f2
md"Alternatively one can fill in the missing values with some standard values. For our artificial dataset the default `FillImputer` fills in the median of all age values rounded to the next integer and gender \"male\" because this dataset contains more \"male\"s than \"female\"s."

# ╔═╡ 69bc41cb-862c-4334-8b71-7f9974fedfe2
MLJ.transform(fit!(machine(FillImputer(), datam)), datam)

# ╔═╡ 381b1ad3-8b7c-4222-800b-6a138518c925
md"### Removing Predictors

Constant predictors should always be removed. For example, consider the following input data:"

# ╔═╡ 26c1ee4f-22b6-42be-9170-4710f7c0ad78
df = DataFrame(a = ones(5), b = randn(5), c = 1:5, d = 2:2:10, e = zeros(5))

# ╔═╡ 481ed397-9729-48a7-b2a7-7f90a2ba6581
md"Now we compute for each column the standard deviation and keep only those columns with standard deviation larger than 0."

# ╔═╡ fec248c4-b213-4617-b63a-c2be1f7017e6
df_clean_const = df[:, std.(eachcol(df)) .!= 0]

# ╔═╡ fa66ac8b-499d-49ae-be7d-2bdd3c1e6e0e
md"We can also check for perfectly correlated predictors using the `cor` function. For our data we find that column 3 and 2 are perfectly correlated:"

# ╔═╡ 68fdc6b3-423c-42df-a067-f91cf3f3a332
findall(≈(1), cor(Matrix(df_clean_const))) |> # find all indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)  # keep only upper off diagonal indices

# ╔═╡ 76e50a60-4056-4f0d-b4e0-3d67d4d771c2
md"Therefore we will only use one of those predictors."

# ╔═╡ 53d18056-7f76-4134-b73e-be4968d88901
df_clean = df_clean_const[:, [1, 2]]

# ╔═╡ 76155286-978b-4a05-b428-4be4089d8b9a
md"
### Standardization
Some methods (like k-nearest neighbors) are sensitive to rescaling of data. For example, in the following artificial dataset the height of people is measured in cm and their weights in kg, but one could have just as well used meters and grams or inches and pounds to report the height and the weight. By standardizing the data these arbitrary choices of the units do not matter anymore."

# ╔═╡ b2f03b4a-b200-4b30-8793-6edc02f8136d
height_weight = DataFrame(height = [165., 175, 183, 152, 171],
	                      weight = [60., 71, 89, 47, 70])

# ╔═╡ 794de3b3-df22-45c1-950d-a515c4f41409
height_weight_mach = fit!(machine(Standardizer(), height_weight));

# ╔═╡ adca81d5-2453-4954-91ba-7cc24d45e8ef
md"In the fitted parameters of the standardizing machine we see the mean and the standard deviation of the different columns of our dataset."

# ╔═╡ bf879ed5-4241-4abe-a42c-a562d49b21d2
fitted_params(height_weight_mach)

# ╔═╡ 2492bd04-cf24-4252-a37f-05fce8c1b87c
data_height_weight_s = MLJ.transform(height_weight_mach, height_weight)

# ╔═╡ 32020e6d-5ab2-4832-b565-4264ebcc82eb
md"The transformed data has now mean 0 and standard deviation 1 in every column."

# ╔═╡ fd19d841-f99e-46f4-a213-46be7a3d598d
combine(data_height_weight_s,
	    [:height, :weight] .=> mean,
	    [:height, :weight] .=> std)

# ╔═╡ 8a1a61a0-d657-4858-b8a7-d4e7d9bbf306
md"Note that standardization does not change the shape of the distribution of the data; it just shifts and scales the data, as we can see in the example below."

# ╔═╡ ee2f2a32-737e-49b5-885b-e7467a7b93f6
data = DataFrame(x = 10*rand(Beta(2, 8), 10^4) .+ 16);

# ╔═╡ cc550977-1f37-4ad1-8170-d6150cc01a9c
plot(histogram(data.x, nbins = 100))

# ╔═╡ f099be37-82d2-4031-955c-6a797c8400f2
begin
	st_mach = fit!(machine(Standardizer(), data))
	st_data = MLJ.transform(st_mach, data)
	plot(histogram(st_data.x, nbins = 100))
end

# ╔═╡ 75dda04c-c0af-4a05-b840-91f5c1aea53c
(mean(st_data.x), std(st_data.x))

# ╔═╡ a7f12901-e155-4f61-b252-c5af6669bfee
function vector_features(data, vectors...)
	q = length(vectors) # number of vector features
	names = [Symbol("H", i) for i in 1:q]
	feature_representation = [max.(0, data.X1 * v[1] .+ data.X2 * v[2])
		                      for v in vectors]
	DataFrame(feature_representation, names)
end

# ╔═╡ ad2cd0c5-e308-4911-aba9-3b56eda8a2e2
transformed_xor_input = vector_features(xor_data, [1, 1], [1, -1], [-1, 1], [-1, -1])

# ╔═╡ 1b8239fb-3894-47d6-87bf-4592b2fd078d
begin
	lin_mach_f = fit!(machine(LogisticClassifier(penalty = :none),
			                  transformed_xor_input,
			                  xor_data.y))
	lin_pred_f = predict_mode(lin_mach_f, transformed_xor_input)
	(training_misclassification_rate = mean(lin_pred_f .!= xor_data.y),)
end

# ╔═╡ cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
md"# Transformations of the Output

Sometimes it is reasonable to transform the output data. For example, if a method works best for normally distributed output data, it may be a good idea to transform the output data such that its distribution is closer to normal.
"

# ╔═╡ 5e6d6155-c254-4ae9-a0e1-366fc6ce6403
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), DataFrame);

# ╔═╡ ed239411-185a-4d9f-b0ec-360400f24dc7
normal_fit = Distributions.fit(Normal, weather.LUZ_wind_peak)

# ╔═╡ 7c04a091-93af-476e-8769-b14c08d3ae10
begin
	histogram(weather.LUZ_wind_peak, normalize = :pdf,
		      xlabel = "LUZ_wind_peak", label = nothing)
	plot!(x -> pdf(normal_fit, x), w = 2, label = "fitted Normal distribution")
end

# ╔═╡ 3c3aa56f-cb19-4115-9268-b73337fb6a49
lognormal_fit = Distributions.fit(Normal, log.(weather.LUZ_wind_peak))

# ╔═╡ 15faf5e7-fa72-4519-a83b-3007486450dd
begin
	histogram(log.(weather.LUZ_wind_peak), normalize = :pdf,
		      xlabel = "log(LUZ_wind_peak)", label = nothing)
	plot!(x -> pdf(lognormal_fit, x), w = 2, label = "fitted Normal distribution")
end

# ╔═╡ 1e4ba3ab-1bf1-495a-b9c4-b9ea657fc91c
md"Instead of transforming the output variable one can also assume that it has a specific distribution, e.g. a Gamma distribution."

# ╔═╡ 74f5178c-a308-4184-bc94-4a04f6f9ffdc
gamma_fit = Distributions.fit_mle(Gamma, weather.LUZ_wind_peak)

# ╔═╡ 62491145-48b9-4ac5-8e7b-0676e9616fe9
begin
	histogram(weather.LUZ_wind_peak, normalize = :pdf,
		      xlabel = "LUZ_wind_peak", label = nothing)
	plot!(x -> pdf(gamma_fit, x), w = 2, label = "fitted Gamma distribution")
end

# ╔═╡ 04397866-4e00-46d2-bda4-0a61cf34e780
md"# Exercises

## Conceptual
1. Suppose, for a regression problem with one-dimensional input ``X`` and one-dimensional output ``Y`` we use the feature functions ``h_1(X) = X``, ``h_2(X) = (X - 1)^2I(X\ge 1)`` (where ``I`` is the indicator function, being 1 when the condition inside the brackets is true and 0 otherwise), fit the linear regression model ``Y = \beta_0 + \beta_1h_1(X) + \beta_2h_2(X) + \varepsilon,`` and obtain coefficient estimates ``\hat{\beta}_0 = 1``, ``\hat{\beta}_1 = 1``, ``\hat{\beta}_2 = -2``. Draw the estimated curve between ``X = -2`` and ``X = 2``.
2. The following table shows one-hot encoded input for columns `mean_of_transport` (car, train, airplane, ship) and `energy_source` (petrol, electric). Indicate for each row the mean of transport and the energy source.
| car | train | airplane | petrol|
|-----|-------|----------|-------|
| 1   | 0     | 0        |  0    |
| 0   | 0     | 0        |  1    |
| 0   | 1     | 0        |  1    |
| 0   | 0     | 0        |  0    |

3. In the \"Vector Features\" section we said that the xor problem has a non-linear decision boundary in the original ``X_1, X_2`` coordinates, but a linear decision boundary in the ``H_1, H_2, H_3, H_4`` coordinates. Here we prove that ``-H_1 + H_2 + H_3 - H_4 = 0`` defines indeed a linear decision boundary that solves the xor-problem.
    * Show that ``-H_1 + H_2 + H_3 - H_4  < 0`` for all points with ``X_1 > 0`` and ``X_2 > 0``.
    * Show that ``-H_1 + H_2 + H_3 - H_4  < 0`` for all points with ``X_1 < 0`` and ``X_2 < 0``.
    * Show that ``-H_1 + H_2 + H_3 - H_4  > 0`` for all other points.

## Applied
1. Load the mushroom data set `OpenML.load(24)` and determine if a linear logistic regression can correctly classify all datapoints if
   * rows with missing values are dropped.
   * rows with missing values are imputed with the most common class.
2. You are given the following artificial dataset. The data is not linearly separable, meaning that there is no linear decision boundary that would perfectly seperate the blue from the red data points.
    * Find a 2-dimensional feature representation ``H_1 = f_1(X_1, X_2), H_2 = f_2(X_1, X_2)``, such that the transformed data is linearly separable.
    * Generate a similar plot as the one below for the transformed data to visualize that the decision boundary has become linear for the transformed data.
"

# ╔═╡ 9a0e0bf7-44e6-4385-ac46-9a6a8e4245bb
begin
	cldata = DataFrame(2*rand(400, 2).-1, :auto)
	cldata.y = cldata.x1.^2 .> cldata.x2
	cldata
end

# ╔═╡ 6f4427a0-18eb-4637-a19a-ec9aa7b6fda8
scatter(cldata.x1, cldata.x2,
        c = Int.(cldata.y) .+ 1, legend = false,
        xlabel = "X₁", ylabel = "X₂")

# ╔═╡ 951f6957-0ff7-4b2e-ba91-1b69122bbe47
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─4e30f307-7b07-4cb7-a34e-ada7e2f6946d
# ╠═34beb7d6-fba9-11eb-15b8-e34afccc9f88
# ╟─4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
# ╠═026de033-b9b2-4165-80e7-8603a386381c
# ╟─69b9c8ed-acf1-43bf-8478-3a9bc57bc36a
# ╠═6532d7c4-2385-401a-9e99-8dc86655687f
# ╟─28cba8c7-c6d9-4a62-92cc-4856353a1969
# ╠═933cf81e-39e7-428c-a270-c638e6ead6fe
# ╟─75a2d9fa-2edd-4506-bb10-ee8e9c780626
# ╠═cc65c9c0-cd8d-432c-b697-8f90a7b831f9
# ╠═d35acc45-7404-4b47-98fe-6e13db8b657e
# ╠═836afa27-9e57-4717-9850-dbd1a470ee17
# ╟─001c88b3-c0db-4ad4-9c0f-04a2ea7b090d
# ╟─0815ba19-5c7d-40d6-b948-8edccfb5c386
# ╠═5774b82f-a09e-46f8-a458-0f85fc4c53d8
# ╟─004958cf-0abb-4d8e-90b1-5966412cba91
# ╠═b43365c6-384e-4fde-95f3-b2ad1ebc421a
# ╠═a91c6298-98c4-4dc6-9450-279da1c34146
# ╠═2755577e-8b4c-4cdc-902e-fe661aa91731
# ╠═81fc52cd-e67b-4188-a028-79abcc77cb92
# ╟─e928e4f7-0ebc-4db5-b83e-0c5d22c0ff3c
# ╠═1bf42b12-5fd1-4b4c-94c6-14a52317b15f
# ╠═0a22b68c-1811-4de1-b3f8-735ee50299d2
# ╠═6323a352-a5f3-4ba7-93bd-903fb1bf22a0
# ╟─85c9650d-037a-4e9f-a15c-464ad479ff44
# ╟─a10b3fa1-358b-4a94-b927-5e0f71edd1f3
# ╠═b66c7efc-8fd8-4152-9b0f-9332f760b51b
# ╠═ebe38042-5160-469b-8c07-484b7c019063
# ╟─fa7cf2ae-9777-4ba9-9fe0-df545b94c5d8
# ╠═99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
# ╟─aa002263-4dc8-4238-ab27-a9094947600c
# ╠═c7dab115-b990-4760-85d0-4214d33ada5d
# ╟─e88f9e54-f86c-4d3a-80d7-1a9c5006476a
# ╠═89cbf454-8d1b-4c82-ae39-0672a225e7dd
# ╟─315f536b-4f10-4471-8e22-18c385ca80f2
# ╠═69bc41cb-862c-4334-8b71-7f9974fedfe2
# ╟─381b1ad3-8b7c-4222-800b-6a138518c925
# ╠═26c1ee4f-22b6-42be-9170-4710f7c0ad78
# ╟─481ed397-9729-48a7-b2a7-7f90a2ba6581
# ╠═fec248c4-b213-4617-b63a-c2be1f7017e6
# ╟─fa66ac8b-499d-49ae-be7d-2bdd3c1e6e0e
# ╠═68fdc6b3-423c-42df-a067-f91cf3f3a332
# ╟─76e50a60-4056-4f0d-b4e0-3d67d4d771c2
# ╠═53d18056-7f76-4134-b73e-be4968d88901
# ╟─76155286-978b-4a05-b428-4be4089d8b9a
# ╠═b2f03b4a-b200-4b30-8793-6edc02f8136d
# ╠═794de3b3-df22-45c1-950d-a515c4f41409
# ╟─adca81d5-2453-4954-91ba-7cc24d45e8ef
# ╠═bf879ed5-4241-4abe-a42c-a562d49b21d2
# ╠═2492bd04-cf24-4252-a37f-05fce8c1b87c
# ╟─32020e6d-5ab2-4832-b565-4264ebcc82eb
# ╠═fd19d841-f99e-46f4-a213-46be7a3d598d
# ╟─8a1a61a0-d657-4858-b8a7-d4e7d9bbf306
# ╠═ee2f2a32-737e-49b5-885b-e7467a7b93f6
# ╠═cc550977-1f37-4ad1-8170-d6150cc01a9c
# ╠═f099be37-82d2-4031-955c-6a797c8400f2
# ╠═75dda04c-c0af-4a05-b840-91f5c1aea53c
# ╠═a7f12901-e155-4f61-b252-c5af6669bfee
# ╠═ad2cd0c5-e308-4911-aba9-3b56eda8a2e2
# ╠═1b8239fb-3894-47d6-87bf-4592b2fd078d
# ╟─cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
# ╠═5e6d6155-c254-4ae9-a0e1-366fc6ce6403
# ╠═ed239411-185a-4d9f-b0ec-360400f24dc7
# ╟─7c04a091-93af-476e-8769-b14c08d3ae10
# ╠═3c3aa56f-cb19-4115-9268-b73337fb6a49
# ╟─15faf5e7-fa72-4519-a83b-3007486450dd
# ╟─1e4ba3ab-1bf1-495a-b9c4-b9ea657fc91c
# ╠═74f5178c-a308-4184-bc94-4a04f6f9ffdc
# ╟─62491145-48b9-4ac5-8e7b-0676e9616fe9
# ╟─04397866-4e00-46d2-bda4-0a61cf34e780
# ╠═9a0e0bf7-44e6-4385-ac46-9a6a8e4245bb
# ╠═6f4427a0-18eb-4637-a19a-ec9aa7b6fda8
# ╟─e688a9de-2dba-4fab-b4e6-9803c5361a62
# ╟─951f6957-0ff7-4b2e-ba91-1b69122bbe47
