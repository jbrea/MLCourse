### A Pluto.jl notebook ###
# v0.19.27

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
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, CSV
import Distributions: Beta, Normal, Gamma
import Distributions, GLM
import PlutoPlotly as PP
const M = MLCourse.JlMod
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ 4e30f307-7b07-4cb7-a34e-ada7e2f6946d
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 000e5d2f-19da-42f8-8bd8-9a2a3943228d
md"The goal of this week is to
1. learn how to clean data.
2. understand and learn to apply feature engineering.
3. understand the implications of transforming the response variable.
"

# ╔═╡ aa002263-4dc8-4238-ab27-a9094947600c
md"# 1. Data Cleaning

## Dealing with Missing Data

In the following artififial dataset the age and the gender is missing for some of the subjects.
"

# ╔═╡ c7dab115-b990-4760-85d0-4214d33ada5d
mlcode(
"""
using DataFrames, MLJ

datam = DataFrame(age = [12, missing, 41, missing, 38, 35],
                  gender = categorical([missing, "female", "male", 
                                        "male", "male", "female"]))
"""
,
"""
import pandas as pd

datam = pd.DataFrame({
		"age":[12,None,41,None,38,35], 
		"gender": pd.Categorical([None,"female","male","male","male","female"])})
datam
"""
,
)

# ╔═╡ e88f9e54-f86c-4d3a-80d7-1a9c5006476a
md"### Drop missing data

Removing rows with missing values is a safe way to deal with such datasets, but it has the disadvantage that the dataset becomes smaller."

# ╔═╡ 89cbf454-8d1b-4c82-ae39-0672a225e7dd
mlcode(
"""
dropmissing(datam)
"""
,
"""
datam1=datam.copy()
datam1.dropna(inplace=True)
datam1
"""
)

# ╔═╡ 315f536b-4f10-4471-8e22-18c385ca80f2
mlstring(md"""### Imputation

Alternatively, one can fill in the missing values with some standard values.
This process is also called "imputation",
How to impute is not a priori clear; it depends a lot on domain knowledge and, for example, for time series one may want to impute differently than for stationary data.

For our artificial dataset the default $(mlstring(md"`FillImputer`", "")) fills in the median of all age values rounded to the next integer and gender \"male\" because this dataset contains more \"male\"s than \"female\"s.""",
"""
Alternatively, one can fill in the missing values with some standard values.
This process is also called "imputation",
How to impute is not a priori clear; it depends a lot on domain knowledge and, for example, for time series one may want to impute differently than for stationary data.

For our artificial dataset the default `SimpleImputer` fills in the the age values by the most_frequent age and gender \"male\" because this dataset contains more \"male\"s than \"female\"s. The parameter `strategy` define different way to create data.
""")

# ╔═╡ 69bc41cb-862c-4334-8b71-7f9974fedfe2
mlcode(
"""
mach = machine(FillImputer(), datam)
fit!(mach, verbosity = 0)
MLJ.transform(mach, datam)
"""
,
"""
from sklearn.impute import SimpleImputer
import numpy as np
imp_mean = SimpleImputer(strategy='most_frequent')
imp_mean.fit_transform(datam)
"""
)

# ╔═╡ 8fe50bf6-7471-4d4f-b099-8e3bd1c1a570
md"The advantage of imputation is that the size of the dataset remains the same. The disadvantage is that the data distribution may differ from the true data generator."

# ╔═╡ 381b1ad3-8b7c-4222-800b-6a138518c925
md"## Removing Predictors

Constant predictors should always be removed. For example, consider the following input data:"

# ╔═╡ 26c1ee4f-22b6-42be-9170-4710f7c0ad78
mlcode(
"""
df = DataFrame(a = ones(5), b = randn(5), c = 1:5, d = 2:2:10, e = zeros(5))
"""
,
"""
df = pd.DataFrame({"a": np.ones(5), "b" : np.random.randn(5), "c" : np.linspace(1,5,5), "d": np.linspace(2,10,5), "e" : np.zeros(5)})
df
"""
,
)

# ╔═╡ 481ed397-9729-48a7-b2a7-7f90a2ba6581
md"Now we compute for each column the standard deviation and keep only those columns with standard deviation larger than 0."

# ╔═╡ fec248c4-b213-4617-b63a-c2be1f7017e6
mlcode(
"""
df_clean_const = df[:, std.(eachcol(df)) .!= 0]
"""
,
"""
df_clean_const = df.loc[:, np.std(df, axis=0) != 0]
df_clean_const
"""
)

# ╔═╡ fa66ac8b-499d-49ae-be7d-2bdd3c1e6e0e
mlstring(md"We can also check for perfectly correlated predictors using the `cor` function. For our data we find that column 3 and 2 are perfectly correlated:"
,
"
We can also check for perfectly correlated predictors using the `cor` function of pandas. We could have also used the pearson correlation function of scipy for exemple. For our data we find that column 3 and 2 are perfectly correlated:
")	

# ╔═╡ 68fdc6b3-423c-42df-a067-f91cf3f3a332
mlcode(
"""
using Statistics

findall(≈(1), cor(Matrix(df_clean_const))) |> # indices with correlation ≈ 1
idxs -> filter(x -> x[1] > x[2], idxs)  # keep only upper off-diagonal indices
"""
,
"""
correlation = np.array(df_clean_const.corr().values) #compute correlation
correlation = np.triu(correlation, k=0) # remove values in double
np.fill_diagonal(correlation,0) # remove "self column" correlation
df_clean_const = df_clean_const.drop(df.columns[np.where(correlation==1)[1]], axis=1) #remove one of the column with exact correlation
df_clean_const
"""
)

# ╔═╡ 76e50a60-4056-4f0d-b4e0-3d67d4d771c2
md"Therefore we will only use one of those predictors."

# ╔═╡ 53d18056-7f76-4134-b73e-be4968d88901
mlcode(
"""
df_clean = df_clean_const[:, [1, 2]]
"""
,
"""
df_clean = df_clean_const
df_clean_const
"""
)

# ╔═╡ 76155286-978b-4a05-b428-4be4089d8b9a
md"
## Standardization
Some methods (like k-nearest neighbors) are sensitive to rescaling of data. For example, in the following artificial dataset the height of people is measured in cm and their weights in kg, but one could have just as well used meters and grams or inches and pounds to report the height and the weight. By standardizing the data these arbitrary choices of the units do not matter anymore."

# ╔═╡ b2f03b4a-b200-4b30-8793-6edc02f8136d
mlcode(
"""
height_weight = DataFrame(height = [165., 175, 183, 152, 171],
	                      weight = [60., 71, 89, 47, 70])
"""
,
"""
height_weight = pd.DataFrame({"height" : [165., 175, 183, 152, 171],
                            "weight" : [60., 71, 89, 47, 70] })
height_weight
"""
)

# ╔═╡ 794de3b3-df22-45c1-950d-a515c4f41409
mlcode(
"""
height_weight_mach = machine(Standardizer(), height_weight)
fit!(height_weight_mach, verbosity = 0)
fitted_params(height_weight_mach)
"""
,
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler 
"""
)

# ╔═╡ adca81d5-2453-4954-91ba-7cc24d45e8ef
md"In the fitted parameters of the standardizing machine we see the mean and the standard deviation of the different columns of our dataset.

In the following cell we see the standardized dataset."

# ╔═╡ 2492bd04-cf24-4252-a37f-05fce8c1b87c
mlcode(
"""
data_height_weight_s = MLJ.transform(height_weight_mach, height_weight)
"""
,
"""
scaled_data = pd.DataFrame(scaler.fit_transform(height_weight), 
							columns = ["height", "weight"])
scaled_data
"""
)

# ╔═╡ 32020e6d-5ab2-4832-b565-4264ebcc82eb
md"The transformed data has now mean 0 and standard deviation 1 in every column."

# ╔═╡ fd19d841-f99e-46f4-a213-46be7a3d598d
mlcode(
"""
combine(data_height_weight_s,
	    [:height, :weight] .=> mean,
	    [:height, :weight] .=> std)
"""
,
"""
("means : ", scaled_data.mean(), "standart deviation : ", scaled_data.std())
"""
)

# ╔═╡ 8a1a61a0-d657-4858-b8a7-d4e7d9bbf306
md"Note that standardization does not change the shape of the distribution of the data; it just shifts and scales the data.

We can see this in the histograms of an artificial dataset and its standardized version: the shape of the distribution remains the same, but the mean and the standard deviation change.

More precisely, assume ``X`` a random variable with probability density function ``f_X(x)`` (red curve in the figure below). Standardization is the (affine) transformation ``Z = \frac{X - \bar x}{\sigma}``, where ``\sigma`` is the standard deviation and ``\bar x`` the mean of the data. Then, the probability density function ``f_Z(z)`` (orange curve) of the scaled data ``Z`` is given by
```math
f_Z(z) = \sigma f_X\left(\sigma z+\bar x\right)
```
"

# ╔═╡ ee2f2a32-737e-49b5-885b-e7467a7b93f6
let d = 10*Beta(2, 6) + 16
    data = DataFrame(x = rand(d, 10^4))
    p1 = histogram(data.x, nbins = 100, normalize = :pdf, xlabel = "x", ylabel = "probability density", label = "original data", title = "original data")
    plot!(x -> pdf(d, x), c = :red, w = 2, label = "\$f_X(x)\$")
	st_mach = fit!(machine(Standardizer(), data), verbosity = 0)
    fp = fitted_params(st_mach)
    a = fp.stds[1]; b = fp.means[1]
	st_data = MLJ.transform(st_mach, data)
	p2 = histogram(st_data.x, nbins = 100, normalize = :pdf, xlabel = "x", ylabel = "probability density", label = "scaled data", title = "scaled data")
    plot!(z -> a*pdf(d, a*z + b), c = :orange, w = 2, label = "\$f_Z(z)\$")
    plot(p1, p2, layout = (1, 2), yrange = (0, .5), size = (700, 400))
end

# ╔═╡ 4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
md"# 2. Feature Engineering
"

# ╔═╡ 004958cf-0abb-4d8e-90b1-5966412cba91
md"## Categorical Predictors

For categorical predictors the canonical transformation is one-hot encoding.
"

# ╔═╡ b43365c6-384e-4fde-95f3-b2ad1ebc421a
mlcode(
"""
cdata = DataFrame(gender = ["male", "male", "female", "female", "female"],
                  treatment = [1, 2, 2, 1, 3])
coerce!(cdata, :gender => Multiclass, :treatment => Multiclass)
"""
,
"""
cdata = pd.DataFrame({
		'gender': pd.Categorical(['male', 'male', 'female', 'female', 'female']),
		'treatment': pd.Categorical([1, 2, 2, 1, 3])})
cdata
"""
)


# ╔═╡ 2755577e-8b4c-4cdc-902e-fe661aa91731
mlcode(
"""
m = fit!(machine(OneHotEncoder(), cdata), verbosity = 0)
MLJ.transform(m, cdata)
"""
,
"""
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
cdata_enc = enc.fit_transform(cdata).toarray()

("categories : ", enc.categories_, "data : ", cdata_enc)
"""
)

# ╔═╡ ffdcfb20-47df-4080-879a-9e41c9e482f4
md"""
!!! note "One-hot coding relative to a standard level"

    For fitting with an intercept it is important to encode categorical predicators relative to a standard level that is given by the intercept. This can be achieved by dropping one of the one-hot predictors.
"""

# ╔═╡ 81fc52cd-e67b-4188-a028-79abcc77cb92
mlcode(
"""
m = fit!(machine(OneHotEncoder(drop_last = true), cdata), verbosity = 0)
MLJ.transform(m, cdata)
"""
,
"""
enc = OneHotEncoder(drop='first')
cdata_enc = enc.fit_transform(cdata).toarray()
cdata_enc
"""
)

# ╔═╡ e928e4f7-0ebc-4db5-b83e-0c5d22c0ff3c
md"### Application to the wage data.

We use an OpenML dataset that was collected in 1985 in the US to determine the factors influencing the wage of different individuals.
Here we apply one-hot coding to the categorical predictors to fit the wage data with linear regression."

# ╔═╡ 8f6f096b-eb5c-42c5-af7b-fb9c6b787cb3
mlcode(
"""
using OpenML

wage = OpenML.load(534) |> DataFrame
"""
,
"""
import openml

wage,_,_,_ = openml.datasets.get_dataset(534).get_data(dataset_format="dataframe")
wage
"""
,
cache = false
)

# ╔═╡ cf54136a-cef4-4784-9f03-74784cdd3a88
mlcode("""
DataFrame(schema(wage))
"""
,
"""
wage.dtypes
""")

# ╔═╡ 1bf42b12-5fd1-4b4c-94c6-14a52317b15f
mlcode(
"""
preprocessor = machine(OneHotEncoder(drop_last = true), select(wage, Not(:WAGE)))
fit!(preprocessor, verbosity = 0)
MLJ.transform(preprocessor, select(wage, Not(:WAGE)))
"""
,
"""
#here we use OneHotEncoder on the all parameters
enc = OneHotEncoder(drop='first')
cdata_enc =enc.fit_transform(wage.drop(["WAGE"], axis=1)).toarray()
cdata_enc
"""
)

# ╔═╡ 0a22b68c-1811-4de1-b3f8-735ee50299d2
mlcode(
"""
using MLJLinearModels

wage_mach = machine(OneHotEncoder(drop_last = true) |> LinearRegressor(),
                    select(wage, Not(:WAGE)),
                    wage.WAGE)
fit!(wage_mach, verbosity = 0)
fitted_params(wage_mach)
"""
,
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer

#here we use OneHotEncoder only on the categorical parameters
categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

preprocessor = make_column_transformer(
    (OneHotEncoder(drop="if_binary"), categorical_columns),
    remainder="passthrough",
    verbose_feature_names_out=False,  # avoid to prepend the preprocessor names
)

pipeline = Pipeline ([("encoder",preprocessor),
					("regressor",LinearRegression())])

pipeline.fit(wage.drop(["WAGE"], axis=1),wage["WAGE"])

feature_names = pipeline[:-1].get_feature_names_out().tolist()
coefs = pd.DataFrame({"Parameters" : feature_names,
					"Coefficients" : pipeline[-1].coef_.reshape(-1), })
coefs
"""
)

# ╔═╡ 85c9650d-037a-4e9f-a15c-464ad479ff44
mlstring(md"From the fitted parameters we can see, for example, that years of education correlate positively with wage, women earn on average ≈2 USD per hour less than men (keeping all other factors fixed) or persons in a management position earn ≈4 USD per hour more than those in a sales position (keeping all other factors fixed).",
"The AGE coefficient is expressed in “dollars/hour per living years” while the EDUCATION one is expressed in “dollars/hour per years of education”. This representation of the coefficients has the benefit of making clear the practical predictions of the model: an increase of year in AGE means a decrease of 0.158028 dollars/hour, while an increase of year in EDUCATION means an increase of  0.812791 dollars/hour.


From the fitted parameters we can see, for example, Women earn on average ≈2 USD per hour less than men (keeping all other factors fixed) or persons in a management position earn ≈3 USD per hour more than those in a sales position (keeping all other factors fixed).")

# ╔═╡ f2955e10-dbe1-47c3-a831-66209d0f426a
md"## Splines

Splines are piecewise polynomial fits. Have a look at the slides for the feature construction. We use to custom code to compute the spline features (you do not need to understand the details of this implementation).
"

# ╔═╡ 026de033-b9b2-4165-80e7-8603a386381c
mlcode(
"""
function spline_features(data, degree, knots)
	q = length(knots) + degree
	names = [Symbol("H", i) for i in 1:q]      # create the feature names
	features = [data[:, 1] .^ d for d in 1:degree] # compute the first features
	append!(features, [(max.(0, data[:, 1] .- k)).^degree
		               for k in knots])        # remaining features
	DataFrame(features, names)
end
"""
,
"""
"""
,
showoutput = false,
collapse = "Custom code to compute spline features"
)

# ╔═╡ 69b9c8ed-acf1-43bf-8478-3a9bc57bc36a
md"We use again the wage dataset (see above). However, here we will just fit the wage as a function of the age, i.e. we will transform the age into a spline representation and perform linear regression on that. Wages are given in USD/hour (see `OpenML.describe_dataset(534)` for more details).

Here are the spline features of the `AGE` predictor.
Note that feature `H1` still contains the unchanged `AGE` data.
"

# ╔═╡ 001c88b3-c0db-4ad4-9c0f-04a2ea7b090d
md"spline degree = $(@bind spline_degree Slider(1:5, show_value = true))

knot 1 = $(@bind knot1 Slider(10:50, default = 30, show_value = true))

knot 2 = $(@bind knot2 Slider(20:60, default = 40, show_value = true))

knot 3 = $(@bind knot3 Slider(30:70, default = 50, show_value = true))

knot 4 = $(@bind knot4 Slider(40:80, default = 60, show_value = true))"

# ╔═╡ 75a2d9fa-2edd-4506-bb10-ee8e9c780626
md"Similarly to polynomial regression, we will use now the `spline_features` function to compute spline features as a first step in a pipeline and then perform linear regression on these spline features."

# ╔═╡ ac4e0ade-5a1f-4a0d-b7ba-f8386f5232f1
function spline_features(data, degree, knots) # doesn't work if taken from M. (don't know why...)
	q = length(knots) + degree
	names = [Symbol("H", i) for i in 1:q]      # create the feature names
	features = [data[:, 1] .^ d for d in 1:degree] # compute the first features
	append!(features, [(max.(0, data[:, 1] .- k)).^degree
		               for k in knots])        # remaining features
	DataFrame(features, names)
end;

# ╔═╡ 933cf81e-39e7-428c-a270-c638e6ead6fe
spline_features(select(M.wage, :AGE), spline_degree, [knot1, knot2, knot3, knot4])

# ╔═╡ cc65c9c0-cd8d-432c-b697-8f90a7b831f9
begin
spline_mod = Pipeline(x -> spline_features(x,
	                                       spline_degree,
				                           [knot1, knot2, knot3, knot4]),
			          LinearRegressor())
spline_fit = machine(spline_mod, select(M.wage, :AGE), M.wage.WAGE)
fit!(spline_fit, verbosity = 0)
end;

# ╔═╡ 5774b82f-a09e-46f8-a458-0f85fc4c53d8
fitted_params(spline_fit)

# ╔═╡ 0815ba19-5c7d-40d6-b948-8edccfb5c386
begin grid = minimum(M.wage.AGE):maximum(M.wage.AGE)
	scatter(M.wage.AGE, M.wage.WAGE)
	plot!(grid, predict(spline_fit, DataFrame(x = grid)), w = 3)
	vline!([knot1, knot2, knot3, knot4], linestyle = :dash, legend = false)
end

# ╔═╡ a10b3fa1-358b-4a94-b927-5e0f71edd1f3
md"## Vector Features

In this section we look at a version of the famous xor problem. It is a binary classification problem with 2-dimensional input. It is called xor problem, because the class label is true if and only if the ``X_1`` and ``X_2`` coordinate of a point have different sign, which is reminiscent of the [logical exclusive or operation](https://en.wikipedia.org/wiki/Exclusive_or). The decision boundary is non-linear. By taking the scalar product with 4 different 2-dimensional vectors and setting to 0 all scalar product that would be negative, we find a 4-dimensional feature representation for which linear logistic regression can classify each point correctly."

# ╔═╡ b66c7efc-8fd8-4152-9b0f-9332f760b51b
mlcode(
"""
function xor_generator(; n = 200)
	x = 2 * rand(n, 2) .- 1
	DataFrame(X1 = x[:, 1], X2 = x[:, 2],
		      y = coerce((x[:, 1] .> 0) .⊻ (x[:, 2] .> 0), Binary))
end
xor_data = xor_generator()
"""
,
"""
def xor_generator(n=200):
    x = 2 * np.random.rand(n, 2) - 1
    df = pd.DataFrame({'X1': x[:, 0], 'X2': x[:, 1], 'y': np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)})
    return df

xor_data = xor_generator()
xor_data
"""
,
cache = false
)

# ╔═╡ 8cd98650-7f52-40d9-b92d-ce564e57cfa7
mlcode("""
scatter(xor_data.X1, xor_data.X2, c = int.(xor_data.y) .+ 1,
		label = nothing, xlabel = "X₁", ylabel = "X₂")
hline!([0], label = nothing, c = :black)
vline!([0], label = nothing, c = :black, size = (400, 300))
""",
"""
import matplotlib.pyplot as plt

plt.scatter(xor_data['X1'], xor_data['X2'], c=np.array(xor_data['y'], dtype=int)+1, label=None, cmap ='coolwarm')
plt.axhline(y=0, color='black', linestyle='-', label=None)
plt.axvline(x=0, color='black', linestyle='-', label=None)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.show()
"""
)

# ╔═╡ 99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
mlcode(
"""
lin_mach = machine(LogisticClassifier(penalty = :none),
                   select(xor_data, Not(:y)),
                   xor_data.y)
fit!(lin_mach, verbosity = 0)
lin_pred = predict_mode(lin_mach, select(xor_data, Not(:y)))
(training_misclassification_rate = mean(lin_pred .!= xor_data.y),)
"""
,
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lin_mach = LogisticRegression(penalty=None)
lin_mach.fit(xor_data[['X1', 'X2']], xor_data['y'])
lin_pred = lin_mach.predict(xor_data[['X1', 'X2']])
training_misclassification_rate = np.mean(xor_data['y'] != lin_pred)

("training_misclassification_rate",training_misclassification_rate,)
"""
)

# ╔═╡ 1c13dd1a-83ce-47e7-98b5-b6a0d8d2426f
md"Now we transform the data to the vector features discussed in the slides."

# ╔═╡ 3da4ce70-d4ab-4d9b-841d-d1f3fb385b81
mlcode(
"""
xor_data_h = DataFrame(H1 = max.(0, xor_data.X1 + xor_data.X2),
	                   H2 = max.(0, xor_data.X1 - xor_data.X2),
	                   H3 = max.(0, -xor_data.X1 + xor_data.X2),
	                   H4 = max.(0, -xor_data.X1 - xor_data.X2),
                       y = xor_data.y)
"""
,
"""
xor_data_h = pd.DataFrame({
		'H1': np.maximum(0, xor_data['X1'] + xor_data['X2']), 
		'H2': np.maximum(0, xor_data['X1'] - xor_data['X2']), 
		'H3': np.maximum(0, -xor_data['X1'] + xor_data['X2']), 
		'H4': np.maximum(0, -xor_data['X1'] - xor_data['X2']), 
		'y': xor_data['y']})

xor_data_h
"""
)

# ╔═╡ d73c777a-f993-4ef8-a3f5-8f6717920436
md"... and we find that the training misclassification rate becomes 0 with this feature representation."

# ╔═╡ 60689196-ca58-4937-8999-4e6f6cd7bdbc
mlcode(
"""
lin_mach_h = machine(LogisticClassifier(penalty = :none),
                     select(xor_data_h, Not(:y)),
                     xor_data_h.y)
fit!(lin_mach_h, verbosity = 0)
lin_pred_h = predict_mode(lin_mach_h, select(xor_data_h, Not(:y)))
(training_misclassification_rate_h = mean(lin_pred_h .!= xor_data_h.y),)
"""
,
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lin_mach = LogisticRegression(penalty=None)
lin_mach.fit(xor_data_h.drop(["y"], axis=1), xor_data_h['y'])
lin_pred = lin_mach.predict(xor_data_h.drop(["y"], axis=1))
training_misclassification_rate = np.mean(xor_data_h['y'] != lin_pred)

("training_misclassification_rate", training_misclassification_rate,)
"""
)


# ╔═╡ cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
md"# 3. Transformations of the Output

Sometimes it is reasonable to transform the output data. For example, if a method works best for normally distributed output data, it may be a good idea to transform the output data such that its distribution is closer to normal.

Another reason can be that the output variable is strictly positive and has strong outliers above the mean: in this case a linear regression will be biased towards the strong outliers, but predictions may nevertheless produce negative results.

Let us illustrate this with the weather data, where the wind peak is a positive variable with strong outliers. In the plot below we see on the left that the wind peak is not at all normally distributed. However, the logarithm of the wind peak looks much closer to normally distributed.
"

# ╔═╡ 1e4ba3ab-1bf1-495a-b9c4-b9ea657fc91c
md"Instead of transforming the output variable one can also assume that it has a specific distribution, e.g. a Gamma distribution."

# ╔═╡ fe93168e-40b9-46cd-bbb6-7c44d198fd57
Markdown.parse("""In fact, the simple linear regression we did in section [Supervised Learning]($(MLCourse._linkname(MLCourse.rel_path("notebooks"), "supervised_learning.jl", ""))) has exactly the above mentioned disadvantages, as you can see by inspecting the red line below.

In contrast, if one fits a linear regression to the logarithm of the wind peak and computes the exponential of the predictions, one gets the orange curve, which is always positive and less biased towards outliers.
""")

# ╔═╡ f40889bb-8035-42fd-b7fc-e81584ed7b1d
MLCourse._eval(:(import GLM)) # hack to make @formula work in cell below

# ╔═╡ 6590e349-4647-4383-a52f-6af4f689a342
md"Here is the code to run the different kinds of regression."

# ╔═╡ ecce64bd-da0b-4d92-8234-00bb167a03e3
mlcode(
"""
using CSV, DataFrames
weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
                   DataFrame)
X = select(weather, :LUZ_pressure)
y = weather.LUZ_wind_peak

# Linear Regression
linreg = machine(LinearRegressor(), X, y)
fit!(linreg, verbosity = 0)
linreg_pred = predict(linreg)

# Linear Regression on Log-Responses
loglinreg = machine(LinearRegressor(), X, log.(y))
fit!(loglinreg, verbosity = 0)
loglinreg_pred = exp.(predict(loglinreg))

# Gamma Regresssion
import GLM

gls = GLM.glm(GLM.@formula(LUZ_wind_peak ~ LUZ_pressure), weather,
              GLM.Gamma(), GLM.InverseLink())
gammapred = GLM.predict(gls)
"""
,
"""
from sklearn import linear_model

weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv")
X = weather[['LUZ_pressure']]
y = weather['LUZ_wind_peak']

# Linear Regression
linreg = linear_model.LinearRegression()
linreg.fit(X, y)
linreg_pred = linreg.predict(X)

# Linear Regression on Log-Responses
loglinreg = linear_model.LinearRegression()
loglinreg.fit(X, np.log(y))
loglinreg_pred = np.exp(loglinreg.predict(X))

# Gamma Regression
gamreg= linear_model.GammaRegressor()
gamreg.fit(X, y)
gamreg_pred = gamreg.predict(X)
"""
)

# ╔═╡ 5e6d6155-c254-4ae9-a0e1-366fc6ce6403
weather = CSV.read(joinpath(@__DIR__, "..", "data", "weather2015-2018.csv"), 
                   DataFrame);

# ╔═╡ 3c3aa56f-cb19-4115-9268-b73337fb6a49
let X = select(weather, :LUZ_pressure), y = weather.LUZ_wind_peak
    mw1 = machine(LinearRegressor(), X, y)
    fit!(mw1, verbosity = 0)
    mw2 = machine(LinearRegressor(), X, log.(y))
    fit!(mw2, verbosity = 0)
    histogram2d(X.LUZ_pressure, y, markersize = 3, xlabel = "LUZ_pressure [hPa]",
	            label = nothing, colorbar = false, ylabel = "LUZ_wind_peak [km/h]", bins = (250, 200))
    xgrid = DataFrame(LUZ_pressure = 920:1020)
    gls = GLM.glm(GLM.@formula(LUZ_wind_peak ~ LUZ_pressure), weather, GLM.Gamma(), GLM.InverseLink())
    gammapred = GLM.predict(gls, DataFrame(LUZ_pressure = 920:1020))
    plot!(xgrid.LUZ_pressure, predict(mw1, xgrid), c = :red, linewidth = 3, label = "linear regression")
    plot!(xgrid.LUZ_pressure, gammapred, c = :green, linewidth = 3, label = "gamma regression")
    hline!([0], linestyle = :dash, c = :gray, label = nothing)
    plot!(xgrid.LUZ_pressure, exp.(predict(mw2, xgrid)), c = :orange, linewidth = 3, label = "linear regression on log(wind_peak)", size = (700, 450))
end

# ╔═╡ 74f5178c-a308-4184-bc94-4a04f6f9ffdc
gamma_fit = Distributions.fit_mle(Gamma, weather.LUZ_wind_peak);

# ╔═╡ 62491145-48b9-4ac5-8e7b-0676e9616fe9
begin
	histogram(weather.LUZ_wind_peak, normalize = :pdf,
		      xlabel = "LUZ_wind_peak", label = nothing)
	plot!(x -> pdf(gamma_fit, x), w = 2, label = "fitted Gamma distribution")
end

# ╔═╡ ed239411-185a-4d9f-b0ec-360400f24dc7
normal_fit = Distributions.fit(Normal, weather.LUZ_wind_peak);

# ╔═╡ 15faf5e7-fa72-4519-a83b-3007486450dd
let
	p1 = histogram(weather.LUZ_wind_peak, normalize = :pdf,
		      xlabel = "LUZ_wind_peak", label = nothing, ylabel = "probability density")
	plot!(x -> pdf(normal_fit, x), w = 2, label = "fitted Normal distribution")
    lognormal_fit = Distributions.fit(Normal, log.(weather.LUZ_wind_peak));
	p2 = histogram(log.(weather.LUZ_wind_peak), normalize = :pdf,
		      xlabel = "log(LUZ_wind_peak)", label = nothing)
	plot!(x -> pdf(lognormal_fit, x), w = 2, label = "fitted Normal distribution")
    plot(p1, p2, layout = (1, 2), size = (700, 500))
end

# ╔═╡ 04397866-4e00-46d2-bda4-0a61cf34e780
md"# Exercises

## Conceptual
#### Exercise 1
Suppose, for a regression problem with one-dimensional input ``X`` and one-dimensional output ``Y`` we use the feature functions ``h_1(X) = X``, ``h_2(X) = (X - 1)^2\chi(X\ge 1)`` (where ``\chi`` is the indicator function, being 1 when the condition inside the brackets is true and 0 otherwise), fit the linear regression model ``Y = \beta_0 + \beta_1h_1(X) + \beta_2h_2(X) + \varepsilon,`` and obtain coefficient estimates ``\hat{\beta}_0 = 1``, ``\hat{\beta}_1 = 1``, ``\hat{\beta}_2 = -2``. Draw the estimated curve between ``X = -2`` and ``X = 2``.
#### Exercise 2
The following table shows one-hot encoded input for columns `mean_of_transport` (car, train, airplane, ship) and `energy_source` (petrol, electric). Indicate for each row the mean of transport and the energy source.

| car | train | airplane | petrol|
|-----|-------|----------|-------|
| 1   | 0     | 0        |  0    |
| 0   | 0     | 0        |  1    |
| 0   | 1     | 0        |  1    |
| 0   | 0     | 0        |  0    |

#### Exercise 3 (optional)
In the \"Vector Features\" section we said that the xor problem has a non-linear decision boundary in the original ``X_1, X_2`` coordinates, but a linear decision boundary in the ``H_1, H_2, H_3, H_4`` coordinates, with ``H_i = \max(0, w_i^TX)`` and we use the vector features ``w_1 = [1, 1]``, ``w_2 = [1, -1]``, ``w_3 = [-1, 1]`` and ``w_4 = [-1, -1]``. Here we prove that ``-H_1 + H_2 + H_3 - H_4 = 0`` defines indeed a linear decision boundary that solves the xor-problem.
* Show that ``-H_1 + H_2 + H_3 - H_4  < 0`` for all points with ``X_1 > 0`` and ``X_2 > 0``.
* Show that ``-H_1 + H_2 + H_3 - H_4  < 0`` for all points with ``X_1 < 0`` and ``X_2 < 0``.
* Show that ``-H_1 + H_2 + H_3 - H_4  > 0`` for all other points.

## Applied
#### Exercise 4
Load the mushroom data set `OpenML.load(24)` and determine if a linear logistic regression can correctly classify all datapoints if
   * rows with missing values are dropped.
   * rows with missing values are imputed with the most common class.
"

# ╔═╡ f706c1fa-6cf1-4b12-b0c3-09565af33fd7
md"#### Exercise 5
You are given the following artificial dataset. The data is not linearly separable, meaning that there is no linear decision boundary that would perfectly seperate the blue from the red data points.
    * Find a 2-dimensional feature representation ``H_1 = f_1(X_1, X_2), H_2 = f_2(X_1, X_2)``, such that the transformed data is linearly separable.
    * Generate a similar plot as the one below for the transformed data to visualize that the decision boundary has become linear for the transformed data.
"

# ╔═╡ 9a0e0bf7-44e6-4385-ac46-9a6a8e4245bb
mlcode(
"""
cldata = DataFrame(2*rand(400, 2).-1, :auto)
cldata.y = cldata.x1.^2 .> cldata.x2
cldata
"""
,
"""
cldata = pd.DataFrame(2*np.random.rand(400, 2)-1, columns=['x1', 'x2'])
cldata['y'] = cldata['x1']**2 > cldata['x2']
cldata
"""
)

# ╔═╡ 6f4427a0-18eb-4637-a19a-ec9aa7b6fda8
mlcode(
"""
using Plots

scatter(cldata.x1, cldata.x2,
        c = Int.(cldata.y) .+ 1, legend = false,
        xlabel = "X₁", ylabel = "X₂")
"""
,
"""
plt.scatter(cldata["x1"], cldata["x2"], c=cldata["y"].astype(int) + 1, label=None, cmap ='coolwarm')
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.show()
"""
)

# ╔═╡ a81db4a1-e561-4796-896c-26133a8efc60
md"#### Exercise 6 (optional)
In Exercise 5 of \"Generalized Linear Regression\" we fitted the bike sharing data using only `:temp` and `:humidity` as predictors. The quality of the fit was not good. Here we try to improve the fit by including more predictors. Many predictors can be treated as categorical, e.g. even the `:hour`, which is actually an ordered, periodic integer, can be treated as categorical to give the linear model a lot of flexibility. Try out different transformations of the input until you find a linear Poisson model that fits the data clearly better than what we had in the previous Exercise. You can measure quality of fit by looking at the same plot as in the previous exercise or by using cross-validation.
"

# ╔═╡ e688a9de-2dba-4fab-b4e6-9803c5361a62
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 951f6957-0ff7-4b2e-ba91-1b69122bbe47
MLCourse.FOOTER

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─4e30f307-7b07-4cb7-a34e-ada7e2f6946d
# ╟─000e5d2f-19da-42f8-8bd8-9a2a3943228d
# ╟─aa002263-4dc8-4238-ab27-a9094947600c
# ╟─c7dab115-b990-4760-85d0-4214d33ada5d
# ╟─e88f9e54-f86c-4d3a-80d7-1a9c5006476a
# ╟─89cbf454-8d1b-4c82-ae39-0672a225e7dd
# ╟─315f536b-4f10-4471-8e22-18c385ca80f2
# ╟─69bc41cb-862c-4334-8b71-7f9974fedfe2
# ╟─8fe50bf6-7471-4d4f-b099-8e3bd1c1a570
# ╟─381b1ad3-8b7c-4222-800b-6a138518c925
# ╟─26c1ee4f-22b6-42be-9170-4710f7c0ad78
# ╟─481ed397-9729-48a7-b2a7-7f90a2ba6581
# ╟─fec248c4-b213-4617-b63a-c2be1f7017e6
# ╟─fa66ac8b-499d-49ae-be7d-2bdd3c1e6e0e
# ╟─68fdc6b3-423c-42df-a067-f91cf3f3a332
# ╟─76e50a60-4056-4f0d-b4e0-3d67d4d771c2
# ╟─53d18056-7f76-4134-b73e-be4968d88901
# ╟─76155286-978b-4a05-b428-4be4089d8b9a
# ╟─b2f03b4a-b200-4b30-8793-6edc02f8136d
# ╟─794de3b3-df22-45c1-950d-a515c4f41409
# ╟─adca81d5-2453-4954-91ba-7cc24d45e8ef
# ╟─2492bd04-cf24-4252-a37f-05fce8c1b87c
# ╟─32020e6d-5ab2-4832-b565-4264ebcc82eb
# ╟─fd19d841-f99e-46f4-a213-46be7a3d598d
# ╟─8a1a61a0-d657-4858-b8a7-d4e7d9bbf306
# ╟─ee2f2a32-737e-49b5-885b-e7467a7b93f6
# ╟─4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
# ╟─004958cf-0abb-4d8e-90b1-5966412cba91
# ╟─b43365c6-384e-4fde-95f3-b2ad1ebc421a
# ╟─2755577e-8b4c-4cdc-902e-fe661aa91731
# ╟─ffdcfb20-47df-4080-879a-9e41c9e482f4
# ╟─81fc52cd-e67b-4188-a028-79abcc77cb92
# ╟─e928e4f7-0ebc-4db5-b83e-0c5d22c0ff3c
# ╟─8f6f096b-eb5c-42c5-af7b-fb9c6b787cb3
# ╟─cf54136a-cef4-4784-9f03-74784cdd3a88
# ╟─1bf42b12-5fd1-4b4c-94c6-14a52317b15f
# ╟─0a22b68c-1811-4de1-b3f8-735ee50299d2
# ╟─85c9650d-037a-4e9f-a15c-464ad479ff44
# ╟─f2955e10-dbe1-47c3-a831-66209d0f426a
# ╟─026de033-b9b2-4165-80e7-8603a386381c
# ╟─69b9c8ed-acf1-43bf-8478-3a9bc57bc36a
# ╟─933cf81e-39e7-428c-a270-c638e6ead6fe
# ╟─001c88b3-c0db-4ad4-9c0f-04a2ea7b090d
# ╟─75a2d9fa-2edd-4506-bb10-ee8e9c780626
# ╟─5774b82f-a09e-46f8-a458-0f85fc4c53d8
# ╟─0815ba19-5c7d-40d6-b948-8edccfb5c386
# ╟─cc65c9c0-cd8d-432c-b697-8f90a7b831f9
# ╟─ac4e0ade-5a1f-4a0d-b7ba-f8386f5232f1
# ╟─a10b3fa1-358b-4a94-b927-5e0f71edd1f3
# ╟─b66c7efc-8fd8-4152-9b0f-9332f760b51b
# ╟─8cd98650-7f52-40d9-b92d-ce564e57cfa7
# ╟─99d3ae30-6bc2-47bf-adc4-1cc1d3ff178d
# ╟─1c13dd1a-83ce-47e7-98b5-b6a0d8d2426f
# ╟─3da4ce70-d4ab-4d9b-841d-d1f3fb385b81
# ╟─d73c777a-f993-4ef8-a3f5-8f6717920436
# ╟─60689196-ca58-4937-8999-4e6f6cd7bdbc
# ╟─cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
# ╟─15faf5e7-fa72-4519-a83b-3007486450dd
# ╟─1e4ba3ab-1bf1-495a-b9c4-b9ea657fc91c
# ╟─62491145-48b9-4ac5-8e7b-0676e9616fe9
# ╟─fe93168e-40b9-46cd-bbb6-7c44d198fd57
# ╟─3c3aa56f-cb19-4115-9268-b73337fb6a49
# ╟─f40889bb-8035-42fd-b7fc-e81584ed7b1d
# ╟─74f5178c-a308-4184-bc94-4a04f6f9ffdc
# ╟─6590e349-4647-4383-a52f-6af4f689a342
# ╟─ecce64bd-da0b-4d92-8234-00bb167a03e3
# ╟─5e6d6155-c254-4ae9-a0e1-366fc6ce6403
# ╟─ed239411-185a-4d9f-b0ec-360400f24dc7
# ╟─04397866-4e00-46d2-bda4-0a61cf34e780
# ╟─f706c1fa-6cf1-4b12-b0c3-09565af33fd7
# ╟─9a0e0bf7-44e6-4385-ac46-9a6a8e4245bb
# ╟─6f4427a0-18eb-4637-a19a-ec9aa7b6fda8
# ╟─a81db4a1-e561-4796-896c-26133a8efc60
# ╟─e688a9de-2dba-4fab-b4e6-9803c5361a62
# ╟─951f6957-0ff7-4b2e-ba91-1b69122bbe47
# ╟─34beb7d6-fba9-11eb-15b8-e34afccc9f88
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
