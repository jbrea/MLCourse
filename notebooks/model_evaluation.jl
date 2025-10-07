### A Pluto.jl notebook ###
# v0.19.46

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

# ╔═╡ 736856ce-f490-11eb-3349-057c86edfe7e
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, CSV
import Distributions, GLM
import Distributions: Normal, Poisson, Beta, Gamma
import MLCourse: poly, Polynomial
import PlutoPlotly as PP
const M = MLCourse.JlMod
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ e1077092-f72a-42af-b6e0-a616f938cba8
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ b70d9848-91a7-4de0-b93e-8078acfd77f8
md"The goal of this week is to
1. Understand the validation set approach for test error estimation and hyper-parameter tuning.
2. Understand the cross-validation approach for test error estimation and hyper-parameter tuning.
3. Learn how to use different resampling strategies, like the validation set approach or cross-validation.
4. Learn how to do hyper-parameter tuning.
5. learn how to clean data.
6. understand and learn to apply feature engineering.
7. understand the implications of transforming the response variable.
"

# ╔═╡ 7eb6a060-f948-4d85-881a-4909e74c15bd
md"# 1. Validation Set Approach

In the validation set approach, the points in a given dataset are shuffled and split into a training and a test set.
"

# ╔═╡ 5049407c-bf93-41a5-97bd-863a69d7016a
md"## Test Error Estimation

We estimate the test error with the validation set approach for fixed hyper-parameters. We use a polynomial fit in the example below.

The `shuffle seed` slider is used to shuffle the 500 training points in different ways.
For `shuffle seed = 0` the data set is not shuffled. For all other values of the `shuffle seed` it is shuffled.
We use a 50%-50% split into training and test set. In this section we use the terms \"validation set\" and \"test set\" interchangeably.

The dataset is obtained from an artificial data generator with mean given by a third-order polynomial and Gaussian noise with variance 1.

Below you see the indices of the training and the validation set"

# ╔═╡ 6cfc32b3-40c0-4149-ba67-2ec67b3938f3
md"shuffle seed = $(@bind split_seed Slider(0:20, show_value = true))"

# ╔═╡ b79fae90-87f3-4f89-933b-d82e76c94d81
md"We see that the resulting curve and the test error depends strongly on the training set, i.e. the indices that are selected for the training set. The training error is always close to the irreducible error in this case."

# ╔═╡ 1320b9b0-d6fc-4e61-8d62-de3988b8b21d
idxs = let idxs = split_seed == 0 ? collect(1:500) : randperm(Xoshiro(split_seed), 500) # shuffle the indices
(training_idxs = idxs[1:250],
 test_idxs = idxs[251:end])
end;

# ╔═╡ 448e4556-9448-4cc0-a00d-bb80b6b0799c
idxs

# ╔═╡ 7d30f043-2915-4bba-aad5-cb7dbe76a3e5
md"## Hyper-Parameter Tuning

Now we use a nested validation set approach to find the optimal degree in polynomial regression and estimate the test error. For this we split the data into training, validation and test set.
* We use a 10%-10%-80% split. 
* For each hyper-parameter value we fit the parameters on the training set and estimate the test error on the validation set. 
* When we have found the best hyper-parameters based on the validation error, we re-fit a model with found degree on the joint training and validation set (20% of the data) 
* Finally, we estimate the test error of this best model using the test set.

Using the `shuffle seed` slider below, you can see that the results depend a lot on how the data is shuffled.
"

# ╔═╡ 0af29197-3e02-4373-b47a-de9a79e9cb55
md"shuffle seed $(@bind split_seed2 Slider(0:20, show_value = true))"

# ╔═╡ 4461d5b0-bc9b-4c89-aa76-70524a5caa7c
md"Let us summarize the results of all 21 seeds in one figure."

# ╔═╡ ffffa3d0-bffb-4ff8-9966-c3312f952ac5
begin
	function train_val_test_set(; n = 500,
                                  seed = 1,
                                  rng = Xoshiro(seed),
                                  shuffle = true,
                                  f_train = 1//10,
                                  f_valid = 1//10,
                                  n_train = floor(Int, n*f_train),
                                  n_valid = floor(Int, n*f_valid),
                                  train_idxs = 1:n_train,
                                  valid_idxs = n_train+1:n_train + n_valid,
                                  test_idxs = n_train+n_valid+1:n
                                  )

        idxs = shuffle ? randperm(rng, n) : collect(1:n)
        (train_idxs = idxs[train_idxs],
         valid_idxs = idxs[valid_idxs],
         test_idxs = idxs[test_idxs])
	end
	function data_split(data; kwargs...)
        idxs = train_val_test_set(; n = size(data, 1), kwargs...)
        (train = data[idxs.train_idxs, :],
         valid = data[idxs.valid_idxs, :],
         test = data[idxs.test_idxs, :])
    end
    function fit_and_evaluate(model, data)
        mach = fit!(machine(model, select(data.train, :x), data.train.y),
                    verbosity = 0)
        (train_rmse = rmse(predict(mach, select(data.train, :x)), data.train.y),
         valid_rmse = length(data.valid.y) > 0 ? rmse(predict(mach, select(data.valid, :x)), data.valid.y) : missing,
         test_rmse = length(data.test.y) > 0 ? rmse(predict(mach, select(data.test, :x)), data.test.y) : missing)
    end
end;

# ╔═╡ 46bffe01-1a7c-405b-b3ec-bfe7570b8a3c
md"# 2. Cross-Validation

## Hyper-Parameter Tuning with Cross-Validation
Here we would like to find the optimal hyper-parameters with cross-validation.
As we will not estimate the final test error of the model with the best hyper-parameters, we use all the data for cross-validation. This setting is therefore quite different to the one in the previous section, where we used a large chunk of the data for testing.


We would like to see, whether hyper-parameter tuning with cross-validation finds the optimal hyper-parameter reliably.
"

# ╔═╡ 4c40a918-b3fb-4528-8508-c36579d0f6fc
md"shuffle seed = $(@bind cv_shuffle_seed Slider(0:20; show_value = true))"

# ╔═╡ 0ef33f9e-6013-445a-8192-9c9c1c047204
md"Let us summarize again all the results for different shuffling of the data set in one figure."

# ╔═╡ 0b81c3d5-277a-4fe6-889b-550e2f83c39d
md"With cross-validation we find quite often degree = 3 as the optimal one. This means that hyper-parameter tuning with cross-validation seems to be able to recover the degree of the generator quite often."

# ╔═╡ b3aed705-5072-4d9b-bb8f-865ac1561bf6
begin
    function cross_validation_sets(n, K; seed = 1,
                                         rng = Xoshiro(seed),
                                         shuffle = false,
                                         idxs = shuffle ? randperm(rng, n) : collect(1:n))
        r = n ÷ K
        [let idx_valid = idxs[(i-1)*r+1:(i == K ? n : i*r)]
             (train_idxs = setdiff(idxs, idx_valid),
              valid_idxs = idx_valid)
         end
         for i in 1:K]
    end
    function cross_validation(model, data; sets = nothing, K = nothing, kwargs...)
        sets = !isnothing(sets) ? sets : cross_validation_sets(size(data, 1), K; kwargs...)
        losses = [fit_and_evaluate(model,
                                   data_split(data;
									          test_idxs = [], # no test set
									          idxs...)) # training and validation
                  for idxs in sets]
        (train_rmse = mean(getproperty.(losses, :train_rmse)),
         valid_rmse = mean(getproperty.(losses, :valid_rmse)))
    end
end;

# ╔═╡ 5747d116-1e61-45f0-a87b-89372c6f270f
md"## Tuning Hyper-parameters with cross-validation and estimating the test error with the validation set approach

Instead of using all data for hyper-parameter search with cross-validation, we could set some data aside as a test set and use this to estimate the test error of the model with the optimal hyper-parameters. This setting is much closer to hyper-parameter tuning with the validation set approach above: in both cases we use 80% of the data for testing and only use 20% of the data for hyper-parameter tuning. In contrast to the validation set approach above, we run hyper-parameter tuning with cross-validation. Once the best degree is found, we refit the model with the best degree on all data except the test set and estimate the test error on the test set.
"

# ╔═╡ c268fb85-e695-4b18-8baf-2ab4e656102f
md"shuffle seed $(@bind cv_shuffle_seed2 Slider(0:20, show_value = true))"

# ╔═╡ 8305fadb-1f70-4668-864b-f5da68baf99c
md"This indicates clearly, that cross-validation is preferable over the validation set approach."

# ╔═╡ e2658128-4053-4484-8e1e-229eceb755ab
md"""# 3. Running Different Resampling Strategies
In this section you find code snippets that illustrate how different resampling strategies are implemented in common machine learning frameworks.

"""

# ╔═╡ 1a5eadf3-6dcb-4f47-8668-58d029032aca
mlstring(
         md"""In `MLJ` there are the following resampling strategies: $(join(last.(split.(string.(subtypes(ResamplingStrategy)), '.')), ", ")). `Holdout` is used for the evaluation set approach, `CV` for cross-validation."""
			 ,
         "")


# ╔═╡ cff653c5-9c20-4952-917a-fb8f265387d8
md"""
## Validation Set Approach

In the following cell we generate an artificial data, fit a polynomial of degree 4 to a training set that consists of 50% of the full dataset (shuffled) and test the fit on the test set.
"""

# ╔═╡ 30fe1445-6ddb-4ef2-8dcd-d41ff703da40
mlcode(
"""
using MLJ, MLJLinearModels
using MLCourse: Polynomial

X, y = make_regression(100, 1) # generate some artificial data
evaluate!(machine(Polynomial(degree = 4) |> LinearRegressor(), X, y),
          resampling = Holdout(shuffle = true, fraction_train = 0.5),
          measure = rmse, verbosity = 0)
"""
,
"""
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100, n_features=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,shuffle=True)

pipeline = Pipeline ([("polynomial",PolynomialFeatures(degree=4)),("regressor",LinearRegression())])
pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred, squared=False)
("MSE " , mse)
"""
)

# ╔═╡ d36a10f4-f61b-4275-9dde-8defba357a0c
md"""## Cross-Validation


In the following cell we generate an artificial data, fit a polynomial of degree 4 to 10 training sets of a 10 fold cross-validation and estimate the test loss on the validation set of each fold.
"""

# ╔═╡ 8ae790b3-3987-4f41-8e21-adbb71081eb9
mlcode(
"""
using MLJ, MLJLinearModels

X, y = make_regression(100, 1) # generate some artificial data
evaluate!(machine(Polynomial(degree = 4) |> LinearRegressor(), X, y),
          resampling = CV(nfolds = 10), measure = rmse, verbosity = 0)
"""
,
"""
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=100, n_features=1)  # generate some artificial data

pipeline = Pipeline ([("polynomial",PolynomialFeatures(degree=4)),("regressor",LinearRegression())])

scores = cross_val_score(pipeline, X, y, cv=10, scoring='neg_root_mean_squared_error')

("NRMSE : mean", scores.mean(), "Standart deviation", scores.std(), "per_fold ",scores)
"""
)

# ╔═╡ 29683c99-6a6a-4f65-bea2-d592895d887e
md"""# 4. Model Tuning

Finding good hyper-parameters (tuning) is such an important step in the process of finding good machine learning models that there exist some nice utility functions to tune the hyper-parameters $(mlstring(md"(see e.g. [tuning section in the MLJ manual](https://juliaai.github.io/MLJ.jl/dev/tuning_models/))", "")).

## Simple Tuning without Test Error Estimation
In the cell below you see an example where the degree of polynomial regression is automatically tuned on a given dataset by performing 10-fold cross-validation on all degrees on a \"grid\", i.e. of all degrees from 1 to 17 are tested."""

# ╔═╡ f93f20db-4fed-481f-b085-ca744b68fa8f
mlcode(
"""
using MLJ, MLJLinearModels, DataFrames

model = Polynomial() |> LinearRegressor()
self_tuning_model = TunedModel(model = model, # the model to be tuned
                               resampling = CV(nfolds = 10), # how to evaluate
                               range = range(model,
                                             :(polynomial.degree), # hyperparameter
                                             values = 1:17),       # values to check
                               measure = rmse) # evaluation measure
X, y = make_regression(100, 1) # generate some artificial data
self_tuning_mach = machine(self_tuning_model, DataFrame(X), y) # make machine
fit!(self_tuning_mach, verbosity = 0)
report(self_tuning_mach) # show the result
"""
,
"""
from sklearn.model_selection import GridSearchCV
import numpy as np

X, y = make_regression(n_samples=100, n_features=1)  # generate some artificial data

pipeline = Pipeline ([("polynomial",PolynomialFeatures()),("regressor",LinearRegression())])

param_grid = [
    {'polynomial__degree': np.arange(1,18,1)} # hyperparameter and values to check
  ]

grid_search = GridSearchCV(pipeline, param_grid, cv=10,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X, y)
("best_accuracy",grid_search.best_score_, "best_parameters", grid_search.best_params_)
"""
)

# ╔═╡ a41302ae-e2d8-4e5d-8dac-198bed16217b
mlstring(
md"The `range(model, :hyper, values = ...)` function in the cell above returns a `NominalRange(polynomial.degree = 1, 2, 3, ...)` to the tuner. This specifies the degrees of the polynomial that should be tested during tuning.

In general, for any given model you may first want to find out, how the parameters are called that you would like to tune. This can be done by inspecting the model.
In the output below you see that the model is a `DeterministicPipeline`, with a `polynomial` and a `linear_regressor`. To tune the degree of the `polynomial` we choose `:(polynomial.degree) in the range function above.

Have a look at the best models found for each fold in the output above.
"
,
md"
In general, for any given model you may first want to find out, how the parameters are called that you would like to tune. This can be done by inspecting the model. The model is a `Pipeline`, with a `polynomial` and a `linear_regressor`.
We can see in the Pipeline that the transform `PolynomialFeatures` is named `polynomial`. To tune the degree of the `polynomial` we choose `'polynomial__degree'` in the param_grid.

Have a look at the best models found for each fold in the output above.
"
)


# ╔═╡ cf3841ba-3963-4716-8fc9-8cce1dc4a5fa
md"""## Nested Cross-Validation

If we want to tune the hyper-parameters with cross-validation and care about the estimated test error of the model found with cross-validation, we can use nested cross-validation, where hyper-parameters are optimized with cross-validation for each fold of the outer cross-validation.

$(mlstring(md"This happens when we `evaluate!` with cross-validation a self-tuning machine that itself uses cross-validation for hyper-parameter tuning. With `resampling = Holdout(fraction_train = 0.5)`, we use the validation set approach to estimate the test error with 50% of the data in the training set. With `resampling = Holdout(fraction_train = 0.5)` in the self-tuning machine, we use the validation set approach for tuning the hyper-parameters.", ""))
"""

# ╔═╡ 3e9c6f1c-4cb6-48d8-8119-07a4b03c2e4b
mlcode(
"""
nested_cv = evaluate!(machine(self_tuning_model, DataFrame(X), y),
                      resampling = CV(nfolds = 5), measure = rmse, verbosity = 0)
"""
,
"""
from sklearn.model_selection import KFold

# Declare the inner and outer cross-validation strategies
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner cross-validation for parameter search
model = GridSearchCV(
	estimator=pipeline, param_grid=param_grid, cv=inner_cv, 
                           scoring='neg_mean_squared_error')
model.fit(X,y)
outer_scores = model.best_score_

# Outer cross-validation to compute the testing score
nested_score=cross_val_score(model, X, y, cv=outer_cv, 
                           scoring='neg_mean_squared_error') 

nested_scores = nested_score.mean()
""",
py_showoutput = false
)

# ╔═╡ 171fbe78-5e1d-4a6b-8d7b-f602db1162fb
md"We can have a closer look at the report per fold:"

# ╔═╡ c8d26ab6-86ec-4b37-9540-0f785fd8cdc2
mlcode(
"""
nested_cv.report_per_fold
"""
,
"""
("Outer scores : ", outer_scores, "Nested scores : ",nested_scores)
"""
)

# ╔═╡ 566be5a7-3eae-4c26-ab6d-605dcf08a57d
md"## Tuning Two Hyperparameters: Lambda and Degree

In this section we tune two hyper-parameters simultaneously.
"

# ╔═╡ 2a32d5ac-f761-41f0-8082-32ca5d0b96b2
mlstring(
md"""
In the cell below we use a `TunedModel` to find with cross validation the best polynomial degree and the best regularization constant for ridge regression. Note how we can just replace the `LinearRegressor()` by a `RidgeRegressor()` in the usual model for polynomial regression (previously we used `Polynomial() |> LinearRegressor()`).
"""
,
md"""
In the cell below we perform cross validation on a grid of hyper-parameters using `GridSearchCV`.
"""
)

# ╔═╡ fe2fe54f-0163-4f5d-9fd1-3d1aa3580875
mlcode(
	"""
using MLJ, MLJLinearModels, DataFrames
import MLCourse: Polynomial
	
model = Polynomial() |> RidgeRegressor()
self_tuning_model = TunedModel(model = model,
                               tuning =  Grid(goal = 500),
                               resampling = CV(nfolds = 5),
                               range = [range(model, :(polynomial.degree),
                                              lower = 1, upper = 20),
                                        range(model, :(ridge_regressor.lambda),
                                              lower = 1e-12, upper = 1e-3,
                                              scale = :log10)],
                               measure = rmse)
f(x) = .3 * sin.(10x) .+ .7x
dataset = let x = rand(50)
    DataFrame(x = x, y = f.(x) .+ .1*randn(50))
end
self_tuning_mach = machine(self_tuning_model, select(dataset, :x), dataset.y)
fit!(self_tuning_mach, verbosity = 0)
report(self_tuning_mach)
"""
,
"""
import pandas as pd

def f(x):
    return 0.3 * np.sin(10*x) + 0.7*x
x = np.random.rand(50)
df = pd.DataFrame({'x': x, 'y': f(x) + 0.1*np.random.randn(50)})

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

pipeline = Pipeline ([("polynomial",PolynomialFeatures()),("model",Ridge())])

param_grid = [
    {'polynomial__degree': np.arange(1,20,1),
	'model__alpha' : np.logspace(-12,-3,10)}]

grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(df["x"].values.reshape(-1,1), df["y"].values.reshape(-1,1))
("best_accuracy:", grid_search.best_score_, "best_parameters:", grid_search.best_params_)
"""
,
)

# ╔═╡ 6b15db59-079c-48c6-97a0-7035e0c6b133
mlcode(
"""
best_deg = report(self_tuning_mach).best_model.polynomial.degree
best_lambda = report(self_tuning_mach).best_model.ridge_regressor.lambda
"""
,
"""
best_deg = grid_search.best_params_['polynomial__degree']
best_lambda = grid_search.best_params_['model__alpha']
"""
,
showinput = false,
showoutput = false,
cache_jl_vars = [:best_deg, :best_lambda],
cache_py_vars = [:best_deg, :best_lambda],
)

# ╔═╡ 7b4daae4-20d3-4992-94e9-46882e47b840
mlstring(md"With the report function we can have a look at the best hyper-parameter values (`polynomial.degree`= $(M.best_deg) and
 `ridge_regressor.lambda` = $(round(M.best_lambda, sigdigits = 2)) found by the self-tuning machine.

The result of the self-tuning machine can be visualized with the `plot` function.",
md"""
The best hyper-parameter values (`polynomial__degree`= $(MLCourse.PyMod._OUTPUT_CACHE[:best_deg]) and
 `model__alpha` = $(round(MLCourse.PyMod._OUTPUT_CACHE[:best_lambda], sigdigits = 2)) found by the self-tuning machine.

Note, we could have also use the sklearn function `RidgeCV` : a Ridge regression with built-in cross-validation.
Use `grid_search.cv_results_` to see all results.

The result of the self-tuning machine can be visualized with the `plot` function.
""")

# ╔═╡ f5057d4a-1103-4728-becc-287d93d682ba
mlcode(
"""
using Plots
plot(self_tuning_mach)
"""
,
"""
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hyperparameters (df) :
  g = sns.PairGrid(results, 
					diag_sharey=False, 
					corner=True, 
					hue = "test_score", 
					x_vars=list(df.columns)[:-1], 
					y_vars=list(df.columns))
  g.map(sns.scatterplot, s=50)
  g.add_legend(fontsize=14, bbox_to_anchor=(1.2,0.55))
  for ax in g.axes.flat:
    if not (ax==None) :
      if ax.get_xlabel() in ["model__alpha"]:
        ax.set(xscale="log")
      if ax.get_ylabel() in ["model__alpha"]:
        ax.set(yscale="log")

results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_model__alpha', 'param_polynomial__degree', 'mean_test_score']]
results.columns = ['model__alpha', 'polynomial__degree', 'test_score']
plot_hyperparameters(results)
plt.show()
""",
)

# ╔═╡ adfe4bb8-f3db-4557-b5d8-6efc33f5e321
mlstring(md"
In the plot **at the top**, we see the root mean square error estimated with cross-validation for different values of the polynomial degree. We see, for example, that for low polynomial degrees (below degree 4) the self-tuning machine did not find any low errors. For every degree we see multiple blue points, because the self-tuning machine tried for every degree multiple values for the regularization constant.
In the plot **at the right** we see the root mean squared error as for different values of the regularization constant `lambda`. We see, for example, that for high regression values (above ``10^{-6}``) the self-tuning machine did not find any low errors. Again, we see multiple blue points for every value of the regularization constant, because multiple polynomial degrees were tested (the line of blue values at the top of the figure is probably produced by low polynomial degrees). In the plot **at the bottom left** we see with a color-and-size code the loss for different values of polynomial degree and regularization constant lambda. The smaller and darker the circle, the lower the error.

Let us have a look how well the model with the best hyper-parameters fits the data."
,
md"
In the plot **at the bottom right**, we see the root mean square error estimated with cross-validation for different values of the polynomial degree. We see, for example, that for low polynomial degrees (below degree 4) the self-tuning machine did not find any low errors. For every degree we see multiple points, because the self-tuning machine tried for every degree multiple values for the regularization constant.
In the plot **at the bottom left** we see the root mean squared error as for different values of the regularization constant `lambda`. We see, for example, that for high regression values (above ``10^{-6}``) the self-tuning machine did not find any low errors. Again, we see multiple points for every value of the regularization constant, because multiple polynomial degrees were tested (the line of blue values at the top of the figure is probably produced by low polynomial degrees). In the plot **at the middle** we see with a color code the loss for different values of polynomial degree and regularization constant lambda. Thedarker the circle, the lower the error.

Let us have a look how well the model with the best hyper-parameters fits the data.
")

# ╔═╡ 596fd0f2-eee0-46ca-a203-e7cbac6f9788
mlcode(
	"""
    p1 = scatter(dataset.x, dataset.y,
                 label = "training data", ylims = (-.1, 1.1))
    plot!(f, label = "generator", c = :green, w = 2)
    grid = 0:.01:1
    pred = predict(self_tuning_mach, (x = grid,))
    plot!(grid, pred,
          label = "fit", w = 3, c = :red, legend = :topleft)
	reducible_error_str = string(round(mean((pred .- f.(grid)).^2), sigdigits = 3))
    annotate!([(.28, .6, "reducible error ≈ " * reducible_error_str)])
"""
,
"""
plt.figure()
p1 = plt.scatter(df["x"], df["y"], label="training data")

x= np.linspace(0, 1, 100)
plt.plot(x,f(x),label="generator", color="green",linewidth=2)

grid = np.arange(0, 1.01, 0.01)
test_pipeline = Pipeline([
				("polynomial", 
				PolynomialFeatures(degree = grid_search.best_params_["polynomial__degree"])), 
				("model",
				Ridge(alpha = grid_search.best_params_["model__alpha"]))
				])
test_pipeline.fit(df["x"].values.reshape(-1,1), df["y"].values.reshape(-1,1))

pred = test_pipeline.predict(grid.reshape(-1,1))
plt.plot(grid, pred, label="fit", linewidth=3, color="red")

# Add annotation
reducible_error_str = str(round(np.mean((pred.flatten() - f(grid))**2), 3))
print((pred - f(grid))**2)
plt.annotate("reducible error ≈ " + reducible_error_str, xy=(0.28, 0.6))

# Add legend and show plot
plt.legend(loc="upper left")
plt.show()
"""
,
)



# ╔═╡ 166b363f-6bae-43be-889d-0b8b9251832e
# md"""# 5. The Bootstrap
# 
# The bootstrap is a resampling strategy to generate "new" datasets of the same size as an existing one by sampling rows with replacement.
# 
# In the following we take as our original dataset a subset of the weather dataset.
# Our goal is to estimate if the sunshine duration positively co-varies with the wind peak.
# """
# 


# ╔═╡ a9d3a6e8-f10f-45bc-85b4-c9faba6e8827
# mlcode("""
# using CSV, DataFrames
# 
# weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
#                    DataFrame)[80:5000, [:LUZ_sunshine_duration, :LUZ_wind_peak]]
# """
# ,
# """
# import numpy as np
# import pandas as pd
# 
# weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv").loc[79:4999, ['LUZ_sunshine_duration', 'LUZ_wind_peak']]
# weather
# """
# )
# 


# ╔═╡ c5bb2ca4-2759-4946-bda3-94993a1dd373
# mlcode(
# """
# function bootstrap(data)
# 	idxs = rand(1:nrow(data), nrow(data))
# 	data[idxs, :]
# end
# bootstrap(weather)
# """
# ,
# """
# def bootstrap(data):
#     idxs = np.random.choice(range(data.shape[0]), size=data.shape[0])
#     return data.iloc[idxs, :]
# 
# bootstrap(weather)
# """
# )
# 
# 


# ╔═╡ c4297b53-6a11-4178-8e47-b552d22f7be7
# mlcode(
# """
# using MLJ, MLJLinearModels, Plots
# 
# function bootstrap_and_fit(data)
#     data_bootstrapped = bootstrap(data)
#     m = machine(LinearRegressor(),
#                 select(data_bootstrapped, Not(:LUZ_wind_peak)),
#                 data_bootstrapped.LUZ_wind_peak)
#     fit!(m, verbosity = 0)
#     fitted_params(m).coefs[1][2] # extract the slope of the linear regression
# end
# slopes = [bootstrap_and_fit(weather) for _ in 1:1000]
# histogram(slopes, label = nothing, xlabel = "slope", ylabel = "counts")
# """
# ,
# """
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# 
# def bootstrap_and_fit(data):
#     data_bootstrapped = bootstrap(data)
#     X = data_bootstrapped.drop(columns=['LUZ_wind_peak'])
#     y = data_bootstrapped['LUZ_wind_peak']
#     model = LinearRegression()
#     model.fit(X, y)
#     return model.coef_[0] # extract the slope of the linear regression
# 
# slopes = [bootstrap_and_fit(weather) for _ in range(1000)]
# plt.figure()
# plt.hist(slopes)
# plt.xlabel("slope")
# plt.ylabel("counts")
# plt.show()
# """
# )
# 


# ╔═╡ 45d6d454-eaac-4401-9a46-390d3d667794
# md"The slope in the figure above refers to the slope of the line found by linear regression. Because we ran linear regression on many bootstrapped datasets, we found many different values for the slope. If the distribution of slopes would be centered at 0, we could conclude that the sunshine_duration is unlikely to be correlated with the wind_peak. However, the distribution of slope values is clearly on the positive side. This means that the wind peak indeed co-varies with the sunshine duration.
# 
# Note that we can be more confident about our statement than if we would have just computed the slope once with the full data set. Why? Because a positive slope could come out randomly for a finite data set, even if the two variables wouldn't be correlated. After bootstrapping, we know that the positive slope is a consistent feature of the dataset that doesn't disappear, when some data points are removed."
#

# ╔═╡ aa002263-4dc8-4238-ab27-a9094947600c
md"# 5. Data Cleaning

## Dealing with Missing Data

In the following artificial dataset the age and the gender is missing for some of the subjects.
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
pd.DataFrame(imp_mean.fit_transform(datam), columns = ['age', 'gender'])
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
df = pd.DataFrame({"a": np.ones(5),
                   "b" : np.random.randn(5),
                   "c" : np.linspace(1,5,5),
                   "d": np.linspace(2,10,5),
                   "e" : np.zeros(5)})
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
"""
,
py_showoutput = false
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
	p2 = histogram(st_data.x, nbins = 100, normalize = :pdf, xlabel = "z", ylabel = "probability density", label = "scaled data", title = "scaled data")
    plot!(z -> a*pdf(d, a*z + b), c = :orange, w = 2, label = "\$f_Z(z)\$")
    plot(p1, p2, layout = (1, 2), yrange = (0, .5), size = (700, 400))
end

# ╔═╡ 4c536f7c-dc4b-4cf7-aefa-694b830a8e6a
md"# 6. Feature Engineering
"

# ╔═╡ 004958cf-0abb-4d8e-90b1-5966412cba91
md"## Categorical Predictors

For categorical predictors the canonical transformation is one-hot encoding.
"

# ╔═╡ b43365c6-384e-4fde-95f3-b2ad1ebc421a
mlcode(
"""
cdata = DataFrame(gender = ["male", "male", "female", "female", "female", "male"],
                  treatment = [1, 2, 2, 1, 3, 2])
coerce!(cdata, :gender => Multiclass, :treatment => Multiclass)
"""
,
"""
cdata = pd.DataFrame({
		'gender': pd.Categorical(['male', 'male', 'female', 'female', 'female', 'male']),
		'treatment': pd.Categorical([1, 2, 2, 1, 3, 2])})
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

cdata_enc_df = pd.DataFrame(cdata_enc,
                            columns = np.concatenate((enc.categories_[0],                                                        enc.categories_[1].astype('str'))))
cdata_enc_df
"""
,
recompute = false
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
nothing
)

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
    (OneHotEncoder(drop="first"), categorical_columns),
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
nothing
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

# ╔═╡ cfcad2d7-7dd9-43fe-9633-7ce9e51e1e27
md"# 7. Transformations of the Output

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
gamreg_pred
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

# ╔═╡ 07734386-6858-46ab-9ce7-9e4092cb7280
md"""## What can we conclude from the marginal ``p(y)``?

Above we plotted the marginal density ``p(y)`` of the wind peak, which differs from the conditional density ``p(y|x)`` that we try to fit in supervised learning. In fact, given the density of inputs ``p(x)``, the density of outputs is given by ``p(y) = \int_{\mathcal X}p(y|x)p(x)dx`` where the integral runs over all possible values of ``x\in\mathcal X``.

To get a better feeling for different shapes of ``p(y)`` we look here at a few artificial examples. We can learn from these examples, that it is very difficult to infer the model or the noise distribution purely by looking at the marginal distribution ``p(y)``.

#### Normal input, linear model, normal noise

``y = 2x + 0.1 + \epsilon``, with normally distributed ``\epsilon``
"""

# ╔═╡ 40419527-aacb-4394-9b2a-98dc63dc3896
md"""The response variable ``y`` looks clearly normally distributed.

#### Normal input, non-linear model, normal noise

``y = 2x^2 + 3 + \epsilon`` with normally distributed ``\epsilon``.
"""

# ╔═╡ 00280812-ad7f-4bb3-825e-9c1d920c84a3
md"""
Neither the marginal ``p(y)`` nor the marginal of ``log(y)`` look normally distributed, although the conditional probability ``p(y|x)`` is actually a normal distribution with mean ``2x^2 + 3``.

#### Non-normal input, linear model, normal noise
``y = 2x + 8.1 + \epsilon``, with normally distributed ``\epsilon``.
"""

# ╔═╡ 9efea3e5-fe80-489d-a5b5-948276064a6b
md"""
Neither the marginal ``p(y)`` nor the marginal of ``\log y`` look normally distributed, even though the conditional probability ``p(y|x)`` is actually linear with linear dependency of the mean on the input ``x``!

#### Normal input, non-linear model, non-normal noise

``y = 2.1 ^ {x/4 + 2 + \epsilon}``, with normally distributed ``\epsilon``. Note that the conditional is not normally distributed, because ``\epsilon`` appears in the exponent.
"""

# ╔═╡ f7a64959-ea12-4aa0-bad1-4ee03c3b1fcc
md"""
The marginal of ``\log(y)`` looks normally distributed.

#### Normal input, linear model, laplace noise
``y = x/4 + 2 + \epsilon``, with Laplacian ``\epsilon``.
"""

# ╔═╡ ee89c448-1e69-4fd1-a4b8-7297a09f2685
md"# Exercises

## Conceptual
#### Exercise 1
We review k-fold cross-validation.
- Explain how k-fold cross-validation is implemented.
- What are the advantages and disadvantages of k-fold cross-validation relative to:
    - The validation set approach?
    - LOOCV?
#### Exercise 2
You are given a model with unknown hyper-parameters and the goal is to find the optimal hyper-parameters and estimate the test error of the optimal model. Suppose of colleague of yours argues that this should be done in the following way: \"Take all the data and run cross-validation to find the best hyper-parameters. Taking all data is better, than taking only a subset, because one has more data to estimate the parameters and estimate the test error on the validation sets. Once the best hyper-parameters are found: estimate the test error by running again cross-validation on all the data with the hyper-parameter fixed to the best value.\" Do you agree with your colleague? If not, explain where you disagree and where you agree.
#### Exercise 3
Suppose, for a regression problem with one-dimensional input ``X`` and one-dimensional output ``Y`` we use the feature functions ``h_1(X) = X``, ``h_2(X) = (X - 1)^2\chi(X\ge 1)`` (where ``\chi`` is the indicator function, being 1 when the condition inside the brackets is true and 0 otherwise), fit the linear regression model ``Y = \beta_0 + \beta_1h_1(X) + \beta_2h_2(X) + \varepsilon,`` and obtain coefficient estimates ``\hat{\beta}_0 = 1``, ``\hat{\beta}_1 = 1``, ``\hat{\beta}_2 = -2``. Draw the estimated curve between ``X = -2`` and ``X = 2``.
#### Exercise 4
The following table shows one-hot encoded input for columns `mean_of_transport` (car, train, airplane, ship) and `energy_source` (petrol, electric). Indicate for each row the mean of transport and the energy source.

| car | train | airplane | petrol|
|-----|-------|----------|-------|
| 1   | 0     | 0        |  0    |
| 0   | 0     | 0        |  1    |
| 0   | 1     | 0        |  1    |
| 0   | 0     | 0        |  0    |


#### Exercise 5 (optional)
Suppose you receive the following email of a colleague of yours. Write an answer to this email.
```
    Hi

    In my internship I am doing this research project with a company that
    manufactures constituents for medical devices. We need to assure high
    quality and detect the pieces that are not fully functional. To do so we
    perform 23 measurements on each piece. My goal is to find a machine
    learning method that detects defective pieces automatically. Both, my
    training set and my test set consist of 4000 pieces that were fine and
    30 defective ones.  I ran logistic regression on that data and found a
    training error of 0.1% and a test error of 0.3%, which is already pretty
    good, I think, no? But with kNN classification it is even better. For
    k = 7 I found a test error of 0.05% which is by far lower than all test
    errors I obtained with other values of k. I was really impressed. Now
    we have a method that predicts with 99.95% accuracy whether a piece is
    defective or not!

    Because you are taking this machine learning class now, I wanted to ask you
    for advice. Does it all make sense to you what I described? If not, do you
    have any suggestion to get even better results?

    Ã bientôt
    Jamie
```


## Applied
"

# 1. Perform k-nearest neighbors regression on data generated with our `data_generator`
#    defined in the first cell of this notebook and find the optimal number k of neighbors
#    with k-fold cross validation.


# ╔═╡ c0e33f73-59cc-4c8b-965b-21d8ea83d1ce
md"""
#### Exercise 6
Take the `classification_data` in our notebook on
   \"flexibility and bias-variance-decomposition notebook\" and find with 10-fold
   cross-validation the optimal number ``k`` of neighbors of kNN
   classification, using the AUC measure. Hint: $(mlstring(md"`MLJ` has the builtin function `auc`.", md"`sklearn` has the builtin metric `sklearn.metric.auc`."))
   Plot the validation AUC for ``k = 1, \ldots, 50``.
"""

# ╔═╡ 186fa41b-5e74-4191-bc2d-e8d865606fc1
md"
#### Exercise 7 (optional)
With the same data as in the previous exercise, estimate test error with the validation set approach for a kNN classifier whose hyper-parameter is tuned with 5 fold cross-validation. Use one quarter of the data for the test set.
"

# ╔═╡ 6f3fea58-8587-4b03-a11a-bac8a46abe67
md"
#### Exercise 8 (optional)
In this exercise you apply our \"recipe for supervised learning\" (see slides). The goal is to predict the miles a car can drive per gallon fuel (mpg) as a function of its horsepower. You can download the dataset from openml; the dataset id is `455`. In the cleaning step we will remove all rows that contain missing values. We select the machine learning methods polynomial regression and k nearest neighbors regression and we take as measure the root mean squared error. Make sure to go trough the steps 2, 5, 9 of the recipe. Plot the predictions of the best method you found. *Hint:* take inspiration from the examples above to tune the hyper-parameters.
"


# ╔═╡ 22679be7-2502-4084-9ce0-35bb73451b52
md"""
#### Exercise 9 (optional)
Write a function that returns the training, validation and test set indices for nested cross-validation on a dataset of size ``n``.
"""


# ╔═╡ a26864cc-fc27-4bc4-8bf2-f703db905624
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 0651292e-3f4e-4263-8235-4caa563403ec
MLCourse.FOOTER

# ╔═╡ 7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
begin
    f(x) = 0.3 + 2x - 0.8x^2 - 0.4x^3
    function data_generator(; n = 500, seed = 12)
        rng = MersenneTwister(seed)
        x = randn(rng, n)
        DataFrame(x = x, y = f.(x) .+ randn(rng, n))
    end
end;

# ╔═╡ 46827e21-0eea-4ec9-83ed-05e41dda1502
data1 = data_generator(seed = 1);

# ╔═╡ 5dc21667-d94f-4c2a-b217-80b9a2262d8c
begin
	data1_train = data1[idxs.training_idxs, :]
	data1_test = data1[idxs.test_idxs, :]
	mach = machine(Polynomial(degree = 8) |> LinearRegressor(),
	               select(data1_train, :x), data1_train.y)
	fit!(mach, verbosity = 0)
	(training_error = rmse(predict(mach), data1_train.y),
	 test_error = rmse(predict(mach, select(data1_test, :x)), data1_test.y))
end

# ╔═╡ 1fe3f6d1-6336-417e-885e-de3fc5821bdf
let
	scatter(data1_train.x, data1_train.y, label = "training set")
	scatter!(data1_test.x, data1_test.y, label = "test set")
	xgrid = -4:.1:4
	pred = predict(mach, DataFrame(x = xgrid))
	plot!(xgrid, pred, label = "fit", w = 3, ylim = (-6, 5), xlim = (-4, 4))
end

# ╔═╡ ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
valid = let
    idxs = [train_val_test_set(n = 500, shuffle = seed != 0, seed = seed) for seed in 0:20]
    losses = [fit_and_evaluate(Polynomial(; degree) |> LinearRegressor(),
                               data_split(data1; sets...))
          for degree in 1:10, sets in idxs]
    valid_losses = getproperty.(losses, :valid_rmse)
    train_losses = getproperty.(losses, :train_rmse)
    best_i = getindex.(argmin(valid_losses, dims = 1), 1)
    test_losses = [fit_and_evaluate(Polynomial(degree = i) |> LinearRegressor(),
                                    data_split(data1; train_idxs = setdiff(1:500, sets.test_idxs),
                                    test_idxs = sets.test_idxs,
                                    valid_idxs = [])).test_rmse
                   for (sets, i) in zip(idxs, best_i)]
    (; idxs, losses, best_i, valid_losses, test_losses)
end;

# ╔═╡ 8428b370-c128-4bc9-802d-bc3aaf6d0959
valid.idxs[split_seed2 + 1]

# ╔═╡ 91eecd2b-af18-4a63-9684-28950e604d1a
let validlosses = valid.valid_losses[:, split_seed2+1],
	testloss = valid.test_losses[split_seed2+1],
	i = valid.best_i[split_seed2 + 1]
    plot(1:10, validlosses, label = "validation loss")
    hline!([testloss], label = "test loss of degree $i")
    scatter!([i], [validlosses[i]], label = "optimal validation loss",
             xlabel = "degree", ylabel = "rmse", yrange = (0, 10))
end

# ╔═╡ b45b32b1-8a65-4823-b3bb-f0b7cc57604b
md"For this seed of the random number generator the optimal degree (x coordinate of the red point in the figure above) found by the validation set approach is close to the actual degree of the data generating process. The estimated test loss of the model with degree $(valid.best_i[split_seed2+1]) is approximately $(round(valid.test_losses[split_seed2+1], sigdigits = 3))."

# ╔═╡ 26362233-b006-423d-8fb5-7cd9150405b4
let validlosses = valid.valid_losses, i = valid.best_i
    plot(validlosses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 5cff19c7-3426-4d91-b555-f059f5f41886
md"""Here we see a large variability of the optimal degree (x coordinates of all the points in the figure above): the best degrees (sorted) are $(join(string.(sort(valid.best_i[:])), ", "))

In the following cell we compare the validation set estimate of the test error to the test set estimation of the test error for the degree that has the smallest validation error. Although sometimes the estimate on the test set is below the estimate on the validaion set, we can confirm that the validation set estimate is much lower than the test set estimate. The estimate on the validation set is biased, because we used the validation set to find the optimal hyperparameter."""

# ╔═╡ df16313f-76be-4d8b-88e7-7cc618ff49d4
let validlosses = valid.valid_losses,
	testlosses = valid.test_losses
    (mean_valid_winner = mean(minimum(validlosses, dims = 1)),
	 mean_test_winner = mean(testlosses))
end


# ╔═╡ 6916273a-d116-4173-acaa-2bcac1d1753b
cv1 = let
    idxs = [cross_validation_sets(500, 4; shuffle = seed != 0, seed)
            for seed in 0:20]
    losses = [cross_validation(Polynomial(; degree) |> LinearRegressor(),
                               data1; sets)
              for degree in 1:10, sets in idxs]
    valid_losses = getproperty.(losses, :valid_rmse)
    best_i = getindex.(argmin(valid_losses, dims = 1), 1)
    (; idxs, losses, valid_losses, best_i)
end;

# ╔═╡ c14c9e49-a993-456e-9115-97da86f8e498
cv1.idxs[cv_shuffle_seed+1]

# ╔═╡ 4b435518-2f12-4921-bb1f-fdd049ddfaed
let validlosses = cv1.valid_losses[:, cv_shuffle_seed+1],
    i = cv1.best_i[cv_shuffle_seed+1]
    plot(1:10, validlosses, label = "cross-validation loss")
    scatter!([i], [validlosses[i]], label = "optimal cross-validation loss",
             xlabel = "degree", ylabel = "rmse")
end

# ╔═╡ a7c88b3f-92cb-4253-a889-c78683722c1d
let validlosses = cv1.valid_losses,
    i = cv1.best_i
    plot(validlosses, label = nothing, ylims = (0.8, 3))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "cross-validation loss")
end


# ╔═╡ 8204d9da-855d-4c66-b4dd-2c18a17b539b
cv2 = let
    idxs = [begin
        s_idxs = seed == 0 ? collect(1:500) : randperm(Xoshiro(seed), 500)
	   (test_idxs = s_idxs[101:500],
	    folds = cross_validation_sets(100, 4, idxs = s_idxs[1:100]))
	    end
		for seed in 0:20]
    losses = [cross_validation(Polynomial(; degree) |> LinearRegressor(),
                               data1; sets = sets.folds)
              for degree in 1:10, sets in idxs]
    valid_losses = getproperty.(losses, :valid_rmse)
    best_i = getindex.(argmin(valid_losses, dims = 1), 1)
    test_losses = [fit_and_evaluate(Polynomial(degree = i) |> LinearRegressor(),
                                    data_split(data1; train_idxs = setdiff(1:500, sets.test_idxs),
                                    test_idxs = sets.test_idxs,
                                    valid_idxs = [])).test_rmse
                   for (sets, i) in zip(idxs, best_i)]
    (; idxs, losses, valid_losses, best_i, test_losses)
end;

# ╔═╡ 890f4d59-ef7d-43d4-bbef-ded8b6b030eb
cv2.idxs[cv_shuffle_seed2+1]

# ╔═╡ ebfac91a-76a7-4b04-b62e-9b9ca038258d
let validlosses = cv2.valid_losses[:, cv_shuffle_seed2+1],
    i = cv2.best_i[cv_shuffle_seed2+1],
    testloss = cv2.test_losses[cv_shuffle_seed2+1]
    plot(1:10, validlosses, label = "validation loss")
    hline!([testloss], label = "test loss of degree $i")
    scatter!([i], [validlosses[i]], label = "optimal validation loss",
             xlabel = "degree", ylabel = "rmse", yrange = (0, 10))
end

# ╔═╡ a568a12a-356c-46d6-8103-81c53014b203
let validlosses = cv2.valid_losses, i = cv2.best_i
    plot(validlosses, label = nothing, ylims = (0, 10))
    scatter!((x -> x[1]).(i), validlosses[i], label = nothing,
             xlabel = "degree", ylabel = "validation loss")
end

# ╔═╡ 5aa96a0f-d7a1-4fe4-aa3a-1fb7b700696f
md"""
Although there is still some variability in the optimal degree (x coordinates of all the points in the figure above), the variability is lower than in the comparable setting with the validation set approach: the best degrees (sorted) are $(join(string.(sort(cv2.best_i[:])), ", "))

The improved hyperparameter search also leads to a better estimate of the test error:
"""

# ╔═╡ 9042e811-9a34-42b8-8069-877f3b3f1e75
(mean_valid_winner = mean(minimum(cv2.valid_losses, dims = 1)), mean_test_winner = mean(cv2.test_losses))

# ╔═╡ 9d250061-e570-4537-b1aa-f6a9019f343d
MLCourse.save_cache(@__FILE__)

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
	scatter(M.wage.AGE, M.wage.WAGE, xlabel = "age [years]", ylabel = "wage [USD/hour]")
	plot!(grid, predict(spline_fit, DataFrame(x = grid)), w = 3)
	vline!([knot1, knot2, knot3, knot4], linestyle = :dash, legend = false)
end

# ╔═╡ 23480396-5bf5-4f31-8a26-5f04ba5537ee
function _example_plots(x, y; with_normal_fit = false, with_log_fit = false)
	f1 = histogram(x, xlabel = "x", ylabel = "p(x)", nbins = 75, normalize = :pdf, label = nothing, title = "marginal input p(x)")
	f2 = histogram2d(x, y, title = "conditional p(y|x)", nbins = 75, xlabel = "x", ylabel = "y", normalize = :pdf)
	f3 = histogram(y, title = "marginal output p(y)", nbins = 75, xlabel = "y", ylabel = "p(y)", normalize = :pdf, label = nothing)
	if with_normal_fit
		normalfit = Distributions.fit(Normal, y)
		laplacefit = Distributions.fit(Distributions.Laplace, y)
		plot!(x -> pdf(normalfit, x), label = "Normal fit", lw = 3)
		plot!(x -> pdf(laplacefit, x), label = "Laplace fit", lw = 3)
	end
	if with_log_fit
		logfit = Distributions.fit(Normal, log.(y))
		f4 = histogram(log.(y), title = "marginal p(log(y))", nbins = 75, xlabel = "log(y)", ylabel = "p(log(y))", normalize = :pdf, label = nothing)
		plot!(x -> pdf(logfit, x), label = "Normal fit", lw = 3)
	else
		f4 = plot(axis = false)
	end
	plot(f1, f2, f3, f4, layout = (2, 2))
end;

# ╔═╡ eb3d3d51-7e67-4833-810f-e006ec0a882f
let n = 5000
	x = randn(n)
	y = 2x .+ .1 .+ .75*randn(n)
	_example_plots(x, y, with_normal_fit = true)
end

# ╔═╡ 9152b724-744f-4203-a46d-c11534379825
let n = 5000
	x = randn(n)
	y = 2x .^ 2 .+ 3 .+ .75*randn(n)
	_example_plots(x, y, with_normal_fit = true, with_log_fit = true)
end

# ╔═╡ 959af8e7-22b5-4563-bb02-d7aabc2fba6d
let n = 6000
	x = [randn(3*n÷4); randn(n÷4) .+ 3 ]
	y = 2x .+ 8.1 .+ .75*randn(n)
	_example_plots(x, y, with_log_fit = true, with_normal_fit = true)
end

# ╔═╡ 38554070-7233-4794-825f-2617a856bf78
let n = 5000
	x = randn(n)
	y = 2.1 .^ (x/4 .+ 2 .+ .75*randn(n))
	_example_plots(x, y, with_normal_fit = true, with_log_fit = true)
end

# ╔═╡ 80593756-ed60-4368-8801-d558a1af7adb
let n = 5000
	x = randn(n)
	y = x/4 .+ 2 .+ rand(Distributions.Laplace(0, .5), n)
	_example_plots(x, y, with_normal_fit = true, with_log_fit = false)
end


# ╔═╡ Cell order:
# ╟─b70d9848-91a7-4de0-b93e-8078acfd77f8
# ╟─7eb6a060-f948-4d85-881a-4909e74c15bd
# ╟─5049407c-bf93-41a5-97bd-863a69d7016a
# ╟─6cfc32b3-40c0-4149-ba67-2ec67b3938f3
# ╟─448e4556-9448-4cc0-a00d-bb80b6b0799c
# ╟─5dc21667-d94f-4c2a-b217-80b9a2262d8c
# ╟─1fe3f6d1-6336-417e-885e-de3fc5821bdf
# ╟─b79fae90-87f3-4f89-933b-d82e76c94d81
# ╟─1320b9b0-d6fc-4e61-8d62-de3988b8b21d
# ╟─7d30f043-2915-4bba-aad5-cb7dbe76a3e5
# ╟─8428b370-c128-4bc9-802d-bc3aaf6d0959
# ╟─0af29197-3e02-4373-b47a-de9a79e9cb55
# ╟─91eecd2b-af18-4a63-9684-28950e604d1a
# ╟─b45b32b1-8a65-4823-b3bb-f0b7cc57604b
# ╟─4461d5b0-bc9b-4c89-aa76-70524a5caa7c
# ╟─ac3c7e84-6c47-4dc1-b862-de4cfb05dad9
# ╟─26362233-b006-423d-8fb5-7cd9150405b4
# ╟─5cff19c7-3426-4d91-b555-f059f5f41886
# ╟─df16313f-76be-4d8b-88e7-7cc618ff49d4
# ╟─ffffa3d0-bffb-4ff8-9966-c3312f952ac5
# ╟─46bffe01-1a7c-405b-b3ec-bfe7570b8a3c
# ╟─c14c9e49-a993-456e-9115-97da86f8e498
# ╟─4c40a918-b3fb-4528-8508-c36579d0f6fc
# ╟─4b435518-2f12-4921-bb1f-fdd049ddfaed
# ╟─0ef33f9e-6013-445a-8192-9c9c1c047204
# ╟─a7c88b3f-92cb-4253-a889-c78683722c1d
# ╟─0b81c3d5-277a-4fe6-889b-550e2f83c39d
# ╟─b3aed705-5072-4d9b-bb8f-865ac1561bf6
# ╟─6916273a-d116-4173-acaa-2bcac1d1753b
# ╟─5747d116-1e61-45f0-a87b-89372c6f270f
# ╟─890f4d59-ef7d-43d4-bbef-ded8b6b030eb
# ╟─c268fb85-e695-4b18-8baf-2ab4e656102f
# ╟─ebfac91a-76a7-4b04-b62e-9b9ca038258d
# ╟─a568a12a-356c-46d6-8103-81c53014b203
# ╟─5aa96a0f-d7a1-4fe4-aa3a-1fb7b700696f
# ╟─9042e811-9a34-42b8-8069-877f3b3f1e75
# ╟─8305fadb-1f70-4668-864b-f5da68baf99c
# ╟─8204d9da-855d-4c66-b4dd-2c18a17b539b
# ╟─e2658128-4053-4484-8e1e-229eceb755ab
# ╟─1a5eadf3-6dcb-4f47-8668-58d029032aca
# ╟─cff653c5-9c20-4952-917a-fb8f265387d8
# ╟─30fe1445-6ddb-4ef2-8dcd-d41ff703da40
# ╟─d36a10f4-f61b-4275-9dde-8defba357a0c
# ╟─8ae790b3-3987-4f41-8e21-adbb71081eb9
# ╟─29683c99-6a6a-4f65-bea2-d592895d887e
# ╟─f93f20db-4fed-481f-b085-ca744b68fa8f
# ╟─a41302ae-e2d8-4e5d-8dac-198bed16217b
# ╟─cf3841ba-3963-4716-8fc9-8cce1dc4a5fa
# ╟─3e9c6f1c-4cb6-48d8-8119-07a4b03c2e4b
# ╟─171fbe78-5e1d-4a6b-8d7b-f602db1162fb
# ╟─c8d26ab6-86ec-4b37-9540-0f785fd8cdc2
# ╟─566be5a7-3eae-4c26-ab6d-605dcf08a57d
# ╟─2a32d5ac-f761-41f0-8082-32ca5d0b96b2
# ╟─fe2fe54f-0163-4f5d-9fd1-3d1aa3580875
# ╟─6b15db59-079c-48c6-97a0-7035e0c6b133
# ╟─7b4daae4-20d3-4992-94e9-46882e47b840
# ╟─f5057d4a-1103-4728-becc-287d93d682ba
# ╟─adfe4bb8-f3db-4557-b5d8-6efc33f5e321
# ╟─596fd0f2-eee0-46ca-a203-e7cbac6f9788
# ╟─166b363f-6bae-43be-889d-0b8b9251832e
# ╟─a9d3a6e8-f10f-45bc-85b4-c9faba6e8827
# ╟─c5bb2ca4-2759-4946-bda3-94993a1dd373
# ╟─c4297b53-6a11-4178-8e47-b552d22f7be7
# ╟─45d6d454-eaac-4401-9a46-390d3d667794
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
# ╟─07734386-6858-46ab-9ce7-9e4092cb7280
# ╟─eb3d3d51-7e67-4833-810f-e006ec0a882f
# ╟─40419527-aacb-4394-9b2a-98dc63dc3896
# ╟─9152b724-744f-4203-a46d-c11534379825
# ╟─00280812-ad7f-4bb3-825e-9c1d920c84a3
# ╟─959af8e7-22b5-4563-bb02-d7aabc2fba6d
# ╟─9efea3e5-fe80-489d-a5b5-948276064a6b
# ╟─38554070-7233-4794-825f-2617a856bf78
# ╟─f7a64959-ea12-4aa0-bad1-4ee03c3b1fcc
# ╟─80593756-ed60-4368-8801-d558a1af7adb
# ╟─5774b82f-a09e-46f8-a458-0f85fc4c53d8
# ╟─0815ba19-5c7d-40d6-b948-8edccfb5c386
# ╟─ee89c448-1e69-4fd1-a4b8-7297a09f2685
# ╟─c0e33f73-59cc-4c8b-965b-21d8ea83d1ce
# ╟─186fa41b-5e74-4191-bc2d-e8d865606fc1
# ╟─6f3fea58-8587-4b03-a11a-bac8a46abe67
# ╟─22679be7-2502-4084-9ce0-35bb73451b52
# ╟─e1077092-f72a-42af-b6e0-a616f938cba8
# ╟─a26864cc-fc27-4bc4-8bf2-f703db905624
# ╟─0651292e-3f4e-4263-8235-4caa563403ec
# ╟─7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
# ╟─46827e21-0eea-4ec9-83ed-05e41dda1502
# ╟─736856ce-f490-11eb-3349-057c86edfe7e
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
# ╟─cc65c9c0-cd8d-432c-b697-8f90a7b831f9
# ╟─ac4e0ade-5a1f-4a0d-b7ba-f8386f5232f1
# ╟─23480396-5bf5-4f31-8a26-5f04ba5537ee
