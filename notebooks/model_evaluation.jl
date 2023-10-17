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

# ╔═╡ 736856ce-f490-11eb-3349-057c86edfe7e
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames
import Distributions: Normal, Poisson
import MLCourse: poly, Polynomial
import PlutoPlotly as PP
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
5. See one application of the bootstrap.
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
         valid_rmse = rmse(predict(mach, select(data.valid, :x)), data.valid.y),
         test_rmse = rmse(predict(mach, select(data.test, :x)), data.test.y))
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

Finding good hyper-parameters (tuning) is such an important step in the process of finding good machine learning models that there exist some nice utility functions to tune the hyper-parameters $(mlstring(md"(see e.g. [tuning section in the MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/))", "")).

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
In the output below you see that the model is a `DeterministiPipeline`, with a `polynomial` and a `linear_regressor`. To tune the degree of the `polynomial` we choose `:(polynomial.degree) in the range function above.

Have a look at the best models found for each fold in the output above.
"
,
"
In general, for any given model you may first want to find out, how the parameters are called that you would like to tune. This can be done by inspecting the model. The model is a `DeterministiPipeline`, with a `polynomial` and a `linear_regressor`.
We can see in the Pipeline that the transform `PolynomialFeatures` is named `polynomial`. To tune the degree of the `polynomial` we choose `:(polynomial__degree)` in the param_grid.

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

# ╔═╡ 166b363f-6bae-43be-889d-0b8b9251832e
md"""# 5. The Bootstrap

The bootstrap is a resampling strategy to generate "new" datasets of the same size as an existing one by sampling rows with replacement.

In the following we take as our original dataset a subset of the weather dataset.
Our goal is to estimate if the sunshine duration positively co-varies with the wind peak.
"""

# ╔═╡ a9d3a6e8-f10f-45bc-85b4-c9faba6e8827
mlcode("""
using CSV, DataFrames

weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
                   DataFrame)[80:5000, [:LUZ_sunshine_duration, :LUZ_wind_peak]]
"""
,
"""
import numpy as np
import pandas as pd

weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv").loc[79:4999, ['LUZ_sunshine_duration', 'LUZ_wind_peak']]
weather
"""
)

# ╔═╡ c5bb2ca4-2759-4946-bda3-94993a1dd373
mlcode(
"""
function bootstrap(data)
	idxs = rand(1:nrow(data), nrow(data))
	data[idxs, :]
end
bootstrap(weather)
"""
,
"""
def bootstrap(data):
    idxs = np.random.choice(range(data.shape[0]), size=data.shape[0])
    return data.iloc[idxs, :]

bootstrap(weather)
"""
)


# ╔═╡ c4297b53-6a11-4178-8e47-b552d22f7be7
mlcode(
"""
using MLJ, MLJLinearModels, Plots

function bootstrap_and_fit(data)
    data_bootstrapped = bootstrap(data)
    m = machine(LinearRegressor(),
                select(data_bootstrapped, Not(:LUZ_wind_peak)),
                data_bootstrapped.LUZ_wind_peak)
    fit!(m, verbosity = 0)
    fitted_params(m).coefs[1][2] # extract the slope of the linear regression
end
slopes = [bootstrap_and_fit(weather) for _ in 1:1000]
histogram(slopes, label = nothing, xlabel = "slope", ylabel = "counts")
"""
,
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def bootstrap_and_fit(data):
    data_bootstrapped = bootstrap(data)
    X = data_bootstrapped.drop(columns=['LUZ_wind_peak'])
    y = data_bootstrapped['LUZ_wind_peak']
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0] # extract the slope of the linear regression

slopes = [bootstrap_and_fit(weather) for _ in range(1000)]
plt.figure()
plt.hist(slopes)
plt.xlabel("slope")
plt.ylabel("counts")
plt.show()
"""
)

# ╔═╡ 45d6d454-eaac-4401-9a46-390d3d667794
md"The distribution of slope values is clearly on the positive side. This means that the wind peak indeed co-varies with the sunshine duration."

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


# ╔═╡ 22679be7-2502-4084-9ce0-35bb73451b52
md"""
#### Exercise 4
Write a function that returns the training, validation and test set indices for nested cross-validation on a dataset of size ``n``.
"""

# ╔═╡ c0e33f73-59cc-4c8b-965b-21d8ea83d1ce
md"""
#### Exercise 5
Take the `classification_data` in our notebook on
   \"flexibility and bias-variance-decomposition notebook\" and find with 10-fold
   cross-validation the optimal number ``k`` of neighbors of kNN
   classification, using the AUC measure. Hint: $(mlstring(md"`MLJ` has the builtin function `auc`.", md"`sklearn` has the builtin metric `sklearn.metric.auc`."))
   Plot the validation AUC for ``k = 1, \ldots, 50``.
"""

# ╔═╡ 186fa41b-5e74-4191-bc2d-e8d865606fc1
md"
#### Exercise 6
With the same data as in the previous exercise, estimate test error with the validation set approach for a kNN classifier whose hyper-parameter is tuned with 5 fold cross-validation. Use one quarter of the data for the test set.
"

# ╔═╡ 6f3fea58-8587-4b03-a11a-bac8a46abe67
md"
#### Exercise 7 (optional)
In this exercise you apply our \"recipe for supervised learning\" (see slides). The goal is to predict the miles a car can drive per gallon fuel (mpg) as a function of its horsepower. You can download the dataset from openml; the dataset id is `455`. In the cleaning step we will remove all rows that contain missing values. We select the machine learning methods polynomial regression and k nearest neighbors regression and we take as measure the root mean squared error. Make sure to go trough the steps 2, 5, 9 of the recipe. Plot the predictions of the best method you found. *Hint:* take inspiration from the examples above to tune the hyper-parameters.
"

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
# ╟─166b363f-6bae-43be-889d-0b8b9251832e
# ╟─a9d3a6e8-f10f-45bc-85b4-c9faba6e8827
# ╟─c5bb2ca4-2759-4946-bda3-94993a1dd373
# ╟─c4297b53-6a11-4178-8e47-b552d22f7be7
# ╟─45d6d454-eaac-4401-9a46-390d3d667794
# ╟─ee89c448-1e69-4fd1-a4b8-7297a09f2685
# ╟─22679be7-2502-4084-9ce0-35bb73451b52
# ╟─c0e33f73-59cc-4c8b-965b-21d8ea83d1ce
# ╟─186fa41b-5e74-4191-bc2d-e8d865606fc1
# ╟─6f3fea58-8587-4b03-a11a-bac8a46abe67
# ╟─e1077092-f72a-42af-b6e0-a616f938cba8
# ╟─a26864cc-fc27-4bc4-8bf2-f703db905624
# ╟─0651292e-3f4e-4263-8235-4caa563403ec
# ╟─7dd7e9a7-9245-4c64-af0c-8f7d2f62b2bf
# ╟─46827e21-0eea-4ec9-83ed-05e41dda1502
# ╟─736856ce-f490-11eb-3349-057c86edfe7e
# ╟─9d250061-e570-4537-b1aa-f6a9019f343d
