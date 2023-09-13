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

# ╔═╡ 94f8e29e-ef91-11eb-1ae9-29bc46fa505a
begin
using Pkg
Base.redirect_stdio(stderr = devnull, stdout = devnull) do
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
end
using Revise, MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames
import Distributions: Normal, Poisson
import MLCourse: fitted_linear_func
import PlutoPlotly as PP
MLCourse.CSS_STYLE
end

# ╔═╡ 12d5824c-0873-49a8-a5d8-f93c73b633ae
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ 661e96c4-9432-4e39-b67e-6d0de2b76635
md"The goal of this week is to
1. Understand which conditional distribution to take, given some response variable ``Y``.
2. Translate the blackboard example of logistic regression into code.
3. Understand Confusion Matrices, ROC and AUC.
4. Know how to perform (multiple) logistic regression (also known as linear classification) on a given dataset.
5. Know how to perform Poisson regression on a given dataset.
"

# ╔═╡ 0e6efce9-093a-4758-a649-6cff525711a5
md"# 1. Distributions

Use the sliders below to get a feeling for how the distributions depend on ``f(x)``.
In the plots, the triangle indicates the current value of ``f(x)``. Note that ``f(x)`` can be an arbitrarily complicated function of ``x``. What matters is that the output of this function is used to shape the probability distribution.
"

# ╔═╡ 4b3b509f-7995-46b3-9130-8d881ee1e0ae
md"## Normal
If the response variable ``Y`` is real valued, one can choose the normal distribution with input-dependent mean ``f(x)`` and standard deviation ``\sigma``. Remember: if ``f(x)`` is linear, the maximum-likelihood estimate of the coefficients of ``f`` is the same as the least-squares solution in linear regression.

```math
p(y|x) = \frac1{\sqrt{2\pi}\sigma}e^{-\frac{(y-f(x))^2}{2\sigma^2}}
```

``f(x)`` = $(@bind ŷₙ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 8060861a-1931-4fae-b6ee-afb94ef2a3d5
md"## Bernoulli
If the response variable ``Y`` is binary, the Bernoulli distribution with input-dependent rate ``\sigma(f(x))`` is a natural choice.

```math
p(Y = \mathrm{A}|x) = p_\mathrm{A} = \sigma(f(x)) = \frac1{1 + e^{-f(x)}}

```

f(x) = $(@bind ŷₛ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 5d3efbe5-8937-45c0-894d-d24ece9b2262
md"## Categorical
If the response variable ``Y`` is a class label, the categorical distribution with input-dependent rates is a natural choice.

```math
p(Y = \mathrm{c}_i|x) = p_{\mathrm{c}_i} = s(f(x))_i = \frac{e^{f_i(x)}}{\sum_{j=1}^{K} e^{f_j(x)}}
```

f₁(x) = $(@bind ŷ₁ Slider(-5:.1:5, show_value = true))

f₂(x) = $(@bind ŷ₂ Slider(-5:.1:5, show_value = true))

f₃(x) = $(@bind ŷ₃ Slider(-5:.1:5, show_value = true))

f₄(x) = $(@bind ŷ₄ Slider(-5:.1:5, show_value = true))
"

# ╔═╡ 6acda59d-cb26-4ec0-8a91-2ec5a71bb2b7
md"
## Poisson

For counts, i.e. non-negative integers, it is not natural to assume a normally distributed noise. Instead, a common choice is to model a count variable ``Y`` with a Poisson distribution defined by
```math
P(Y = k|f(x)) = \frac{e^{-f(x)}f(x)^k}{k!}
```
where ``f(x)`` is the expected value of ``Y`` conditioned on ``x``.
In the cell below you can see the distribution of counts for different values of ``f(x)``.
"

# ╔═╡ 058d6dff-d6a5-47db-bba2-756d67b873d4
md"``f(x)`` = $(@bind λ Slider(0:.1:15, show_value = true))"

# ╔═╡ 2767c63e-eea4-4227-9d54-293826091e70
scatter(pdf.(Poisson(λ), 0:25), xlabel = "k", ylabel = "probability", legend = false)

# ╔═╡ f63c0616-eefe-11eb-268a-271fdc2ddd98
md"# 2. Blackboard Example

### Data Generator and Training Set
"


# ╔═╡ f63c0616-eefe-11eb-1cea-dfdaa64e6233
mlcode(
"""
using DataFrames, MLJ, Random

logistic(x) = 1 / (1 + exp(-x))
function data_generator(x; rng = Xoshiro(432))
    y = ifelse.(logistic.(2x .- 1) .> rand(rng, length(x)), "A", "B")
    DataFrame(x = x, y = categorical(y, levels = ["A", "B"]))
end
classification_data = data_generator([0., 2., 3.])
"""
,
"""
import numpy as np
import pandas as pd

np.random.seed(24)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def data_generator(x):
    y = np.where(logistic(2*np.array(x) - 1) > np.random.random(len(x)), "A", "B")
    df = pd.DataFrame({"x": x, "y": pd.Categorical(y, categories=["A", "B"])})
    return df

classification_data = data_generator([0., 2., 3.])
classification_data
"""
)

# ╔═╡ 7dfa42e3-c409-43a7-9952-a64fbad63c7f
mlstring(md"In the cell above we used the function `categorical` to tell the computer that the strings \"A\" and \"B\" indicate membership in different categories."
,
"
In the cell above we used the function `pd.Categorical` to tell the computer that the strings \"A\" and \"B\" indicate membership in different categories.
"
)

# ╔═╡ 5233d53e-0104-4eeb-a742-687bd9c9978a
md"
### Fitting

Next we will fit a logistic classifier to this data."

# ╔═╡ f63c061e-eefe-11eb-095b-8be221b33d49
mlcode(
"""
using MLJLinearModels
mach3 = machine(LogisticClassifier(penalty = :none), # model
                select(classification_data, :x),     # input
                classification_data.y);              # output
fit!(mach3, verbosity = 0);
fitted_params(mach3)
"""
,
"""
from sklearn.linear_model import LogisticRegression

mach3 = LogisticRegression(penalty=None)
mach3.fit(
    classification_data['x'].values.reshape(-1, 1), 
    classification_data['y']
    )
("coeff : ", mach3.coef_ , "intercept", mach3.intercept_)
"""
	
)

# ╔═╡ a9c7ca33-ce22-49b3-a976-8c180916fa5e
md"**Training data and conditional probabilites.** The training data is shown as blue dots (y-labels on the left), the probability of \"A\" under the data generator as a green curve (y-labels on the right). Note that ``Y = \mbox{B}`` at ``X = 3`` is very unlikely under the given data generator, but also unlikely events happen occasionally. The fitted probabilities are shown in red.
"

# ╔═╡ 0e775dfb-0da4-4536-886c-ada8c176a073
md"
### Making Predictions
"

# ╔═╡ 9322fa8d-772d-43ac-a6ff-10fe500c3244
mlstring(md"

For the `LogisticClassifier` the `predict` method returns the conditional probabilities of the classes. Click on the little gray triangle below to toggle the display of the full output."
,
"
For the `LogisticClassifier` the `predict_proba` method returns the conditional probabilities of the classes.
")

# ╔═╡ f63c061e-eefe-11eb-3b91-7136b4a16616
mlcode(
"""
p̂ = predict(mach3, DataFrame(x = -1:.5:2))
"""
,
"""
p = mach3.predict_proba( np.arange(-1, 2.5, 0.5).reshape(-1,1))
p
"""
)

# ╔═╡ 0c90f5b6-8a3b-41d8-9f51-d7d7c6b06ba0
mlstring(md"

If we want to extract the probability of a given response, we can use the `pdf` function."
,
"
The probability of a given response is store in one column of p. For the probability of response A : 
")

# ╔═╡ 5224d406-4e02-424d-9502-a22e0614cb96
mlcode(
"""
pdf.(p̂, "A")
"""
,
"""
("Probability of A : ", p[:,0])
"""
)

# ╔═╡ 34c49e49-a5e7-48ad-807b-c0624a59a367
mlstring(md"If we want to get as a response the class with the highest probability, we can use the function `predict_mode`."
,
"If we want to get as a response the class with the highest probability, we can use the function `predict`.
"
)

# ╔═╡ f63c0628-eefe-11eb-3125-077e533456d9
mlcode(
"""
predict_mode(mach3, DataFrame(x = -1:.5:2))
"""
,
"""
mach3.predict(np.arange(-1, 2.5, 0.5).reshape(-1,1))
"""
)

# ╔═╡ 38ccf735-5195-4e00-868f-95a895c05985
md"
### Computing the Log-Likelihood

In the following cell we define the log-likelihood loss function `ll` to compute the loss of the optimal parameters. This is the formula we derived in the slides. Note that the training data is somewhat hidden in this formula: the response of the first data point is a B (see above) at x = 0, therefore its probability is `logistic(-(θ[1] + θ[2]*0)) = logistic(-θ[1])` etc."

# ╔═╡ d0c4804f-c66d-4405-a7dc-1e85974e261f
mlcode(
"""
ll(θ) = log(logistic(-θ[1])) +
        log(logistic(θ[1] + 2θ[2])) +
        log(logistic(-θ[1] - 3θ[2]))

ll([-1.28858, .338548]) # the parameters we obtained above
"""
,
"""
import math

def ll(theta):
    return math.log(logistic(-theta[0])) +  math.log(logistic(theta[0] + 2 * theta[1])) + math.log(logistic(-theta[0] - 3 * theta[1]))

ll([-1.28858, 0.338548]) # the parameters we obtained above
"""
)

# ╔═╡ b8b81c1b-0faf-4ce9-b690-2f6cc9542b0f
md"""The likelihood of the optimal parameters is approximately -1.85.

$(mlstring(md"We could have obtained the same result using the `MLJ` function `log_loss`. This function computes the negative log-likelihood for each individual data point. To get the total likelihood we need to take the negative of the sum of these values.",
"We could have obtained the same result using the `sklearn` function `log_loss`. This function computes the negative log-likelihood for each individual data point. To get the total likelihood we need to take the negative of the sum of these values."))
"""

# ╔═╡ d034a44b-e331-4929-9053-351e7fe9aa94
mlcode(
"""
-sum(log_loss(predict(mach3), classification_data.y))
"""
,
"""
from sklearn.metrics import log_loss
classification_data["y"].values
-log_loss(classification_data["y"].values, mach3.predict_proba(classification_data['x'].values.reshape(-1, 1)), normalize=False)
"""
)

# ╔═╡ f7117513-283f-4e32-a2a1-3594c794c94d
md"### Multiple Logistic Regression

In the figure below we see on the left the probability of class A for the selected
parameter values and the decision threshold as a red plane. On the right we see samples (large points, red = class A)
obtained with this probability distribution and predictions (small points)
at the given decision threshold.
Play with the parameters to get a feeling for how they affect the probability
and the samples.

θ₀ = $(@bind θ₀ Slider(-3:3, default = 0, show_value = true))

θ₁ = $(@bind θ₁ Slider(-8:8, default = 3, show_value = true))

θ₂ = $(@bind θ₂ Slider(-8:8, default = 0, show_value = true))

decision threshold = $(@bind thresh Slider(.01:.01:1, default = .5, show_value = true))
"

# ╔═╡ fd4165dc-c3e3-4c4c-9605-167b5b4416da
md"# 3. Confusion Matrix, ROC and AUC

The parameters `s` controls the \"steepness\" of the data generator (blue).

Different datasets can be sampled by selecting different `seed`s.

By changing the decision threshold, the number of true positives and false positives changes. The curve that one obtains by changing the decision threshold in this way is called the receiver operating characteristic (ROC) curve. The area under the ROC (AUC) indicates how well the data can be separated into two classes: AUC ≈ 0.5 means that it is basically impossible to separate the data, whereas AUC ≈ 1 means that the data can almost perfectly be separated into two classes.
"

# ╔═╡ 7738c156-8e1b-4723-9818-fba364822171
md"s = $(@bind s Slider(-4:.1:4, default = 0, show_value = true))

seed = $(@bind seed Slider(1:100, show_value = true))

threshold = $(@bind threshold Slider(.01:.01:.99, default = 0.5, show_value = true))
"

# ╔═╡ 0fcfd7d2-6ea3-4c75-bad3-7d0fdd6fde11
begin
    logodds(p) = log(p/(1-p))
    function error_rates(x, y, t)
        P = sum(y)
        N = length(y) - P
        pos_pred = y[x .> t]
        TP = sum(pos_pred)
        FP = sum(1 .- pos_pred)
        FP/N, TP/P
    end
end;

# ╔═╡ ad5b293d-c0f4-4693-84f4-88308639a501
md"# 4. Applications of Logistic Regression

## Preparing the spam data

The text in our spam data set is already preprocessed. But we do not yet have
a format similar to our weather prediction data set with a fixed number ``p`` of
predictors for each email. In this section we create a very simple feature
representation of our emails:
1. We create a lexicon of words that are neither very frequent nor very rare.
2. For each email we count how often every word in this lexicon appears.
3. Our feature matrix will consist of ``n`` rows (one for each email) and ``p``
    predictors (one for each word in the lexicon) with ``x_{ij}`` measuring how
    often word ``j`` appears in document ``i``, normalized by the number of
    lexicon words in each email (such that the elements in every row sum to 1).

The code in this section is a bit advanced and it can be skipped at first reading.
The important point to remember here is that one may need to do quite some preprocessing of the raw data in order to apply certain machine learning methods to the data. Later in the course, we will see that one can apply other machine learning methods directly on the character-level representation of the emails.
"

# ╔═╡ 210b977d-7136-407f-a1c9-eeea869d0312
mlcode(
"""
using CSV, DataFrames
spamdata = CSV.read(download("https://go.epfl.ch/bio322-spam.csv"), DataFrame,
                    limit = 6000)
dropmissing!(spamdata) # remove entries without any text (missing values).
"""
,
"""
spamdata = pd.read_csv("https://go.epfl.ch/bio322-spam.csv", nrows=5000)
spamdata.dropna(inplace=True) # Drop rows with missing values
spamdata
"""
)

# ╔═╡ 4cbb3057-01f4-4e80-9029-4e80d6c9e5e6
mlstring(md"In the next cell we create the full lexicon of words appearing in the first
2000 emails. Each lexicon entry is of the form `\"word\" => count`. "
,
"
In the next cell we create the full lexicon of words appearing in the first
2000 emails. Use `vectorizer.get_feature_names_out()` to see the list of words. `X` is the matrix of occurance of these words for each email.
"
)

# ╔═╡ c50c529f-d393-4854-b5ed-91e90d557d12
mlcode(
"""
import TextAnalysis: Corpus, StringDocument, DocumentTermMatrix, lexicon,
                     update_lexicon!, tf
crps = Corpus(StringDocument.(spamdata.text[1:2000]))
update_lexicon!(crps)
lexicon(crps)
"""
,
"""
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
word_counts = vectorizer.fit_transform(spamdata["text"].values[:2000]).toarray()
"""
)

# ╔═╡ a37baeec-4252-40bd-8022-88cbedc504ed
mlstring(md" Then we select only those words of the full lexicon that appear at least 100 times and at most 10^3 times. These numbers are pulled out of thin air (like
all the design choses of this very crude feature engineering)."
,
"We select only those words that appear at least 100
times and at most 10^3 times.
"
)

# ╔═╡ f61773d4-cedb-4cb5-8bb1-82e0664fbf19
mlcode(
"""
small_lex = Dict(k => lexicon(crps)[k]
                 for k in findall(x -> 100 <= last(x) <= 10^3, lexicon(crps)))
m = DocumentTermMatrix(crps, small_lex)
"""
,
"""
words_occurence = np.sum(word_counts, axis=0) # count words occurences in all emails
index = np.where((words_occurence>10**3) | (words_occurence<100))[0] # select words by their count score
word_counts = np.delete(word_counts, index, 1)
word_counts
"""
)

# ╔═╡ 534681d5-71d8-402a-b455-f491cfbb353e
mlcode(
"""
spam_or_ham = coerce(String.(spamdata.label[1:2000]), OrderedFactor)
normalized_word_counts = float.(DataFrame(tf(m), :auto))
"""
,
"""
spam_or_ham = list(spamdata["label"][:2000])
normalized_word_counts = word_counts/word_counts.sum(axis= 0)[None,:] #normalized
words_list = np.delete(vectorizer.get_feature_names_out(),index)
pd.DataFrame(normalized_word_counts,columns=words_list)
"""
)

# ╔═╡ ec1c2ea5-29ce-4371-be49-08798305ff50
mlstring(md"Here we go: now we have a matrix of size 2000 x 801 as input and a vector of binary label as output. We will be able to use this as input in multiple logistic regression.",
"Here we go: now we have a matrix of size 2000 x 781 as input and a vector of binary label as output. We will be able to use this as input in multiple logistic regression.
")

# ╔═╡ 62ad57e5-1366-4635-859b-ccdab2efd3b8
md"## Multiple Logistic Regression on the spam data"

# ╔═╡ 29e1d9ff-4375-455a-a69b-8dd0c2cac57d
mlcode(
"""
m3 = fit!(machine(LogisticClassifier(penalty = :none),
                  normalized_word_counts,
                  spam_or_ham), verbosity = 0)
predict(m3)
"""
,
"""
m3 = LogisticRegression(penalty=None)
m3.fit(normalized_word_counts, spam_or_ham)
"""
)

# ╔═╡ 21b66582-3fda-401c-9421-73ae2f455a75
mlcode(
"""
predict_mode(m3)
"""
,
"""
m3.predict(normalized_word_counts)
"""
)

# ╔═╡ 32bafa9e-a35e-4f54-9857-d269b47f95c3
mlcode(
"""
confusion_matrix(predict_mode(m3), spam_or_ham)
"""
,
"""
from sklearn.metrics import confusion_matrix
confusion_matrix(m3.predict(normalized_word_counts), spam_or_ham)
"""
)

# ╔═╡ 4e4f4adf-364f-49b9-9391-5050a4c1286a
md"With our simple features, logistic regression can classify the training data
almost always correctly. Let us see how well this works for test data.
"

# ╔═╡ 50c035e6-b892-4157-a52f-824578366977
mlcode(
"""
test_crps = Corpus(StringDocument.(spamdata.text[2001:end]))
test_input = float.(DataFrame(tf(DocumentTermMatrix(test_crps, small_lex)), :auto))
test_labels = coerce(String.(spamdata.label[2001:end]), OrderedFactor)
confusion_matrix(predict_mode(m3, test_input), test_labels)
"""
,
"""
vectorizer = CountVectorizer(vocabulary = words_list)
test_input = vectorizer.fit_transform(spamdata["text"].values[2000:]).toarray()
test_labels = list(spamdata["label"][2000:])
confusion_matrix(m3.predict(test_input), test_labels)
"""
)

# ╔═╡ ef9489c3-2bff-431b-92c4-f1b9778040cf
md"In the following we use the functions `roc_curve` and `auc` to plot the ROC curve and compute the area under the curve."

# ╔═╡ e7d48a13-b4e6-4633-898c-c13b3e7f68ea
mlcode(
"""
using Plots
fprs1, tprs1, _ = roc_curve(predict(m3), spam_or_ham)
fprs2, tprs2, _ = roc_curve(predict(m3, test_input), test_labels)
plot(fprs1, tprs1, label = "training ROC")
plot!(fprs2, tprs2, label = "test ROC", legend = :bottomright)
"""
,
"""
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

train_input_int = [ 1 if x == "ham" else 0 for x in list(m3.predict(normalized_word_counts))]
train_labels_int = [ 1 if x == "ham" else 0 for x in spam_or_ham ]
test_input_int = [ 1 if x == "ham" else 0 for x in m3.predict(test_input)]
test_labels_int = [ 1 if x == "ham" else 0 for x in test_labels ]

fprs1, tprs1, _ = roc_curve(train_input_int, train_labels_int)
fprs2, tprs2, _ = roc_curve(test_input_int, test_labels_int)

plt.plot(fprs1, tprs1, label="training ROC")
plt.plot(fprs2, tprs2, label="test ROC")
plt.legend(loc="lower right")
plt.show()
"""
)

# ╔═╡ 8b851c67-0c6e-4081-a8ed-b818c2902c2f
mlcode(
"""
(training_auc = auc(predict(m3), spam_or_ham),
 test_auc = auc(predict(m3, test_input), test_labels))
"""
,
"""
from sklearn.metrics import roc_curve, auc
training_auc = auc(fprs1, tprs1)
test_auc = auc(fprs2, tprs2)
(training_auc, test_auc)
"""
)

# ╔═╡ a30578dd-aecb-46eb-b947-f009282cf2fc
md"Let us evaluate the fit in terms of commonly used losses for binary classification."

# ╔═╡ 8ed39cdc-e99e-48ff-9973-66df41aa0f78
mlcode(
"""
function losses(machine, input, response)
    (negative_loglikelihood = sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = auc(predict(machine, input), response)
	)
end
losses(m3, normalized_word_counts, spam_or_ham)
"""
,
"""
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

def losses(machine, input, response):
  negative_loglikelihood = log_loss(response, machine.predict_proba(input), normalize=False)
  misclassification_rate = np.mean(response != machine.predict(input))
  accuracy = accuracy_score(response, machine.predict(input))
  auc = roc_auc_score([ 1 if x == "ham" else 0 for x in list(m3.predict(input))], [ 1 if x == "ham" else 0 for x in response ])
  return negative_loglikelihood, misclassification_rate, accuracy, auc

losses(m3, normalized_word_counts, spam_or_ham)
"""
)


# ╔═╡ 935adbcd-48ab-4a6f-907c-b04137ca3abe
mlcode(
"""
losses(m3, test_input, test_labels)
"""
,
"""
losses(m3, test_input, test_labels)
"""
)

# ╔═╡ b6689b27-e8a2-44e4-8791-ce237767ee63
md"# 5. Poisson Regression

In this section we have a look at a regression problem where the response is a count variable. As an example we use a Bike sharing data set. In this data set, the number of rented bikes in Washington D.C. at a given time is recorded together with the weather condition. Remove the semicolon in the cell below, if you want to learn more about this dataset.

In this section our goal will be to predict the number of rented bikes at a given temperature and windspeed. Here is the description of the dataset:
"

# ╔═╡ 310999a0-f212-4e69-a4cb-346b3f49f202
OpenML.describe_dataset(42712)

# ╔═╡ 6384a36d-1dac-4d72-9d7b-84511f20e6ca
mlcode(
"""
bikesharing = OpenML.load(42712) |> DataFrame
"""
,
"""
import openml

bikesharing,_,_,_ = openml.datasets.get_dataset(42712).get_data(dataset_format="dataframe")
bikesharing
"""
)

# ╔═╡ db0f6302-333f-4e65-bff8-cd6c64f72cce
mlcode(
"""
dropmissing!(bikesharing) # remove rows with missing data
DataFrame(schema(bikesharing))
"""
,
"""
bikesharing.dropna(inplace=True) # remove rows with missing data
bikesharing.dtypes
"""
)

# ╔═╡ b9ba1df0-5086-4c0f-a2c9-200c2be27294
md"Above we see that the `:count` column is detected as `Continuous`, whereas it should be `Count`. We will therefore coerce it to the correct scientific type in the first line of the cell below.

For count variables we can use Poisson regression. Following the standard recipe, we parametrize ``f(x) = \theta_0 + \theta_1 x_1 + \cdots +\theta_d x_d``, plug this into the formula of the Poisson distribution and fit the parameters ``\theta_0, \ldots, \theta_d`` by maximizing the log-likelihood. In `MLJ` this is done by the `CountRegressor()`."

# ╔═╡ 81c55206-bf59-4c4e-ac5e-77a46e31bec7
mlcode(
"""
coerce!(bikesharing, :count => Count)

import MLJGLMInterface: LinearCountRegressor
m4 = machine(LinearCountRegressor(),
	         select(bikesharing, [:temp, :humidity]),
             bikesharing.count);
fit!(m4, verbosity = 0);
fitted_params(m4)
"""
,
"""
from sklearn import linear_model

m4 = linear_model.PoissonRegressor() # Creating a linear count regression model
m4.fit(bikesharing[['temp', 'humidity']], bikesharing['count']) # Fitting the model

m4.coef_ # Retrieving the fitted parameters
"""
)

# ╔═╡ 6ea40424-22a0-42b9-bfab-8d4903ab8d64
md"Not suprisingly, at higher temperatures more bikes are rented than at lower temperatures, but humitidy coefficient is negative.

In the next cell, we see that the predictions with this machine are Poisson distributions."

# ╔═╡ f071c985-7be7-454e-8541-28416400882f
mlcode(
"""
predict(m4)
"""
,
"""
m4.predict(bikesharing[['temp', 'humidity']])
"""
)

# ╔═╡ aa96bbe7-49f4-4244-9c71-8d9b2b3ee065
mlstring(md"And we can obtain the mean or the mode of this conditional distribution with:", "")

# ╔═╡ d5b394ac-b243-4825-a5c1-b30146500ef6
mlcode(
"""
predict_mean(m4)
"""
,
"""
nothing
"""
)

# ╔═╡ caa11dd3-577d-4692-b889-3a38d0bf61e0
mlcode(
"""
predict_mode(m4)
"""
,
"""
nothing
"""
)

# ╔═╡ 9ec91fbc-b756-4074-a623-1d47925c8239
mlstring(md"##### Side remark

`MLJ` distinguishes between models that make deterministic point predictions, like `LinearRegressor()` that predicts the expected response (which is the same as the mean of the conditional normal distribution in the probabilistic view), and models that predict probability distributions, like `LogisticClassifier()` or `LinearCountRegressor()`. If you are unsure about the prediction type of a model you can use the function `prediction_type`:","")

# ╔═╡ 41e5133c-db89-4407-9501-70e869616e9d
mlcode(
"""
prediction_type(LinearRegressor())
"""
,
"""
nothing
"""
)

# ╔═╡ b8bb7c85-0be8-4a87-96da-4e1b37aea96d
mlcode(
"""
prediction_type(LogisticClassifier())
"""
,
"""
nothing
"""
)

# ╔═╡ d3d7fa67-ca7d-46e1-b705-e30ec9b09f6a
mlcode(
"""
prediction_type(LinearCountRegressor())
"""
,
"""
nothing
"""
)

# ╔═╡ 8b0451bf-59b0-4e71-be84-549e23b5bfe7
md"""# Exercises

## Conceptual
#### Exercise 1
Suppose we have a data set with three predictors, ``X_1`` = Final Grade, ``X_2`` = IQ, ``X_3`` = Level (1 for College and 0 for High School).  The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get ``\hat\beta_0 = 25, \hat\beta_1 = 2, \hat\beta_2 = 0.07, \hat\beta_3 = 15``.
   - Which answer is correct, and why?
      - For a fixed value of IQ and Final Grade, high school graduates earn more, on average, than college graduates.
      - For a fixed value of IQ and Final Grade, college graduates earn more, on average, than high school graduates.
   - Predict the salary of a college graduate with IQ of 110 and a Final Grade of 4.0.
#### Exercise 2
Suppose we collect data for a group of students in a machine learning class with variables ``X_1 =`` hours studied, ``X_2 =`` grade in statistics class, and ``Y =`` receive a 6 in the machine learning class. We fit a logistic regression and produce estimated coefficients, ``\hat{\beta}_0 = -6``, ``\hat{\beta}_1 = 0.025``, ``\hat{\beta}_2 = 1``.
   - Estimate the probability that a student who studies for 75 hours and had a 4 in the statistics class gets a 6 in the machine learning class.
   - How many hours would the above student need to study to have a 50% chance of getting an 6 in the machine learning class?
#### Exercise 3
In this exercise we will familiarize ourselves with the loss function implicitly defined by maximum likelihood estimation of the parameters in a classification setting with multiple classes. Remember that the input ``f(x)`` of the softmax function ``s`` is a vector-valued function. Here we assume a linear function ``f`` and write the ``i``th component of this function as ``f_i(x) = \theta_{i0} + \theta_{i1}x_1 + \cdots + \theta_{ip}x_p``. Note that each component ``i`` has now its own parameters ``\theta_{i0}`` to ``\theta_{ip}``. Using matrix multiplication we can also write ``f(x) = \theta x`` where ``\theta = \left(\begin{array}{ccc}\theta_{10} & \cdots & \theta_{1p}\\\vdots & \ddots & \cdots\\\theta_{K0} & \cdots & \theta_{Kp}\end{array}\right)`` is a ``K\times(p+1)`` dimensional matrix and ``x = (1, x_1, x_2, \ldots, x_p)`` is a column vector of length ``p+1``.
- Write the log-likelihood function for a classification problem with ``n`` data points, ``p`` input dimensions and ``K`` classes and simplify the expression as much as you can. *Hint*: to simplify the notation we can use the convention ``s_y(f(x)) = P(y|x)`` to write the conditional probability of class ``y`` given input ``x`` and $\theta_{y_ij}$ for the parameter in the $y_i$th row and $j$th column of the parameter matrix. This convention makes sense when the classes are identified by the integers ``1, 2, \ldots, K``; in this case ``s_y(f(x))`` is the ``y``th component of ``s(f(x))``. Otherwise we would could specify a mapping from classes ``C_1, C_2, \ldots, C_K`` to the integers ``1, 2, \ldots, K`` for this convention to make sense.
- Assume now ``K = 3`` and ``p = 2``. Explicitly write the log-likelihood function for the training set ``\mathcal D = ((x_1 = (0, 0), y_1 = C), (x_2 = (3, 0), y_2 = A), (x_3 = (0, 2), y_3 = B))``. *Hint:* Choose yourselve a mapping from classes ``A, B, C`` to indices ``1, 2, 3``.
- Assume ``K = 2`` and ``p = 1`` and set ``\theta_{20} = 0`` and ``\theta_{21} = 0``. Show that we recover standard logistic regression in this case. *Hint*: show that ``s_1(f(x)) = \sigma(f_1(x))`` and ``s_2(f(x)) = 1 - \sigma(f_1(x))``, where ``s`` is the softmax function and ``\sigma(x) = 1/(1 + e^{-x})`` is the logistic function.
- Show that one can always set ``\theta_{K0}, \theta_{K1}, \ldots, \theta_{Kp}`` to zero. *Hint* Show that the softmax function with the transformed parameters ``\tilde\theta_{ij}=\theta_{ij} - \theta_{Kj}`` has the same value as the softmax function in the original parameters.

## Applied
#### Exercise 4
In the multiple linear regression of the weather data set above we used all
   available predictors. We do not know if all of them are relevant. In this exercise our aim is to find models with fewer predictors and quantify the loss in prediction accuracy.
- Systematically search for the model with at most 2 predictors that has the lowest test rmse. *Hint* write a function `train_and_evaluate` that takes the training and the test data as input as well as an array of two predictors; remember that `data[:, [\"A\", \"B\"]]` returns a sub-dataframe with columns \"A\" and \"B\". This function should fit a `LinearRegressor` on the training set with those two predictors and return the test rmse for the two predictors. To get a list of all pairs of predictors you can use something like `predictors = setdiff(names(train), ["time", "LUZ_wind_peak"]); predictor_pairs = [[p1, p2] for p1 in predictors, p2 in predictors if p1 != p2 && p1 > p2]`
- How much higher is the test error compared to the fit with all available predictors?
- How many models did you have to fit to find your result above?
- How many models would you have to fit to find the best model with at most 5 predictors? *Hint* the function `binomial` may be useful.
"""

# ╔═╡ b3a2cc60-f58c-4d07-93fc-19c80a6dd4da
md"""
#### Exercise 5
- Read the section on [scientific types in the MLJ manual](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Data-containers-and-scientific-types).
- Coerce the `count` variable of the bike sharing data to `Continuous` and fit a linear model (`LinearRegressor`) with predictors `:temp` and `:humidity`. 
Create a scatter plot with the true counts `bikesharing.count` on the x-axis and the predicted mode (`predict_mode`) of the counts for the linear regression model and the Poisson model on the y-axis. If the model perfectly captures the data, the plotted points should lie on the diagonal; you can add `plot!(identity)` to the figure to display the diagonal.
Comment on the differences you see in the plot between the Poisson model and the linear regression model.
"""

# ╔═╡ a52e3700-db5b-439e-9e43-0cde9a283c38
md"""
#### Exercise 6
In this exercise we perform linear classification of the MNIST handwritten digits
   dataset.
   - Load the MNIST data set with `using OpenML; mnist = OpenML.load(554) |> DataFrame; dropmissing!(mnist);`
   - Usually the first 60'000 images are taken as training set, but for this exercise I recommend to use fewer rows, e.g. the first 5000.
   - Scale the input values to the interval [0, 1) with `mnist[:, 1:784] ./= 255`
   - Fit a `MLJLinearModels.MultinomialClassifier(penalty = :none)` to the data. Be patient! This can take a few minutes.
   - Compute the misclassification rate and the confusion matrix on the training set.
   - Use as test data rows 60001 to 70000 and compute the misclassification rate
     and the confusion matrix on this test set.
   - Plot some of the correctly classified test images.
   - Plot some of the wrongly classified training and test images.
     Are they also difficult for you to classify?
"""

# ╔═╡ 20fe6494-2214-48e9-9c05-61a5faf9f91f
md"""
#### Exercise 7
Write a data generator function that samples inputs ``x`` normally distributed
   with mean 2 and standard deviation 3. The response ``y\in\{\mbox{true, false}\}``
   should be sampled from a Bernoulli distribution with rate of ``\mbox{true}``
   equal to ``\sigma(0.5x - 2.7)`` where ``\sigma(x) = 1/(1 + e^{-x})`` is the sigmoid (or logistic) function.
   - Create a training set of size ``n = 20``.
   - Fit the data with logistic regression.
   - Look at the fitted parameters.
   - Predict the probability of class `true` on the training input.  *Hint*: use the `pdf` function.
   - Determine the class with highest predicted probability and compare the result to the labels of the training set.
   - Create a test set of size ``n = 10^4`` where the input is always at ``x = 4``. Estimate the average test error at ``x = 4`` using this test set. Use the negative log-likelihood as error function.
   - Compute the test error at ``x = 4`` using the fitted parameters and compare your result to the previous result. *Hint:* Have a look at the slides for how to compute the test error when the parameters of the generator and the fitted function are known.
   - Rerun your solution with different training sets of size ``n = 20`` and write down your observations.
   - Rerun your solution multiple times with training sets of size ``n = 10^4``. Compare the fitted parameters to the one of the generator and look at the test error. Write down your observations.
"""

# ╔═╡ 20c5c7bc-664f-4c04-8215-8f3a9a2095c9
begin
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 7f08fcaa-000d-422d-80b4-e58a2f489d74
MLCourse.FOOTER

# ╔═╡ fbc70eaa-df15-423a-9885-93a5fa27fbc5
begin
	normalpdf(m) = x -> pdf(Normal(m, 1), x)
	logistic(x) = 1 / (1 + exp(-x))
	softmax(x) = exp.(x)/sum(exp.(x))
end;

# ╔═╡ 2adab164-1b4f-4f2f-ade3-dd70544c692e
begin
    plot(normalpdf(ŷₙ), xrange = (-8, 8), label = nothing,
         yrange = (0, .41), xlabel = "y", ylabel = "probability density p(y|x)", size = (350, 250))
    scatter!([ŷₙ], [0], markersize = 8, shape = :utriangle, label = nothing)
end

# ╔═╡ 6124e1eb-0aec-4ac9-aee6-383b25865220
let
	p1 = bar([0, 1], [logistic(ŷₛ), logistic(-ŷₛ)],
               xtick = ([0, 1], ["A", "B"]), ylabel = "probability P(class|x)",
               xlabel = "class", ylim = (0, 1), xlim = (-.5, 1.5), label = nothing)
    p2 = plot(logistic, xlim = (-5, 5), ylim = (0, 1),
               xlabel = "f(x)", ylabel = "probability of A", label = nothing)
    scatter!([ŷₛ], [0], markersize = 8, shape = :utriangle, label = nothing)
    vline!([ŷₛ], linestyle = :dash, color = :black)
    hline!([logistic(ŷₛ)], linestyle = :dash, color = :black)
    plot(p1, p2, size = (700, 350), legend = false)
end

# ╔═╡ 897fc69d-f1e9-49e9-9a61-25eb6849f7ec
bar(1:4, softmax([ŷ₁, ŷ₂, ŷ₃, ŷ₄]), ylabel = "probability P(class|x)",
    ylim = (0, 1), xlim = (.5, 4.5), xlabel = "class", legend = false,
    xtick = (1:4, ["A", "B", "C", "D"]), size = (350, 250))

# ╔═╡ 88741216-4736-4491-a167-31a7852a54e4
let xgrid = -.5:.1:3.5
    scatter([0, 2, 3], [0, 1, 0], xlabel = "x", ylabel = "y", yticks = ([0, 1], ["B","A"]), label = "data")
    plot!(twinx(), xgrid, logistic.(2xgrid .- 1), c = :green, label = "data generator", yaxis = "probability P(Y = \"A\"|x)", yrange = (0, 1))
    plot!(twinx(), xgrid, logistic.(.338548xgrid .- 1.28858), c = :red, label = "fitted model", yrange = (0, 1))
end

# ╔═╡ 285c6bfc-5f29-46e0-a2c1-8abbec74501b
begin
    Random.seed!(seed)
    auc_samples_x = 2 * randn(200)
    auc_samples_y = logistic.(2.0^s * auc_samples_x) .> rand(200)
end;

# ╔═╡ f1a48773-2971-4069-a240-fd1e10aeb1ed
confusion_matrix(auc_samples_x .> 1/(2.0^s) * logodds(threshold),
                 categorical(auc_samples_y, levels = [false, true], ordered = true))

# ╔═╡ c98524b5-d6b3-469c-82a1-7d231cc792d6
begin
    errs = [error_rates(auc_samples_x, auc_samples_y, 1/(2.0^s) * logodds(t))
           for t in .01:.01:.99]
    push!(errs, (0., 0.))
    prepend!(errs, [(1., 1.)])
end;

# ╔═╡ 3336ab15-9e9b-44af-a7d5-1d6472241e62
let
p1 = scatter(auc_samples_x, auc_samples_y, markershape = :vline, label = nothing, color = :black, yticks = ([0, 1], ["A", "B"]), xlabel = "x", ylabel = "y")
plot!(twinx(), x -> logistic(2.0^s * x), color = :blue, label = nothing, xlims = (-8, 8), ylabel = "probability P(Y = A|x)", yrange = (0, 1))
    vline!([1/(2.0^s) * logodds(threshold)], w = 3, color = :red,
           label = nothing,)
    p2 = plot(first.(errs), last.(errs), title = "ROC", fillrange = 0, fillstyle = :/, label = nothing, left_margin = 5Plots.mm)
    annotate!([.7], [.2], ["AUC ≈ $(round(sum(last.(errs) * .01), sigdigits = 2))"])
    fp, tp = errs[floor(Int, threshold * 100)]
    scatter!([fp], [tp], color = :red, xlims = (-.01, 1.01), ylims = (-.01, 1.01),
            labels = nothing, ylabel = "true positive rate",
            xlabel = "false positive rate")
    plot(p1, p2, size = (700, 400))
end

# ╔═╡ 26d957aa-36d4-4b90-9b91-2d9d883877ea
begin
	Random.seed!(123)
	samples = (X1 = 6 * rand(200) .- 3, X2 = 6 * rand(200) .- 3)
    f(x1, x2, θ₀ = θ₀, θ₁ = θ₁, θ₂ = θ₂) = logistic(θ₀ + θ₁ * x1 + θ₂ * x2)
end;

# ╔═╡ 4f89ceab-297f-4c2c-9029-8d2d7fad084f
let 
	Random.seed!(17)
    xgrid = -3:.25:3; ygrid = -3:.25:3
    wireframe = [[PP.scatter3d(x = fill(x, length(ygrid)),
                               y = ygrid, z = f.(x, ygrid),
                               mode = "lines", line_color = "blue")
                      for x in xgrid];
                 [PP.scatter3d(x = xgrid,
                               y = fill(y, length(xgrid)),
                               z = f.(xgrid, y), mode = "lines",
                               line_color = "blue")
                  for y in ygrid];
                 PP.mesh3d(opacity = .2, color = "red",
                           x = repeat(xgrid, length(ygrid)),
                           y = repeat(ygrid, inner = length(xgrid)),
                           z = fill(thresh, length(xgrid)*length(ygrid)))]
    l1 = PP.Layout(scene1 = PP.attr(xaxis_title_text = "X₁",
                                    xaxis_title_standoff = 0,
                                    yaxis_title_text = "X₂",
                                    yaxis_title_standoff = 0,
                                    zaxis_title_text = "probability of A",
                                    zaxis_title_standoff = 0,
                                    camera_eye = PP.attr(x = -1, y = 2.2, z = .2),
                                    domain = PP.attr(x = [-.1, .65], y = [-.1, 1.1])
                                   ),
                   xaxis2_domain = [.65, 1], yaxis2_domain = [.1, .9],
                   xaxis2_title_text = "X₁", yaxis2_title_text = "X₂",
                   xaxis2_title_standoff = 0, yaxis2_title_standoff = 0,
                   uirevision = 1, showlegend = false)
    labels = f.(samples.X1, samples.X2) .> rand(200)
    grid = MLCourse.grid2D(-3:.2:3, -3:.2:3, names = (:X1, :X2))
    plabels = f.(grid.X1, grid.X2) .> thresh
    pdata = [PP.scatter(x = grid.X1[plabels],
                        y = grid.X2[plabels],
                        mode = "markers", marker_color = "green",
                        marker_size = 3),
             PP.scatter(x = grid.X1[(!).(plabels)],
                        y = grid.X2[(!).(plabels)],
                        mode = "markers", marker_color = "red",
                        marker_size = 3),
             PP.scatter(x = samples.X1[labels], y = samples.X2[labels],
                        mode = "markers",
                        marker_color = "green"),
             PP.scatter(x = samples.X1[(!).(labels)], y = samples.X2[(!).(labels)],
                        mode = "markers",
                        marker_color = "red")]
    plot = hcat(PP.Plot(wireframe), PP.Plot(pdata))
    PP.relayout!(plot, l1)
    PP.PlutoPlot(plot)
end


# ╔═╡ Cell order:
# ╟─661e96c4-9432-4e39-b67e-6d0de2b76635
# ╟─0e6efce9-093a-4758-a649-6cff525711a5
# ╟─4b3b509f-7995-46b3-9130-8d881ee1e0ae
# ╟─2adab164-1b4f-4f2f-ade3-dd70544c692e
# ╟─8060861a-1931-4fae-b6ee-afb94ef2a3d5
# ╟─6124e1eb-0aec-4ac9-aee6-383b25865220
# ╟─5d3efbe5-8937-45c0-894d-d24ece9b2262
# ╟─897fc69d-f1e9-49e9-9a61-25eb6849f7ec
# ╟─6acda59d-cb26-4ec0-8a91-2ec5a71bb2b7
# ╟─058d6dff-d6a5-47db-bba2-756d67b873d4
# ╟─2767c63e-eea4-4227-9d54-293826091e70
# ╟─f63c0616-eefe-11eb-268a-271fdc2ddd98
# ╟─f63c0616-eefe-11eb-1cea-dfdaa64e6233
# ╟─7dfa42e3-c409-43a7-9952-a64fbad63c7f
# ╟─5233d53e-0104-4eeb-a742-687bd9c9978a
# ╟─f63c061e-eefe-11eb-095b-8be221b33d49
# ╟─88741216-4736-4491-a167-31a7852a54e4
# ╟─a9c7ca33-ce22-49b3-a976-8c180916fa5e
# ╟─0e775dfb-0da4-4536-886c-ada8c176a073
# ╟─9322fa8d-772d-43ac-a6ff-10fe500c3244
# ╟─f63c061e-eefe-11eb-3b91-7136b4a16616
# ╟─0c90f5b6-8a3b-41d8-9f51-d7d7c6b06ba0
# ╟─5224d406-4e02-424d-9502-a22e0614cb96
# ╟─34c49e49-a5e7-48ad-807b-c0624a59a367
# ╟─f63c0628-eefe-11eb-3125-077e533456d9
# ╟─38ccf735-5195-4e00-868f-95a895c05985
# ╟─d0c4804f-c66d-4405-a7dc-1e85974e261f
# ╟─b8b81c1b-0faf-4ce9-b690-2f6cc9542b0f
# ╟─d034a44b-e331-4929-9053-351e7fe9aa94
# ╟─f7117513-283f-4e32-a2a1-3594c794c94d
# ╟─4f89ceab-297f-4c2c-9029-8d2d7fad084f
# ╟─fd4165dc-c3e3-4c4c-9605-167b5b4416da
# ╟─7738c156-8e1b-4723-9818-fba364822171
# ╟─3336ab15-9e9b-44af-a7d5-1d6472241e62
# ╟─f1a48773-2971-4069-a240-fd1e10aeb1ed
# ╟─0fcfd7d2-6ea3-4c75-bad3-7d0fdd6fde11
# ╟─285c6bfc-5f29-46e0-a2c1-8abbec74501b
# ╟─c98524b5-d6b3-469c-82a1-7d231cc792d6
# ╟─ad5b293d-c0f4-4693-84f4-88308639a501
# ╟─210b977d-7136-407f-a1c9-eeea869d0312
# ╟─4cbb3057-01f4-4e80-9029-4e80d6c9e5e6
# ╟─c50c529f-d393-4854-b5ed-91e90d557d12
# ╟─a37baeec-4252-40bd-8022-88cbedc504ed
# ╟─f61773d4-cedb-4cb5-8bb1-82e0664fbf19
# ╟─534681d5-71d8-402a-b455-f491cfbb353e
# ╟─ec1c2ea5-29ce-4371-be49-08798305ff50
# ╟─26d957aa-36d4-4b90-9b91-2d9d883877ea
# ╟─62ad57e5-1366-4635-859b-ccdab2efd3b8
# ╟─29e1d9ff-4375-455a-a69b-8dd0c2cac57d
# ╟─21b66582-3fda-401c-9421-73ae2f455a75
# ╟─32bafa9e-a35e-4f54-9857-d269b47f95c3
# ╟─4e4f4adf-364f-49b9-9391-5050a4c1286a
# ╟─50c035e6-b892-4157-a52f-824578366977
# ╟─ef9489c3-2bff-431b-92c4-f1b9778040cf
# ╟─e7d48a13-b4e6-4633-898c-c13b3e7f68ea
# ╟─8b851c67-0c6e-4081-a8ed-b818c2902c2f
# ╟─a30578dd-aecb-46eb-b947-f009282cf2fc
# ╟─8ed39cdc-e99e-48ff-9973-66df41aa0f78
# ╟─935adbcd-48ab-4a6f-907c-b04137ca3abe
# ╟─b6689b27-e8a2-44e4-8791-ce237767ee63
# ╟─310999a0-f212-4e69-a4cb-346b3f49f202
# ╟─6384a36d-1dac-4d72-9d7b-84511f20e6ca
# ╟─db0f6302-333f-4e65-bff8-cd6c64f72cce
# ╟─b9ba1df0-5086-4c0f-a2c9-200c2be27294
# ╟─81c55206-bf59-4c4e-ac5e-77a46e31bec7
# ╟─6ea40424-22a0-42b9-bfab-8d4903ab8d64
# ╟─f071c985-7be7-454e-8541-28416400882f
# ╟─aa96bbe7-49f4-4244-9c71-8d9b2b3ee065
# ╟─d5b394ac-b243-4825-a5c1-b30146500ef6
# ╟─caa11dd3-577d-4692-b889-3a38d0bf61e0
# ╟─9ec91fbc-b756-4074-a623-1d47925c8239
# ╟─41e5133c-db89-4407-9501-70e869616e9d
# ╟─b8bb7c85-0be8-4a87-96da-4e1b37aea96d
# ╟─d3d7fa67-ca7d-46e1-b705-e30ec9b09f6a
# ╟─8b0451bf-59b0-4e71-be84-549e23b5bfe7
# ╟─b3a2cc60-f58c-4d07-93fc-19c80a6dd4da
# ╟─a52e3700-db5b-439e-9e43-0cde9a283c38
# ╟─20fe6494-2214-48e9-9c05-61a5faf9f91f
# ╟─20c5c7bc-664f-4c04-8215-8f3a9a2095c9
# ╟─7f08fcaa-000d-422d-80b4-e58a2f489d74
# ╟─fbc70eaa-df15-423a-9885-93a5fa27fbc5
# ╟─12d5824c-0873-49a8-a5d8-f93c73b633ae
# ╟─94f8e29e-ef91-11eb-1ae9-29bc46fa505a
