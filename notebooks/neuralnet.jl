### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 87f59dc7-5149-4eb6-9d81-440ee8cecd72
begin

using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, Flux
import PlutoPlotly as PP
import MLCourse: heaviside
const M = MLCourse.JlMod
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ 83e2c454-042f-11ec-32f7-d9b38eeb7769
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ c1cbcbf4-f9a9-4763-add8-c5ed7cbd2889
md"The goal of this week is to go through a case study, where we follow a recipe of a supervised machine learning project (see slides).
"

# ╔═╡ 60a740a1-4dc2-488e-9583-750688fe7b36
md"# Case Study: Bike Rental
## Data Loading"

# ╔═╡ c5270f07-ca15-4767-add9-dbef0ce52e69
mlcode("",
"
import openml
import pandas as pd

bikesharing , *_= openml.datasets.get_dataset(42712).get_data()
bikesharing = bikesharing.iloc[:1296,:]
bikesharing
")

# ╔═╡ 0297a717-ebe6-41bb-9923-3e8f1c0193f3
md"## Data Inspection"

# ╔═╡ 71507967-74e0-4d54-b863-c5f2f2f7ac0d
mlcode("",
"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
g = sns.PairGrid(bikesharing)
g.map_lower(sns.regplot, line_kws={'color': 'black'})
g.map_diag(sns.histplot, color = 'darkorange' )
g.map_upper(sns.kdeplot, fill=True,cmap="Reds")
plt.show()
""")

# ╔═╡ 56c33b61-858f-474b-9f0f-94bbb76f5bfe
md"
- There is only one year. We can drop this constant predictor
- Not surprisingly, temp and feel_temp are highly correlated. We could investigate if both predictors are needed.
- Also not surprisingly, there is a non-linear, periodic relationship between hour and the number of rented bicycles. Maybe one-hot encoding, or cos- sine-features of the hour predictor would be a good idea.
"

# ╔═╡ 8f3467c1-69c6-4140-b43c-13d5d28e25b3
md"## Feature engineering and a linear fit"

# ╔═╡ b50a062e-7875-4cf2-9dc6-e50d9606ea08
mlcode("",
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

selected_features = ['temp', 'humidity', 'hour', 'weekday', 'weather', 'holiday']
target = bikesharing['count']

preprocessor1 = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['hour', 'weekday', 'holiday']),
        ('passthrough', 'passthrough', ['temp', 'humidity'])
    ])

preprocessor2 = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['holiday']),
        ('passthrough', 'passthrough', ['hour', 'weekday', 'temp', 'humidity'])
    ])

linear_model1 = Pipeline([
    ('preprocessor', preprocessor1),
    ('regressor', PoissonRegressor(max_iter=5000))
])

linear_model2 = Pipeline([
    ('preprocessor', preprocessor2),
    ('regressor', PoissonRegressor(max_iter=5000))
])

"""
)

# ╔═╡ 3461e59d-870a-428a-af9c-1f8435c6154c
mlcode("",
"
cv_scores1 = cross_val_score(linear_model1,X=bikesharing[selected_features], y=target, cv=5, scoring='neg_mean_poisson_deviance')
cv_scores1
")

# ╔═╡ 3abaa571-bee0-4ef8-aea0-7726739988ee
mlcode("",
"cv_scores2 = cross_val_score(linear_model2,X=bikesharing[selected_features], y=target, cv=5, scoring='neg_mean_poisson_deviance')
cv_scores2
")

# ╔═╡ 9e1d6da1-4feb-47d3-b2ed-36e99d23275e
md"Clearly, one-hot coding of hours and weekday works better than the raw input for this linear model (this does not need to be the case for the neural network). This is confirmed in the plot below."

# ╔═╡ e4cfe1da-4797-4446-ba87-42f31cc8ef85
mlcode("",
"""
import matplotlib.pyplot as plt

linear_model1.fit(bikesharing[selected_features], target)
linear_model2.fit(bikesharing[selected_features], target)

plt.figure()
plt.scatter(bikesharing['count'], linear_model1.predict(bikesharing[selected_features]), label="with hour, weekday, holiday 1-hot", s=25, linewidth=.5, edgecolor='black')
plt.scatter(bikesharing['count'], linear_model2.predict(bikesharing[selected_features]), label="only holiday 1-hot", s=25, linewidth=.5, edgecolor='orange')
plt.plot([0, bikesharing['count'].max()], [0, bikesharing['count'].max()], linestyle='--', color='black')
plt.xlabel("true counts")
plt.ylabel("predicted mode of counts")
plt.legend(loc='upper left')
plt.show()
"""
)

# ╔═╡ ae463314-4bcb-4153-966e-70edf7d0b07b
md"There is still a lot of unexplained variance, but we shouldn't conclude here that the irreducible error is high. In particular, because the negative mean poisson deviance (see below) is roughly as large as the cross-validation estimate of the test error. Despite of the test error estimate across the different folds, this is indicative of the linear model having a large bias, i.e. not being flexible enough.
"

# ╔═╡ 88fbf339-94cd-427a-b602-a5449da31f96
mlcode("",
"
from sklearn.metrics import mean_poisson_deviance

linear_model1.fit(bikesharing[selected_features], target)
predictions = linear_model1.predict(bikesharing[selected_features])
-mean_poisson_deviance(target, predictions)
"
)

# ╔═╡ e1e17233-34de-4f86-b227-83f5230d86e9
md"## Neural Network fit"

# ╔═╡ 11b2642a-ed6d-46b4-8661-721600c115bf
mlcode("",
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X = torch.tensor(preprocessor1.fit_transform(bikesharing[selected_features]).toarray(), dtype = torch.float32)
y = torch.tensor(target)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Neural network to predict Poisson rate (lambda)
class PoissonNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
			nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
			nn.Linear(hidden_dim, hidden_dim, bias = False),
			nn.Dropout(dropout_rate),
			nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        rate = torch.exp(self.fc(x))
        return rate.squeeze(-1)

model = PoissonNet(input_dim=32, hidden_dim=256)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Poisson negative log-likelihood loss
class PoissonLoss(torch.nn.Module):
    def forward(self, rate, target):
        return torch.mean(rate - target * torch.log(rate + 1e-8))

loss_fn = PoissonLoss()

# Training loop
losses = []
for epoch in range(600):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        rate = model(batch_X)
        loss = loss_fn(rate, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()/batch_X.shape[0]
    losses.append(epoch_loss)


plt.figure()
plt.plot(range(10, 601), losses[9:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.show()
"""
,
recompute = true
)

# ╔═╡ 8735326c-6c53-4b3a-a962-487ae4d79013
md"Always looking at the learning curve and the plot below, I tried different values  hidden\_dim, batch\_size, learning\_rate and number of epochs, until I had something decent. This could be taken as a starting point for a more thorough hyper-parameter tuning. At this point, I did not care about overfitting. I wanted to find some hyper-parameter values that lead to an accurate fit."

# ╔═╡ 0128cd24-7959-462d-ad1c-432e010962c2
mlcode("",
"""
plt.figure()
plt.scatter(bikesharing['count'], model(X).detach().numpy(), s=25, linewidth=.5, edgecolor='black')
plt.plot([0, bikesharing['count'].max()], [0, bikesharing['count'].max()], linestyle='--', color='black')
plt.xlabel("true counts")
plt.ylabel("predicted mode of counts")
plt.show()
"""
,
recompute=true
)

# ╔═╡ 2e4387f3-598f-4c01-891d-d1577b65ab95
md"This looks much better than the linear model above."

# ╔═╡ a55694fb-cb72-4805-be44-cc9bee31c6e9
mlcode("",
"""
from skorch import NeuralNetRegressor
from sklearn.model_selection import cross_val_score

net_poisson = NeuralNetRegressor(
    PoissonNet,
    module__input_dim=32,
    module__hidden_dim=256,
    max_epochs=500,
    lr=1e-3,
    criterion=loss_fn,
    optimizer=torch.optim.AdamW,
    iterator_train__shuffle=False,
    verbose=0
)

scores_poisson = cross_val_score(net_poisson, X, y, cv=5, scoring='neg_mean_poisson_deviance')
scores_poisson
""",
recompute=true
)

# ╔═╡ 22a8f42c-c7ca-4547-92d9-b6b557780377
mlcode("","{'number of parameters': sum(p.numel() for p in model.parameters()), 'number of samples': X.shape[0]}")

# ╔═╡ 710331e7-5b08-42f0-809e-8d32750ded63
md"The model does indeed seem to overfit quite a bit, which is not surprising, given that the number of parameters is much higher than the number of samples!"

# ╔═╡ 6cafba73-50bb-4fcd-abb6-9449ec41aebd
mlcode("",
"""
net_poisson_reg = NeuralNetRegressor(
    PoissonNet,
    module__input_dim=32,
    module__hidden_dim=256,
    module__dropout_rate=0.1,
    max_epochs=500,
    lr=1e-3,
    criterion=loss_fn,
    optimizer=torch.optim.AdamW,
    iterator_train__shuffle=False,
    verbose=0
)

scores_poisson_reg = cross_val_score(net_poisson_reg, X, y, cv=5, scoring='neg_mean_poisson_deviance')
scores_poisson_reg
""")

# ╔═╡ e43e7f0b-879f-4f04-a390-7bfa3bc4c806
mlcode("", "{'linear model': cv_scores1.mean(), 'without regularization': scores_poisson.mean(), 'with dropout': scores_poisson_reg.mean()}", recompute = true)

# ╔═╡ 0440f6b1-b422-4209-9c72-bda3313d986a
md"With a little bit of dropout the estimate of the test score is much higher, and in fact better than the linear model. Again, this could be the starting point of more detailed hyper-parameter tuning."

# ╔═╡ cd977593-439f-45a6-88b3-473ff97e52e0
md"""
# Exercises
## Conceptual
#### Exercise 1
Here is some LLM generated code.
```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 10, bias=false),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(10, 4)
        )
    def forward(self, x):
        return self.model(x)

data = torch.randn(10_000, 3)
target = torch.randint(0, 4, (10_000,))

batch_size = 256
num_samples = data.size(0)
num_epochs = 20
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(num_epochs):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        batch_data = data[indices]
        batch_target = target[indices]

        optimizer.zero_grad()
        outputs = net(batch_data)
        loss = criterion(outputs, batch_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_data.size(0)
    avg_loss = epoch_loss / num_samples
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

```
1. How many parameters does the `net` have?
2. List all hyper-parameters in this code.
3. Write a summary of 5-10 sentences to explain this code to a SV semester 5 student who didn't take this course.
"""

# ╔═╡ d15f353e-9a23-4466-af56-336b19872bb2
Markdown.parse("""
The following text can be found in a [research paper on dropout](https://jmlr.org/papers/v15/srivastava14a.html)
#### Exercise 2
$(MLCourse.embed_figure("dropout.png"))

Show with a simple example of a neural network with one hidden layer of relu neurons and a linear readout layer, that

a) the approximate model matches the true model average if dropout is applied after the relu layer.

b) the approximate model doesn't match the true model average if dropout is applied before the relu layer.
"""
)

# ╔═╡ 46597df0-4386-4567-b424-29a69c856568
md"## Applied
#### Exercise 3
In this exercise we will compare different regularization methods for neural networks.

We use the following generator of synthetic data:
"

# ╔═╡ 7545547b-db7e-444d-b924-b6a6fa0e2ac9
mlcode("",
"
import numpy as np

def generate_nonlinear_3class_dataset(n_samples=300, random_state=None):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples, dtype=int)
    # Non-linear boundaries: circles
    for i in range(n_samples):
        r = np.sqrt(X[i,0]**2 + X[i,1]**2)
        if r < 0.8:
            y[i] = 0
        elif r < 1.6:
            y[i] = 1
        else:
            y[i] = 2
    return X, y

X, y = generate_nonlinear_3class_dataset(100, random_state=142)

"
)

# ╔═╡ 2a666ce7-f074-4e62-a5ba-0f7ab185da48
md"
a) Train a neural network with two hidden layers of ``n`` relu neurons, and a third hidden layer of 2 relu neurons. Choose `n` and the number of epochs large enough to reach zero classification error on the training set.

b) Plot the raw 2-dimensional input data, as well as the activity in the third (2D) hidden layer. Color the points with the class label.

c) Estimate the test error on a large test set.

d) Regularize your neural network in 3 different ways: dropout in the first two hidden layers, L2 regularization, and early stopping. Compare the 3 results (also with the solution found in a) in terms of the L2 norm of the final parameter values, representation in the third hidden layer (plot as in b), and test error.
"

# ╔═╡ 8d6dfbd5-90d3-4d61-98c9-ae73879e803e
MLCourse.FOOTER

# ╔═╡ 14bc31ad-c5b9-44c9-ad6a-028dbea2b06b
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─83e2c454-042f-11ec-32f7-d9b38eeb7769
# ╟─c1cbcbf4-f9a9-4763-add8-c5ed7cbd2889
# ╟─60a740a1-4dc2-488e-9583-750688fe7b36
# ╟─c5270f07-ca15-4767-add9-dbef0ce52e69
# ╟─0297a717-ebe6-41bb-9923-3e8f1c0193f3
# ╟─71507967-74e0-4d54-b863-c5f2f2f7ac0d
# ╟─56c33b61-858f-474b-9f0f-94bbb76f5bfe
# ╟─8f3467c1-69c6-4140-b43c-13d5d28e25b3
# ╟─b50a062e-7875-4cf2-9dc6-e50d9606ea08
# ╟─3461e59d-870a-428a-af9c-1f8435c6154c
# ╟─3abaa571-bee0-4ef8-aea0-7726739988ee
# ╟─9e1d6da1-4feb-47d3-b2ed-36e99d23275e
# ╟─e4cfe1da-4797-4446-ba87-42f31cc8ef85
# ╟─ae463314-4bcb-4153-966e-70edf7d0b07b
# ╟─88fbf339-94cd-427a-b602-a5449da31f96
# ╟─e1e17233-34de-4f86-b227-83f5230d86e9
# ╟─11b2642a-ed6d-46b4-8661-721600c115bf
# ╟─8735326c-6c53-4b3a-a962-487ae4d79013
# ╟─0128cd24-7959-462d-ad1c-432e010962c2
# ╟─2e4387f3-598f-4c01-891d-d1577b65ab95
# ╟─a55694fb-cb72-4805-be44-cc9bee31c6e9
# ╟─22a8f42c-c7ca-4547-92d9-b6b557780377
# ╟─710331e7-5b08-42f0-809e-8d32750ded63
# ╟─6cafba73-50bb-4fcd-abb6-9449ec41aebd
# ╟─e43e7f0b-879f-4f04-a390-7bfa3bc4c806
# ╟─0440f6b1-b422-4209-9c72-bda3313d986a
# ╟─cd977593-439f-45a6-88b3-473ff97e52e0
# ╟─d15f353e-9a23-4466-af56-336b19872bb2
# ╟─46597df0-4386-4567-b424-29a69c856568
# ╟─7545547b-db7e-444d-b924-b6a6fa0e2ac9
# ╟─2a666ce7-f074-4e62-a5ba-0f7ab185da48
# ╟─8d6dfbd5-90d3-4d61-98c9-ae73879e803e
# ╟─87f59dc7-5149-4eb6-9d81-440ee8cecd72
# ╟─14bc31ad-c5b9-44c9-ad6a-028dbea2b06b
