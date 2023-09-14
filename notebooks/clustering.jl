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

# ╔═╡ 48d87103-4c23-4144-a121-1e33d2bb87f3
begin
using Pkg
Base.redirect_stdio(stderr = devnull, stdout = devnull) do
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
end
using Revise, MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, MLJMultivariateStatsInterface, OpenML, MLJClusteringInterface, StatsPlots, Distributions, Distances
import PlutoPlotly as PP
const M = MLCourse.JlMod
MLCourse.CSS_STYLE
end

# ╔═╡ 7b013132-0ee2-11ec-1dd2-25a9f16f0568
begin
    using PlutoUI
    function MLCourse.embed_figure(name)
        "![](data:img/png; base64,
             $(open(MLCourse.base64encode,
                    joinpath(Pkg.devdir(), "MLCourse", "notebooks", "figures", name))))"
    end
    PlutoUI.TableOfContents()
end

# ╔═╡ da8f8ce7-c3ad-4e1a-bb9e-c7be15646d72
md"# K-means Clustering"

# ╔═╡ 49a1493b-ba68-4e7a-8c3d-47423f7a8204
mlcode("""
using MLJ, DataFrames, OpenML, StatsPlots
iris = DataFrame(OpenML.load(61))
iris = rename!(iris, replace.(names(iris), "'" => ""))
coerce!(iris, :class => Multiclass)
"""
,
"""
import pandas as pd
from sklearn.datasets import load_iris

iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['class'] = iris_data.target
iris
"""
)

# ╔═╡ 67b8f27f-b5b5-4dc0-ab51-83ac3dcaa2f7
mlcode("""
iris.class
""",
"""
iris["class"]
""")

# ╔═╡ 1ebd74bc-4479-41d2-aba4-934cdd2b778a
mlcode("""
@df iris corrplot([:sepallength :sepalwidth :petallength :petalwidth],
                    compact = true, fillcolor=cgrad())
""",
"""
import seaborn as sns
g = sns.PairGrid(iris.drop(["class"], axis =1))
g.map_lower(sns.regplot, line_kws={'color': 'black'})
g.map_diag(sns.histplot, color = 'darkorange' )
g.map_upper(sns.kdeplot, fill=True,cmap="Reds")
""")

# ╔═╡ 4dfeee20-ac92-48d2-9ba1-40baea227100
md"
If you want to dive a bit deeper into some quantitative approaches that are used to select the number of clusters please have a look at [this wikipedia article](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) or this [opinionated tutorial](https://towardsdatascience.com/silhouette-method-better-than-elbow-method-to-find-optimal-clusters-378d62ff6891).
"

# ╔═╡ e187d7a8-4ccd-4b0a-8c71-f79219623142
mlcode("""r'Number of clusters $(@bind nclust Slider(2:8, show_value = true))'"""
,
"r'Number of clusters $(@bind nclust Slider(2:8, show_value = true))'"
)

# ╔═╡ 99cd7b6b-1d1f-4712-a83e-7969ae7fecd4
mlcode(
md""" $(name): $(Child(name,Slider(1:100)))"""
@bind x Slider(5:15)
,"""""")

# ╔═╡ d1c88a44-be52-4b0e-bc23-cca00d10ffb6
mlcode("""
using MLJClusteringInterface
mach1 = machine(KMeans(k = nclust), select(iris, Not(:class)))
fit!(mach1, verbosity = 0)
""",
"""
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=nclust, random_state=0, n_init="auto").fit(iris.drop(["class"], axis =1))
kmeans
"""
)

# ╔═╡ 4df24b22-47a7-446a-8e27-9f2031a75764
mlcode("""
prediction = MLJ.predict(mach1, select(iris, Not(:class)))
""",
"""
prediction = kmeans.predict(iris.drop(["class"], axis =1))
prediction
""")

# ╔═╡ 0e7f34b9-e24f-447b-840b-e2750d2e778b
mlcode("""
p1 = scatter(iris.petallength, iris.petalwidth,
			 legend = false, c = Int.(int(prediction)),
			 title = "prediction", xlabel = "petal length", ylabel = "petal width")
p2 = scatter(iris.petallength, iris.petalwidth,
			 legend = false, c = Int.(int(iris.class)),
			 title = "ground truth", xlabel = "petal length", ylabel = "petal width")
plot(p1, p2, layout = (1, 2), size = (700, 500))
""",
"""
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

ax1.scatter(iris['petal length (cm)'], iris['petal width (cm)'], c=prediction.astype(int))
ax1.set_title('prediction')
ax1.set_xlabel('petal length')
ax1.set_ylabel('petal width')

ax2.scatter(iris['petal length (cm)'], iris['petal width (cm)'], c=iris['class'].astype(int))
ax2.set_title('ground truth')
ax2.set_xlabel('petal length')
ax2.set_ylabel('petal width')

plt.show()
""")

# ╔═╡ 70b3f1bb-7c47-4bb0-aa17-cda6fdbe0469
md"## KMeans Clustering Applied to MNIST"

# ╔═╡ 58220aaf-9f85-4de7-82fa-dec8bc9842fd
Markdown.parse("
Let us try to apply K-Means Clustering to the MNIST data set.")

# ╔═╡ 8e71fb7a-b83f-4ff1-b78b-4814dd6d6a6c
mlcode("""
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    select(df, Not(:class)),
    df.class
end

mnist_mach = fit!(machine(KMeans(k = 10), mnist_x));

mnist_pred = predict(mnist_mach)

idxs = [findall(==(k), mnist_pred)[1:15] for k in 1:10]
img = vcat([[hcat([[reshape(Array(mnist_x[idx, :]), 28, :)' fill(255, 28)]
				   for idx in row]...); fill(255, 15*29)']
			for row in idxs]...)./255
plot(Gray.(img), showaxis = false, ticks = false,
	 xlabel = \"examples\", ylabel = \"cluster 1 to 10\")
""",
"""
import openml
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

mnist,_,_,_ = openml.datasets.get_dataset(554).get_data(dataset_format="dataframe")
mnist["class"] = mnist["class"].astype("category")

kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=10, random_state=0,
				n_init="auto")).fit(mnist.drop(["class"], axis =1))

prediction = kmeans.predict(mnist.drop(["class"], axis =1))

idxs = [list(np.where(prediction == k)[0][1:10]) for k in range(1, 11)]

fig, ax = plt.subplots(9,5)
for i in range(9):
  for j in range(5) :
    ax[i][j].imshow(mnist.iloc[idxs[i][j], :-1].values.reshape(28, 28).astype(float)/255,
           cmap='gray')

plt.show()
""")

# ╔═╡ a0af5510-8585-4c38-9166-5284d3b2923e
Markdown.parse("$(MLCourse.embed_figure("mnist_kmeans.png"))

Some clusters (rows in the figure above) seem indeed to have specialised on some
digit classes. This can further be seen by looking at the confusion_matrix.")

# ╔═╡ 6310f2b4-ea1b-48d1-a23d-b5a43291772f
mlcode("""
confusion_matrix(mnist_pred, mnist_y) |> x -> DataFrame(x.mat, x.labels)
""",
"""
from sklearn.metrics import confusion_matrix
confusion_matrix(prediction, np.array(mnist["class"].values, dtype=int))
""")

# ╔═╡ 46816b6c-490d-426b-befd-735601aa3d10
Markdown.parse("$(MLCourse.embed_figure("mnist_kmeans_table.png"))")

# ╔═╡ 5dad8e78-2862-47cc-b693-806b307733e1
mlstring(md"We see that the ninth cluster (9th row in this table and in the figure above) has indeed specialized on images with class label \"0\" and the eigth cluster has specialized on images with class label \"2\", but e.g. cluster 5 contains similar amounts of 0s, 5s and 6s.",
"
We see that the ninth cluster (9th row in this table and in the figure above) has indeed specialized on images with class label \"0\" and the eigth cluster has specialized on images with class label \"2\", but e.g. cluster 5 contains similar amounts of 0s, 5s and 6s.
")

# ╔═╡ 12317841-c1f8-4022-970e-8ef613c79b78
md"# Hierarchical Clustering"

# ╔═╡ 872e95fd-fac0-4c39-bc65-e4cdcf0050bb
mlcode("""
using Distributions, Distances
hc = machine(HierarchicalClustering(k = 3, linkage = :complete, metric = Euclidean()))
""",
"""
from sklearn.cluster import AgglomerativeClustering
hc = make_pipeline(StandardScaler(), AgglomerativeClustering(n_clusters = 3, metric = "euclidean", linkage = "complete")).fit(iris.drop(["class"], axis =1))
hc
"""
)

# ╔═╡ f4d74b33-e05c-445a-9d59-c91dfff97eab
mlcode("""predict(hc, select(iris, Not(:class)))""",
"""
hc.fit_predict(iris.drop(["class"], axis =1))
"""
)

# ╔═╡ 2f4e3d67-d88e-4f9b-9572-6be1bc30106b
md"After running the `predict` function on the data, we can get from the `report` the fitted dendrogram and a cutter object with which we can make further predictions at other heights `h` or numbers of clusters `k`."

# ╔═╡ c4bde04d-23e8-4fb0-9e66-88b443a12926
mlcode("""hc_report = report(hc); cutter = hc_report.cutter; dendrogram = hc_report.dendrogram;"""
,
"""
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(hc, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# code from sklearn
""";
eval = false
)

# ╔═╡ 84d2b372-1c4a-4cae-958c-a7a7b19f24f6
mlcode("""cutter(h = 2)""",
"""
""";
eval = false
)

# ╔═╡ 41e0f1e0-19a7-441e-a6ea-be0fe4678288
mlcode("""cutter(k = 4)""",
"""
""";
eval = false
)

# ╔═╡ 23958c06-c037-452d-ae15-36d4d8cc9038
md"This can be used to visualize the dendrogram and the predictions."

# ╔═╡ 6cc7a2a0-414f-499d-9f45-f525b0a5c1cc
mlcode("""
h = $(@bind hclusth Slider(range(minimum(report(M.hc).dendrogram.heights), stop = maximum(report(M.hc).dendrogram.heights), length = 100), default = maximum(report(M.hc).dendrogram.heights)/2))
""",
"""
""")

# ╔═╡ f6b3f577-d893-4c7f-a0c5-df5a267f2d1e
mlcode("""
hclusth=

pred = cutter(h = hclusth)
p1 = scatter(iris.petallength, iris.petalwidth,
			 legend = false, c = Int.(int(pred)),
			 title = "prediction", xlabel = "petal length",
			 ylabel = "petal width")
p2 = plot(dendrogram)
hline!([hclusth], c = :red, w = 3)
plot(p2, p1, layout = (1, 2), size = (700, 500))
""",
"""
""")

# ╔═╡ e266ada3-ba4d-4177-8129-f1220f293c72
let
	pred = cutter(h = hclusth)
    p1 = scatter(iris.petallength, iris.petalwidth,
                 legend = false, c = Int.(int(pred)),
                 title = "prediction", xlabel = "petal length",
                 ylabel = "petal width")
    p2 = plot(dendrogram)
    hline!([hclusth], c = :red, w = 3)
    plot(p2, p1, layout = (1, 2), size = (700, 500))
end


# ╔═╡ 675e3a37-8044-4e8a-9821-cf2e71cf38f2
md"# DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is another popular clustering method.
It captures the insight that clusters are dense groups of points.
The idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.
The following illustration is based on a nice blog post by [Naftali Harris](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/).

Change the seed and the different parameters of the methods to investigate their sensitivity!
"

# ╔═╡ 9ca4cac1-f378-42cd-ba60-d174a47e23a8
md"""Seed of random number generator $(@bind seed Slider(collect(1:50), show_value = true)).

Cluster assignment with $(@bind method Select(["DBSCAN", "k-means", "hierarchical clustering"]))"""

# ╔═╡ 88cd7a57-5d9d-45ae-968e-b3b5a8d3a824
mlcode("""
nothing""",
"""
# here is the python code to run DBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
X = np.array([[1, 2], [2, 2], [2, 3],
               [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
"""
)

# ╔═╡ 9d54fbb8-44f8-46c8-90ef-de85746c410b
function smiley_data(; n = 400, pimples = n÷10,
                       a = 6, b = 8, d = 4, c = .15, C = (c - a/b^2))
    i = 1
    res = 20 * rand(n + pimples, 2) .- 10
    while i ≤ n
        x = 20*rand() - 10
		y = 20*rand() - 10
        if 81 < x^2 + y^2 < 100 ||
           (x + 4)^2 + (y - 4)^2 < 1 ||
           (x - 4)^2 + (y - 4)^2 < 1 ||
           (-b < x < b && y > c*x^2 - a && y < C* x^2 - d)
            res[i, 1] = x
            res[i, 2] = y
            i += 1
        end
    end
    res
end;

# ╔═╡ 6d845685-ac31-4df7-9d18-f1fab6c08e3d
begin
	Random.seed!(seed) # the seed is defined by the slider below
	smiley = DataFrame(smiley_data(), :auto)
	smiley_kmeans_pred = predict(fit!(machine(KMeans(k = 4), smiley), verbosity = 0),
		                         smiley)
	smiley_hclust_pred = predict(machine(HierarchicalClustering(k = 4)), smiley)
	smiley_dbscan_pred = predict(machine(DBSCAN(min_cluster_size = 10, radius = 1)), smiley)
end;

# ╔═╡ 8ea10eb7-8b37-4026-a7ec-e44bba7532ea
let d = smiley
	scatter(d.x1, d.x2, aspect_ratio = 1,
            color = Int.(int.(method == "k-means" ? smiley_kmeans_pred :
                              method == "hierarchical clustering" ? smiley_hclust_pred :
                              smiley_dbscan_pred)),
            ylabel = "X₂", xlabel = "X₁", label = nothing)
end

# ╔═╡ 85d574c2-b823-4dcf-b711-efc755e724b7
md"# Exercises
## Conceptual

#### Exercise 1
In this exercise, you will perform clustering manually, with ``K = 2``, on a small example with ``n = 6`` observations and ``p = 2`` features. The observations are as follows.

 ````\begin{array}{c|cc}
 \hline
 \mbox{Obs.} &X_1 &X_2 \cr
 \hline
 1 &1 &4 \cr
 2 &1 &3 \cr
 3 &0 &4 \cr
 4 &5 &1 \cr
 5 &6 &2 \cr
 6 &4 &0 \cr
 \hline
 \end{array}````

(a) Plot the observations.

(b) Randomly assign a cluster label to each observation. You can use `rand(1:2, 6)` to do this, if you do not want to do this manually. Report the cluster labels for each observation.

(c) Compute the centroid for each cluster.

(d) Assign each observation to the centroid to which it is closest, in terms of Euclidean distance. Report the cluster labels for each observation.

(e) Repeat (c) and (d) until the answers obtained stop changing.

(f) In your plot from (a), color the observations according to the clusters labels obtained.

(g) Apply hierarchical clustering with complete linkage and Euclidean metric to the same data set.  Draw the dendrogram with the links at the correct heights.

## Applied

###
# Exercise 2
In this exercise, you will generate simulated data and perform K-means clustering on the data.

(a) Generate a simulated data set with 20 observations in each of three classes (i.e. 60 observations total), and 50 variables with the following generator
"

# ╔═╡ 272f3f11-c753-4cee-93e0-ec990d6c2660
mlcode(
"""
function data_generator(d; n = 20)
    cluster1 = rand(MvNormal([0; -d; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster2 = rand(MvNormal([0; 0; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster3 = rand(MvNormal([0; d; fill(0, 48)], [3; fill(.5, 49)]), n)
    (data = DataFrame(vcat(cluster1', cluster2', cluster3'), :auto),
     true_labels = [fill(1, n); fill(2, n); fill(3, n)])
end
""",
"""
def data_generator(d, n=20):
    cluster1 = np.random.multivariate_normal([0,-d]+[0]*48, np.tile(([3] +[0.5]*49),(50,1)), n)
    cluster2 = np.random.multivariate_normal([0, 0] + [0]*48, np.tile(([3] +[0.5]*49),(50,1)), n)
    cluster3 = np.random.multivariate_normal([0, d] + [0]*48, np.tile(([3] +[0.5]*49),(50,1)), n)
    data = pd.DataFrame(np.vstack((cluster1, cluster2, cluster3)), columns=['x'+str(i) for i in range(50)])
    true_labels = np.hstack((np.ones(n), 2*np.ones(n), 3*np.ones(n)))
    return data, true_labels
""")

# ╔═╡ 7a29009c-3020-44aa-89e1-a8712c75d567
md"(b) PCA is an unsupervised machine learning method (you will learn more about it next week) that allows to visualize high-dimensional data in lower dimensions. You can use the command"

# ╔═╡ bfb97a80-4bb8-423f-b936-ecc3456a4327
mlcode("""
using MultivariateStats
MLJ.transform(fit!(machine(PCA(maxoutdim = 2), data)), data)
""",
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
""";
eval = false
)

# ╔═╡ 884e4f0e-e53f-4f26-b285-6d30d1d68fc3
md"to fit an unsupervised machine called `PCA` (with at most 2 output dimensions) to the data and transform the data according to this machine to 2 dimensions. Plot the result. Use a different color to indicate the observations in each of the three classes.

(c) Perform K-means clustering of the observations with ``K = 3``. Repeat K-means clustering for multiple random initializations and keep the best result. *Hint:* for reproducibility you can use `Random.seed!(SEED)`, where SEED is positive integer. How well do the clusters that you obtained in K-means clustering compare to the true class labels? *Hint:* a `confusion_matrix` may be helpful to comare the result to the true labels.


(d) Perform K-means clustering with ``K = 2``. Describe your results.

(e) Now perform K-means clustering with ``K = 4``, and describe your results.

(f) Now perform K-means clustering with ``K = 3`` on the 2-dimensional data you obtained with PCA, rather than on the raw data. Comment on the results.

(g) Perform K-means clustering with ``K = 3`` on the data after scaling each variable to have standard deviation one. How do these results compare to those obtained in (c)? Explain.

(h) *Optional:* Create other artificial data sets where scaling does not affect the result of clustering or where it improves the result of clustering.

#### Exercise 3

On the website of the text book, there is a gene expression data set that consists of 40 tissue samples with measurements on 1000 genes. Some samples are from healthy patients, while others are from a diseased group.

(a) Download the data set with"

# ╔═╡ b6ed3729-06d5-4d3f-ae20-ba7079a7f7b0
mlcode("""
using CSV
CSV.read(download(\"https://www.statlearning.com/s/Ch12Ex13.csv\"),
         DataFrame, header = false, transpose = true)
""",
"""
pd.read_csv(\"https://www.statlearning.com/s/Ch12Ex13.csv\")
""";
eval = false
)

# ╔═╡ 482e94c0-cff8-4082-9b95-38402258398a
md"
(b) Apply hierarchical clustering and plot the result.

(c) Do the results change qualitatively with different types of linkage?

(d) How many patients do you think are in each group?
"

# ╔═╡ 15830699-57c5-4bc2-bc92-54105597ab26
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 35ac2056-ab72-44b0-9972-723a0887a622
MLCourse.FOOTER

# ╔═╡ Cell order:
# ╟─48d87103-4c23-4144-a121-1e33d2bb87f3
# ╟─da8f8ce7-c3ad-4e1a-bb9e-c7be15646d72
# ╟─49a1493b-ba68-4e7a-8c3d-47423f7a8204
# ╟─67b8f27f-b5b5-4dc0-ab51-83ac3dcaa2f7
# ╟─1ebd74bc-4479-41d2-aba4-934cdd2b778a
# ╟─4dfeee20-ac92-48d2-9ba1-40baea227100
# ╠═e187d7a8-4ccd-4b0a-8c71-f79219623142
# ╠═99cd7b6b-1d1f-4712-a83e-7969ae7fecd4
# ╟─d1c88a44-be52-4b0e-bc23-cca00d10ffb6
# ╟─4df24b22-47a7-446a-8e27-9f2031a75764
# ╟─0e7f34b9-e24f-447b-840b-e2750d2e778b
# ╟─70b3f1bb-7c47-4bb0-aa17-cda6fdbe0469
# ╟─58220aaf-9f85-4de7-82fa-dec8bc9842fd
# ╠═8e71fb7a-b83f-4ff1-b78b-4814dd6d6a6c
# ╟─a0af5510-8585-4c38-9166-5284d3b2923e
# ╟─6310f2b4-ea1b-48d1-a23d-b5a43291772f
# ╟─46816b6c-490d-426b-befd-735601aa3d10
# ╟─5dad8e78-2862-47cc-b693-806b307733e1
# ╟─12317841-c1f8-4022-970e-8ef613c79b78
# ╟─872e95fd-fac0-4c39-bc65-e4cdcf0050bb
# ╟─f4d74b33-e05c-445a-9d59-c91dfff97eab
# ╟─2f4e3d67-d88e-4f9b-9572-6be1bc30106b
# ╟─c4bde04d-23e8-4fb0-9e66-88b443a12926
# ╠═84d2b372-1c4a-4cae-958c-a7a7b19f24f6
# ╠═41e0f1e0-19a7-441e-a6ea-be0fe4678288
# ╟─23958c06-c037-452d-ae15-36d4d8cc9038
# ╠═6cc7a2a0-414f-499d-9f45-f525b0a5c1cc
# ╠═f6b3f577-d893-4c7f-a0c5-df5a267f2d1e
# ╠═e266ada3-ba4d-4177-8129-f1220f293c72
# ╟─675e3a37-8044-4e8a-9821-cf2e71cf38f2
# ╠═6d845685-ac31-4df7-9d18-f1fab6c08e3d
# ╟─9ca4cac1-f378-42cd-ba60-d174a47e23a8
# ╟─8ea10eb7-8b37-4026-a7ec-e44bba7532ea
# ╟─88cd7a57-5d9d-45ae-968e-b3b5a8d3a824
# ╟─9d54fbb8-44f8-46c8-90ef-de85746c410b
# ╟─85d574c2-b823-4dcf-b711-efc755e724b7
# ╟─272f3f11-c753-4cee-93e0-ec990d6c2660
# ╟─7a29009c-3020-44aa-89e1-a8712c75d567
# ╟─bfb97a80-4bb8-423f-b936-ecc3456a4327
# ╟─884e4f0e-e53f-4f26-b285-6d30d1d68fc3
# ╟─b6ed3729-06d5-4d3f-ae20-ba7079a7f7b0
# ╟─482e94c0-cff8-4082-9b95-38402258398a
# ╟─15830699-57c5-4bc2-bc92-54105597ab26
# ╟─7b013132-0ee2-11ec-1dd2-25a9f16f0568
# ╟─35ac2056-ab72-44b0-9972-723a0887a622
