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
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, MLJMultivariateStatsInterface, OpenML, MLJClusteringInterface, Distributions, Distances
import PlutoPlotly as PP
const M = MLCourse.JlMod
MLCourse.load_cache(@__FILE__)
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

# ╔═╡ eb6a77fa-fc7e-4546-8748-2438e9a519b8
md"The goal of this week is to
1. Understand and learn how to use K-means clustering.
2. Understand and learn how to use Hierarchical clustering.
3. See a comparison of K-means clustering, hierarchical clustering and DBSCAN.
"

# ╔═╡ da8f8ce7-c3ad-4e1a-bb9e-c7be15646d72
md"# 1. K-means Clustering

The [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is an old and famous dataset containing measurements on 150 iris flowers of three different species.
For each flower, the length and width of sepal and petal were measured.
Although the species is known for each flower, we explore here, if clustering methods can discover three clusters that correspond to the three flower species.
"

# ╔═╡ 49a1493b-ba68-4e7a-8c3d-47423f7a8204
mlcode(
"""
using OpenML, DataFrames, MLJ

iris = DataFrame(OpenML.load(61))
iris = rename!(iris, replace.(names(iris), "'" => ""))
coerce!(iris, :class => Multiclass)
iris
"""
,
"""
"""
,
cache = false,
collapse = "Loading Iris Dataset"
)

# ╔═╡ 2fd14aae-17c2-4791-8241-9602e3e0069c
MLCourse.JlMod.eval(:(using StatsPlots)) # hack for macro below

# ╔═╡ 1ebd74bc-4479-41d2-aba4-934cdd2b778a
mlcode(
"""
using StatsPlots

@df iris corrplot([:sepallength :sepalwidth :petallength :petalwidth],
                    compact = true, fillcolor=cgrad())
"""
,
"""
"""
,
showinput = false
)

# ╔═╡ 472f0d57-cf50-4c3d-b251-dcf2f7c121c2
md"
If you want to dive a bit deeper into some quantitative approaches that are used to select the number of clusters please have a look at [this wikipedia article](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) or this [opinionated tutorial](https://towardsdatascience.com/silhouette-method-better-than-elbow-method-to-find-optimal-clusters-378d62ff6891).

Number of clusters $(@bind nclust Slider(2:8, show_value = true))
"

# ╔═╡ d1c88a44-be52-4b0e-bc23-cca00d10ffb6
begin
    # nclust is defined by the slider value above
    mach1 = machine(KMeans(k = nclust), select(MLCourse.JlMod.iris, Not(:class)))
    fit!(mach1, verbosity = 0)
    prediction = MLJ.predict(mach1, select(MLCourse.JlMod.iris, Not(:class)))
end;

# ╔═╡ 0e7f34b9-e24f-447b-840b-e2750d2e778b
let iris = MLCourse.JlMod.iris
    p1 = scatter(iris.petallength, iris.petalwidth,
                 legend = false, c = Int.(int(prediction)),
                 title = "prediction", xlabel = "petal length", ylabel = "petal width")
    p2 = scatter(iris.petallength, iris.petalwidth,
                 legend = false, c = Int.(int(iris.class)),
                 title = "ground truth", xlabel = "petal length", ylabel = "petal width")
    plot(p1, p2, layout = (1, 2), size = (700, 500))
end

# ╔═╡ 70b3f1bb-7c47-4bb0-aa17-cda6fdbe0469
md"## KMeans Clustering Applied to MNIST

Let us apply K-Means Clustering to the MNIST data set.
"

# ╔═╡ 260a1fb7-58b8-4d83-b8d7-a0bd8e6836ac
mlcode(
"""
using OpenML, DataFrames, MLJ, MLJClusteringInterface, Plots, Random

mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    select(df, Not(:class)),
    df.class
end

Random.seed!(213)

mnist_mach = machine(KMeans(k = 10), mnist_x)
fit!(mnist_mach, verbosity = 0)

mnist_pred = predict(mnist_mach)

idxs = [findall(==(k), mnist_pred)[1:15] for k in 1:10]
img = vcat([[hcat([[reshape(Array(mnist_x[idx, :]), 28, :)' fill(255, 28)]
                   for idx in row]...); fill(255, 15*29)']
            for row in idxs]...)./255
plot(Gray.(img), showaxis = false, ticks = false, xrange = (0, 430),
     xlabel = "examples", ylabel = "cluster 1 to 10", size = (700, 400))
"""
,
"""
"""
)


# ╔═╡ b5165fda-1bdd-4837-9eed-42ce9db40529
md"
Some clusters (rows in the figure above) seem indeed to have specialised on some
digit classes. This can further be seen by looking at the confusion_matrix.
"

# ╔═╡ fdf190cc-0426-4327-a710-80fe2ead632c
mlcode(
"""
confusion_matrix(mnist_pred, mnist_y) |> x -> DataFrame(x.mat, x.labels)
"""
,
"""
"""
,
recompute = false,
showinput = false
)

# ╔═╡ ab161204-bbcd-4608-ae67-fcde39b2539b
md"
We see that the 8th cluster (8th row in this table and in the figure above) has indeed specialized on images with class label \"0\" and the 4th cluster has specialized on images with class label \"2\", but other clusters contain mixtures of different digits.

"

# ╔═╡ 12317841-c1f8-4022-970e-8ef613c79b78
md"# 2. Hierarchical Clustering

In the following we run hierarchical clustering to find 3 clusters in the iris dataset.
"

# ╔═╡ 872e95fd-fac0-4c39-bc65-e4cdcf0050bb
mlcode(
"""
using MLJClusteringInterface, Distances

hc = machine(HierarchicalClustering(k = 3,
                                    linkage = :complete,
                                    metric = Euclidean()))
predict(hc, select(iris, Not(:class)))
"""
,
"""
"""
,
cache_jl_vars = [:hc]
)

# ╔═╡ 2f4e3d67-d88e-4f9b-9572-6be1bc30106b
md"More interesting than mere predictions is, however, the dendrogram of hierarchical clustering. By changing the height at which the dendrogram is cut, we can get different predictions."

# ╔═╡ 6a603cb6-7a1b-4a1d-9358-c4c811efa2bb
md"h = $(@bind hclusth Slider(range(minimum(report(M.hc).dendrogram.heights), stop = maximum(report(M.hc).dendrogram.heights), length = 100), default = maximum(report(M.hc).dendrogram.heights)/2))"

# ╔═╡ e266ada3-ba4d-4177-8129-f1220f293c72
let rep = report(M.hc)
	pred = rep.cutter(h = hclusth, k = nothing)
    p1 = scatter(M.iris.petallength, M.iris.petalwidth,
                 legend = false, c = Int.(int(pred)),
                 title = "prediction", xlabel = "petal length",
                 ylabel = "petal width")
    p2 = plot(rep.dendrogram)
    hline!([hclusth], c = :red, w = 3)
    plot(p2, p1, layout = (1, 2), size = (700, 500))
end


# ╔═╡ c4bde04d-23e8-4fb0-9e66-88b443a12926
mlcode(
"""
hc_report = report(hc)
cutter = hc_report.cutter
cutter(h = 2, k = nothing) # get predictions at different cutting heights
dendrogram = hc_report.dendrogram # get the dendrogram
plot(dendrogram)
"""
,
"""
"""
,
showoutput = false,
collapse = "How to show and cut a dendrogram"
)


# ╔═╡ 675e3a37-8044-4e8a-9821-cf2e71cf38f2
md"# 3. DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is another popular clustering method.
It captures the insight that clusters are dense groups of points.
The idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.
The following illustration is based on a nice blog post by [Naftali Harris](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/).

Change the seed and the different parameters of the methods to investigate their sensitivity!
"

# ╔═╡ 9ca4cac1-f378-42cd-ba60-d174a47e23a8
md"""Seed of random number generator $(@bind seed Slider(collect(1:50), show_value = true)).

Cluster assignment with $(@bind method Select(["DBSCAN", "k-means", "hierarchical clustering"]))"""

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

"

# ╔═╡ 1ed55c9f-1301-4553-84ee-26ada25f9b76
md"""
## Applied

#### Exercise 2
In this exercise, you will generate simulated data and perform K-means clustering on the data.

(a) Generate a simulated data set with 20 observations in each of three classes (i.e. 60 observations total), and 50 variables with the following generator
"""

# ╔═╡ d6adf823-be4e-4dd3-87e0-867fbd8aa7a9
mlcode(
"""
function data_generator(d; n = 20)
    cluster1 = rand(MvNormal([0; -d; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster2 = rand(MvNormal([0; 0; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster3 = rand(MvNormal([0; d; fill(0, 48)], [3; fill(.5, 49)]), n)
    (data = DataFrame(vcat(cluster1', cluster2', cluster3'), :auto),
     true_labels = [fill(1, n); fill(2, n); fill(3, n)])
end
"""
,
"""
"""
)


# ╔═╡ add72e27-5e6e-4112-b1bf-9f579a845384
md"""
(b) PCA is an unsupervised machine learning method (you will learn more about it next week) that allows to visualize high-dimensional data in lower dimensions. You can use the command
$(mlstring(md"`MLJ.transform(fit!(machine(PCA(maxoutdim = 2), data)), data)`", ""))
to fit an unsupervised machine (with at most 2 output dimensions) to the data and transform the data according to this machine to 2 dimensions. Plot the result. Use a different color to indicate the observations in each of the three classes.

(c) Perform K-means clustering of the observations with ``K = 3``. Repeat K-means clustering for multiple random initializations and keep the best result. *Hint:* for reproducibility you can use `Random.seed!(SEED)`, where SEED is positive integer. How well do the clusters that you obtained in K-means clustering compare to the true class labels? *Hint:* a `confusion_matrix` may be helpful to comare the result to the true labels.


(d) Perform K-means clustering with ``K = 2``. Describe your results.

(e) Now perform K-means clustering with ``K = 4``, and describe your results.

(f) Now perform K-means clustering with ``K = 3`` on the 2-dimensional data you obtained with PCA, rather than on the raw data. Comment on the results.

(g) Perform K-means clustering with ``K = 3`` on the data after scaling each variable to have standard deviation one. How do these results compare to those obtained in (c)? Explain.

(h) *Optional:* Create other artificial data sets where scaling does not affect the result of clustering or where it improves the result of clustering.

"""

# ╔═╡ 9bf65d2c-b8ab-4f84-9774-8d7c6a3fc6f0
md"""
#### Exercise 3

On the website of the text book, there is a gene expression data set that consists of 40 tissue samples with measurements on 1000 genes. Some samples are from healthy patients, while others are from a diseased group.

(a) Download the data set with
"""

# ╔═╡ 98a3326b-f12a-4309-bfaa-ad06216d62c9
mlcode(
"""
CSV.read(download(\"https://www.statlearning.com/s/Ch12Ex13.csv\"),
         DataFrame, header = false, transpose = true)
"""
,
"""
"""
,
eval = false
)


# ╔═╡ 9cce2cda-a8a7-4b07-a9a1-178588284435
md"""
(b) Apply hierarchical clustering and plot the result.

(c) Do the results change qualitatively with different types of linkage?

(d) How many patients do you think are in each group?
"""

# ╔═╡ 15830699-57c5-4bc2-bc92-54105597ab26
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 35ac2056-ab72-44b0-9972-723a0887a622
MLCourse.FOOTER

# ╔═╡ b24d7fbf-bd2b-4bd7-90ef-c77ec9099bc8
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─eb6a77fa-fc7e-4546-8748-2438e9a519b8
# ╟─da8f8ce7-c3ad-4e1a-bb9e-c7be15646d72
# ╟─49a1493b-ba68-4e7a-8c3d-47423f7a8204
# ╟─2fd14aae-17c2-4791-8241-9602e3e0069c
# ╟─1ebd74bc-4479-41d2-aba4-934cdd2b778a
# ╟─472f0d57-cf50-4c3d-b251-dcf2f7c121c2
# ╟─d1c88a44-be52-4b0e-bc23-cca00d10ffb6
# ╟─0e7f34b9-e24f-447b-840b-e2750d2e778b
# ╟─70b3f1bb-7c47-4bb0-aa17-cda6fdbe0469
# ╟─260a1fb7-58b8-4d83-b8d7-a0bd8e6836ac
# ╟─b5165fda-1bdd-4837-9eed-42ce9db40529
# ╟─fdf190cc-0426-4327-a710-80fe2ead632c
# ╟─ab161204-bbcd-4608-ae67-fcde39b2539b
# ╟─12317841-c1f8-4022-970e-8ef613c79b78
# ╟─872e95fd-fac0-4c39-bc65-e4cdcf0050bb
# ╟─2f4e3d67-d88e-4f9b-9572-6be1bc30106b
# ╟─6a603cb6-7a1b-4a1d-9358-c4c811efa2bb
# ╟─e266ada3-ba4d-4177-8129-f1220f293c72
# ╟─c4bde04d-23e8-4fb0-9e66-88b443a12926
# ╟─675e3a37-8044-4e8a-9821-cf2e71cf38f2
# ╟─6d845685-ac31-4df7-9d18-f1fab6c08e3d
# ╟─9ca4cac1-f378-42cd-ba60-d174a47e23a8
# ╟─8ea10eb7-8b37-4026-a7ec-e44bba7532ea
# ╟─9d54fbb8-44f8-46c8-90ef-de85746c410b
# ╟─85d574c2-b823-4dcf-b711-efc755e724b7
# ╟─1ed55c9f-1301-4553-84ee-26ada25f9b76
# ╟─d6adf823-be4e-4dd3-87e0-867fbd8aa7a9
# ╟─add72e27-5e6e-4112-b1bf-9f579a845384
# ╟─9bf65d2c-b8ab-4f84-9774-8d7c6a3fc6f0
# ╟─98a3326b-f12a-4309-bfaa-ad06216d62c9
# ╟─9cce2cda-a8a7-4b07-a9a1-178588284435
# ╟─15830699-57c5-4bc2-bc92-54105597ab26
# ╟─7b013132-0ee2-11ec-1dd2-25a9f16f0568
# ╟─35ac2056-ab72-44b0-9972-723a0887a622
# ╟─48d87103-4c23-4144-a121-1e33d2bb87f3
# ╟─b24d7fbf-bd2b-4bd7-90ef-c77ec9099bc8
