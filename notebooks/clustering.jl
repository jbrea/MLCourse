### A Pluto.jl notebook ###
# v0.17.1

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
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML, Plots,
          LinearAlgebra, Statistics, Random, MLJClusteringInterface, StatsPlots,
          Distributions
end


# ╔═╡ 7b013132-0ee2-11ec-1dd2-25a9f16f0568
begin
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ da8f8ce7-c3ad-4e1a-bb9e-c7be15646d72
md"# K-means Clustering"

# ╔═╡ 49a1493b-ba68-4e7a-8c3d-47423f7a8204
begin
    iris = DataFrame(OpenML.load(61))
    iris = rename!(iris, replace.(names(iris), "'" => ""))
    coerce!(iris, :class => Multiclass)
end

# ╔═╡ 1ebd74bc-4479-41d2-aba4-934cdd2b778a
@df iris cornerplot([:sepallength :sepalwidth :petallength :petalwidth],
                    compact = true)

# ╔═╡ 472f0d57-cf50-4c3d-b251-dcf2f7c121c2
md"Number of clusters $(@bind nclust Slider(2:8, show_value = true))"

# ╔═╡ d1c88a44-be52-4b0e-bc23-cca00d10ffb6
mach1 = fit!(machine(KMeans(k = nclust), select(iris, Not(:class)))); # nclust is defined by the slider value above

# ╔═╡ 4df24b22-47a7-446a-8e27-9f2031a75764
prediction = MLJ.predict(mach1, select(iris, Not(:class)));

# ╔═╡ 0e7f34b9-e24f-447b-840b-e2750d2e778b
let
    p1 = scatter(iris.petallength, iris.petalwidth,
                 legend = false, c = Int.(int(prediction)),
                 title = "prediction", xlabel = "petal length", ylabel = "petal width")
    p2 = scatter(iris.petallength, iris.petalwidth,
                 legend = false, c = Int.(int(iris.class)),
                 title = "ground truth", xlabel = "petal length", ylabel = "petal width")
    plot(p1, p2, layout = (1, 2), size = (700, 500))
end

# ╔═╡ 70b3f1bb-7c47-4bb0-aa17-cda6fdbe0469
md"## KMeans Clustering Applied to MNIST"

# ╔═╡ 8b458fe0-8de4-46a1-b14b-1ef430a3c0f3
Markdown.parse("
Let us try to apply K-Means Clustering to the MNIST data set.

```julia
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    select(df, Not(:class)),
    df.class
end

mnist_mach = fit!(machine(KMeans(k = 10), mnist_x));

mnist_pred = predict(mnist_mach)

let
    idxs = [findall(==(k), mnist_pred)[1:15] for k in 1:10]
    img = vcat([[hcat([[reshape(Array(mnist_x[idx, :]), 28, :)' fill(255, 28)]
                       for idx in row]...); fill(255, 15*29)']
                for row in idxs]...)./255
    plot(Gray.(img), showaxis = false, ticks = false,
         xlabel = \"examples\", ylabel = \"cluster 1 to 10\")
end
```
$(MLCourse.embed_figure("mnist_kmeans.png"))

Some clusters (rows in the figure above) seem indeed to have specialised on some
digit classes. This can further be seen by looking at the confusion_matrix.

```julia
confusion_matrix(mnist_pred, mnist_y) |> x -> DataFrame(x.mat, x.labels)
```

$(MLCourse.embed_figure("mnist_kmeans_table.png"))

We see that the ninth cluster (9th row in this table and in the figure above) has indeed specialized on images with class label \"0\" and the eigth cluster has specialized on images with class label \"2\", but e.g. cluster 5 contains similar amounts of 0s, 5s and 6s.
")

# ╔═╡ 12317841-c1f8-4022-970e-8ef613c79b78
md"# Hierarchical Clustering"

# ╔═╡ 20297924-5e09-40d2-9f7e-27e0e1b8a968
mach2 = fit!(machine(HierarchicalClustering(linkage = :complete,
                                            metric = MLJClusteringInterface.Euclidean()),
                     select(iris, Not(:class))));

# ╔═╡ 6a603cb6-7a1b-4a1d-9358-c4c811efa2bb
md"h = $(@bind hclusth Slider(range(minimum(mach2.fitresult.height), stop = maximum(mach2.fitresult.height), length = 100), default = maximum(mach2.fitresult.height)/2))"

# ╔═╡ e266ada3-ba4d-4177-8129-f1220f293c72
let
    mach2.model.k = nothing
    mach2.model.h = hclusth
    pred = predict(mach2, select(iris, Not(:class)))
    p1 = scatter(iris.petallength, iris.petalwidth,
                 legend = false, c = pred,
                 title = "prediction", xlabel = "petal length",
                 ylabel = "petal width")
    p2 = plot(report(mach2))
    hline!([mach2.model.h], c = :red, w = 3)
    plot(p2, p1, layout = (1, 2), size = (700, 500))
end


# ╔═╡ 85d574c2-b823-4dcf-b711-efc755e724b7
md"# Exercises
## Conceptual

**Exercise 1.** In this exercise, you will perform clustering manually, with ``K = 2``, on a small example with ``n = 6`` observations and ``p = 2`` features. The observations are as follows.

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

**Exercise 1.** In this exercise, you will generate simulated data and perform K-means clustering on the data.

(a) Generate a simulated data set with 20 observations in each of three classes (i.e. 60 observations total), and 50 variables with the following generator
```julia
function data_generator(d; n = 20)
    cluster1 = rand(MvNormal([0; -d; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster2 = rand(MvNormal([0; 0; fill(0, 48)], [3; fill(.5, 49)]), n)
    cluster3 = rand(MvNormal([0; d; fill(0, 48)], [3; fill(.5, 49)]), n)
    (data = DataFrame(vcat(cluster1', cluster2', cluster3'), :auto),
     true_labels = [fill(1, n); fill(2, n); fill(3, n)])
end
```

(b) PCA is an unsupervised machine learning method (you will learn more about it next week) that allows to visualize high-dimensional data in lower dimensions. You can use the command
```julia
MLJ.transform(fit!(machine(PCA(maxoutdim = 2), data)), data)
```
to fit an unsupervised machine called `PCA` (with at most 2 output dimensions) to the data and transform the data according to this machine to 2 dimensions. Plot the result. Use a different color to indicate the observations in each of the three classes.

(c) Perform K-means clustering of the observations with ``K = 3``. Repeat K-means clustering for multiple random initializations and keep the best result. *Hint:* for reproducibility you can use `Random.seed!(SEED)`, where SEED is positive integer. How well do the clusters that you obtained in K-means clustering compare to the true class labels? *Hint:* a `confusion_matrix` may be helpful to comare the result to the true labels.


(d) Perform K-means clustering with ``K = 2``. Describe your results.

(e) Now perform K-means clustering with ``K = 4``, and describe your results.

(f) Now perform K-means clustering with ``K = 3`` on the 2-dimensional data you obtained with PCA, rather than on the raw data. Comment on the results.

(g) Perform K-means clustering with ``K = 3`` on the data after scaling each variable to have standard deviation one. How do these results compare to those obtained in (c)? Explain.

(h) *Optional:* Create other artificial data sets where scaling does not affect the result of clustering or where it improves the result of clustering.

**Exercise 2**

On the website of the text book, there is a gene expression data set that consists of 40 tissue samples with measurements on 1000 genes. Some samples are from healthy patients, while others are from a diseased group.

(a) Download the data set with
```julia
CSV.read(download(\"https://www.statlearning.com/s/Ch12Ex13.csv\"),
         DataFrame, header = false, transpose = true)
```

(b) Apply hierarchical clustering and plot the result.

(c) Do the results change qualitatively with different types of linkage?

(d) How many patients do you think are in each group?

"

# ╔═╡ 15830699-57c5-4bc2-bc92-54105597ab26
MLCourse.list_notebooks(@__FILE__)

# ╔═╡ 35ac2056-ab72-44b0-9972-723a0887a622
MLCourse.footer()

# ╔═╡ Cell order:
# ╠═48d87103-4c23-4144-a121-1e33d2bb87f3
# ╟─da8f8ce7-c3ad-4e1a-bb9e-c7be15646d72
# ╠═49a1493b-ba68-4e7a-8c3d-47423f7a8204
# ╟─1ebd74bc-4479-41d2-aba4-934cdd2b778a
# ╟─472f0d57-cf50-4c3d-b251-dcf2f7c121c2
# ╠═d1c88a44-be52-4b0e-bc23-cca00d10ffb6
# ╠═4df24b22-47a7-446a-8e27-9f2031a75764
# ╟─0e7f34b9-e24f-447b-840b-e2750d2e778b
# ╟─70b3f1bb-7c47-4bb0-aa17-cda6fdbe0469
# ╟─8b458fe0-8de4-46a1-b14b-1ef430a3c0f3
# ╟─12317841-c1f8-4022-970e-8ef613c79b78
# ╠═20297924-5e09-40d2-9f7e-27e0e1b8a968
# ╟─6a603cb6-7a1b-4a1d-9358-c4c811efa2bb
# ╟─e266ada3-ba4d-4177-8129-f1220f293c72
# ╟─85d574c2-b823-4dcf-b711-efc755e724b7
# ╟─15830699-57c5-4bc2-bc92-54105597ab26
# ╟─7b013132-0ee2-11ec-1dd2-25a9f16f0568
# ╟─35ac2056-ab72-44b0-9972-723a0887a622
