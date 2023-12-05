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

# ╔═╡ 8bd459cb-20bb-483e-a849-e18caae3beef
begin
using Pkg
stdout_orig = stdout
stderr_orig = stderr
redirect_stdio(stdout = devnull, stderr = devnull)
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse, HypertextLiteral, Plots, Random, MLJ, MLJLinearModels, DataFrames, LinearAlgebra, MLJMultivariateStatsInterface, OpenML, MLJClusteringInterface, Distributions, Distances
import PlutoPlotly as PP
const M = MLCourse.JlMod
M.chop(x, eps = 1e-12) = abs(x) > eps ? x : zero(typeof(x))
redirect_stdio(stdout = stdout_orig, stderr = stderr_orig)
MLCourse.load_cache(@__FILE__)
MLCourse.CSS_STYLE
end

# ╔═╡ b97724e4-d7b0-4085-b88e-eb3c5bcbe441
begin
    using PlutoUI
    function MLCourse.embed_figure(name)
        "![](data:img/png; base64,
             $(open(MLCourse.base64encode,
                    joinpath(Pkg.devdir(), "MLCourse", "notebooks", "figures", name))))"
    end
    PlutoUI.TableOfContents()
end

# ╔═╡ a7ba035f-7433-46f3-a64f-8bf3d6c6cbad
md"The goal of this week is to
1. Understand PCA from different perspectives.
2. Learn how to run PCA.
3. Understand the limitations of PCA and learn how to use alternatives like T-SNE or UMAP
4. Learn how to use PCA for principal component regression.
"

# ╔═╡ 78d7653f-1139-4833-b65c-43adb4fca9b1
md"# 1. Principal Component Analysis
## For Dimensionality Reduction"

# ╔═╡ 1bb3d957-06dc-4be4-bed7-d14da43c25ee
md"If all the 3-dimensional data are parallel to the X₁-X₂-plane we can simply take the first and the second coordinate of each point to obtain a 2-dimensional representation of the data."

# ╔═╡ d40fa053-39ed-4d36-ad4c-54356d0146cf
let
    Random.seed!(1247)
    x = randn(50);
    y = randn(50);
    z = fill(2, 50);
    c = ["red"; "blue"; "green"; fill("black", 47)]
    p1 = PP.Plot(PP.scatter3d(; x, y, z,
                              mode = "markers",
                              marker_color = c,
                              marker_size = 3))
    p2 = PP.Plot(PP.scatter(; x = x, y = y, marker_color = c,
                            mode = "markers"))
    l = PP.Layout(yaxis2_scaleanchor = "x",
                  xaxis2_title = "X₁", yaxis2_title = "X₂",
                  scene1 = PP.attr(xaxis_title = "X₁",
                                   yaxis_title_text = "X₂",
                                   zaxis_title = "X₃",
                                   camera_eye = PP.attr(x = 2.8, y = -1, z = 1)),
                  uirevision = 1, showlegend = false)
    plot = [p1 p2]
    PP.relayout!(plot, l)
    PP.PlutoPlot(plot)
end

# ╔═╡ 029c165b-8d34-42e3-9b68-4d97cfad074e
md"If all the 3-dimensional data are parallel to the X₁-X₃-plane we can take the first and the *third* coordinate of each point to obtain a 2-dimensional representation of the data. Note that it would not make much sense to take the first and the *second* coordinate in this example, because the data points just lie on a line in the X₁-X₂-plane."

# ╔═╡ 4f960f96-4218-44db-801c-a67222848062
let
    Random.seed!(2468)
    x = randn(50);
    y = randn(50);
    z = fill(2, 50);
    c = ["red"; "blue"; "green"; fill("black", 47)]
    p1 = PP.Plot(PP.scatter3d(; x, y = z, z = y,
                              mode = "markers",
                              marker_color = c,
                              marker_size = 3))
    p2 = PP.Plot(PP.scatter(; x = x, y = y, marker_color = c,
                            mode = "markers"))
    l = PP.Layout(yaxis2_scaleanchor = "x",
                  xaxis2_title = "X₁", yaxis2_title = "X₃",
                  scene1 = PP.attr(xaxis_title = "X₁",
                                   yaxis_title_text = "X₂",
                                   zaxis_title = "X₃",
                                   camera_eye = PP.attr(x = 2.8, y = -1, z = 1)),
                  uirevision = 1, showlegend = false)
    plot = [p1 p2]
    PP.relayout!(plot, l)
    PP.PlutoPlot(plot)
end


# ╔═╡ 59654b1f-011a-4f2d-8665-d2c885b76553
md"If all the 3-dimensional data lie in an arbitrarily oriented plane we can rotate the coordinate system first, and then take the first and the second coordinate of the roated points to obtain a 2-dimensional representation of the data."


# ╔═╡ a935143d-52f8-4a5e-bede-58467065caed
let
    Random.seed!(2468)
    x = randn(50);
    y = randn(50);
    z = fill(2, 50);
    c = ["red"; "blue"; "green"; fill("black", 47)]
    p1 = PP.Plot(PP.scatter3d(; x = .2x+.4y-.3z, y = .3x+.2z, z = -.4x+.4y-.2z,
                              mode = "markers",
                              marker_color = c,
                              marker_size = 3))
    p2 = PP.Plot(PP.scatter(; x = x, y = y, marker_color = c,
                            mode = "markers"))
    l = PP.Layout(yaxis2_scaleanchor = "x",
                  xaxis2_title = "P₁", yaxis2_title = "P₂",
                  scene1 = PP.attr(xaxis_title = "X₁",
                                   yaxis_title_text = "X₂",
                                   zaxis_title = "X₃",
                                   camera_eye = PP.attr(x = 2.8, y = -1, z = 1)),
                  uirevision = 1, showlegend = false)
    plot = [p1 p2]
    PP.relayout!(plot, l)
    PP.PlutoPlot(plot)
end


# ╔═╡ 08401f05-7e4a-4d51-8f96-d2bee22db808
md"Even if the data points do not lie perfectly on a plane, one may want to find a low-dimensional representation of the data that shows as much as possible."

# ╔═╡ c1262304-6846-4811-ae82-397863255415
md"""## PCA Finds the Directions of Largest Variance

Let us start with dimensionality reduction from 2 to 1 dimension. Below we can try to find the direction of the first principal component (the first principal component loading vector ϕ₁) by moving the slider. By definition, it is one of the two vectors for which the average of squared scores zᵢ₁² is maximal.
"""

# ╔═╡ 89f0af1e-bb2f-428e-b2c3-a6985fe3bc8e
@bind ϕ Slider(0:.1:2π)

# ╔═╡ 8c4dab97-52b5-46a2-8025-65c98cdc5b78
md"""The second principal component points in the direction of maximal variance *under the constraint, that it is orthogonal to the first principal component vector*. In the figure below you see a visualization of the variance along the first, second or third principal component (loading vectors ``\phi_1, \phi_2, \phi_3``). The colored lines are parallel to the loading vector you pick in the dropdown menu. They go from each data point to the "center"-plane that goes through the origin ``(0, 0, 0)`` and stands perpendicular to the picked component (loading vector). **The length of the line that starts at data point ``i`` is given by the score ``z_{ij}`` for principal component ``j``**. The average squared length of these lines is the variance along the chosen principal component."""

# ╔═╡ f9a506bb-6bdf-48aa-8f81-8fb6de078533
@bind pc Select(["1 PC", "2 PC", "3 PC"])

# ╔═╡ 18ad4980-2960-4ea9-8c9a-bf17cd3546ff
function data_generator(; seed = 71,  N = 100, rng = MersenneTwister(seed))
	h = [2.5 1.5 .8] .* randn(rng, N, 3)
    a1 = rand(rng); a2 = rand(rng)
	G = [1 0 0
         0 cos(a1) -sin(a1)
         0 sin(a1) cos(a1)] * [cos(a2) 0 sin(a2)
                               0       1   0
                               -sin(a2) 0 cos(a2)]
	x = h * G
end;

# ╔═╡ b2688c20-a578-445f-a5c4-335c85931862
pca_data = data_generator(seed = 241);

# ╔═╡ 743477bb-6cf0-419e-814b-e774738e8d89
m2 = fit!(machine(PCA(maxoutdim = 2), DataFrame(pca_data, :auto)), verbosity = 0);

# ╔═╡ 4a8188a0-c809-4014-9fef-cf6b5f830b1b
begin
    function pc_vectors(m)
        proj = fitted_params(m).projection
        a1 = proj[:, 1]
        a2 = proj[:, 2]
        a3 = cross(a1, a2)
        a3 ./= norm(a3)
        (a1, a2, a3)
    end
    function pca_plane(x, y, m)
        _, _, a3 = pc_vectors(m)
        a3 ./= a3[3]
        -a3[1]*x-a3[2]*y
    end
    pca_plane(x, y) = pca_plane(x, y, m2)
end;

# ╔═╡ 884a7428-2dd4-4091-b889-3acfbfa45d5b
let
    p1 = PP.scatter3d(x = pca_data[:, 1], y = pca_data[:, 2], z = pca_data[:, 3],
                      mode = "markers", marker_size = 3)
    a = pc_vectors(m2)
    col = ["red", "green", "orange"]
    k = Meta.parse(pc[1:1])
    var = 0.
    lines = [begin
                s = pca_data[i, :]
                e = s - (a[k]'*s) * a[k]
                PP.scatter3d(x = [s[1], e[1]], y = [s[2], e[2]], z = [s[3], e[3]],
                             line_color = col[k], mode = "lines")
            end
            for i in 1:size(pca_data, 1)]
    PP.PlutoPlot(PP.Plot([p1; lines],
                         PP.Layout(scene = PP.attr(camera_eye = PP.attr(x = 1,
                                                                        y = 2,
                                                                        z = .4),
                                                   xaxis_title = "X₁",
                                                   yaxis_title_text = "X₂",
                                                   zaxis_title = "X₃"),
                                   uirevision = 1,
                                   showlegend = false)))
end


# ╔═╡ df9a5e8d-0692-44e4-8fd0-43e8643c4b7c
md"## PCA Finds Linear Subspaces Closest to the Data

The `fitted_params` of a `PCA` machine contain as columns the the principal component vectors. The first two components span a plane (see figure below). This plane is closest to the data in the sense that the average squared distance to each data point is minimal. There is no other plane that would be closer to the data points."

# ╔═╡ d3e5ca43-f70e-4a11-a3c9-fc0755179e4c
let
    p1 = PP.scatter3d(x = pca_data[:, 1],
                      y = pca_data[:, 2],
                      z = pca_data[:, 3],
                      mode = "markers",
                      name = "data",
                      marker_size = 3)
    a1, a2, a3 = pc_vectors(m2)
    a1 *= 2; a2 *= 2
    xgrid = -8:.1:8
    p2 = PP.mesh3d(x = repeat(xgrid, length(xgrid)),
                    y = repeat(xgrid, inner = length(xgrid)),
                    z = reshape(pca_plane.(xgrid, xgrid'), :),
                    opacity = .5, color = "gray")
    p3 = PP.scatter3d(x = [-a1[1], a1[1]], y = [-a1[2], a1[2]], z = [-a1[3], a1[3]],
                      mode = "lines", line_color = "red",
                      line_width = 3, name = "first PC")
    p4 = PP.scatter3d(x = [-a2[1], a2[1]], y = [-a2[2], a2[2]], z = [-a2[3], a2[3]],
                      line_color = "green", mode = "lines", line_width = 3,
                      name = "second PC")
    residuals = [begin
                    s = pca_data[i, :]
                    e = s - (a3'*s) * a3
                    PP.scatter3d(x = [s[1], e[1]], y = [s[2], e[2]], z = [s[3], e[3]],
                            line_color = "orange",
                            mode = "lines",
                            name = "distance to plane", showlegend = i == 1)
                end
                for i in 1:size(pca_data, 1)]
    PP.PlutoPlot(PP.Plot([p1; p2; p3; p4; residuals],
                         PP.Layout(scene = PP.attr(camera_eye = PP.attr(x = 1,
                                                                        y = 1,
                                                                        z = .4),
                                                   xaxis_title = "X₁",
                                                   yaxis_title_text = "X₂",
                                                   zaxis_title = "X₃"),
                                   uirevision = 1,
                                   showlegend = true)))
end

# ╔═╡ 936570b9-43da-4c32-b15a-d595f690b1a6
md"
### Reconstructions of the Data

With ``X`` the ``100\times 3``-dimensional matrix of the original data, the reconstruction is given by ``X\Phi_L\Phi_L^T = Z_L\Phi_L^T`` with ``\Phi_L`` the ``3\times L``-dimensional matrix constructed from the first ``L`` loading vectors as columns. The reconstructions are closest to the original data while being forced to vary only along the first ``L`` principal components.

Reconstruction using the first $(@bind firstkpc Select(string.(1:3))) component(s).
"

# ╔═╡ 9032ad56-123c-4585-9b67-c94e5ecb3899
let
    p1 = PP.scatter3d(x = pca_data[:, 1], y = pca_data[:, 2], z = pca_data[:, 3],
                      marker_size = 2, name = "data", mode = "markers")
    a1, a2, a3 = pc_vectors(m2)
    ϕ = if firstkpc == "1"
        a1
    elseif firstkpc == "2"
        [a1 a2]
    else
        [a1 a2 a3]
    end
    rec = pca_data * ϕ * ϕ'
    p2 = PP. scatter3d(x = rec[:, 1], y = rec[:, 2], z = rec[:, 3],
                       marker_size = 2, name = "reconstruction", mode = "markers")
    PP.PlutoPlot(PP.Plot([p1, p2],
                         PP.Layout(scene = PP.attr(camera_eye = PP.attr(x = 1,
                                                                        y = 1,
                                                                        z = .4),
                                                   xaxis_title = "X₁",
                                                   yaxis_title_text = "X₂",
                                                   zaxis_title = "X₃"),
                                   uirevision = 1,
                                   showlegend = true)))
end


# ╔═╡ ab17ef21-ea29-4d62-b14e-b87cffc0a1f9
md"## PCA Decorrelates

A principal component analysis can be performed by computing the covariance matrix ``X^T X`` and its eigenvalues. Below you can see that the covariance matrix is not diagonal and that the eigenvalues (divided by ``n-1``) correspond to the variances along the different principal components.
"

# ╔═╡ da99f084-522b-4969-a65c-31d1548d3073
mlcode(
"""
using LinearAlgebra, Statistics

pca_data = randn(100, 5) * randn(5, 5)   # generate some arbitrary data
X = pca_data .- mean(pca_data, dims = 1) # subtract the mean of each column
(XtX = X'X, eigen_values = eigvals(X'X))
"""
,
"""
import numpy as np
from numpy.linalg import eigvals

pca_data = np.random.randn(100, 5) @ np.random.randn(5, 5)   # generate some arbitrary data
X = pca_data - np.mean(pca_data, axis=0) # subtract the mean of each column
XtX = X.T @ X
eigen_values = eigvals(XtX)
("XtX",XtX,"eigen_values", eigen_values)
"""
)

# ╔═╡ d88509c3-51ca-4ff9-9c1f-c3e13ec0e133
mlcode(
"""
using MLJ, MLJMultivariateStatsInterface, DataFrames

pca = machine(PCA(variance_ratio = 1), DataFrame(pca_data, :auto))
fit!(pca, verbosity = 0)
report(pca).principalvars * (size(pca_data, 1) - 1)
"""
,
"""
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(pca_data)
(pca.singular_values_)
"""
,
)

# ╔═╡ 615a823f-32df-469d-b2a3-e9d2c35e36a7
md"An interesting property of the scores is that their covariance matrix is diagonal, with the eigenvalues of ``X^TX`` on the diagonal. Hence we call PCA a decorrelating transformation."

# ╔═╡ ded0e6cd-e969-47d5-81cf-e2a967bc0a34
mlcode(
"""
Z = MLJ.matrix(MLJ.transform(pca, pca_data))
(ZtZ = chop.(Z'*Z),) # chop sets values close to zero to zero.
"""
,
"""
import numpy as np
from sklearn.decomposition import PCA

pca = PCA()
Z = pca.fit_transform(pca_data)
ZtZ = np.where(np.abs(Z.T @ Z) < 1e-10, 0, Z.T @ Z)
ZtZ
"""
)

# ╔═╡ 31bcc3f0-88c1-4035-b0de-b4217623dc22
md"# 2. Applications of PCA
## Running PCA on artificial data
"

# ╔═╡ 2b73abbb-8cef-4da0-b4cf-b640fe0ad102
mlstring(md"The `PCA` function in `MLJ` has three important keyword arguments as explained in the doc string. If the keyword argument `variance_ratio = 1` all principal components are computed.", "The `PCA` function in `sklearn` has important keyword arguments as explained in the doc string. n_components is the number of components to keep. if n_components is not set all components are kept")

# ╔═╡ 927627e0-9614-4234-823d-eb2e13229784
mlcode(
"""
using MLJ, MLJMultivariateStatsInterface, DataFrames

pca_data = randn(100, 5) * randn(5, 5)   # generate some arbitrary data
pca = machine(PCA(variance_ratio = 1), DataFrame(pca_data, :auto))
fit!(pca, verbosity = 0)
fitted_params(pca) # shows the loadings as columns
"""
,
"""
pca_data = np.random.randn(100, 5) @  np.random.randn(5, 5)  # generate some arbitrary data

pca = PCA()
pca.fit(pca_data)
(pca.explained_variance_ratio_, pca.singular_values_)
"""
)

# ╔═╡ 45ee714c-9b73-492e-9422-69f27be79de2
mlstring(md"The report of a `PCA` machine returns the number of dimensions `indim` of the data. the number of computed principal components `outdim`, the total variance `tvar` and the `mean` of the data set, together with the variances `principalvars` along the computed principal components. The sum of the variances `principalvars` is given in `tprincipalvar` and the rest in `tresidualvar = tvar - tprincipalvar`. If you change the `variance_ratio` to a lower value above you will see that not all principal components are computed.", "The `PCA` machine returns several informations. components_ : Principal axes in feature space, representing the directions of maximum variance in the data. explained_variance_ : The amount of variance explained by each of the selected components. explained_variance_ratio_ : Percentage of variance explained by each of the selected components. See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html ")

# ╔═╡ b52d72e3-64a3-4b7e-901b-3512b97aec23
mlcode(
"""
report(pca)
"""
,
"""
(pca.explained_variance_ratio_, pca.singular_values_)
"""
,
)

# ╔═╡ a8188f31-204a-406f-96a2-86768918a2f9
md"## PCA for Visualization"

# ╔═╡ 51720585-09c8-40bb-8ec2-c2f25804d731
mlcode(
"""
function biplot(m, data;
                pc = (1, 2),
                score_style = :text,
                loadings = :all)
    scores = MLJ.transform(m, data)
    p = scatter(getproperty(scores, Symbol(:x, pc[1])),
                getproperty(scores, Symbol(:x, pc[2])),
                label = nothing, aspect_ratio = 1, size = (600, 600),
                xlabel = "PC" * string(pc[1]), ylabel = "PC" * string(pc[2]),
                framestyle = :axis, markeralpha = score_style == :text ? 0 : 1,
                c = :gray,
                txt = score_style == :text ? Plots.text.(1:nrows(scores), 8, :gray) :
                                             nothing,
                markersize = score_style == :text ? 0 :
                             isa(score_style, Int) ? score_style : 3)
    plot!(p[1], inset = (1, Plots.bbox(0, 0, 1, 1)),
          right_margin = 10Plots.mm, top_margin = 10Plots.mm)
    p2 = p[1].plt.subplots[end]
    plot!(p2, aspect_ratio = 1, mirror = true, legend = false)
    params = fitted_params(m)
    ϕ = if hasproperty(params, :pca)
        params.pca.projection
    else
        params.projection
    end
    n = isa(data, AbstractDataFrame) ? names(data) : keys(data)
    idxs = if loadings == :all
        1:length(n)
    elseif isa(loadings, Int)
        l = reshape(sum(abs2.(ϕ[:, [pc...]]), dims = 2), :)
        sortperm(l, rev = true)[1:loadings]
    else
        loadings
    end
    for i in idxs
        plot!(p2, [0, .9*ϕ[i, pc[1]]], [0, .9*ϕ[i, pc[2]]],
              c = :red, arrow = true)
        annotate!(p2, [(ϕ[i, pc[1]], ϕ[i, pc[2]],
                        (n[i], :red, :center, 8))])
    end
    scatter!(p2, 1.1*ϕ[:, pc[1]], 1.1*ϕ[:, pc[2]],
             markersize = 0, markeralpha = 0) # dummy for sensible xlims and xlims
    plot!(p2,
          background_color_inside = Plots.RGBA{Float64}(0, 0, 0, 0),
          tickfontcolor = Plots.RGB(1, 0, 0))
    p
end

"""
,
"""
import matplotlib.pyplot as plt

def biplot(PC1, PC2, scalePC1, scalePC2, features, ldngs) :
  fig, ax = plt.subplots(figsize=(14, 9))
  
  for i, feature in enumerate(features):
    print(i)
    ax.arrow(0, 0, ldngs[0, i], 
	             ldngs[1, i], 
	             head_width=0.03, 
	             head_length=0.03, 
	             color="red")
    ax.text(ldngs[0, i] * 1.15, 
	            ldngs[1, i] * 1.15, 
	            feature,color="red", fontsize=18)
    ax.scatter(PC1 * scalePC1,
	                    PC2 * scalePC2, s=5)
	 
  for i in range(0,len(PC1)):
    ax.text(PC1[i] * scalePC1, 
	            PC2[i] * scalePC2, str(i), 
	            fontsize=10)
	
  ax.set_xlabel('PC1', fontsize=20)
  ax.set_ylabel('PC2', fontsize=20)
  ax.set_title('Biplot', fontsize=20)
  return fig
"""
,
collapse = "Custom biplot code",
)

# ╔═╡ 265594dd-ba59-408a-925a-c2a173343df7
mlcode(
"""
using OpenML, DataFrames, MLJ, MLJMultivariateStatsInterface, Plots

wine = OpenML.load(187) |> DataFrame;
X = float.(select(wine, Not(:class)))
mwine = machine(Standardizer() |> PCA(), X)
fit!(mwine, verbosity = 0)
biplot(mwine, X)
"""
,
"""
import openml
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

wine,_,_,_ = openml.datasets.get_dataset(187).get_data(dataset_format="dataframe")
X = wine.drop(["class"], axis =1)

mwine = make_pipeline(StandardScaler(), PCA())

PC1 = mwine.fit_transform(X)[:,0]
PC2 = mwine.fit_transform(X)[:,1]
ldngs = mwine[1].components_
scalePC1 = 1.0/(PC1.max() - PC1.min())
scalePC2 = 1.0/(PC2.max() - PC2.min())
features = X.columns

biplot (PC1, PC2, scalePC1, scalePC2, features, ldngs)
plt.show()
"""
,
recompute = true
)

# ╔═╡ ecc77003-c138-4ea9-93a7-27df46aa1e89
md"Biplots are useful to see loadings and scores at the same time. The scores of each data point are given by the coordinates of the numbers plotted in gray when reading the bottom and the left axis. For example, the score for point 15 would be approximately (-4.2, 2.1). The loadings are given by the red labels (not the tips of the arrows) when reading the top and the right axis. For examples the loadings of the first principal component are approximately (0.65, 0.09, 0.76) (see also `fitted_params` above)."

# ╔═╡ e72c09e3-91cb-4455-9880-628428730efe
md"### Scaling the Data Matters!!!

Note that PCA is not invariant under rescaling of the data.
If we would not standardize the wine data, our biplot would look like this:
"

# ╔═╡ 9f6b1079-209f-4a24-96c8-81bdcf8bfc32
mlcode(
"""
mwine2 = machine(PCA(variance_ratio = 1), X)
fit!(mwine2, verbosity = 0)
biplot(mwine2, X)
"""
,
"""
mwine2 = PCA()
PC1 = mwine2.fit_transform(X)[:,0]
PC2 = mwine2.fit_transform(X)[:,1]
ldngs = mwine2.components_
scalePC1 = 1.0/(PC1.max() - PC1.min())
scalePC2 = 1.0/(PC2.max() - PC2.min())
features = X.columns

biplot (PC1, PC2, scalePC1, scalePC2, features, ldngs)
plt.show()
"""
)


# ╔═╡ 87c8b712-aa89-42f7-9529-fdea5d1d41f6
md"""
## Reconstructing a Generative Model with PCA

Let us assume a given data set ``X`` with ``p`` predictors was generated by some mixture of ``q \leq p`` hidden (=unobserved) predictors. Under some conditions PCA can recover these unobserved predictors from the data set ``X``.

### An Artificial Toy Example

In this example we generate 2-dimensional (hidden = unobserved) data and project it to 3 dimensions. This 3-dimensional input is our measured data that we take as input for PCA. We will find that PCA recovers in this case the original data.

"""

# ╔═╡ e24cff15-a966-405a-b083-4b86be09c6f1
mlcode(
"""
using Random

Random.seed!(1)
h = [1.5 .5] .* randn(500, 2) # 2D data with higher variance along first dimension
G = [-.3 .4 .866
     .885 .46 .1]             # projection matrix to 3D
x = h * G                     # projected data
"""
,
"""
import numpy as np

np.random.seed(1)
h = np.multiply([1.5, 0.5], np.random.randn(500, 2))
G = np.array([[-0.3, 0.4, 0.866], [0.885, 0.46, 0.1]])
x = np.dot(h, G) # projected data
"""
,
showoutput = false,
cache_jl_vars = [:h, :x, :G]
)

# ╔═╡ 9e60bd23-bd0e-4311-a059-a39a70143307
let h = M.h, x = M.x
	p1 = PP.Plot(PP.scatter(x = h[:, 1], y = h[:, 2], mode = "markers"),
                 PP.Layout(title = "original (hidden) 2D data"))
    p2 = PP.Plot(PP.scatter3d(x = x[:, 1], y = x[:, 2], z = x[:, 3],
                              mode = "markers", marker_size = 2),
                 PP.Layout(title = "measured 3D data"))
    l = PP.Layout(yaxis1_scaleanchor = "x",
                  xaxis1_title = "H₁", yaxis1_title = "H₂",
                  scene2 = PP.attr(xaxis_title = "X₁",
                                   yaxis_title_text = "X₂",
                                   zaxis_title = "X₃",
                                   camera_eye = PP.attr(x = -2.8, y = 1, z = 1)),
                  uirevision = 1, showlegend = false)
    plot = [p1 p2]
    PP.relayout!(plot, l)
    PP.PlutoPlot(plot)
end

# ╔═╡ a0bb0a63-67e2-414e-a3f0-b76875c1295d
mlcode(
"""
using DataFrames, MLJ, MLJMultivariateStatsInterface

m1 = machine(PCA(), DataFrame(x, :auto))
fit!(m1, verbosity = 0)
h_reconstructed = MLJ.transform(m1, x)
"""
,
"""
pca = PCA(n_components =2)
m1 = pca.fit_transform(x)
"""
,
showoutput = false,
cache_jl_vars = [:h_reconstructed]
)

# ╔═╡ e38cdfcf-362f-4e6e-81a2-e58334bb8d5f
let ĥ = M.h_reconstructed, h = M.h
    scatter(h[:, 1], h[:, 2],
            xlabel = "H₁", ylabel = "H₂",
            aspect_ratio = 1, label = "original hidden data")
    scatter!(sign(h[1, 1]) * sign(ĥ.x1[1]) * ĥ.x1, ĥ.x2, label = "reconstructed hidden data", markersize = 3)
end

# ╔═╡ 16843059-269e-4557-a4b9-5d7034f99ba8
mlstring( md"We can see that the reconstructed hidden data almost perfectly matches the true hidden data. We can also see that the transposed of the reconstructed projection matrix `fitted_params(m1).projection` is close to the projection matrix `G` we used.",
"We can see that the reconstructed hidden data almost perfectly matches the true hidden data. We can also see that the transposed of the reconstructed projection matrix is close to the projection matrix `G` we used.")

# ╔═╡ 6e033717-5815-4ffe-b991-81b72b99df9f
mlcode(
"""
(fitted_params(m1).projection', G)
"""
,
"""
P, _, _, _ = np.linalg.lstsq(pca.transform(x), x, rcond=None)
(P,G)
"""
)

# ╔═╡ b25687fc-aa11-4843-9a89-31c4c9b5bc23
md"This example is specifically constructed for PCA to succeed. The conditions on which success relies are
1. the original data set has the largest variance along ``H_1``. You can change the data set such that it has higher variance along dimension ``H_2``. What happens in this case?
2. The rows of the projection matrix `G` have norm 1. What happens if you multiply the original `G` by 2?
3. `x` was obtained by linear transformation with `G` from `h`.
"

# ╔═╡ 14bfdb2d-fd71-4c62-8823-6e292a424bf5
md"### Reconstructing Spike Trains

Here is a more realistic - but still artificial - example. Let us assume we were interested in the activity of 2 neurons somewhere in a brain, firing at different moments in time an action potential with a shape that is characteristic for the neuron (see blue and red curve in the figure below). Each one of three electrodes that are placed in the brain tissue near these two neurons sense the two neurons in a way that depends on the relative positions of the electrodes and the neurons. The goal is to reconstruct from the measurements at the electrodes the activity of the neurons.
"

# ╔═╡ f6117412-f6d8-41d7-9524-dc25fc891deb
begin
    s1(t) = t < 0 ? 0 : (exp(-t/.2) - exp(-t/.01)) * sin(10*t)
    s2(t) = t < 0 ? 0 : (exp(-t/.5) - exp(-t/.3)) * sin(4*t)
end;

# ╔═╡ 9da3a52e-23ca-4ce4-8524-0abeb4be2306
let
    gr()
    t = -1:.02:3
    plot(t, s1.(t), label = "spike of neuron 1",
         xlabel = "time [arbitrary units]", ylabel = "voltage [arbitrary units]")
    plot!(t, s2.(t), label = "spike of neuron 2",
         xlabel = "time [arbitrary units]", ylabel = "voltage [arbitrary units]")
end

# ╔═╡ 00e15b1c-a12d-4ac2-b3c6-6fece1abef31
begin
    v1 = sum(s1.((0:.02:1000) .- ts) for ts in unique(rand(0:2:1000, 200))) .+ 1e-2*randn(50001)
	v2 = sum(s2.((0:.02:1000) .- ts) for ts in unique(rand(0:2:1000, 200))) .+ 1e-2*randn(50001)
end;

# ╔═╡ 60ba84d8-4f48-4d8a-b695-11edbe91ccc8
let
    gr()
    p1 = plot(v1, xlim = (0, 1000), title = "unobserved spike train",
              label = "v1", xlabel = "time", ylabel = "voltage")
    p2 = plot(v2, xlim = (0, 1000), label = "v2", xlabel = "time", ylabel = "voltage")
    p3 = scatter(v1, v2, aspect_ratio = 1,
                 label = nothing, xlabel = "v1", ylabel = "v2")
    plot(plot(p1, p2, layout = (2, 1)), p3, layout = (1, 2))
end

# ╔═╡ b256ff25-431a-4c47-8195-d958d3d28c56
sdata = [v2 v1] * M.G;

# ╔═╡ abb36d40-ad0e-442c-ab87-b15837a2541c
md"In the figure above you see on the left some time steps of our artificially generated neural activity patterns. Each dot in the figure on the right shows the voltages of the two neurons at a given moment in time.
We assume now that the electrodes measure `sdata`, which is obtained by mixing the originial data in three different ways; think of the 3 columns of `G` as the mixing coefficients. In reality the dependence of the electrode signal on the neural activity might not be as simple as we assume here."

# ╔═╡ e81b5dee-dffb-4a4c-b471-0974599fd87d
let
    gr()
    p1 = plot(sdata[:, 1], xlim = (0, 1000), title = "measured signal",
              label = "electrode 1",
              xlabel = "time", ylabel = "voltage", c = :black)
    p2 = plot(sdata[:, 2], xlim = (0, 1000), label = "electrode 2",
              xlabel = "time", ylabel = "voltage", c = :black)
    p3 = plot(sdata[:, 3], xlim = (0, 1000), label = "electrode 3",
              xlabel = "time", ylabel = "voltage", c = :black)
    p4 = scatter3d(sdata[:, 1], sdata[:, 2], sdata[:, 3],
                   xlabel = "electrode 1" , ylabel = "electrode 2",
                   zlabel = "electode 3", label = nothing, c = :black)
    plot(plot(p1, p2, p3, layout = (3, 1)),
         p4, layout = (1, 2), size = (700, 500))
end

# ╔═╡ 48528bf7-8a40-4773-944c-54d0446a5cac
sma = fit!(machine(PCA(mean = 0), DataFrame(sdata, :auto)), verbosity = 0);

# ╔═╡ 9a5e0976-1c0e-45fc-855e-b0fbec801e8c
srec = MLJ.transform(sma, sdata);

# ╔═╡ b5d2986e-3723-4463-ab05-cc176aea60d8
let
	gr()
	p1 = plot(v1, label = "original v1")
	plot!(srec.x1, label = "reconstruction v1", xlim = (0, 1000))
	p2 = plot(v2, label = "original v2")
	plot!(srec.x2, label = "reconstruction v2", xlim = (0, 1000))
    p3 = scatter(v1, v2, label = "original")
	scatter!(srec.x1, srec.x2, aspect_ratio = 1, label = "reconstruction")
	plot(plot(p1, p2, layout = (2, 1)), p3, layout = (1, 2))
end

# ╔═╡ eb421c10-a023-4faa-bcfb-374b8ebb9c38
md"Also here we see that PCA recovers the original activity patterns. Note that the almost perfect reconstruction relies again on the three conditions specified in the previous section."

# ╔═╡ 9724b203-ae86-4cb6-afec-68e26c01ab83
md"## PCA for Denoising"

# ╔═╡ 715a0e0c-5105-4fee-8b93-03a966a1bd8e
mlcode(
"""
data = let t = 0:.1:8
    Random.seed!(1)
    DataFrame(hcat([rand((sin, cos)).(t) .+ .2*randn(length(t))
				    for _ in 1:2000]...)', :auto)
end
"""
,
"""
import random
import pandas as pd
import numpy as np

t = np.arange(0, 8.1, 0.1)
random.seed(1)
data = pd.DataFrame(np.column_stack([np.random.choice([np.sin, np.cos])(t) + 0.2*np.random.randn(len(t)) for _ in range(2000)]).T)
"""
,
showoutput = false,
cache_jl_vars = [:data]
)

# ╔═╡ 95ba7abd-c425-46d4-8382-fedec7ed58f6
let data = M.data
    gr()
	plot(Array(data[1, :]))
	for i in 2:20
		plot!(Array(data[i, :]))
	end
	plot!(legend = false, xlabel = "time", ylabel = "voltage", title = "raw data")
end

# ╔═╡ d2bfdda4-4496-4daa-bcb5-51d9a7519823
mlcode(
"""
mdenoise = machine(PCA(maxoutdim = 2, mean = 0), data)
fit!(mdenoise, verbosity = 0)
reconstruction = MLJ.inverse_transform(mdenoise, MLJ.transform(mdenoise, data));
"""
,
"""
mdenoise = PCA(n_components=2)

mdenoise.fit(data)

# transform the data using the fitted PCA object and then inverse transform it
reconstruction = mdenoise.inverse_transform(mdenoise.transform(data))
"""
,
showoutput = false,
cache_jl_vars = [:reconstruction]
)

# ╔═╡ b9383ef0-8499-4e56-a8d6-606ec128676c
let reconstruction = M.reconstruction
	plot(Array(reconstruction[1, :]))
	for i in 2:20
		plot!(Array(reconstruction[i, :]))
	end
# 	plot!(s1.(0:.02:3), w = 2)
# 	plot!(s2.(0:.02:3), w = 2)
# 	plot!(s3.(0:.02:3), w = 2)
    plot!(legend = false, xlabel = "time", ylabel = "voltage",
          title = "denoised reconstruction")
end

# ╔═╡ af03d989-b19f-44ac-a275-d8af57bbeaeb
md"## PCA for Compression"

# ╔═╡ 026a362d-3234-409b-b9cb-e6a2a50c6a0d
mlcode(
"""
using OpenML, DataFrames
faces = OpenML.load(41083) |> DataFrame
"""
,
"""
import openml
faces,_,_,_ = openml.datasets.get_dataset(41083).get_data(dataset_format="dataframe")
faces
"""
,
showoutput = false,
cache_jl_vars = [:faces]
)

# ╔═╡ e967a9cb-ad95-4bd2-8fbd-491dc6fe0475
let faces = M.faces
    gr()
    plot(Gray.(vcat([[hcat([[reshape(Array(faces[j*8 + i, 1:end-1]), 64, 64)' ones(64)] for i in 1:8]...); ones(8*65)'] for j in 0:5]...)), ticks = false)
end

# ╔═╡ a00d99c2-e580-47ee-acb6-23d097d47bff
mlstring(nothing,
md"Helpful documentation for plotting pca can be found [here](https://plotly.com/python/pca-visualization).")

# ╔═╡ 0cb282aa-81d4-408d-a919-0d28cdbb7bed
mlcode(
"""
pca_faces = machine(PCA(), faces[:, 1:end-1])
fit!(pca_faces, verbosity = 0)

faces_vars = report(pca_faces).principalvars ./ report(pca_faces).tvar
faces_pcs = fitted_params(pca_faces).projection

let vars = faces_vars
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end
"""
,
"""
scaler = StandardScaler()
faces_scaled = scaler.fit_transform(faces)
pca_faces = PCA(100)
pca_faces.fit(faces_scaled[:,1:-1])

def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    fig, ax = plt.subplots(figsize=(14, 9))
    scree = pca.explained_variance_ratio_*100
    ax.bar(np.arange(len(scree))+1, scree)
    ax.plot(np.arange(len(scree))+1, scree.cumsum(), c="red",marker='o')
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Percentage explained variance")
    ax.set_title("Scree plot")
    return fig

display_scree_plot(pca_faces)
plt.show()
"""
,
showoutput = true,
cache_jl_vars = [:faces_vars, :faces_pcs],
)


# ╔═╡ 0742ea40-1895-48f4-b149-d472397e83d0
let pcs = M.faces_pcs
    scale = x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    plots = []
    for i in [1:5; size(pcs, 2)-4:size(pcs, 2)]
        push!(plots, plot(Gray.(scale(reshape(pcs[:, i], 64, 64))'),
                          title = "PC $i"))
    end
    plot(plots..., layout = (2, 5), showaxis = false, ticks = false)
end

# ╔═╡ 7c020b97-384d-43fc-af1c-a4bf75a6f06c
md"cumulative proportion of variance explained $(@bind variance_ratio Slider(.9:.001:.999, show_value = true))"

# ╔═╡ cddfed10-7c9c-47b0-83fc-b3322da522ed
begin faces = M.faces
    first_few_pc_faces = fit!(MLJ.machine(PCA(variance_ratio = variance_ratio), faces[:, 1:end-1]), verbosity = 0)
    npc = size(fitted_params(first_few_pc_faces).projection, 2)
    md"Using the first ``L = ``$npc components. The compression ratio is ``\frac{n\times p}{(p+n)\times L}`` = $(round(400*64^2/((64^2 + 400)*npc), sigdigits = 2))."
end

# ╔═╡ 03f92562-7419-4d8e-baa2-67f5b021219e
md"Image number $(@bind imgid Select(string.(1:nrow(faces))))"

# ╔═╡ 8e466825-0c84-41c7-93c6-e3cc81a60ef5
let
    gr()
    i = Meta.parse(imgid)
    p1 = plot(Gray.(reshape(Array(faces[i, 1:end-1]), 64, 64)'),
              ticks = false, title = "orginal", showaxis = false)
    p2 = plot(Gray.(reshape(Array(MLJ.inverse_transform(first_few_pc_faces,
                                                       MLJ.transform(first_few_pc_faces,
                                                                     faces[i:i,1:end-1]))), 64, 64)'), ticks = false, showaxis = false, title = "reconstruction")
    plot(p1, p2, layout = (1, 2), size = (700, 500))
end


# ╔═╡ 218d4c11-45a1-4d92-8b48-d08251804638
md"## PCA Applied to the Weather Data"

# ╔═╡ 0943f000-a148-4470-ae5e-9d8d39cd8207
mlcode(
"""
using CSV, DataFrames, MLJ

weather = CSV.read(download("https://go.epfl.ch/bio322-weather2015-2018.csv"),
                   DataFrame)
weather = select(weather, (x -> match(r"direction", x) == nothing &&
                                match(r"time", x) == nothing).(names(weather)))
weather_mach = machine(Standardizer() |> PCA(), weather)
fit!(weather_mach, verbosity = 0)

p1 = biplot(weather_mach, weather, score_style = 2)
p2 = biplot(weather_mach, weather, score_style = 2, pc = (3, 4))
vars = report(weather_mach).pca.principalvars ./
        report(weather_mach).pca.tprincipalvar
p3 = plot(vars, label = nothing, yscale = :log10,
          xlabel = "component", ylabel = "proportion of variance explained")
p4 = plot(cumsum(vars),
          label = nothing, xlabel = "component",
          ylabel = "cumulative prop. of variance explained")
plot(p1, p2, p3, p4, layout = (2, 2), size = (700, 800))
"""
,
"""
weather = pd.read_csv("https://go.epfl.ch/bio322-weather2015-2018.csv")
# select columns of the data frame "weather" that do not contain the words "direction" or "time" in their column names
weather = weather.loc[:, ~(weather.columns.str.contains('direction|time'))]

mweather = make_pipeline(StandardScaler(), PCA()).fit(X)
"""
,
)

# ╔═╡ d44be4c6-a409-421a-8e68-39fa752db4c0
md"# 3. Limitations of PCA as a Dimension Reduction Tool

With PCA, high dimensional data can be transformed to low-dimensional data using a linear projection. The result of this is that pairwise distances are perfectly preserved between points in the plane spanned by the first two principal components and distances orthogonal to this plane are projected to zero. This can occasionally lead to unexpected results, like in the example below, where 8 point clouds are projected to five point clouds by PCA (only thanks to coloring the merged point clouds can still be distinguished).

Alternative methods like t-SNE and UMAP do not try to preserve pairwise distances globally, but rather try to rearrange the data such that all point clouds are still distinguishable in the lower-dimensional space.

## Eight Clouds
"

# ╔═╡ d3b769d7-ec71-4496-926a-e838c3b50c2a
mlcode(
"""
Random.seed!(123)
eight_clouds = DataFrame(vcat([[.5 .5 20] .* randn(50, 3) .+ [x y 0]
                         for y in (-2, 2), x in (-6, -2, 2, 6)]...), :auto);
eight_clouds = MLJ.transform(fit!(machine(Standardizer(), eight_clouds),
                                  verbosity = 0)) # standardizing data
"""
,
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(123)
eight_clouds = pd.DataFrame(np.vstack([[0.5, 0.5, 20] * np.random.randn(50, 3) + [x, y, 0] for y in [-2, 2] for x in [-6, -2, 2, 6]]), columns=['x', 'y', 'z'])

scaler = StandardScaler()
eight_clouds = pd.DataFrame(scaler.fit_transform(eight_clouds), columns = eight_clouds.columns)
"""
,
showoutput = false,
cache_jl_vars = [:eight_clouds]
)

# ╔═╡ 67395b94-46bd-43e5-9e12-0bd921de8203
let eight_clouds = M.eight_clouds
    gr()
    c = vcat([fill(i, 50) for i in 1:8]...)
    p0 = scatter3d(eight_clouds.x1, eight_clouds.x2, eight_clouds.x3,
                   c = c, title = "original 3D data")
    p1 = scatter(eight_clouds.x1, eight_clouds.x2, c = c,
                 xlabel = "x", ylabel = "y", title = "projection onto X-Y")
    pc = MLJ.transform(fit!(machine(PCA(maxoutdim = 2), eight_clouds), verbosity = 0))
    p2 = scatter(pc.x1, pc.x2, xlabel = "PC 1", ylabel = "PC 2", c = c,
                 title = "PCA projection")
    plot(p0, p1, p2, layout = (1, 3), legend = false, size = (700, 400))
end

# ╔═╡ e4302d86-5e4f-4c56-bdca-8b4eed2af47c
mlcode(
"""
using TSne

tsne_proj = tsne(Array(eight_clouds), 2, 0, 2000, 50.0, progress = false);
"""
,
"""
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, n_iter =2000, learning_rate='auto', init='random', perplexity=50).fit_transform(eight_clouds)
"""
,
showoutput = false,
cache_jl_vars = [:tsne_proj]
)

# ╔═╡ 6b319dbe-c626-4294-92b6-cab574aabfb0
let tsne_proj = M.tsne_proj
    gr()
    scatter(tsne_proj[:, 1], tsne_proj[:, 2], legend = false,
            c = vcat([fill(i, 50) for i in 1:8]...), xlabel = "tSNE 1", ylabel = "tSNE 2")
end

# ╔═╡ c079cac4-8789-416d-bc4f-a84a3120ab65
md"We see that tSNE correctly shows the 4 groups and also shows which groups are close to each other."

# ╔═╡ ea1caedc-932a-4d13-bcb7-4c661641596d
mlcode(
"""
using UMAP

umap_proj = umap(Array(eight_clouds)', 2, min_dist = .5, n_neighbors = 10);
"""
,
"""
from umap import UMAP
umap_proj = UMAP(n_components=2, n_neighbors=10, random_state=42, min_dist = 0.5).fit(np.array(eight_clouds))
"""
,
showoutput = false,
cache_jl_vars = [:umap_proj]
)

# ╔═╡ 82e4dc4d-3e75-43e0-aa33-b48a76d09404
let umap_proj = M.umap_proj
    gr()
    scatter(umap_proj[1, :], umap_proj[2, :], legend = false,
            c = vcat([fill(i, 50) for i in 1:8]...), xlabel = "UMAP 1", ylabel = "UMAP 2")
end

# ╔═╡ a30af1e5-eb42-4e23-bdb0-386bf76a43a4
md"Also UMAP finds 4 clusters but some of them are strongly connected, such that it would be difficult to distinguish them without coloring.

Note that the results of tSNE and UMAP change each time you run the cell! The reason is that these methods depend on some gradient descent procedure with random initialization. This is in contrast to PCA that gives always the same lower dimensional result."

# ╔═╡ 716da1a0-266f-41ec-8ef3-a2593ac7b74b
md"## MNIST

Also for real data PCA sometimes fails to find the clusters in
the data. We demonstrate this here with the MNIST data set.
"

# ╔═╡ a6942d97-a091-4a18-bc3b-cd376da263e5
mlcode(
"""
mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    select(df, Not(:class)),
    df.class
end
mnist_pca = machine(PCA(maxoutdim = 2), mnist_x)
fit!(mnist_pca, verbosity = 0)
mnist_proj = MLJ.transform(mnist_pca)

plot()
for i in 0:9
    idxs = findall(x -> x == string(i), mnist_y)
    scatter!(mnist_proj.x1[idxs], mnist_proj.x2[idxs],
             markersize = 1, markerstrokewidth = 0,
             label = string(i))
end
plot!(xlabel = "PC 1", ylabel = "PC 2", legend_position = (0.1, .95))
"""
,
"""
import seaborn as sns 

mnist,_,_,_ = openml.datasets.get_dataset(554).get_data(dataset_format="dataframe")
mnist['class'] = mnist["class"].astype('category')

# Convert the features to continuous
mnist = mnist.astype(float)

# Separate the features and the target variable
mnist_x = mnist.drop('class', axis=1)
mnist_y = mnist['class']

# Perform PCA
mnist_pca = PCA(n_components=2)
mnist_proj = mnist_pca.fit_transform(mnist_x)

df = pd.DataFrame()
df['pca-one'] = mnist_proj[:,0]
df['pca-two'] = mnist_proj[:,1]
df["y"] =  mnist_y

plt.figure(figsize=(16,10))
sns.scatterplot(x="pca-one", y="pca-two", hue="y",
                palette=sns.color_palette("hls", 10), data=df,
                legend="full", alpha=0.3)
plt.show()
"""
,
showoutput = true,
cache_jl_vars = [:mnist_proj, :mnist_y],
)

# ╔═╡ fd8f7308-c38c-420c-a2a1-56ecbb208671
md"""
The dots are colored according to their MNIST class label.

Looking at this it may appear puzzling that K-Means found somewhat reasonable
clusters in the MNIST dataset.

However, let us look at a TSne dimensionality reduction:
"""

# ╔═╡ 87805205-9bb8-40de-b37e-80442a8cd0e7
mlcode(
"""
using TSne

tsne_proj_mnist = tsne(Array(mnist_x[1:7000, :]), 2, 50, 1000, 20, progress = false)
scatter(tsne_proj_mnist[:, 1], tsne_proj_mnist[:, 2],
        c = Int.(int(mnist_y[1:7000])),
        xlabel = \"TSne 1\", ylabel = \"TSne 2\",
        legend = false)
"""
,
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(mnist_x[:7000])

plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=mnist_y[:7000].astype(int), cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Visualizing MNIST through t-SNE', fontsize=24)
plt.show()
"""
)


# ╔═╡ daa928e2-ce2a-4593-a4cd-8faf62c0028d
md"""
Here the images with different class labels are more clearly grouped into different clusters.

Above we ran tSNE only on 10% of the data, because tSNE requires quite some computation.
UMAP scales better to larger datasets and takes on the order of 10 minutes to compute the dimension reduction for the whole MNIST data set.
"""

# ╔═╡ 2eb56328-d39d-40c4-8b73-06e2a9a3e191
mlcode(
"""
using UMAP

umap_proj_mnist = umap(Array(Array(mnist_x)'), 2)

plot()
for i in 0:9
    idxs = findall(x -> x == string(i), mnist_y)
    scatter!(umap_proj_mnist[1, idxs], umap_proj_mnist[2, idxs],
             markersize = 1, markerstrokewidth = 0,
             label = string(i))
end
plot!(xlabel = "UMAP 1", ylabel = "UMAP 2", legend_position = -5)
"""
,
"""
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="white")

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist_x)

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist_y.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()
"""
)

# ╔═╡ 273de5a4-4400-48c9-8a0d-296e57cf26a4
md"""
It is interesting to see in the figure above that the images of digits 4, 7 and 9 are projected to nearby groups (top left corner) and the groups of digits 3, 5, 8 are also nearby each other, but the groups 0, 1, 2 and 6 are clearly separated. In each group there are also some images from the wrong group.

If you want to know more about tSNE and UMAP have a look e.g. at the following [blog post](https://pair-code.github.io/understanding-umap/).
"""

# ╔═╡ 3087540f-19e1-49cb-9b3b-c23207775ea7
md"# 4. Principal Component Regression

As an alternative to regularization, PCA can be used as a preprocessing step in linear regression. This is called principal component regression. It is particularly useful in cases where linear regression could overfit the noise. Note, however, that ridge regression or the Lasso often leads to very similar results as principal component regression.
"

# ╔═╡ 187253a7-78af-4793-8017-48845803a3b8
pcr_data_x, pcr_data_y = let p1 = 10, p2 = 50, n = 120
	hidden = randn(n, p1)
	β = randn(p1)
	y = hidden * β .+ .1 * randn(n)
	transformation = randn(p1, p2)
	x = hidden * transformation .+ .1 * randn(n, p2)
	DataFrame(x, :auto), y
end

# ╔═╡ 60c37ce2-50d7-426f-9dd8-597878425f9c
mlcode(
"""
using MLJ, MLJLinearModels, DataFrames, MLJMultivariateStatsInterface

# artificial data
pcr_data_x, pcr_data_y = let p1 = 10, p2 = 50, n = 120
	hidden = randn(n, p1)
	β = randn(p1)
	y = hidden * β .+ .1 * randn(n)
	transformation = randn(p1, p2)
	x = hidden * transformation .+ .1 * randn(n, p2)
	DataFrame(x, :auto), y
end

# data split
train_idx = rand((true, false), 120)
test_idx = (!).(train_idx)

# linear regression as a benchmark
mach1 = machine(MLJLinearModels.LinearRegressor(),
		        pcr_data_x[train_idx, :],
		        pcr_data_y[train_idx])
fit!(mach1, verbosity = 0)
rmse_lr = rmse(predict(mach1, pcr_data_x[test_idx, :]), pcr_data_y[test_idx])

# principle component regression
mach2 = machine(PCA() |> MLJLinearModels.LinearRegressor(),
		        pcr_data_x[train_idx, :],
		        pcr_data_y[train_idx])
fit!(mach2, verbosity = 0)
rmse_pcr = rmse(predict(mach2, pcr_data_x[test_idx, :]), pcr_data_y[test_idx])

# result
(; rmse_lr, rmse_pcr)
"""
,
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# artificial data
p1, p2, n = 10, 50, 120
hidden = np.random.randn(n, p1)
beta = np.random.randn(p1)
y = hidden @ beta + 0.1 * np.random.randn(n)
transformation = np.random.randn(p1, p2)
x = hidden @ transformation + 0.1 * np.random.randn(n, p2)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.5, random_state=42)

m = LinearRegression()  # Create a linear regression model
m.fit(X_train,y_train)
y_pred = m.predict(X_test)
rmse_lr = mean_squared_error(y_test,y_pred)**0.5

pca = PCA(20)
logistic = LinearRegression()
pipe = Pipeline(steps=[("pca", pca), ("logistic", logistic)])
pipe.fit(X_train,y_train)
y_pred_pca = pipe.predict(X_test)
rmse_pca_lr = mean_squared_error(y_test,y_pred_pca)**0.5

(rmse_lr, rmse_pca_lr)
"""
)


# ╔═╡ eea3f7f8-83ab-4942-9017-504f340ad95b
md"""# Exercises

## Conceptual

#### Exercise 1

(a) The figure below shows 2-dimensional data points. Let us perform PCA on this dataset, i.e. estimate graphically the principal component loadings ``\phi_{11}, \phi_{21}, \phi_{12}, \phi_{22}``. *Hint*: start with determining graphically the direction of the first and the second principal component and normalize the direction vectors afterwards.

$(Random.seed!(818); let x = ([6 4] .* randn(400, 2)) * [-1 1
                                      -2 -.01]
    gr()
    scatter(x[:, 1], x[:, 2], xlim = (-32, 32), ylim = (-16, 16),
            aspect_ratio = 1, legend = false, xlabel = "X₁", ylabel = "X₂")
  end)

(b) In the figure below you see the scores of ``n = 40`` measuments and the loadings of a principal component analysis. How many features ``p`` were measured in this data set?

$(Random.seed!(17); let data = DataFrame(Array(DataFrame(X1 = 6*randn(40), X2 = 4 * randn(40), X3 = randn(40)*2, X4 = randn(40)*2)) * randn(4, 4), :auto)
    gr()
    biplot(fit!(machine(PCA(), data), verbosity = 0))
end
)

(c) Estimate from the figure in (b) the loadings of the first two principal components and the scores of data point 35.

#### Exercise 2
The figures show biplots of principal component analysis applied to 30 three-dimensional points.

$(Random.seed!(123); let d = DataFrame(randn(30, 2) * randn(2, 3) .+ randn(30, 3) * 1e-6, :auto)
    p = fit!(machine(PCA(variance_ratio = 1, mean = 0), d), verbosity = 0)
    gr()
    plot(biplot(p), biplot(p, pc = (1, 3)), layout = (1, 2), size = (700, 350))
end)

(a) In which direction does this dataset have the highest variance? Give your answer as a vector with coefficients determined approximately from the figure.

(b) Is the two-dimensional plane that is closest to the data parallel to the plane spanned by x1 and x3? Justify your answer with a few words.

(c) How large approximately is the proportion of variance explained by the third component?


#### Exercise 3 (optional)

In this exercise you will explore the connection between PCA and SVD.

(a) In the lecture we said that the first principal component is the eigenvector ``\phi_1`` with the largest eigenvalue ``\lambda_1`` of the matrix ``X^T X``. From linear algebra you know that a real, symmetric matrix ``A`` can be diagonalized such that ``A = W \Lambda W^T`` holds, where ``\Lambda`` is a diagonal matrix that contains the eigenvalues and ``W`` is an orthogonal matrix that satisfies ``W^T W = I``, where ``I`` is the identity matrix. The columns of ``W`` are the eigenvectors of ``A``. Let us assume we have found the diagonalization ``X^T X = W \Lambda W^T``. Multiply this equation from the right with ``W`` and show that the columns of ``W`` contain the eigenvectors of ``X^T X``.

(b) The singular value decomposition (SVD) of ``X`` is ``X = U\Sigma V^T`` where ``U`` and ``V`` are orthogonal matrices and ``\Sigma`` is a diagonal matrix. Use the singular value decomposition to express the right-hand-side of ``X^TX = ...`` in terms of ``U``, ``V`` and ``\Sigma``, show how ``U, V, \Sigma, W`` and ``\Lambda`` relate to each other and prove the statements on slide 11 of the lecture. *Hint*: the product of two diagonal matrices is again a diagonal matrix.


## Applied

#### Exercise 4
Look at the following three artificial data sets.

"""

# ╔═╡ a2cd5095-1c95-4107-93eb-d129baf65fd0
mlcode(
"""
data1 = DataFrame(randn(50, 2) * randn(2, 3), :auto)
data2 = DataFrame(randn(50, 2) * randn(2, 3) .+ 0.5*randn(50, 3), :auto)
data3 = DataFrame(randn(50, 2) * randn(2, 10), :auto)
"""
,
"""
data1 = pd.DataFrame(np.random.randn(50, 2) @ (np.random.randn(2, 3)), columns=['x1', 'x2',"x3"])
data2 = pd.DataFrame(np.random.randn(50, 2) @ (np.random.randn(2, 3)) + 0.5*np.random.randn(50, 3), columns=['x1', 'x2',"x3"])
data3 = pd.DataFrame(np.random.randn(50, 2) @ (np.random.randn(2, 10)))
"""
,
showoutput = false
)


# ╔═╡ 00aeeea2-2421-4711-be17-870e87e5851e
md"""
(a) Write down your expectations on the curve of proportion of variance
explained for the tree datasets.

(b) Perform PCA and plot the proportion of variance explained.
"""

# ╔═╡ 4b2652ce-cf0d-4891-bad5-c44d43e61587
md"""
#### Exercise 5 (optional)
In this exercise we look at a food consumption dataset with the relative
consumption of certain food items in European countries. The
numbers represent the percentage of the population consuming that food type.
You can load the data with
"""

# ╔═╡ ec5e88b0-9a4b-4911-9bd3-3261a3fc055f
mlcode(
"""
using CSV, DataFrames

data = CSV.read(download(\"https://openmv.net/file/food-consumption.csv\"), DataFrame)
"""
,
"""
data = pd.read_csv("https://openmv.net/file/food-consumption.csv")
"""
,
showoutput = false
)


# ╔═╡ 6ea7b016-ad29-4a02-ba8d-ed3025682746
md"""
(a) Perform PCA on this dataset and produce a two biplots: one with PC1 versus
PC2 and another with PC1 versus PC3. *Hint*: before applying PCA you may have to deal with missing data (see notebook on transformations) $(mlstring(md"and convert integers to floating point numbers with `coerce!(data, Count => Continuous)`.", "."))

(b) Which are the important variables in the first three components?

(c) What does it mean for a country to have a high score in the first component?
Pick one country that has a high score in the first component and verify that
this country does indeed have this interpretation.
"""

# ╔═╡ a0d57582-0dc8-476b-a908-1d0c34454505
md"""
#### Exercise 6 (optional)

Take a random image e.g.
"""

# ╔═╡ 7de5205c-d86d-4ad5-8b67-089c33bcb233
mlcode(
"""
using FileIO, Plots
img = FileIO.load(download(\"http://kenia.pordescubrir.com/wp-content/uploads/2008/08/windsurf.jpg\"))
"""
,
"""
# importing modules
import urllib.request
from PIL import Image
  
urllib.request.urlretrieve(
  'http://kenia.pordescubrir.com/wp-content/uploads/2008/08/windsurf.jpg',
   "gfg.jpg")
  
img = Image.open("gfg.jpg")

"""
)

# ╔═╡ cc2136d5-7d79-47ea-8b8c-6ce7f7a98ac0
md"""
Transform it to grayscale with $(mlstring(md"`Gray.(img)`","")) and to a data frame with
$(mlstring(md"`DataFrame(float.(real.(Gray.(img))), :auto)`", "")).

The goal is to compress this image with PCA.
To do so, treat each row of this image as a single data point.

(a) Perform PCA on this data.

(b) Reconstruct the image with as many principal components needed to explain 99% of variance.

(c) How many numbers are needed to store the image in this compressed form?

(d) How big is the compression factor?
"""

# ╔═╡ b574a023-21e4-4ae2-a198-7c9962368713
MLCourse.list_notebooks(@__FILE__)


# ╔═╡ 147f9c9c-72dd-4003-90b1-33fd68d762ce
MLCourse.FOOTER

# ╔═╡ b24d7fbf-bd2b-4bd7-90ef-c77ec9099bc8
MLCourse.save_cache(@__FILE__)

# ╔═╡ Cell order:
# ╟─a7ba035f-7433-46f3-a64f-8bf3d6c6cbad
# ╟─78d7653f-1139-4833-b65c-43adb4fca9b1
# ╟─1bb3d957-06dc-4be4-bed7-d14da43c25ee
# ╟─d40fa053-39ed-4d36-ad4c-54356d0146cf
# ╟─029c165b-8d34-42e3-9b68-4d97cfad074e
# ╟─4f960f96-4218-44db-801c-a67222848062
# ╟─59654b1f-011a-4f2d-8665-d2c885b76553
# ╟─a935143d-52f8-4a5e-bede-58467065caed
# ╟─08401f05-7e4a-4d51-8f96-d2bee22db808
# ╟─c1262304-6846-4811-ae82-397863255415
# ╟─89f0af1e-bb2f-428e-b2c3-a6985fe3bc8e
# ╟─8c4dab97-52b5-46a2-8025-65c98cdc5b78
# ╟─b2688c20-a578-445f-a5c4-335c85931862
# ╟─f9a506bb-6bdf-48aa-8f81-8fb6de078533
# ╟─884a7428-2dd4-4091-b889-3acfbfa45d5b
# ╟─743477bb-6cf0-419e-814b-e774738e8d89
# ╟─18ad4980-2960-4ea9-8c9a-bf17cd3546ff
# ╟─4a8188a0-c809-4014-9fef-cf6b5f830b1b
# ╟─df9a5e8d-0692-44e4-8fd0-43e8643c4b7c
# ╟─d3e5ca43-f70e-4a11-a3c9-fc0755179e4c
# ╟─936570b9-43da-4c32-b15a-d595f690b1a6
# ╟─9032ad56-123c-4585-9b67-c94e5ecb3899
# ╟─ab17ef21-ea29-4d62-b14e-b87cffc0a1f9
# ╟─da99f084-522b-4969-a65c-31d1548d3073
# ╟─d88509c3-51ca-4ff9-9c1f-c3e13ec0e133
# ╟─615a823f-32df-469d-b2a3-e9d2c35e36a7
# ╟─ded0e6cd-e969-47d5-81cf-e2a967bc0a34
# ╟─31bcc3f0-88c1-4035-b0de-b4217623dc22
# ╟─2b73abbb-8cef-4da0-b4cf-b640fe0ad102
# ╟─927627e0-9614-4234-823d-eb2e13229784
# ╟─45ee714c-9b73-492e-9422-69f27be79de2
# ╟─b52d72e3-64a3-4b7e-901b-3512b97aec23
# ╟─a8188f31-204a-406f-96a2-86768918a2f9
# ╟─51720585-09c8-40bb-8ec2-c2f25804d731
# ╟─265594dd-ba59-408a-925a-c2a173343df7
# ╟─ecc77003-c138-4ea9-93a7-27df46aa1e89
# ╟─e72c09e3-91cb-4455-9880-628428730efe
# ╟─9f6b1079-209f-4a24-96c8-81bdcf8bfc32
# ╟─87c8b712-aa89-42f7-9529-fdea5d1d41f6
# ╟─e24cff15-a966-405a-b083-4b86be09c6f1
# ╟─9e60bd23-bd0e-4311-a059-a39a70143307
# ╟─a0bb0a63-67e2-414e-a3f0-b76875c1295d
# ╟─e38cdfcf-362f-4e6e-81a2-e58334bb8d5f
# ╟─16843059-269e-4557-a4b9-5d7034f99ba8
# ╟─6e033717-5815-4ffe-b991-81b72b99df9f
# ╟─b25687fc-aa11-4843-9a89-31c4c9b5bc23
# ╟─14bfdb2d-fd71-4c62-8823-6e292a424bf5
# ╟─f6117412-f6d8-41d7-9524-dc25fc891deb
# ╟─9da3a52e-23ca-4ce4-8524-0abeb4be2306
# ╟─00e15b1c-a12d-4ac2-b3c6-6fece1abef31
# ╟─60ba84d8-4f48-4d8a-b695-11edbe91ccc8
# ╟─b256ff25-431a-4c47-8195-d958d3d28c56
# ╟─abb36d40-ad0e-442c-ab87-b15837a2541c
# ╟─e81b5dee-dffb-4a4c-b471-0974599fd87d
# ╟─48528bf7-8a40-4773-944c-54d0446a5cac
# ╟─9a5e0976-1c0e-45fc-855e-b0fbec801e8c
# ╟─b5d2986e-3723-4463-ab05-cc176aea60d8
# ╟─eb421c10-a023-4faa-bcfb-374b8ebb9c38
# ╟─9724b203-ae86-4cb6-afec-68e26c01ab83
# ╟─715a0e0c-5105-4fee-8b93-03a966a1bd8e
# ╟─95ba7abd-c425-46d4-8382-fedec7ed58f6
# ╟─d2bfdda4-4496-4daa-bcb5-51d9a7519823
# ╟─b9383ef0-8499-4e56-a8d6-606ec128676c
# ╟─af03d989-b19f-44ac-a275-d8af57bbeaeb
# ╟─026a362d-3234-409b-b9cb-e6a2a50c6a0d
# ╟─e967a9cb-ad95-4bd2-8fbd-491dc6fe0475
# ╟─a00d99c2-e580-47ee-acb6-23d097d47bff
# ╟─0cb282aa-81d4-408d-a919-0d28cdbb7bed
# ╟─0742ea40-1895-48f4-b149-d472397e83d0
# ╟─7c020b97-384d-43fc-af1c-a4bf75a6f06c
# ╟─cddfed10-7c9c-47b0-83fc-b3322da522ed
# ╟─03f92562-7419-4d8e-baa2-67f5b021219e
# ╟─8e466825-0c84-41c7-93c6-e3cc81a60ef5
# ╟─218d4c11-45a1-4d92-8b48-d08251804638
# ╟─0943f000-a148-4470-ae5e-9d8d39cd8207
# ╟─d44be4c6-a409-421a-8e68-39fa752db4c0
# ╟─d3b769d7-ec71-4496-926a-e838c3b50c2a
# ╟─67395b94-46bd-43e5-9e12-0bd921de8203
# ╟─e4302d86-5e4f-4c56-bdca-8b4eed2af47c
# ╟─6b319dbe-c626-4294-92b6-cab574aabfb0
# ╟─c079cac4-8789-416d-bc4f-a84a3120ab65
# ╟─ea1caedc-932a-4d13-bcb7-4c661641596d
# ╟─82e4dc4d-3e75-43e0-aa33-b48a76d09404
# ╟─a30af1e5-eb42-4e23-bdb0-386bf76a43a4
# ╟─716da1a0-266f-41ec-8ef3-a2593ac7b74b
# ╟─a6942d97-a091-4a18-bc3b-cd376da263e5
# ╟─fd8f7308-c38c-420c-a2a1-56ecbb208671
# ╟─87805205-9bb8-40de-b37e-80442a8cd0e7
# ╟─daa928e2-ce2a-4593-a4cd-8faf62c0028d
# ╟─2eb56328-d39d-40c4-8b73-06e2a9a3e191
# ╟─273de5a4-4400-48c9-8a0d-296e57cf26a4
# ╟─3087540f-19e1-49cb-9b3b-c23207775ea7
# ╟─187253a7-78af-4793-8017-48845803a3b8
# ╟─60c37ce2-50d7-426f-9dd8-597878425f9c
# ╟─eea3f7f8-83ab-4942-9017-504f340ad95b
# ╟─a2cd5095-1c95-4107-93eb-d129baf65fd0
# ╟─00aeeea2-2421-4711-be17-870e87e5851e
# ╟─4b2652ce-cf0d-4891-bad5-c44d43e61587
# ╟─ec5e88b0-9a4b-4911-9bd3-3261a3fc055f
# ╟─6ea7b016-ad29-4a02-ba8d-ed3025682746
# ╟─a0d57582-0dc8-476b-a908-1d0c34454505
# ╟─7de5205c-d86d-4ad5-8b67-089c33bcb233
# ╟─cc2136d5-7d79-47ea-8b8c-6ce7f7a98ac0
# ╟─b97724e4-d7b0-4085-b88e-eb3c5bcbe441
# ╟─b574a023-21e4-4ae2-a198-7c9962368713
# ╟─147f9c9c-72dd-4003-90b1-33fd68d762ce
# ╟─8bd459cb-20bb-483e-a849-e18caae3beef
# ╟─b24d7fbf-bd2b-4bd7-90ef-c77ec9099bc8
