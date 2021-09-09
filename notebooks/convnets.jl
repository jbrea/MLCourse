### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 64bac944-8008-498d-a89b-b9f8ee54aa98
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI
    PlutoUI.TableOfContents()
end

# ╔═╡ 01a467a5-7389-44d1-984d-244dfb1ea39f
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 7cfd274d-75d8-4cd9-b38b-4176466c9e26
Markdown.parse("# Fitting MNIST with a Convolutional Neural Networks

In the following we fit a convolutional neural network to the MNIST data set.

Running this example takes more than 10 minutes.

We load the data
```julia
using MLJ, Plots, MLJFlux, Flux, MLJOpenML, DataFrames
mnist_x, mnist_y = let df = MLJOpenML.load(554) |> DataFrame
    coerce!(df, :class => Multiclass)
    coerce!(df, Count => MLJ.Continuous)
    Float32.(df[:, 1:end-1] ./ 255),
    df.class
end
```
and transform the input to an image representation
```julia
images = coerce(PermutedDimsArray(reshape(Array(mnist_x), :, 28, 28), (3, 2, 1)),
                GrayImage);
plot(images[1])
```
$(MLCourse.embed_figure("mnist_example1.png"))

Now we define our network builder
```julia
builder = MLJFlux.@builder(
              begin
                  front = Chain(Conv((8, 8), n_channels => 16, relu),
                                Conv((4, 4), 16 => 32, relu),
                                Flux.flatten)
                  d = first(Flux.outputsize(front, (n_in..., n_channels, 1)))
                  Chain(front, Dense(d, n_out))
              end)
```
The `front` consists of two convolutional layers. The volume of the second
convolutional layer is flattened and taken as input to a single dense layer.
```julia
m = machine(ImageClassifier(builder = builder,
                            batch_size = 32,
                            epochs = 5),
            images, mnist_y)
fit!(m, rows = 1:60000, verbosity = 2)
```
Finally, let us compute the test error rate
```julia
mean(predict_mode(m, images[60001:70000]) .!= mnist_y[60001:70000])
```
The result is a misclassification rate of 1.4% which is again a strong improvement
over the MLP result. Note that we achieved this value without tuning of the hyper-parameters. Well-tuned convolutional networks achieve performances below 1% error.
")


# ╔═╡ 2bbf9876-0676-11ec-3985-73f4dcaea02f
Markdown.parse("# Exercises
## Conceptual

1. Here below is an image (with padding 1 already applied). We would like to process it with a convolutional network with one convolution layer with two ``3 \\times 3`` filters (depicted below the image), stride 1 and relu non-linearity.
    - Determine the width, height and depth of the volume after the convolutional layer.
    - Compute the output of the convolutional layer assuming the two biases to be zero.
$(MLCourse.embed_figure("conv_exercise.png"))
2. Given a volume of width ``n``, height ``n`` and depth ``c``:
    - Convince yourself that the convolution with ``k`` filters of size ``f\\times f\\times c`` with stride ``s`` and padding ``p`` leads to a new volume of size ``\\left( \\frac{n + 2p - f}{s} + 1\\right)\\times \\left( \\frac{n + 2p - f}{s} + 1\\right)\\times k\\, .``
    - Flux knows the padding option `pad = SamePad()`. It means that the output volume should have the same x and y dimension as the input volume. In this exercise we compute the padding needed this option.  What padding ``p`` do you have to choose for a given ``n`` and ``f`` such that the input and output volumes have the same width and depth for stride ``s=1``. Check you result for the special case of ``n=4`` and ``f=3``.

## Applied
1. Try to find a convolutional network that works better than all methods you considered so far for the Fashion-MNIST dataset.
")

# ╔═╡ Cell order:
# ╟─7cfd274d-75d8-4cd9-b38b-4176466c9e26
# ╟─2bbf9876-0676-11ec-3985-73f4dcaea02f
# ╟─01a467a5-7389-44d1-984d-244dfb1ea39f
# ╟─64bac944-8008-498d-a89b-b9f8ee54aa98
