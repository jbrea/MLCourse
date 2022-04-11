### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 706c7561-e27c-4f57-b97d-65fbca8e85b5
begin
    using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using MAT, CodecZlib, Flux, Tar, ONNXNaiveNASflux, NaiveNASlib, Images,Plots, ImageTransformations, Downloads, MLJFlux, DataFrames, MLJ
end

# ╔═╡ ffbc24d3-b2d8-4fb2-9502-5fa20744b09f
using MLCourse; MLCourse.footer()

# ╔═╡ 3cf31b8e-0507-4a59-9e98-8d9aa303766a
md"""# Transfer Learning

We will download a convolutional network (ResNet18) pretrained on ImageNet,
keep everything but the last layer and retrain a new final layer on a dataset to
classify [images of flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
Many other pretrained models can be downloaded from the [Open Neural Network Exchange (ONNX) Model Zoo](https://github.com/onnx/models).
"""

# ╔═╡ 644fcaae-714b-4c7e-bd78-d7b2184205eb
md"""
In the following cell we define some functions to load the pretrained neural network and the dataset.
"""

# ╔═╡ 80885955-3896-4f52-80b0-0452b5ba3e7b
begin
    const DATADIR = joinpath(@__DIR__, "transfer", "data")
	mkpath(DATADIR)
    function download_data(; path = DATADIR)
        url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102"
        imagefile = joinpath(path, "102flowers.tgz")
        if !isfile(imagefile)
            Downloads.download("$url/102flowers.tgz", imagefile)
        end
        labelfile = joinpath(path, "imagelabels.mat")
        if !isfile(labelfile)
            Downloads.download("$url/imagelabels.mat", labelfile)
        end
        imagedir = joinpath(path, "imgs")
        if !isdir(imagedir)
            open(GzipDecompressorStream, joinpath(path, "102flowers.tgz")) do io
                Tar.extract(io, imagedir)
            end
        end
    end
    image_list(path = DATADIR) = readdir(joinpath(path, "imgs", "jpg"))
    function load_labels(path = DATADIR)
        Int.(MAT.matread(joinpath(path, "imagelabels.mat"))["labels"])
    end
    function load_img(file; path = joinpath(DATADIR, "imgs", "jpg"))
        i = Images.load(joinpath(path, file))
        imresize(i, 224, 224)
    end
    function as_input(ir)
        Float32.(reshape(permutedims(channelview(ir), (2, 3, 1)), 224, 224, 3, 1))
    end
    function load_net(; urlbase = "https://github.com/onnx/models/raw/main/",
            urlmodel = "vision/classification/resnet/model/resnet18-v1-7.onnx",
            filename = joinpath(DATADIR, last(splitpath(urlmodel))))
        if !isfile(filename)
            Downloads.download(joinpath(urlbase, urlmodel), filename)
        end
        ONNXNaiveNASflux.load(filename)
    end
end;

# ╔═╡ ae200510-dcac-4582-a1ff-399ddaa6109f
md"""
Let us load the images and labels. The names corresponding to the image classes can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html).
"""

# ╔═╡ 1e487cd5-4885-4d82-9fe9-18a9a53da3f2
begin
    download_data()
    images = load_img.(image_list())
    labels = load_labels()
end;

# ╔═╡ 5a538b95-ad68-4c4d-8f36-068cd7df9f08
let idxs = rand(1:length(images), 15)
    sample_images = images[idxs]
    plot([begin
              plot(sample_images[i], ticks = false);
              annotate!((.5, -.2, labels[idxs[i]]))
          end for i in 1:15]...,
         layout = (3, 5))
end

# ╔═╡ ba973dcb-aca8-4dcd-a417-28f8e341a4af
begin
    resnet = load_net()
    name.(vertices(resnet)) # shows the names of all the layers
end

# ╔═╡ 7ae8c5ac-9dac-4975-b879-2a280b70116d
frozen_base = CompGraph(vertices(resnet)[1],    # input layer
	                    vertices(resnet)[end-1] # we use the second to last as output
                        );

# ╔═╡ fed6db68-c5bf-40f2-baf0-5adfa43159ac
md"Let us transform all the images to feature representations with the pretrained convolutional network. The output of the second to last layer is our feature representation. This takes quite a bit of time."

# ╔═╡ 08e45235-4f63-485b-8522-d84402ab122b
input = @. frozen_base(as_input(images));

# ╔═╡ 9f70fab4-a974-45b7-a4e8-67d418d6ba39
md"Now we train a standard linear model with gradient descent on the feature representation."

# ╔═╡ 4d5457a2-371e-4ba6-963f-5d6e339f5953
begin
    df = DataFrame(hcat(input...)', :auto);
    model = NeuralNetworkClassifier(builder = MLJFlux.Linear(σ = identity),
                                    epochs = 100,
                                    batch_size = 128)
	mach = machine(model, df, coerce(reshape(labels, :), Multiclass))
    evaluate!(mach, resampling = CV(nfolds = 5, shuffle = true),
              force = true, repeats = 3,
              measure = [mcr, log_loss], verbosity = 0)
end

# ╔═╡ bf7ef997-5ce9-4712-9bd3-b244a1ec8e4c
md"We find that the misclassification rate on the test set is around 11%.

## Training Convolutional Networks From Scratch

Instead of using a pretrained convolutional network we could have trained a new one from scratch. Below are two examples with convolutional networks trained from scratch. The first one is a small custom convolutional neural network. The second one uses the modern MobileNetv3 architecture with a small modification in the final layers. Training these networks on a Tesla V100 GPU takes a few minutes. The misclassification rate is above 60% (!) for both models and thus far higher than the 11% we obtained with the transfer learning approach (which took little training time on a CPU).
"

# ╔═╡ 44382c06-6d32-459f-931c-71a894f13430
md"""
```julia
# First convolutional model
input_raw = coerce(cat(as_input.(images)..., dims = 4), ColorImage);
builder = MLJFlux.@builder(
          begin
              front = Chain(Conv((7, 7), n_channels => 16, relu,
                                 stride = 5,
                                 pad = SamePad()),
                            Conv((5, 5), 16 => 16, relu,
                                 stride = 3,
                                 pad = SamePad()),
                            Conv((3, 3), 16 => 16, relu,
                                 stride = 3, pad = SamePad()),
                            Flux.flatten)
              d = first(Flux.outputsize(front, (n_in..., n_channels, 1)))
              Chain(front, Dense(d, 256, relu), Dense(256, n_out))
          end)
cnnmodel = ImageClassifier(builder = builder,
                           batch_size = 128,
                           acceleration=CUDALibs(),
                           epochs = 30)
cnn = machine(cnnmodel, input_raw, coerce(reshape(labels, :), Multiclass))
evaluate!(cnn, resampling = Holdout(fraction_train = .8, shuffle = true),
          measure = [mcr, log_loss], verbosity = 2)

# MobileNetv3
using Pkg; Pkg.add(name = "Metalhead", rev = "178a0ff")
using Metalhead
mnbuilder = MLJFlux.@builder(
     begin
         net = MobileNetv3()
         newnet = Chain(net.layers[1:end-1]...,
                        Chain(net.layers[end].layers[1:2]...,
                              Dense(576, 128, hardswish),
                              net.layers[end].layers[4],
                              Dense(128, n_out)))
     end)
cnnmodel2 = ImageClassifier(builder = mnbuilder,
                           batch_size = 128,
                           acceleration=CUDALibs(),
                           epochs = 30)
cnn2 = machine(cnnmodel2, input_raw, coerce(reshape(labels, :), Multiclass))
evaluate!(cnn2, resampling = Holdout(fraction_train = .8, shuffle = true),
          measure = [mcr, log_loss], verbosity = 2)
```
"""

# ╔═╡ Cell order:
# ╟─3cf31b8e-0507-4a59-9e98-8d9aa303766a
# ╠═706c7561-e27c-4f57-b97d-65fbca8e85b5
# ╟─644fcaae-714b-4c7e-bd78-d7b2184205eb
# ╠═80885955-3896-4f52-80b0-0452b5ba3e7b
# ╟─ae200510-dcac-4582-a1ff-399ddaa6109f
# ╠═1e487cd5-4885-4d82-9fe9-18a9a53da3f2
# ╠═5a538b95-ad68-4c4d-8f36-068cd7df9f08
# ╠═ba973dcb-aca8-4dcd-a417-28f8e341a4af
# ╠═7ae8c5ac-9dac-4975-b879-2a280b70116d
# ╟─fed6db68-c5bf-40f2-baf0-5adfa43159ac
# ╠═08e45235-4f63-485b-8522-d84402ab122b
# ╟─9f70fab4-a974-45b7-a4e8-67d418d6ba39
# ╠═4d5457a2-371e-4ba6-963f-5d6e339f5953
# ╟─bf7ef997-5ce9-4712-9bd3-b244a1ec8e4c
# ╟─44382c06-6d32-459f-931c-71a894f13430
# ╟─ffbc24d3-b2d8-4fb2-9502-5fa20744b09f
