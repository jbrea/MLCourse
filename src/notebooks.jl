const NOTEBOOKS = [("Introduction", "introduction.jl"),
                   ("Supervised Learning", "supervised_learning.jl"),
                   ("Generalized Linear Regression", "generalized_linear_regression.jl"),
                   ("Flexibility and Bias-Variance-Decomposition", "flexibility.jl"),
                   ("Model Assessment", "model_evaluation.jl"),
                   ("Regularization", "regularization.jl"),
                   ("Transformations of Input or Output", "transformations.jl"),
                   ("Gradient Descent", "gradient_descent.jl"),
                   ("Multilayer Perceptrons", "mlp.jl"),
                   ("Convolutional Neural Networks", "convnets.jl"),
                   ("Recurrent Neural Networks", "rnn.jl")
                  ]

function _linkname(path, file)
    if haskey(ENV, "html_export") && ENV["html_export"]
        "$(splitext(file)[1]).html"
    else
        "open?path=" * joinpath(path, nb)
    end
end
function list_notebooks(file)
    sp = splitpath(file)
    path = joinpath(@__DIR__, "..", "notebooks")
    filename = split(sp[end], "#")[1]
    list = join(["1. " * (nb == filename ?
                            name * " (this notebook)" :
                            "[$name](" * _linkname(path, file) * ")")
                 for (name, nb) in NOTEBOOKS], "\n")
Markdown.parse("""# Course Overview

               $list
               """)
end
