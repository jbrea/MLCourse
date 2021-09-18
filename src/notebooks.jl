const NOTEBOOKS = [("Introduction to Julia", "introduction.jl"),
                   ("Supervised Learning", "supervised_learning.jl"),
#                    ("Generalized Linear Regression", "generalized_linear_regression.jl"),
#                    ("Flexibility and Bias-Variance-Decomposition", "flexibility.jl"),
#                    ("Model Assessment", "model_evaluation.jl"),
#                    ("Regularization", "regularization.jl"),
#                    ("Transformations of Input or Output", "transformations.jl"),
#                    ("Gradient Descent", "gradient_descent.jl"),
#                    ("Multilayer Perceptrons", "mlp.jl"),
#                    ("Convolutional Neural Networks", "convnets.jl"),
#                    ("Recurrent Neural Networks", "rnn.jl"),
#                    ("Clustering", "clustering.jl"),
#                    ("Principal Component Analysis", "pca.jl"),
#                    ("Reinforcement Learning", "rl.jl")
                  ]

function _linkname(path, nb, basedir)
    if haskey(ENV, "html_export") && ENV["html_export"] == "true"
        joinpath(basedir, "$(splitext(nb)[1]).html")
    else
        "open?path=" * joinpath(path, nb)
    end
end
function list_notebooks(file, basedir = "")
    sp = splitpath(file)
    path = joinpath(@__DIR__, "..", "notebooks")
    filename = split(sp[end], "#")[1]
    list = join(["1. " * (nb == filename ?
                            name * " (this notebook)" :
                            "[$name](" * _linkname(path, nb, basedir) * ")")
                 for (name, nb) in NOTEBOOKS], "\n")
Markdown.parse("""# Course Overview

               $list
               """)
end

function footer()
    html"""
        <style>

        #launch_binder {
            display: none;
        }

        body.disable_ui main {
                max-width : 95%;
            }

        @media screen and (min-width: 1081px) {
            body.disable_ui main {
                margin-left : 12%;
                max-width : 700px;
                align-self: flex-start;
            }
        }
        </style>
        <p> This page is part of an <a href="https://github.com/jbrea/MLCourse">introductory machine learning course</a> taught by Johanni Brea.</p> <a href="https://www.epfl.ch"><img src="https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg"></img></a>
    """
end
