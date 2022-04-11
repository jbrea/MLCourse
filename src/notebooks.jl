const NOTEBOOKS = [("Introduction to Julia", "introduction.jl"),
                   ("Supervised Learning", "supervised_learning.jl"),
                   ("Generalized Linear Regression", "generalized_linear_regression.jl"),
                   ("Flexibility and Bias-Variance-Decomposition", "flexibility.jl"),
                   ("Model Assessment", "model_evaluation.jl"),
                   ("Regularization", "regularization.jl"),
                   ("Transformations of Input or Output", "transformations.jl"),
                   ("Gradient Descent", "gradient_descent.jl"),
                   ("Multilayer Perceptrons", "mlp.jl"),
                   ("Other Nonlinear Methods", "other_nonlinear.jl"),
#                    ("Recurrent Neural Networks", "rnn.jl"),
#                    ("Tree-Based Methods", "trees.jl"),
                   ("Clustering", "clustering.jl"),
                   ("Principal Component Analysis", "pca.jl"),
                   ("Reinforcement Learning", "rl.jl")
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

               Optional Extras:
               1. [Transfer Learning](https://lcnwww.epfl.ch/bio322/transfer_learning.html)
               2. [Generative Models](https://lcnwww.epfl.ch/bio322/ae_gan.html)
               """)
end

function footer()
    html"""
        <style>
            #launch_binder {
                display: none;
            }
        </style>
        <p> This page is part of an <a href="https://bio322.epfl.ch">introductory machine learning course</a> taught by Johanni Brea.<br>The course is strongly inspired by <a href="https://www.statlearning.com/">"An Introduction to Statistical Learning"</a>.</p> <a href="https://www.epfl.ch"><img src="https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg"></img></a>
    """
end
