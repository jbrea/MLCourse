# MLCourse

This repository contains teaching material for an introductory machine learning course.
You can find an interactive preview of the Pluto notebooks of this course [here](https://bio322.epfl.ch) and you can run some notebooks on [mybinder](https://mybinder.org/v2/gh/jbrea/MLCourse/binder?urlpath=pluto/open?path%3D/home/jovyan/MLCourse/index.jl) (some notebooks will crash on mybinder when they hit the memory limit).

To use the code, please download [julia version 1.7](https://julialang.org/downloads)
or newer, open julia and install the code in this repository with
```julia
julia> using Pkg
       Pkg.develop(url = "https://github.com/jbrea/MLCourse")
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       Pkg.instantiate()
       using MLCourse
       MLCourse.create_sysimage()
```

To use the notebooks, restart julia and type
```julia
julia> using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       using MLCourse
       MLCourse.start()
```

You can update the course material with
```julia
julia> using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       using MLCourse
       MLCourse.update()
```

![](https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg)
