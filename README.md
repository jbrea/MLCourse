# MLCourse

This repository contains teaching material for an introductory machine learning course.
You can find a static preview (i.e. sliders etc. won't work) of the Pluto notebooks used in this course [here](https://jbrea.github.io/MLCourse/notebooks/welcome.html).

To use the code, please download [julia](julialang.org/downloads) (at least version 1.6)
open julia and install the code in this repository with
```julia
julia> using Pkg
       Pkg.develop(url = "https://github.com/jbrea/MLCourse")
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       Pkg.instantiate()
```

To use the notebooks, start julia and type
```julia
julia> using MLCourse
       MLCourse.start()
```

To reduce loading times of the pluto notebooks you can create a custom system image with
```julia
julia> using MLCourse
       MLCourse.create_sysimage()
```
