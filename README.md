# MLCourse

This repository contains teaching material for an introductory machine learning course.
You can find an interactive preview of the Pluto notebooks of this course [here](https://bio322.epfl.ch) and a static preview (where sliders etc. don't work)  [here](https://jbrea.github.io/MLCourse/notebooks/welcome.html).

To use the code, please download [julia](https://julialang.org/downloads) (at least version 1.6.2)
open julia and install the code in this repository with
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
       MLCourse.update(create_sysimage = true)
```

![](https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg)
