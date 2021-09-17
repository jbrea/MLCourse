# MLCourse

This repository contains teaching material for an introductory machine learning course.
You can find an interactive preview of the Pluto notebooks of this course [here](https://bio322.epfl.ch) and a static preview (where sliders etc. don't work)  [here](https://jbrea.github.io/MLCourse/notebooks/welcome.html).

To use the code, please download [julia](https://julialang.org/downloads) (at least version 1.6)
open julia and install the code in this repository with
```julia
julia> using Pkg
       Pkg.develop(url = "https://github.com/jbrea/MLCourse")
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       Pkg.instantiate()
```

To use the notebooks, start julia and type
```julia
julia> using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       using MLCourse
       MLCourse.start()
```

If git is installed on your system (otherwise download it e.g.
[here](https://git-scm.com/downloads)) you can update the course material with
```julia
julia> using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       using MLCourse
       MLCourse.update()
```

To reduce loading times of the pluto notebooks you can create a custom system image with
```julia
julia> using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       using MLCourse
       MLCourse.create_sysimage()
```

![](https://www.epfl.ch/wp/5.5/wp-content/themes/wp-theme-2018/assets/svg/epfl-logo.svg)
