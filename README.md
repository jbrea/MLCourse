# MLCourse

This repository contains teaching material for an introductory machine learning course.
You can find an interactive preview of the Pluto notebooks of this course [here](https://bio322.epfl.ch) and you can run some notebooks on [mybinder](https://mybinder.org/v2/gh/jbrea/MLCourse/binder?urlpath=pluto/open?path%3D/home/jovyan/MLCourse/index.jl) (some notebooks will crash on mybinder when they hit the memory limit).

To use the code, please download Julia version 1.7.3 with `juliaup` or by downloading the
binary from [this page](https://julialang.org/downloads/oldreleases/).

For example, **Windows** users can download `juliaup` on a command line with
```
winget install julia -s msstore
```
and then run
```
juliaup add 1.7.3
juliaup default 1.7.3
```
Alternatively you can download [this installer](https://julialang-s3.julialang.org/bin/winnt/x64/1.7/julia-1.7.3-win64.exe).

**Mac** users can download `juliaup` on a command line with
```
curl -fsSL https://install.julialang.org | sh
```
or if `brew` is available on the system
```
brew install juliaup
```
Once `juliaup` is installed, please run
```
juliaup add 1.7.3
juliaup default 1.7.3
```
You may need to `source ~/.bashrc` if `juliaup` is not found after installation.
Alternatively you can download [this installer](https://julialang-s3.julialang.org/bin/mac/x64/1.7/julia-1.7.3-mac64.dmg)

**Linux** users can download `juliaup` on a command line with
```
curl -fsSL https://install.julialang.org | sh
```
Or use the AUR if you are on Arch Linux or `zypper` if you are on openSUSE Tumbleweed.
Once `juliaup` is installed, please run
```
juliaup add 1.7.3
juliaup default 1.7.3
```
You may need to `source ~/.bashrc` if `juliaup` is not found after installation.
Alternatively you can download and unpack [this archive](https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz)

Once Julia 1.7.3 is installed, open a julia REPL and run the following commands

```julia
julia> using Pkg
       Pkg.activate(temp = true)
       Pkg.develop(url = "https://github.com/jbrea/MLCourse")
       Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
       Pkg.instantiate()
       using MLCourse
       MLCourse.start()
```
Many packages and binaries are downloaded in the `Pkg.instantiate()` step.
If you encounter an error message like `ERROR: Unable to automatically install
'sysimage'` or `ERROR: failed to clone from ...` rerun `Pkg.instantiate()` a moment later.

To use the at a later point notebooks, restart julia and type
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
