# MLCourse

This repository contains teaching material for an introductory machine learning course.

**Students of my course do not need to pull this repository or follow the instructions
below. This repository is mostly used to create the interactive websites on [https://bio322.epfl.ch](https://bio322.epfl.ch).**

## Installation

To use the code, we recommend downloading Julia version 1.9.3 with `juliaup`.

<details>
<summary>Windows</summary>

#### 1. Install juliaup
```
winget install julia -s msstore
```
#### 2. Add Julia 1.9.3
```
juliaup add 1.9.3
```
#### 3. Make 1.9.3 default
```
juliaup default 1.9.3
```

<!---#### Alternative
Alternatively you can download [this installer](https://julialang-s3.julialang.org/bin/winnt/x64/1.7/julia-1.9.3-win64.exe).--->

</details>


<details>
<summary>Mac</summary>

#### 1. Install juliaup
```
curl -fsSL https://install.julialang.org | sh
```
You may need to run `source ~/.bashrc` or `source ~/.bash_profile` or `source ~/.zshrc` if `juliaup` is not found after installation.

Alternatively, if `brew` is available on the system you can install juliaup with
```
brew install juliaup
```
#### 2. Add Julia 1.9.3
```
juliaup add 1.9.3
```
#### 3. Make 1.9.3 default
```
juliaup default 1.9.3
```

<!---#### Alternative
Alternatively you can download [this installer](https://julialang-s3.julialang.org/bin/mac/x64/1.7/julia-1.9.3-mac64.dmg)--->

</details>

<details>
<summary>Linux</summary>

#### 1. Install juliaup

```
curl -fsSL https://install.julialang.org | sh
```
You may need to run `source ~/.bashrc` or `source ~/.bash_profile` or `source ~/.zshrc` if `juliaup` is not found after installation.

Alternatively, use the AUR if you are on Arch Linux or `zypper` if you are on openSUSE Tumbleweed.
#### 2. Add Julia 1.9.3
```
juliaup add 1.9.3
```
#### 3. Make 1.9.3 default
```
juliaup default 1.9.3
```

<!---#### Alternative
Alternatively you can download and unpack [this archive](https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.9.3-linux-x86_64.tar.gz)--->

</details>

#### 4. Installing MLCourse
Once you have finished the above steps 1.-3. for your operating system, launch julia and
run the following code to install the course material.
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

## Usage
Once MLCourse is installed, you can open the notebooks in a Julia REPL anytime with
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
