### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 065a1e67-6b63-43df-9d6d-303af08d8434
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 034f7cc2-eacd-11eb-29e2-810a573295cf
using LinearAlgebra

# ╔═╡ 034f7c90-eacd-11eb-1b40-8722209689fa
using DataFrames # load the DataFrames package

# ╔═╡ 034f7d1a-eacd-11eb-1d48-591daf597cd6
using Random

# ╔═╡ 034f7d3a-eacd-11eb-1144-65f9acfcf24c
using Plots

# ╔═╡ 034f7d58-eacd-11eb-3189-bff16ec27739
using Distributions

# ╔═╡ 034f7d76-eacd-11eb-24cc-fd8d53d736c2
using Statistics

# ╔═╡ d6e5fe02-21a5-486c-a237-878be1d95439
begin
    using MLCourse
    MLCourse.list_notebooks(@__FILE__)
end

# ╔═╡ 2b67233e-76ba-40be-81e2-787fbe5d3641
using PlutoUI; PlutoUI.TableOfContents()

# ╔═╡ f3508747-da29-47c9-a98e-22ea15caaf2f
Markdown.parse("Hi and welcome to an introduction to the Julia programming language.

$(haskey(ENV, "html_export") ? "This page was created with an interactive [Pluto notebook](https://plutojl.org/). If you want to run it locally on your machine, follow the instructions [here](https://github.com/jbrea/MLCourse)." :
"This is an interactive Pluto notebook. You can create new cells by clicking any \"+\" above or below existing cells. In these cells you can write code and run it by clicking the play button or [Shift] + [Enter] on your keyboard (or [Ctrl] + [Enter]).
The output gets displayed above the cell. Have a look at [this website](https://plutojl.org/) if you want to learn more about Pluto notebooks.

To get help, please open the Live docs at the bottom right of this page and click on the code you want to get help for. For example, click on the ÷ symbol in the 8th cell below.")

If you want to learn more about Julia visit [julialang.org](https://julialang.org).
In the following sections you find links to different chapters of the [manual](https://docs.julialang.org/en/v1/). There is also this [cheat sheet](https://juliadocs.github.io/Julia-Cheat-Sheet/). Noteworthy differences to other programming languages can be found [here](https://docs.julialang.org/en/v1/manual/noteworthy-differences/).

Before we start, we activate the environment of this course. The environment defines all the versions of the packages used in this course.")

# ╔═╡ d92c0729-c7ab-42cc-b713-30f00e237833
md"# Mathematical Operations

Here is the [link](https://docs.julialang.org/en/v1/manual/mathematical-operations/) to the respective section in the Julia manual."

# ╔═╡ b8630ee9-d2c0-4cd0-b3f7-77f66e997a80


# ╔═╡ 034f7b46-eacd-11eb-1aa2-f380d08206cc
1 + 2

# ╔═╡ 034f7b64-eacd-11eb-1618-6de6def66405
2 * 3

# ╔═╡ 034f7b6e-eacd-11eb-1733-650ad4da2730
3 ^ 2

# ╔═╡ 034f7b78-eacd-11eb-003a-5b5801c19d80
3 == 9 / 3 # test for equality

# ╔═╡ 034f7b6e-eacd-11eb-184a-d504fab31681
4 / 3 # compute the fraction as a floating point number

# ╔═╡ fb3677a9-1610-49dd-96e4-3bd7badc58db
4 // 3 # leave the fraction as a fraction

# ╔═╡ 884f568b-079c-47d6-a597-34b67925f195
4//3 * 3//2 * 1//7 # simplify fractions

# ╔═╡ 034f7b84-eacd-11eb-2836-11cebedb05b9
5 ÷ 2 # Write this symbol with \div [TAB]. Try it in a new cell!

# ╔═╡ 52efdece-3c21-43d4-a826-3b96e3a23e60
md"# Variables and Simple Functions

See also [here](https://docs.julialang.org/en/v1/manual/variables/)."

# ╔═╡ 034f7ba0-eacd-11eb-3fc1-9dca0e072755
x = 2

# ╔═╡ 034f7ba0-eacd-11eb-23ae-cd327563ec60
y = 4.5

# ╔═╡ 034f7ba0-eacd-11eb-3347-49deb13761d9
z = x * y

# ╔═╡ 121b3dbb-f86c-44b8-8b0f-f6b4d0b9992f
md"In the following cells we apply simple built-in functions to the variables defined above. If you change the values of these variables, the output of the cells below will also change, thanks to the reactivity of this notebook. Try it by assigning another value to `x` in the cell above."

# ╔═╡ 034f7baa-eacd-11eb-33f2-5f619e4a7832
sin(x)

# ╔═╡ 034f7bb2-eacd-11eb-006b-7b935e35af66
exp(x)

# ╔═╡ 034f7bb2-eacd-11eb-214b-cbfc83965c26
log(x)

# ╔═╡ ec1dd4c4-d61d-4fdb-b576-3fe3b059eb3a
typeof(2)

# ╔═╡ 1a3387ee-b59d-49d7-9cca-2a8bd04eaaad
typeof(2.0)

# ╔═╡ 034f7b96-eacd-11eb-0c81-ad33210de120
typeof(4//3)

# ╔═╡ 034f7b96-eacd-11eb-2ee1-cd8e6813a726
typeof(2f0)

# ╔═╡ 0f4e3f45-71dd-41b2-a0b3-c55a04ace61f
md"In the following cells you see 3 different ways to define custom functions. See also [here](https://docs.julialang.org/en/v1/manual/functions/), and, in particular, [here](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions) for anonymous functions."

# ╔═╡ 034f7bbe-eacd-11eb-2e95-a93ad2d5e867
myfunc(x, y) = 2 * x + 3 - y # creates a new function called `myfunc`

# ╔═╡ 034f7bbe-eacd-11eb-2c36-d3949367ce5a
myfunc(x, y)

# ╔═╡ e0ec7ba3-4858-4572-994d-953caa98080b
anonfunc = x -> x^2 # creates an anonymous function called `anonfunc`

# ╔═╡ 04da39a1-e3bc-4086-b555-6cea2cf1689d
anonfunc(4)

# ╔═╡ 034f7bc8-eacd-11eb-0396-5b743b505b2a
function longfunc(x, y)
    tmp = x + y
    tmp ^ 2 - 0.25 # the result of the last line is the output
end

# ╔═╡ 034f7bc8-eacd-11eb-31f9-c119d1dfb8a5
longfunc(x, y)

# ╔═╡ ce3c7687-c7a9-4b5d-9a55-51514ed90fd9
md"We will sometimes use the pipe operator to chain functions."

# ╔═╡ 034f7bd2-eacd-11eb-28f3-b707012e1db0
x |> sin |> exp |> log10

# ╔═╡ 034f7bd2-eacd-11eb-1789-7380a89d5339
log10(exp(sin(x)))

# ╔═╡ 034f7cd6-eacd-11eb-0f8c-11179c3ab480
md" # Strings and Markdown

See also [here](https://docs.julialang.org/en/v1/manual/strings/)
and [here](https://docs.julialang.org/en/v1/stdlib/Markdown/).
"

# ╔═╡ 034f7cd6-eacd-11eb-0f8c-11179c3ab48f
s1 = "Hi! "

# ╔═╡ 034f7ce0-eacd-11eb-3c49-e39b8d3e917f
s2 = "Welcome to Machine Learning"

# ╔═╡ 034f7ce0-eacd-11eb-18a1-bd2d1849b9bc
s1 * s2

# ╔═╡ 034f7cec-eacd-11eb-3262-bdc0f2e13a3a
s1 ^ 3 * s2

# ╔═╡ 034f7cec-eacd-11eb-27da-9bb861643df2
split(s2)

# ╔═╡ 832ce046-ab34-4623-9d02-793dc5748208
md"Sometimes we want to paste something computed into a string. This is called [interpolation](https://docs.julialang.org/en/v1/manual/strings/#string-interpolation)."

# ╔═╡ 91b0de0c-221c-468e-9077-5233aedeb4ca
answer = 42

# ╔═╡ 0c7ae3ec-a40f-4e3e-85d8-3f79f091e3b4
"The answer is $answer." # we interpolate with the dollar sign

# ╔═╡ 732ae082-b90f-4108-9872-36077fb2c54c
md"Whenever you see just text here, it is actually the output of a markdown cell.
Markdown cells can be written in two ways: either as a markdown string `md\"this is a markdown string\"` (note the trailing `md`) or as `Markdown.parse(\"this is a markdown string\")`.
You can look at the code of this cell by toggling its visibility with the \"eye\"
button on the left. For you to take notes, it may be useful to know some of the
markdown features. We look at some in the cell below.
"

# ╔═╡ 6bfb60b4-964e-46ec-adc2-ec9e9a0f8158
md"#### This is a fourth level header

[This is a link](https://docs.julialang.org/en/v1/)

- here we have a list
- of multiple items
   1. with different levels.
   1. The number you put here doesn't matter.
   0. The markdown interpreter knows how to count.

Below you see a Julia code block:

```julia
my_fancy_function(x) = x^x # here I define my function
my_fancy_function(3) == 27
```

and here is the same thing in python:

```python
def my_fancy_function(x): # here I define my function
    return x**x

my_fancy_function(3) == 27
```

This is a formula ``E = mc^2``.

And below we have some nicely colored text boxes.

!!! note

    Sometimes I just note random stuff.

!!! tip

    It is useful to know markdown.

!!! warning \"Don't forget the exercises!\"

    No pain, no gain!

!!! danger \"Machine Learning Can Be Addictive\"

    Rumours say that in the past there were students that liked their first
    Machine Learning course so much that they decided to pursue a career in
    Machine Learning.
"

# ╔═╡ 0efb1f1a-8051-4843-aa68-1f4ad8618f92
md"# Vectors, Matrices and Arrays

See also [here](https://docs.julialang.org/en/v1/manual/arrays/)."

# ╔═╡ 034f7bdc-eacd-11eb-1999-9b62efdb4128
v = [1, 2, 3] # this is a vector

# ╔═╡ 034f7cba-eacd-11eb-2f1a-836ccc185dd9
push!(v, 12)

# ╔═╡ 034f7cba-eacd-11eb-3901-2bc3d0ff0e39
pop!(v)

# ╔═╡ 50d83a01-f4c5-45c1-9c81-ed71d8313642
md"If you wonder about the ! in the function names: have a look [here](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)."

# ╔═╡ 034f7bfa-eacd-11eb-1839-371285e2c6ed
w = [1 2 3] # this is a matrix with 1 row and 3 columns

# ╔═╡ 034f7be4-eacd-11eb-1616-2df92f244e73
v[1] # access to the first element of the vector

# ╔═╡ 034f7bf0-eacd-11eb-0b3f-69f172657051
v[2] = 4 # assign value 4 to the second element of the vector

# ╔═╡ 034f7bf0-eacd-11eb-012e-997132cd41c6
v # here we check that the second element is indeed 4 now

# ╔═╡ 034f7bfa-eacd-11eb-2bf8-c795453a8959
m = [1 2 3
     4 5 6
     7 8 9] # this is a 3 x 3 matrix.

# ╔═╡ 034f7c04-eacd-11eb-3448-832bd75c3356
m[3, 2] # accessing the element in the 3 row, second column

# ╔═╡ 9c34e4ef-4eba-459f-a7ff-b20708155e65
m' # transpose the matrix

# ╔═╡ e576bf44-9cc1-4053-b40f-b82c8c39c89d
md"Any function can be applied element-wise to arrays using the [dot syntax](https://docs.julialang.org/en/v1/manual/functions/#man-vectorized)."

# ╔═╡ 034f7c0e-eacd-11eb-114f-9dbad3e74dd6
sin.(v)

# ╔═╡ 034f7c0e-eacd-11eb-1b73-27539a68f26d
exp.(m)

# ╔═╡ 034f7c16-eacd-11eb-0d0b-518b28558fda
v .+ 3

# ╔═╡ 034f7c16-eacd-11eb-2cb0-8b1198bfcb78
myfunc.(v, v)

# ╔═╡ 034f7ccc-eacd-11eb-1613-cbdf2386b60b
m * v  # matrix vector multiplication

# ╔═╡ 034f7ccc-eacd-11eb-2ab4-599e0ed86304
v' * v # inner product

# ╔═╡ 034f7cd6-eacd-11eb-37bb-f1930f2f2772
eigvals(m) # compute the Eigen values

# ╔═╡ c997a07f-bc89-4c29-9945-62046b6889d2
md"To concatenate vectors and matrices we have the following syntax (see also [here](https://docs.julialang.org/en/v1/manual/arrays/#man-array-concatenation))."

# ╔═╡ 07fd5baf-ae0b-4e74-97a6-5a6e14644122
[[1, 2, 3]; [77, 88, 99]] # the semicolon concatenates the two vectors

# ╔═╡ 11422ca4-c096-4167-aa53-de312615348e
md"The same can be achieved with"

# ╔═╡ 0ace8c81-e481-4323-9b20-116de412330a
vcat([1, 2, 3], [77, 88, 99])

# ╔═╡ 5d09ccbf-5a51-41f7-a7ef-84cf2cc5fa1f
[[1, 2, 3] [77, 88, 99]] # alternatively you can use hcat([1, 2, 3], [77, 88, 99])

# ╔═╡ 9fd6e8c1-7b84-4d76-abef-9e3da4a56dc7
md"Sometimes it is easier to construct a vector or matrix with [comprehension](https://docs.julialang.org/en/v1/manual/arrays/#man-comprehensions)."

# ╔═╡ e33c0eea-4f5c-4fc8-800d-fa16335eedd0
[i^2 for i in [3, 4, 9, 7]]

# ╔═╡ 315e5a3b-dc28-4c3b-8fc2-3442cb7590be
[i^2 for i in [3, 4, 9, 7] if isodd(i)]

# ╔═╡ 45deab46-9a8c-4d22-9460-980a3900396f
["$i, $j" for i in 1:5, j in 1:5] # see string interpolation above for the $

# ╔═╡ b91eda3c-95cb-4372-9c6b-03fa6a6b02b2
md"# Ranges

See also [here](https://docs.julialang.org/en/v1/base/math/#Base.::)."

# ╔═╡ 034f7c16-eacd-11eb-3877-6ba06a933698
collect(1:10) # 1:10 is a range; the function `collect` turns it into a vector

# ╔═╡ 034f7c22-eacd-11eb-3ab7-b30813d11e94
r1 = 1:.1:10 # a range from 1 to 10 with step-size 0.1

# ╔═╡ 034f7c2c-eacd-11eb-1aae-97b030af71e9
r3 = 10:-.1:1 # a range from 10 to 1 with step-size -0.1

# ╔═╡ 034f7c2c-eacd-11eb-1f56-052874c5c6ab
collect(r3)

# ╔═╡ 034f7c36-eacd-11eb-0188-cd307154555c
exp.(r1) # apply exp to the range

# ╔═╡ 034f7c40-eacd-11eb-26d9-956b65235a10
v[2:3] # you can use ranges to access multiple elements of a vector or matrix

# ╔═╡ 4237f5df-208d-44c4-90d7-db0efbe7b710
v[[1, 3]] # a vector of indices can also be used

# ╔═╡ 034f7c40-eacd-11eb-1beb-bb9e9bb325db
v[2:end]

# ╔═╡ 034f7c4a-eacd-11eb-0de4-537297318c85
v[begin:2]

# ╔═╡ 034f7c4a-eacd-11eb-03ed-bfc7b532501b
m[:, 2] # the colon : alone is a short-hand for begin:end

# ╔═╡ 034f7c54-eacd-11eb-2918-ef8f2befddc1
m[1:2, 2:3]

# ╔═╡ 034f7c54-eacd-11eb-2fe9-5330d3cd3cd5
function is_larger_than_10(x)
	result = ""
    for i in 1:10 # ranges are useful to construct for loops
        if x < i
            return result * " is smaller than 10"
        else
            result *= "|"
        end
    end
    result * "... is larger than 10"
end

# ╔═╡ 034f7c5e-eacd-11eb-24c6-d7e7213a7a5e
is_larger_than_10(4)

# ╔═╡ 034f7c5e-eacd-11eb-1ff7-55f09fb796fe
is_larger_than_10(78)

# ╔═╡ d9ad1d6d-a3b5-4a83-bda9-05ee43ded9d6
md"# Tuples

See also [here](https://docs.julialang.org/en/v1/manual/functions/#Tuples)."

# ╔═╡ 034f7c68-eacd-11eb-1e11-d77379c677c8
t = (2, 3, 4., "bla")

# ╔═╡ 034f7c68-eacd-11eb-12e6-51022c54f6ed
typeof(t)

# ╔═╡ 034f7c72-eacd-11eb-28a5-d7d6f3b9f75a
t[1]

# ╔═╡ 034f7c72-eacd-11eb-1cc2-91eaa2de5e25
t[4]

# ╔═╡ 034f7c7c-eacd-11eb-1aaa-297c951dd8cd
nt = (year = 1789,
	  event = "French Revolution",
	  slogan = "liberté, égalité, fraternité")

# ╔═╡ 034f7c88-eacd-11eb-28f2-c1c7e05279c6
typeof(nt)

# ╔═╡ 034f7c88-eacd-11eb-14fe-b3d49ff91622
nt.year

# ╔═╡ 034f7c90-eacd-11eb-0d1c-5dbd7b62d55c
nt.slogan

# ╔═╡ df8588cb-400b-4423-9766-e6cac2c9717a
md"# DataFrames

DataFrames are tables with named columns.
We will constantly use DataFrames to organize data.
You can find [here](https://dataframes.juliadata.org/stable/man/comparisons/) a list with useful commands. There is also a [cheat sheet](https://www.ahsmart.com/assets/pages/data-wrangling-with-data-frames-jl-cheat-sheet/DataFramesCheatSheet_v1.x_rev1.pdf).
See also [the documentation of the DataFrames.jl package](https://dataframes.juliadata.org/stable/)."

# ╔═╡ 034f7c9a-eacd-11eb-2e85-29c5bffbc39a
df = DataFrame(year = [1789, 1863],
               event = ["French Revolution", "Foundation of Red Cross"],
               slogan = ["liberté, égalité, fraternité", missing])

# ╔═╡ 034f7cae-eacd-11eb-39dc-ab39e97cc716
push!(df, [1789,
           "George Washington becomes first president of the USA",
           "Deeds, not Words."]) # append a new row to the already created dataframe

# ╔═╡ 034f7c9a-eacd-11eb-3aee-0bab7ab5d502
df.year

# ╔═╡ 034f7ca4-eacd-11eb-2f3b-c1dd9be6ad84
df[:, 1] # we can also access columns like for matrices

# ╔═╡ 034f7ca4-eacd-11eb-054e-1f930c99d304
dropmissing(df) # removes all rows with missing entries.

# ╔═╡ 034f7cae-eacd-11eb-38c0-57df4ba09992
df[1, :] # show the first row

# ╔═╡ 034f7cc2-eacd-11eb-0f0f-ff16d1ac454b
df[df.year .== 1789, [:year, :slogan]] # select all rows where year == 1789 and columns year and slogan

# ╔═╡ 034f7cf4-eacd-11eb-302f-55b164df3b90
md"# Random Numbers

See also [here](https://docs.julialang.org/en/v1/stdlib/Random/).
"

# ╔═╡ 034f7cf4-eacd-11eb-302f-55b164df3b9f
rand() # a sample from the uniform distribution over the interval [0, 1).

# ╔═╡ 034f7cf4-eacd-11eb-003c-133f34cca508
rand() # different number than in the cell above.

# ╔═╡ 034f7cfe-eacd-11eb-2514-4b6fdbc42e8f
rand(10)

# ╔═╡ 034f7d08-eacd-11eb-2694-51c1baa997e6
rand(3, 3)

# ╔═╡ 034f7d12-eacd-11eb-1673-c94969427bbf
randn() # a sample from the normal distribution with mean 0 and variance 1

# ╔═╡ 034f7d12-eacd-11eb-261c-a5de2f06f420
rand((:bla, "bli", 3, 1.2))

# ╔═╡ c142c7ba-3ea4-4c1b-834f-90bc6f5be36a
md"When functions like `rand` or `randn` are called, a [pseudo-random-number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) works in the background to produce numbers that look random. When we want to assure reproducibility, it is best to explicitly define and use the (pseudo-)random-number generator. Standard choices are the `Xoshiro` and the `MersenneTwister` generators."

# ╔═╡ 034f7d1a-eacd-11eb-1e34-21d5eef62940
rng = Xoshiro(123) # creates a Xoshiro pseudo-random-number generator with seed 123

# ╔═╡ 034f7d26-eacd-11eb-066c-538e5b614536
rand(rng)

# ╔═╡ 07decfab-875e-4a02-8893-a11e494df943
rand(rng)

# ╔═╡ 034f7d30-eacd-11eb-081f-970c4daee184
Random.seed!(rng, 123) # resets the seed

# ╔═╡ 034f7d30-eacd-11eb-0b09-7b3b9ac47570
rand(rng) # same result as when `rand()` was called the first time.

# ╔═╡ 313bf559-eca7-4eb9-bd49-c8f4b1b8234c
md"As before, we can use this random number generator to sample from other distributions:"

# ╔═╡ 7dd9b949-464a-4db9-ad61-b0753b68b92e
rand(rng, (:bla, "bli", 3, 1.2))

# ╔═╡ 0f6bd75f-34dc-4793-b0e8-6e54950aeff6
randn(rng)

# ╔═╡ 61c2ebc7-6586-4d5b-afd1-ed2b27964ebc
rand(rng, 3, 3)

# ╔═╡ 034f7d3a-eacd-11eb-3fbe-6dc4ab1efd50
md"# Plotting

See also the [documentation of the Plots.jl package](http://docs.juliaplots.org/latest/)
"

# ╔═╡ 034f7d3a-eacd-11eb-3fbe-6dc4ab1efd56
plot(1:4, rand(4))

# ╔═╡ 034f7d3a-eacd-11eb-1ef3-4f89b8e5aee9
scatter(rand(100), rand(100))

# ╔═╡ 034f7d44-eacd-11eb-0ab2-2f0d2caed9d6
plotattr() # get some help on plotting attributes

# ╔═╡ 034f7d4c-eacd-11eb-3960-d10d1c16075d
plotattr(:Series)

# ╔═╡ 034f7d4c-eacd-11eb-3841-97ca37e34c9f
plotattr(:Series, "label")

# ╔═╡ 034f7d58-eacd-11eb-123f-9b563a7ae27e
scatter(rand(100), rand(100), label = "my data", xlabel = "X1", ylabel = "X2")

# ╔═╡ 4ca1cd98-a24b-491f-b9ac-28a7daf96d50
md"To compose figures with multiple elements, you can use the `!` version of the plotting functions"

# ╔═╡ 65e73d27-3709-4894-87ab-432343d314c6
begin # we use begin to write a cell with multiple lines
	scatter(1:50, sqrt.(1:50) .+ randn(50), label = "data points")
	plot!(sqrt, color = :red, label = "square root function")
end

# ╔═╡ 9f69c2ab-2940-40ce-b164-07bfa6cc1697
md"Instead of `begin`-`end`-blocks we will also use `let`-`end`-blocks. Variables computed in `let`-blocks are local to that cell, whereas variables defined in `begin`-blocks are global and can be accessed by other cells. (With the `;` at the end of the cell we suppress the output)."

# ╔═╡ e7ed9e83-7f17-4d9a-a49b-e87bc0747035
begin
	a_global_variable = 17
	another_global_variable = "Hi"
end;

# ╔═╡ 0d66592a-d098-4ab5-9323-503db8fc73f6
a_global_variable * 3

# ╔═╡ a9a3f29e-4931-4464-9bc5-af6598815f24
"$another_global_variable there!"

# ╔═╡ 6e65247f-b285-49e5-a16e-31b4541225bf
let
	a_local_variable = 17
	another_local_variable = "Hi"
end;

# ╔═╡ fcd77620-f18a-499b-87c4-fffaa876e59f
a_local_variable * 3

# ╔═╡ 034f7d62-eacd-11eb-3840-51c0622eb8a0
md"# Distributions

See also the [documentation of the Distributions.jl package](https://juliastats.org/Distributions.jl/stable/)
"

# ╔═╡ 034f7d62-eacd-11eb-3840-51c0622eb8a9
d1 = Normal(3, 2.5)

# ╔═╡ 034f7d6c-eacd-11eb-0eb6-277bf88c13d3
rand(d1) # draw a sample from this normal distribution

# ╔═╡ 034f7d6c-eacd-11eb-2792-d773a2c77cb6
rand(d1, 10) # draw 10 samples from this distriution

# ╔═╡ 034f7d6c-eacd-11eb-0dc0-3bd2b5a24d38
mean(rand(d1, 10^5))

# ╔═╡ 034f7d76-eacd-11eb-141d-0f90eef73f08
std(rand(d1, 10^5)) # standard deviation

# ╔═╡ 034f7d7e-eacd-11eb-0945-2dda6be01349
d2 = Bernoulli(.7)

# ╔═╡ 034f7d7e-eacd-11eb-2a1b-933e1231c220
rand(d2, 10)

# ╔═╡ 034f7d8a-eacd-11eb-1dc0-d99d481ab6fa
md"# Other Packages

Here are some other popular Julia packages. They are not relevant for this course, but maybe you want to explore them at some point.
* [Symbolics](https://github.com/JuliaSymbolics/Symbolics.jl)
* [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl)
* [Optim](https://github.com/JuliaNLSolvers/Optim.jl)
* [Images](https://github.com/JuliaImages/Images.jl)
* [Turing](https://github.com/TuringLang/Turing.jl)
"

# ╔═╡ 4a03cfae-9876-4cf0-a498-d750853191cb
md"""# Exercises

#### Exercise 1
Create a data frame with 3 columns named A, B and C.
   1. Column A contains 5 random numbers sampled from a Bernoulli distribution with rate 0.3, column B contains 5 random numbers from the uniform distribution over the interval [0, 1), and column C contains 5 samples from the set `(:hip, :hop)`.
   2. Create a vector whose i'th element contains the sum of the i'th entries of columns A and B of the data frame created in 1.
   3. Select all rows with `:hop` in column C and display the resulting data frame.

"""

# ╔═╡ 2944de86-82fd-4409-8504-f5ea3deae21d
md"""
#### Exercise 2
   Very often the first version of some code does not run as it should. Therefore we need good debugging strategies. For this it is important to know how to read error messages. In this exercise we will produce different error messages and try to interpret them.
   1. Write `longfunc(1, [1, 2, 3])` in a new cell and read the error message. At the bottom of this error message you can see where the error occurred. You can click on the yellowish field with text `Other: 2` to jump to the relevant code. Based on this you should know now that `x = 1` and `y = [1, 2, 3]` are tried to be added in `longfunc`. Let us now look at the first line of the error message: it says `no method matching +(::Int64, ::Vector{Int64})`, which means that Julia doesn't know how to add the integer `x = 1` to the vector `y = [1, 2, 3]`. Now there are multiple ways to fix the error, depending on what you want. If you wanted to add `1` to every element of the vector, you could modify `longfunc` such that `tmp = x .+ y`. Try this fix. You will run into another error. Can you also fix the next error?
   2. Create a cell with content `A = []`, another cell with `mean(x) = sum(x)/length(x)`, and a third cell with content `mean(A)`. This produces a scary error message :). Usually you can ignore all the lines that do not have a link to jump to some code (e.g. you can ignore the line `1. zero(::Type{Any})` etc.; it is telling you that the error occurred somewhere in the `zero` function which is a builtin function of Julia). What matters is the line `mean(::Vector{Any})` which tells you that you called the function `mean` with a vector of type `Any` (this is the type of the empty vector `A = []`). You can get rid of the error by changing e.g. `A = [1, 2, 3]`.
   3. Create a cell with content `f(x) = longfunc(x)` and another one with content `f([2, 3])`. This produces another commonly occuring error message, including a hint about closest alternatives. If you carefully look at the error message `longfunc(::Any, !Matched::Any)` you see that the function `longfunc` did not get a second argument (this is the meaning of `!Matched`). You can fix this e.g. with `f(x) = longfunc(x[1], x[2])`.
   4. By now you should be (close to) a debugging wizzard and you can find fixes to all bugs in the following lines of code (copy them over to a new cell). The goal is just to have something that runs without errors; the result does not need to be meaningful :) *Hint:* Click on the topmost clickalble link in the error messages to jump to the line of code that creates the error.
```julia
begin
   function full_of_bugs(x)
       tmp = [myfunc(3); x]
       tmp2 = tmp^2
       tmp3 = tmp2 + log(sum(x))
       sqrt(tmp3 - length(x))
   end
   full_of_bugs([2, 1, -5])
end
```
"""

# ╔═╡ e15d86d2-6902-4c8a-90ca-b4ed60fbab1d
md"""
#### Exercise 3
The state of a Pluto notebook is usually consistent, because cells `B, C, …` that depend on a given cell `A` are reevaluated when the code or the result of cell `A` changes. However, there is one thing that can lead to unexpected behavior: functions that mutate their argument (for the afficionados, Julia uses an evaluation strategy called [call by sharing](https://en.wikipedia.org/wiki/Evaluation_strategy#Call_by_sharing)). Here is an example:
```julia
function multiply_by_two!(x)
    x .*= 2
end
```
1. Paste this function to a new cell.
2. In another new cell define `my_vector = [1, 2, 3, 4]`.
3. In the next cell run `multiply_by_two!(my_vector)`.
4. In the next cell run `my_vector` to simply show it's value.
5. Run the cell in step 3 multiple times. Does the cell in step 4 automatically show the updated value? What happens when you re-run the cell in step 4?
6. A related, somewhat unexpected behavior can be observed with pseudo-random-number generators. Paste the code `rng2 = MersenneTwister(123)` in a cell and in the next cell `rand(rng2)`. Run the second cell multiple times. What do you observe? What happens when you rerun the cell that defines the pseudo-random-number generator?

Remember, the exclamation mark at the end of the function name is a [convention in Julia](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention) to remind the programmer that the argument may change.
"""

# ╔═╡ dbe2c72c-bbc6-4912-af98-e8f473b7ac27
md"""
#### Exercise 4
   1. Use comprehension (see "Vectors, Matrices, Arrays") to create a vector with all numbers of the form ``x^y`` with ``x=1, \ldots, 10``, ``y = 2, \ldots, 7`` and ``y > x``.
   2. Compute the sum of the square root of these numbers.
"""

# ╔═╡ 2c527d8d-5e11-4089-ad32-355fc7ac5e10
md"""
#### Exercise 5
   1. Write a function that returns the smallest entry of a vector (without using the built-in function `minimum`, `argmin` or `findmin`).
   2. Test your function on a vector of 10 randomly sampled integers in the range 1 to 100.
"""

# ╔═╡ af94fccd-bb3e-498a-8d2a-f8e75740cd29
md"""
#### Exercise 6
   1. Plot the `cos` function on the interval 0 to 4π. Hint: type `\pi + [Tab]` to enter the symbol π. To learn how to place custom tick labels on the x-axis, type `xticks` in a cell and open the \"Live docs\" at the bottom-right.
   2. Add a scatter plot with 100 points whose ``x`` and ``y`` coordinates are randomly sampled from the interval ``[0, 1)`` on top of the figure with the cosine.
"""

# ╔═╡ 0314376e-ff8c-4ad0-8a4b-f94f04f31f2c
MLCourse.footer()

# ╔═╡ Cell order:
# ╟─f3508747-da29-47c9-a98e-22ea15caaf2f
# ╠═065a1e67-6b63-43df-9d6d-303af08d8434
# ╟─d92c0729-c7ab-42cc-b713-30f00e237833
# ╠═b8630ee9-d2c0-4cd0-b3f7-77f66e997a80
# ╠═034f7b46-eacd-11eb-1aa2-f380d08206cc
# ╠═034f7b64-eacd-11eb-1618-6de6def66405
# ╠═034f7b6e-eacd-11eb-1733-650ad4da2730
# ╠═034f7b78-eacd-11eb-003a-5b5801c19d80
# ╠═034f7b6e-eacd-11eb-184a-d504fab31681
# ╠═fb3677a9-1610-49dd-96e4-3bd7badc58db
# ╠═884f568b-079c-47d6-a597-34b67925f195
# ╠═034f7b84-eacd-11eb-2836-11cebedb05b9
# ╟─52efdece-3c21-43d4-a826-3b96e3a23e60
# ╠═034f7ba0-eacd-11eb-3fc1-9dca0e072755
# ╠═034f7ba0-eacd-11eb-23ae-cd327563ec60
# ╠═034f7ba0-eacd-11eb-3347-49deb13761d9
# ╟─121b3dbb-f86c-44b8-8b0f-f6b4d0b9992f
# ╠═034f7baa-eacd-11eb-33f2-5f619e4a7832
# ╠═034f7bb2-eacd-11eb-006b-7b935e35af66
# ╠═034f7bb2-eacd-11eb-214b-cbfc83965c26
# ╠═ec1dd4c4-d61d-4fdb-b576-3fe3b059eb3a
# ╠═1a3387ee-b59d-49d7-9cca-2a8bd04eaaad
# ╠═034f7b96-eacd-11eb-0c81-ad33210de120
# ╠═034f7b96-eacd-11eb-2ee1-cd8e6813a726
# ╟─0f4e3f45-71dd-41b2-a0b3-c55a04ace61f
# ╠═034f7bbe-eacd-11eb-2e95-a93ad2d5e867
# ╠═034f7bbe-eacd-11eb-2c36-d3949367ce5a
# ╠═e0ec7ba3-4858-4572-994d-953caa98080b
# ╠═04da39a1-e3bc-4086-b555-6cea2cf1689d
# ╠═034f7bc8-eacd-11eb-0396-5b743b505b2a
# ╠═034f7bc8-eacd-11eb-31f9-c119d1dfb8a5
# ╟─ce3c7687-c7a9-4b5d-9a55-51514ed90fd9
# ╠═034f7bd2-eacd-11eb-28f3-b707012e1db0
# ╠═034f7bd2-eacd-11eb-1789-7380a89d5339
# ╟─034f7cd6-eacd-11eb-0f8c-11179c3ab480
# ╠═034f7cd6-eacd-11eb-0f8c-11179c3ab48f
# ╠═034f7ce0-eacd-11eb-3c49-e39b8d3e917f
# ╠═034f7ce0-eacd-11eb-18a1-bd2d1849b9bc
# ╠═034f7cec-eacd-11eb-3262-bdc0f2e13a3a
# ╠═034f7cec-eacd-11eb-27da-9bb861643df2
# ╟─832ce046-ab34-4623-9d02-793dc5748208
# ╠═91b0de0c-221c-468e-9077-5233aedeb4ca
# ╠═0c7ae3ec-a40f-4e3e-85d8-3f79f091e3b4
# ╟─732ae082-b90f-4108-9872-36077fb2c54c
# ╠═6bfb60b4-964e-46ec-adc2-ec9e9a0f8158
# ╟─0efb1f1a-8051-4843-aa68-1f4ad8618f92
# ╠═034f7bdc-eacd-11eb-1999-9b62efdb4128
# ╠═034f7cba-eacd-11eb-2f1a-836ccc185dd9
# ╠═034f7cba-eacd-11eb-3901-2bc3d0ff0e39
# ╟─50d83a01-f4c5-45c1-9c81-ed71d8313642
# ╠═034f7bfa-eacd-11eb-1839-371285e2c6ed
# ╠═034f7be4-eacd-11eb-1616-2df92f244e73
# ╠═034f7bf0-eacd-11eb-0b3f-69f172657051
# ╠═034f7bf0-eacd-11eb-012e-997132cd41c6
# ╠═034f7bfa-eacd-11eb-2bf8-c795453a8959
# ╠═034f7c04-eacd-11eb-3448-832bd75c3356
# ╠═9c34e4ef-4eba-459f-a7ff-b20708155e65
# ╟─e576bf44-9cc1-4053-b40f-b82c8c39c89d
# ╠═034f7c0e-eacd-11eb-114f-9dbad3e74dd6
# ╠═034f7c0e-eacd-11eb-1b73-27539a68f26d
# ╠═034f7c16-eacd-11eb-0d0b-518b28558fda
# ╠═034f7c16-eacd-11eb-2cb0-8b1198bfcb78
# ╠═034f7ccc-eacd-11eb-1613-cbdf2386b60b
# ╠═034f7ccc-eacd-11eb-2ab4-599e0ed86304
# ╠═034f7cc2-eacd-11eb-29e2-810a573295cf
# ╠═034f7cd6-eacd-11eb-37bb-f1930f2f2772
# ╟─c997a07f-bc89-4c29-9945-62046b6889d2
# ╠═07fd5baf-ae0b-4e74-97a6-5a6e14644122
# ╟─11422ca4-c096-4167-aa53-de312615348e
# ╠═0ace8c81-e481-4323-9b20-116de412330a
# ╠═5d09ccbf-5a51-41f7-a7ef-84cf2cc5fa1f
# ╟─9fd6e8c1-7b84-4d76-abef-9e3da4a56dc7
# ╠═e33c0eea-4f5c-4fc8-800d-fa16335eedd0
# ╠═315e5a3b-dc28-4c3b-8fc2-3442cb7590be
# ╠═45deab46-9a8c-4d22-9460-980a3900396f
# ╟─b91eda3c-95cb-4372-9c6b-03fa6a6b02b2
# ╠═034f7c16-eacd-11eb-3877-6ba06a933698
# ╠═034f7c22-eacd-11eb-3ab7-b30813d11e94
# ╠═034f7c2c-eacd-11eb-1aae-97b030af71e9
# ╠═034f7c2c-eacd-11eb-1f56-052874c5c6ab
# ╠═034f7c36-eacd-11eb-0188-cd307154555c
# ╠═034f7c40-eacd-11eb-26d9-956b65235a10
# ╠═4237f5df-208d-44c4-90d7-db0efbe7b710
# ╠═034f7c40-eacd-11eb-1beb-bb9e9bb325db
# ╠═034f7c4a-eacd-11eb-0de4-537297318c85
# ╠═034f7c4a-eacd-11eb-03ed-bfc7b532501b
# ╠═034f7c54-eacd-11eb-2918-ef8f2befddc1
# ╠═034f7c54-eacd-11eb-2fe9-5330d3cd3cd5
# ╠═034f7c5e-eacd-11eb-24c6-d7e7213a7a5e
# ╠═034f7c5e-eacd-11eb-1ff7-55f09fb796fe
# ╟─d9ad1d6d-a3b5-4a83-bda9-05ee43ded9d6
# ╠═034f7c68-eacd-11eb-1e11-d77379c677c8
# ╠═034f7c68-eacd-11eb-12e6-51022c54f6ed
# ╠═034f7c72-eacd-11eb-28a5-d7d6f3b9f75a
# ╠═034f7c72-eacd-11eb-1cc2-91eaa2de5e25
# ╠═034f7c7c-eacd-11eb-1aaa-297c951dd8cd
# ╠═034f7c88-eacd-11eb-28f2-c1c7e05279c6
# ╠═034f7c88-eacd-11eb-14fe-b3d49ff91622
# ╠═034f7c90-eacd-11eb-0d1c-5dbd7b62d55c
# ╟─df8588cb-400b-4423-9766-e6cac2c9717a
# ╠═034f7c90-eacd-11eb-1b40-8722209689fa
# ╠═034f7c9a-eacd-11eb-2e85-29c5bffbc39a
# ╠═034f7cae-eacd-11eb-39dc-ab39e97cc716
# ╠═034f7c9a-eacd-11eb-3aee-0bab7ab5d502
# ╠═034f7ca4-eacd-11eb-2f3b-c1dd9be6ad84
# ╠═034f7ca4-eacd-11eb-054e-1f930c99d304
# ╠═034f7cae-eacd-11eb-38c0-57df4ba09992
# ╠═034f7cc2-eacd-11eb-0f0f-ff16d1ac454b
# ╟─034f7cf4-eacd-11eb-302f-55b164df3b90
# ╠═034f7cf4-eacd-11eb-302f-55b164df3b9f
# ╠═034f7cf4-eacd-11eb-003c-133f34cca508
# ╠═034f7cfe-eacd-11eb-2514-4b6fdbc42e8f
# ╠═034f7d08-eacd-11eb-2694-51c1baa997e6
# ╠═034f7d12-eacd-11eb-1673-c94969427bbf
# ╠═034f7d12-eacd-11eb-261c-a5de2f06f420
# ╠═034f7d1a-eacd-11eb-1d48-591daf597cd6
# ╟─c142c7ba-3ea4-4c1b-834f-90bc6f5be36a
# ╠═034f7d1a-eacd-11eb-1e34-21d5eef62940
# ╠═034f7d26-eacd-11eb-066c-538e5b614536
# ╠═07decfab-875e-4a02-8893-a11e494df943
# ╠═034f7d30-eacd-11eb-081f-970c4daee184
# ╠═034f7d30-eacd-11eb-0b09-7b3b9ac47570
# ╟─313bf559-eca7-4eb9-bd49-c8f4b1b8234c
# ╠═7dd9b949-464a-4db9-ad61-b0753b68b92e
# ╠═0f6bd75f-34dc-4793-b0e8-6e54950aeff6
# ╠═61c2ebc7-6586-4d5b-afd1-ed2b27964ebc
# ╟─034f7d3a-eacd-11eb-3fbe-6dc4ab1efd50
# ╠═034f7d3a-eacd-11eb-1144-65f9acfcf24c
# ╠═034f7d3a-eacd-11eb-3fbe-6dc4ab1efd56
# ╠═034f7d3a-eacd-11eb-1ef3-4f89b8e5aee9
# ╠═034f7d44-eacd-11eb-0ab2-2f0d2caed9d6
# ╠═034f7d4c-eacd-11eb-3960-d10d1c16075d
# ╠═034f7d4c-eacd-11eb-3841-97ca37e34c9f
# ╠═034f7d58-eacd-11eb-123f-9b563a7ae27e
# ╟─4ca1cd98-a24b-491f-b9ac-28a7daf96d50
# ╠═65e73d27-3709-4894-87ab-432343d314c6
# ╟─9f69c2ab-2940-40ce-b164-07bfa6cc1697
# ╠═e7ed9e83-7f17-4d9a-a49b-e87bc0747035
# ╠═0d66592a-d098-4ab5-9323-503db8fc73f6
# ╠═a9a3f29e-4931-4464-9bc5-af6598815f24
# ╠═6e65247f-b285-49e5-a16e-31b4541225bf
# ╠═fcd77620-f18a-499b-87c4-fffaa876e59f
# ╟─034f7d62-eacd-11eb-3840-51c0622eb8a0
# ╠═034f7d58-eacd-11eb-3189-bff16ec27739
# ╠═034f7d62-eacd-11eb-3840-51c0622eb8a9
# ╠═034f7d6c-eacd-11eb-0eb6-277bf88c13d3
# ╠═034f7d6c-eacd-11eb-2792-d773a2c77cb6
# ╠═034f7d6c-eacd-11eb-0dc0-3bd2b5a24d38
# ╠═034f7d76-eacd-11eb-24cc-fd8d53d736c2
# ╠═034f7d76-eacd-11eb-141d-0f90eef73f08
# ╠═034f7d7e-eacd-11eb-0945-2dda6be01349
# ╠═034f7d7e-eacd-11eb-2a1b-933e1231c220
# ╟─034f7d8a-eacd-11eb-1dc0-d99d481ab6fa
# ╟─4a03cfae-9876-4cf0-a498-d750853191cb
# ╟─2944de86-82fd-4409-8504-f5ea3deae21d
# ╟─e15d86d2-6902-4c8a-90ca-b4ed60fbab1d
# ╟─dbe2c72c-bbc6-4912-af98-e8f473b7ac27
# ╟─2c527d8d-5e11-4089-ad32-355fc7ac5e10
# ╟─af94fccd-bb3e-498a-8d2a-f8e75740cd29
# ╟─d6e5fe02-21a5-486c-a237-878be1d95439
# ╟─2b67233e-76ba-40be-81e2-787fbe5d3641
# ╟─0314376e-ff8c-4ad0-8a4b-f94f04f31f2c
