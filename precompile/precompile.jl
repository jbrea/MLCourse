using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using PackageCompiler
PackageCompiler.create_sysimage([:Plots, :Pluto, :PlutoUI, :MLJ, :Distributions,
                                 :Flux, :Zygote, :MLJLinearModels, :Random,
                                 :NearestNeighborModels, :Statistics, :LinearAlgebra,
                                 :DataFrames, :CSV, :StatsPlots];
                                sysimage_path=joinpath(@__DIR__, "mlcourse.so"),
                                precompile_execution_file=joinpath(@__DIR__, "warmup.jl"))
