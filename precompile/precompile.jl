using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using PackageCompiler
PackageCompiler.create_sysimage([
                                 :Pluto,
                                 :MLJ,
                                 :MLJLinearModels,
#                                  :DataFrames,
#                                  :Plots,
#                                  :PlutoUI, :Distributions,
#                                  :Flux, :Zygote, :Random,
#                                  :NearestNeighborModels, :Statistics, :LinearAlgebra,
#                                  :DataFrames, :CSV, :StatsPlots
                                ];
                                 replace_default = true,
#                                 sysimage_path=joinpath(@__DIR__, "mlcourse.so"),
                                precompile_execution_file=joinpath(@__DIR__, "warmup.jl"))
