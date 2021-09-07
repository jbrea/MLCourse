using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CSV, DataFrames

const DATADIR = joinpath(@__DIR__, "..", "data")
d1 = CSV.read(joinpath(DATADIR, "order_95966_data.txt"),
              DataFrame, missingstring = "-") |> dropmissing
d2 = CSV.read(joinpath(DATADIR, "order_95967_data.txt"),
              DataFrame, missingstring = "-") |> dropmissing
d = vcat(d1, d2)
rename!(d, :fu3010h1 => :wind_peak,
           :prestah0 => :pressure,
           :tre200h0 => :temperature,
           :rre150h0 => :precipitation,
           :sre000h0 => :sunshine_duration,
           :fu3010h0 => :wind_mean,
           :dkl010h0 => :wind_direction)
gs = [rename!(DataFrame(g), [n => "$(g.stn[1])_$n"
                             for n in names(g)[3:end]]...)[:, 2:end]
      for g in groupby(d, :stn)]
df = innerjoin(gs..., on = :time) |> dropmissing
train = df.time .< 2018000000;
CSV.write(joinpath(DATADIR, "weather2015-2018.csv"), df[train, :])
CSV.write(joinpath(DATADIR, "weather2019-2020.csv"), df[(!).(train), :])
