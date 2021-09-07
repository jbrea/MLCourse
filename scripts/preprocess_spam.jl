using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using TextAnalysis, DataFrames, CSV, Random
spamdata = DataFrame(label = String[], text = String[])
for i in 3:4 # 1:6
    olddir = pwd()
    dir = tempname()
    mkdir(dir)
    cd(dir)
    file = joinpath(dir, "enron$i.tar.gz")
    download("http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron$i.tar.gz", file)
    run(`tar -xzvf $file`)
    for l in ["spam", "ham"]
        for f in readdir(joinpath(dir, "enron$i", l))
            s = read(joinpath(dir, "enron$i", l, f), String)[9:end]
            s = replace(s, "\n" => " ")
            s = replace(s, "\r" => " ")
            sd = StringDocument(s)
            remove_corrupt_utf8!(sd)
            prepare!(sd, strip_html_tags | strip_numbers | strip_punctuation)
            remove_case!(sd)
            stem!(sd)
            push!(spamdata, [l, text(sd)])
        end
    end
    cd(olddir)
    rm(dir, recursive = true)
end
spamdata = spamdata[shuffle(1:nrow(spamdata)), :]

const DATADIR = joinpath(@__DIR__, "..", "data")

CSV.write(joinpath(DATADIR, "spam.csv"), spamdata)
