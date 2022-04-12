using CSV
# using ROCAnalysis
using MLBase
using StatsBase
using GLM
using Plots
# using Lathe
using DataFrames
using CategoricalArrays
using FreqTables
using Random

function binvec(x::AbstractVector, n::Int,
    rng::AbstractRNG=Random.default_rng())
    n > 0 || throw(ArgumentError("number of bins must be positive"))
    l = length(x)

    # find bin sizes
    d, r = divrem(l, n)
    lens = fill(d, n)
    lens[1:r] .+= 1
    # randomly decide which bins should be larger
    shuffle!(rng, lens)

    # ensure that we have data sorted by x, but ties are ordered randomly
    df = DataFrame(id=axes(x, 1), x=x, r=rand(rng, l))
    sort!(df, [:x, :r])

    # assign bin ids to rows
    binids = reduce(vcat, [fill(i, v) for (i, v) in enumerate(lens)])
    df.binids = binids

    # recover original row order
    sort!(df, :id)
    return df.binids
end




function return_df()    
    df = DataFrame(CSV.File("data.csv"))
    select!(df, Not([:id]))
    for column in names(df)
        if column!="diagnosis"
            df[!,column]=binvec(df[!,column],5)
        end
    end
    return df
end