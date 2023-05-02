using CSV
# using ROCAnalysis
using MLBase
using StatsBase
#using Plots
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

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function data_pre(df; indices = [])
    train, test = splitdf(df, 0.6);
    train=Matrix(train)
    test=Matrix(test)
    y_train=permutedims(train[:,1])
    X_train=indices == [] ? permutedims(train[:,2:end]) : permutedims(train[:,indices])
    y_test=test[:,1]
    X_test=indices == [] ? test[:,2:end] : test[:,indices]
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end



function return_df()    
    df = DataFrame(CSV.File("data/data.csv"))
    select!(df, Not([:id]))
    for column in names(df)
        if column!="diagnosis"
            df[!,column]=binvec(df[!,column],5)
        end
    end
    return df
end

function return_MNIST_df()    
    df = DataFrame(CSV.File("data/mnist_3_5_train.csv"))
    return df
end

function return_MNIST_df_t()    
    df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
    return df
end