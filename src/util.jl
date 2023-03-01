using CSV
using DataFrames
using Random
function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

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

function data_pre(df)
    train, test = splitdf(df, 0.6);
    train=Matrix(train)
    test=Matrix(test)
    y_train=permutedims(train[:,1])
    X_train=permutedims(train[:,2:end])
    y_test=test[:,1]
    X_test=test[:,2:end]
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end

function data_pre_adult()
    y_train=permutedims(Matrix(DataFrame(CSV.File("data/adult/y_train.csv"))))
    X_train=permutedims(Matrix(DataFrame(CSV.File("data/adult/x_train.csv"))))
    y_test=Matrix(DataFrame(CSV.File("data/adult/y_test.csv")))
    X_test=Matrix(DataFrame(CSV.File("data/adult/x_test.csv")))
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end