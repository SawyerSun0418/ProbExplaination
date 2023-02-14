using CSV
using MLBase
#using StatsBase
using DataFrames
using CategoricalArrays
using FreqTables
using Random
#using ScikitLearn
using Flux
using MLUtils
using BSON: @save
using BSON: @load
#@sk_import linear_model: LogisticRegression

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

#= function train_LR()
    df = DataFrame(CSV.File("data/data.csv"))
    select!(df, Not([:id]))
    for column in names(df)
        if column!="diagnosis"
            df[!,column]=binvec(df[!,column],5)
        end
    end
    train, test = splitdf(df, 0.6);
    train=Matrix(train)
    test=Matrix(test)
    model = LogisticRegression(fit_intercept=true,max_iter=500)
    y_train=train[:,1]
    X_train=train[:,2:end]
    y_test=test[:,1]
    X_test=test[:,2:end]
    ScikitLearn.fit!(model, X_train, y_train)   
    prediction = ScikitLearn.predict(model, X_test)
    prediction_class = [if x < 0.5 0 else 1 end for x in prediction];
  
    prediction_df = DataFrame(y_actual = y_test,y_predicted = prediction_class, prob_predicted = ScikitLearn.predict_proba(model, X_test)[:,2]);
    prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
    accuracy = mean(prediction_df.correctly_classified)
    #temp=sum(ScikitLearn.predict_proba(model, X_test)[:,2])
    #println(temp)
    println(accuracy)
    #println(first(prediction_df,2))
    #instance=Array(test[test[:,1] .== 0,:])
    #i=instance[2,:]
    #popfirst!(i)
    #println(i)
    return model
end =#

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
    X_train=permutedims(Matrix(DataFrame(CSV.File("data/adult/x_train_oh.csv"))))
    y_test=Matrix(DataFrame(CSV.File("data/adult/y_test.csv")))
    X_test=Matrix(DataFrame(CSV.File("data/adult/x_test_oh.csv")))
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end

function train_NN_flux()
    df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
    train_data, test_data=data_pre(df)
    model = Chain(
        Dense(784=>128, relu),
        Dense(128=>10, relu),
        Dense(10=>1, sigmoid)
        )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()
    temp=0
    history=0
    h_a=0
    for epochs in 1:500
        Flux.train!(loss, Flux.params(model), train_data, opt)
        a=test(model,test_data)
        if a>0.9
            if a > h_a
                history=model
                h_a=a
                temp=0
            else temp+=1 end

            if temp>=10
                return history, test_data
            end
        end
    end
    println("failed")
    return model, test_data
end

function train_LR_flux()
    df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
    train_data, test_data=data_pre(df)
    model = Chain(
        Dense(784 => 1, sigmoid)
        )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()
    Flux.@epochs 250 Flux.train!(loss, Flux.params(model), train_data, opt)
    
    return model, test_data
end

function accuracy(x,y,model)
    prediction_class = [if i< 0.5 0 else 1 end for i in model(permutedims(x))];
    count=0
    s=size(y)[1]
    for i in 1:s
        if prediction_class[i]==y[i]
            count+=1
        end
    end
    a = count/s
    return a
end

function test(model, test)
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)
    println("\nAccuracy: $accuracy_score")
    return accuracy_score
end

function save_model()
    model, test_data=train_NN_flux()
    a=test(model,test_data)
    if a > 0.9
        @save "src/model/flux_NN_MNIST_new.bson" model
    else
        println("not accurate")
    end
end

function load_model(model_add)
    @load model_add model
    return model
end
