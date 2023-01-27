using CSV
#using ROCAnalysis
using MLBase
using StatsBase
#using Plots
#using Lathe
using DataFrames
using CategoricalArrays
using FreqTables
using Random
#using Lathe.preprocess: TrainTestSplit
#using MLJ
using ScikitLearn
using Flux
using MLUtils
using BSON: @save
using BSON: @load
@sk_import linear_model: LogisticRegression
include("./dataframe.jl")

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end


function train_LR()
    df=return_df()
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
end

function data_pre()
    df=return_MNIST_df()
    train, test = splitdf(df, 0.6);
    train=Matrix(train)
    test=Matrix(test)
    y_train=train[:,1]'
    X_train=train[:,2:end]'
    y_test=test[:,1]
    X_test=test[:,2:end]
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end

function data_pre_adult()
    y_train=Matrix(DataFrame(CSV.File("data/adult/y_train.csv")))'
    X_train=Matrix(DataFrame(CSV.File("data/adult/x_train_oh.csv")))'
    y_test=Matrix(DataFrame(CSV.File("data/adult/y_test.csv")))
    X_test=Matrix(DataFrame(CSV.File("data/adult/x_test_oh.csv")))
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end

function train_NN_flux()
    train_data, test_data=data_pre_adult()
    model = Chain(
        Dense(76=>152, relu),
        Dense(152=>100, relu),
        Dense(100=>25, relu),
        Dense(25=>1, sigmoid)
        )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()
    temp=0
    history=0
    h_a=0
    for epochs in 1:500
        Flux.train!(loss, Flux.params(model), train_data, opt)
        a=test(model,test_data)
        if a>0.7
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
    train_data, test_data=data_pre_adult()
    model = Chain(
        Dense(784=>128, relu),
        Dense(128=>64, relu),
        Dense(64=>1, sigmoid)
        )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()
    Flux.@epochs 250 Flux.train!(loss, Flux.params(model), train_data, opt)
    
    return model, test_data
end

function accuracy(x,y,model)
    prediction_class = [if i< 0.5 0 else 1 end for i in model(x')];
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
    if a > 0.7
        @save "src/model/flux_NN_adult.bson" model
    else
        println("not accurate")
    end
end

function load_model(model_add)
    @load model_add model
    return model
end
