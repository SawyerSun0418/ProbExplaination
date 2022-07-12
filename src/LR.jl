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
    df=return_MNIST_df()
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
    #println(accuracy)
    #println(first(prediction_df,2))
    #instance=Array(test[test[:,1] .== 0,:])
    #i=instance[2,:]
    #popfirst!(i)
    #println(i)
    return model
end

#train_LR()