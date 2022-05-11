using CSV
#using ROCAnalysis
using MLBase
using StatsBase
using GLM
#using Plots
#using Lathe
using DataFrames
using CategoricalArrays
using FreqTables
using Random
#using Lathe.preprocess: TrainTestSplit
#using MLJ
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
    fm = @formula(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + smoothness_mean + compactness_mean + 
              concavity_mean + concave_points_mean + symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
              perimeter_se + area_se + smoothness_se + compactness_se + concavity_se + concave_points_se + symmetry_se+ fractal_dimension_se + 
              radius_worst + texture_worst + perimeter_worst + area_worst + smoothness_worst + compactness_worst + concavity_worst + concave_points_worst + symmetry_worst + fractal_dimension_worst)

    logis = glm(fm, train, GLM.Binomial(), ProbitLink())   
    prediction = predict(logis, test)
    prediction_class = [if x < 0.5 0 else 1 end for x in prediction];
  
    prediction_df = DataFrame(y_actual = test.diagnosis,y_predicted = prediction_class, prob_predicted = prediction);
    prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
    accuracy = mean(prediction_df.correctly_classified)
    #println(accuracy)
    #println(prediction_df)
    instance=Array(test[test.diagnosis .== 1,:])
    i=instance[1,:]
    popfirst!(i)
    println(i)
    return logis,i
end

#train_LR()