using CSV
using ROCAnalysis
using MLBase
using StatsBase
using GLM
using Plots
using DataFrames
using Lathe
df = DataFrame(CSV.File("iris.csv"))
println(describe(df))
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df, .75);
fm = @formula(variety ~ sepal_length + sepal_width + petal_length + petal_width)
logis = glm(fm, train, Binomial(), ProbitLink())
prediction = predict(logis, test)
prediction_class = [if x < 0.5 0 else 1 end for x in prediction];
  
prediction_df = DataFrame(y_actual = test.variety,y_predicted = prediction_class, prob_predicted = prediction);
prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
accuracy = mean(prediction_df.correctly_classified)