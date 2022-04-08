using CSV
using ROCAnalysis
using MLBase
using StatsBase
using GLM
using Plots
using Lathe
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





df = DataFrame(CSV.File("data.csv"))
println(describe(df))
select!(df, Not([:id]))
println("without id")
println(describe(df))
println(size(df))
for column in names(df)
    if column!="diagnosis"
        df[column]=binvec(df[column],5)
    end
end
println("binned")
println(describe(df))
println(size(df))
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df, .75);
fm = @formula(diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + smoothness_mean + compactness_mean + 
              concavity_mean + concave_points_mean + symmetry_mean + fractal_dimension_mean + radius_se + texture_se + 
              perimeter_se + area_se + smoothness_se + compactness_se + concavity_se + concave_points_se + symmetry_se+ fractal_dimension_se + 
              radius_worst + texture_worst + perimeter_worst + area_worst + smoothness_worst + compactness_worst + concavity_worst + concave_points_worst + symmetry_worst + fractal_dimension_worst)

logis = glm(fm, train, Binomial(), ProbitLink())
prediction = predict(logis, test)
prediction_class = [if x < 0.5 0 else 1 end for x in prediction];
  
prediction_df = DataFrame(y_actual = test.diagnosis,y_predicted = prediction_class, prob_predicted = prediction);
prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
accuracy = mean(prediction_df.correctly_classified)
println(accuracy)