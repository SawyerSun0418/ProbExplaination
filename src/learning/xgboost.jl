using CSV
using DataFrames
using XGBoost
using Metrics
using BSON: @save
X_test = Matrix(DataFrame(CSV.File("data/adult/x_test.csv")))
X_train = Matrix(DataFrame(CSV.File("data/adult/x_train.csv")))
y_test = vec(Matrix(DataFrame(CSV.File("data/adult/y_test.csv"))))
y_train = vec(Matrix(DataFrame(CSV.File("data/adult/y_train.csv"))))

#num_round = 10
dtrain = DMatrix(X_train, label=y_train)
#params = Dict("max_depth" => 3, "eta" => 0.1, "objective" => "binary:logistic")
model = XGBoost.xgboost(dtrain, num_round=6, max_depth = 6, eta = 0.5, objective = "binary:logistic", eval_metric = "error")
dtest = DMatrix(X_test, label=y_test)
y_pred = XGBoost.predict(model, dtest)
y_pred_labels = round.(Int, y_pred)  # Convert probabilities to binary labels
accuracy = sum(y_pred_labels .== y_test) / length(y_test)
println("Test set accuracy: $accuracy")
if accuracy>0.8
    @save "models/XGBoost_adult.bson" model
end