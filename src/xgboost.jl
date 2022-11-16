using CSV
using DataFrames
using XGBoost
using Metrics
x_test = Matrix(DataFrame(CSV.File("data/adult/x_test_oh.csv")))
x_train = Matrix(DataFrame(CSV.File("data/adult/x_train_oh.csv")))
y_test = vec(Matrix(DataFrame(CSV.File("data/adult/y_test.csv"))))
y_train = vec(Matrix(DataFrame(CSV.File("data/adult/y_train.csv"))))
dtrain = DMatrix(x_train, label=y_train)
dtest = DMatrix(x_test, label=y_test)
bst = xgboost(dtrain, num_round = 6, max_depth = 6, eta = 0.5, eval_metric = "error", objective = "binary:logistic")
#ins=[0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0]
#ins=reshape(ins,1,:)
#print(predict(bst,ins))
#yhat_train = [if x > 0 1 else 0 end for x in predict(bst, dtrain)]
yhat_train = predict(bst, dtrain)
#yhat_test = [if x > 0 1 else 0 end for x in predict(bst, dtest)]
yhat_test = predict(bst, dtest)
print("Training Accuracy (Full Data): ")
print(Metrics.binary_accuracy(yhat_train, y_train))
print("Test     Accuracy (Full Data):")
print(Metrics.binary_accuracy(yhat_test, y_test))
#XGBoost.save(bst,"src/model/adult/xgb_adult")
#logis = Booster
#XGBoost.load(logis, "src/model/adult/xgb_adult.model")