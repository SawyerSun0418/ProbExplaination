using CSV
#using MLBase
using StatsBase
using DataFrames
#using CategoricalArrays
#using FreqTables
using Random
#using ScikitLearn
using Flux
#using Zygote
#using MLUtils
using BSON: @save
using BSON: @load
#@sk_import linear_model: LogisticRegression
include("util.jl")


function train_NN_flux()
    #df = DataFrame(CSV.File("data/mnist_3_5_train.csv"))
    train_data, test_data=data_pre_adult()
    model = Chain(
        Dense(11=>64, relu),
        Dropout(0.2),
        Dense(64=>32, relu),
        Dropout(0.2),
        Dense(32=>2, relu),
        Dense(2=>1, sigmoid)
        )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.RMSProp(0.001)
    temp=0
    history=0
    h_a=0
    for epochs in 1:50
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

function train_CNN_flux()
    df = DataFrame(CSV.File("data/mnist_3_5_train.csv"))
    train_data, test_data=data_pre(df)
    n_features, n_instances  = size(train_data[1][1])
    height, width, n_channels = 28, 28, 1
    train_imgs = permutedims(reshape(train_data[1][1], (height, width, n_channels, n_instances)), (2,1,3,4))
    train_labels = train_data[1][2]
    n_instances, n_features = size(test_data[1])
    test_imgs = permutedims(reshape(permutedims(test_data[1]), (height, width, n_channels, n_instances)), (2,1,3,4))
    test_labels = test_data[2]
    model = Chain(
        #= Conv((3, 3), 1=>16, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        Conv((3, 3), 16=>32, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        Conv((3, 3), 32=>32, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        x -> reshape(x, :, size(x, 4)),
        Dense(288, 10), =#
        Conv((3, 3), 1=>32, relu),
        BatchNorm(32),
        Conv((3, 3), 32=>32, relu),
        BatchNorm(32),
        Conv((5, 5), 32=>32, stride=(2, 2), pad=(2, 2), relu),
        BatchNorm(32),
        Dropout(0.4),
        Conv((3, 3), 32=>64, relu),
        BatchNorm(64),
        Conv((3, 3), 64=>64, relu),
        BatchNorm(64),
        Conv((5, 5), 64=>64, stride=(2, 2), pad=(2, 2), relu),
        BatchNorm(64),
        Dropout(0.4),
       # x -> @show(size(x)),
        x -> reshape(x, :, size(x, 4)),
        Dense(1024=>128, relu),
        BatchNorm(128),
        Dropout(0.4),
        Dense(128=>10, relu),
        Dense(10=>1, sigmoid)
    )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()
    temp=0
    history=0
    h_a=0
    for epochs in 1:100
        @show epochs
        train = [(train_imgs, train_labels)]
        Flux.train!(loss, Flux.params(model), train, opt)
        prediction_class = [if i< 0.5 0 else 1 end for i in model(test_imgs)];
        a = mean(vec(prediction_class) .== vec(test_labels))
        @show a
        if a>0.9
            if a > h_a
                history=model
                h_a=a
                temp=0
            else temp+=1 end

            if temp>=5
                return history, test_data
            end
        end
    end
    println("failed")
    return model, test_data
end

function train_LR_flux()
    df = DataFrame(CSV.File("data/mnist_3_5_train.csv"))
    indices = [255, 256, 257, 258, 259, 260, 261, 283, 284, 285, 286, 287, 288, 289, 311, 312, 313, 314, 315, 316, 317, 339, 340, 341, 342, 343, 344, 345, 367, 368, 369, 370, 371, 372, 373, 395, 396, 397, 398, 399, 400, 401, 423, 424, 425, 426, 427, 428, 429]
    train_data, test_data=data_pre(df, indices = indices)
    train_data
    model = Chain(
        Dense(49 => 1, sigmoid)
        )
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.ADAM()
    for i = 1:250
        Flux.train!(loss, Flux.params(model), train_data, opt)
    end
    X_train, y_train = train_data[1]
    prediction_class = [if i< 0.5 0 else 1 end for i in model(X_train)];
    a = mean(vec(prediction_class) .== vec(y_train))
    println("\nAccuracy: $a")
    return model, test_data
end

function accuracy(x,y,model)
    prediction_class = [if i< 0.5 0 else 1 end for i in model(permutedims(x))];
    a = mean(vec(prediction_class) .== vec(y))
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
        @save "models/flux_NN_adult.bson" model
    else
        println("not accurate")
    end
end

function load_model(model_add)
    @load model_add model
    return model
end
