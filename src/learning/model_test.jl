using Flux
using CUDA
using BSON: @load
using ProbabilisticCircuits
using Revise
includet("../models.jl")


function train_model(df)
    train_data, test_data=data_pre(df)
    CUDA.@time begin
        model = Chain(
            Dense(784=>128, relu),
            Dense(128=>10, relu),
            Dense(10=>1, sigmoid)
            )
        loss(x, y) = Flux.binarycrossentropy(model(x), y)
        opt = Flux.Optimise.ADAM()
        for epochs in 1:5
            Flux.train!(loss, Flux.params(model), train_data, opt)
        end
    end
    return model
end

function run(model, batch_size=100)
    println("====================")
    println("Train Data Size", size(train_data))
    train_batch = train_data[:, 1:batch_size]
    @time model(train_batch)
    @time model(train_batch)
    train_batch_gpu = gpu(train_batch)
    model_gpu = gpu(model)
    CUDA.@time model_gpu(train_batch_gpu)
    CUDA.@time model_gpu(train_batch_gpu)

    nothing
end

function run_3d(model, sample_size, beam_size)
    data_gpu = cu(permutedims(train_data[:, 1:beam_size], [2, 1])) # just want the correct shape for now, (beam_size, features)
    S_gpu = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)    # -> (sample_size, data_size, num_features)
    
    @show typeof(S_gpu)

    S = Array(S_gpu)
    S2 = permutedims(S, [3, 1, 2]) # (num_features, sample_size*, data_size*); only matters that num_features is first for using Flux
    @show size(S)
    @show size(S2)

    println("CPU timing...")
    @time model(S2)
    @time y2 = model(S2)
    @show size(y2)

    # 1. premute dims to match the need for the NN, num_features should be first 
    # 2. convert from CuArray{Missing, Int64} -> CuArray{Int64}
    CUDA.@time S2_gpu = permutedims(S_gpu, [3, 1, 2])::CuArray{Union{Int64, Missing}}
    @show typeof(S2_gpu)
    CUDA.@time S2_gpu = convert(CuArray{Int64}, S2_gpu)
    @show typeof(S2_gpu)
    @show size(S2_gpu)
    model_gpu = gpu(model)

    println("GPU timing...")
    CUDA.@time model_gpu(S2_gpu)
    CUDA.@time y2 = model_gpu(S2_gpu)
    @show size(y2)
    return nothing
end


pc = Base.read("circuits/mnist35.jpc", ProbCircuit);
CUDA.@time bpc = CuBitsProbCircuit(pc);


df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
train_data, test_data=data_pre(df)
train_data = train_data[1][1]
train_data = collect(train_data)
@show size(train_data)

# @load "src/model/flux_NN_MNIST.bson" model
model = train_model(df);

# run(model)
# run_3d(model, 100, 28*28*3)

