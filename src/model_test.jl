using Flux
using CUDA
using BSON: @load
include("models.jl")

@load "src/model/flux_NN_MNIST.bson" model
train_data,test_data=data_pre()
train_data = train_data[1][1]
print(size(train_data))
train_batch = train_data[:,1:300]
@time model(train_batch)
@time model(train_batch)
train_batch_gpu = gpu(train_batch)
model_gpu = gpu(model)
CUDA.@time model_gpu(train_batch_gpu)
CUDA.@time model_gpu(train_batch_gpu)