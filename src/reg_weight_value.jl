using ProbabilisticCircuits
using CUDA
using DataFrames
using CSV
using Flux
using Flux: params
include("./models.jl")
include("regenerate.jl")

# Read the CSV file
csv_file = "data/mnist_3_5_test.csv" # Replace with your CSV file path
df = CSV.read(csv_file, DataFrame)

# Number of rows to select randomly
num_rows = 10

# Randomly select rows
sampled_df = first(shuffle(df), num_rows)

# Convert the DataFrame to a Matrix
sampled_matrix = Matrix(sampled_df)

logis=load_model("models/flux_LR_MNIST.bson")
# Extract the weights of the first layer
first_layer_weights = params(logis.layers[1])[1]

result=Vector{Union{Missing, Int64}}[]

for i in 1:num_rows
    row = sampled_matrix[i,:]
    pixels = row[2:end]
    label = row[1]
    target_indices = findall(x -> x == 1, pixels)
    temp = label == 1 ? fill(-Inf, size(first_layer_weights)) : fill(Inf, size(first_layer_weights))
    for index in target_indices
        temp[1, index] = first_layer_weights[1, index]
    end
    sorted_indices = label == 1 ? sortperm(vec(temp), rev=true) : sortperm(vec(temp), rev=false)
    top_k_indices = sorted_indices[1:min(10, end)]
    output = Array{Union{Missing, Int64}}(missing, 1, 784)
    for index in top_k_indices
        output[index] = pixels[index]
    end
    push!(result, vec(output))
end

result=reduce(vcat,result')
df=DataFrame(result,:auto)
output_dir = "experiments/MNIST/weights/10"
id = "1"
CSV.write(output_dir*"/"*"experiment_plot"*"_"*id*".csv",df)
pc = Base.read("circuits/mnist35.jpc", ProbCircuit)
df_r = regenerate(pc,output_dir*"/"*"experiment_plot"*"_"*id*".csv",size(result,1))
CSV.write(output_dir*"/"*"experiment_sampled"*"_"*id*".csv",df_r)

CUDA.@time bpc = CuBitsProbCircuit(pc);
explanation_gpu=cu(result)
S=ProbabilisticCircuits.sample(bpc, 1000, explanation_gpu)
S = Array{Int64}(S)
sdp=Vector{Float64}[]
exp=Vector{Float64}[]
size_3 = 0
size_5 = 0
exp_3 = 0
exp_5 = 0
for i in 1:size(sampled_matrix,1)
    pred=logis(sampled_matrix[i,2:end])
    pred = [if x < 0.5 0 else 1 end for x in pred];
    temp=0
    exp_temp = 0
    for j in 1:1000
        exp_pred=logis(S[j,i,:])
        exp_temp += exp_pred[1]
        exp_pred = [if x < 0.5 0 else 1 end for x in exp_pred];
        if exp_pred==pred
            temp+=1
        end
    end
        temp=temp/1000
        exp_temp = exp_temp/1000
    push!(sdp,[temp])
    push!(exp, [exp_temp])
    if sampled_matrix[i,1]==1
        global exp_3+=exp_temp
        global size_3+=1
    else
        global exp_5+=exp_temp
        global size_5+=1
    end
end
ave_3=exp_3/size_3
ave_5=exp_5/size_5
sdp=reduce(vcat,sdp')
df_sdp=DataFrame(sdp,:auto)
exp=reduce(vcat,exp')
df_exp=DataFrame(exp,:auto)
df_ori=DataFrame(sampled_matrix[:,2:end],:auto)
CSV.write(output_dir*"/"*"experiment_sdp"*"_"*id*".csv",df_sdp)
CSV.write(output_dir*"/"*"experiment_exp"*"_"*id*".csv",df_exp)
CSV.write(output_dir*"/"*"experiment_original_ins"*"_"*id*".csv",df_ori)
println("ave exp for 3:",ave_3)
println("ave exp for 5:",ave_5)