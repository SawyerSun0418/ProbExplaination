using ArgParse
using Random
using StatsBase
include("./beam_search.jl");


function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--exp-id", "-o"
            help = "Experiment ID"
            arg_type = String
        "--dataset", "-d"
            help = "Dataset"
            arg_type = String
            default = "mnist"
        "--classifier"
            help = "Classifier [Flux_LR, Flux_NN]"
            arg_type = String
            default = "Flux_NN"
        "--circuit-path"
            help = "circuit path"
            arg_type = String
        "--beam-size"
            help = "Beam Size"
            arg_type = Int64
            default = 3
        "--features-k"
            help = "k: number of features in explanation"
            arg_type = Int64
            default = 30
        "--num-sample"
            help = "number of samples"
            arg_type = Int64
            default = 100
        "--num-output"
            help = "number of output"
            arg_type = Int64
            default = 100
        "--output-dir"
            help = "output directory"
            arg_type = String
        "--feature-selection"
            help = "if or if not use feature selection for mnist"
            arg_type = Bool
            default = false
        "--cuda"
            help = "which gpu (starting from 1)"
            arg_type = Int64
            default = 1
    end

    return parse_args(s);
end

#= function binvec(x::AbstractVector, n::Int,
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
end =#

function rand_instance(num, dataset)
    if dataset == "mnist"
        df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
    elseif dataset == "adult"
        df=DataFrame(CSV.File("data/adult/x_test_oh_util.csv"))
    elseif dataset == "cancer"
        df = DataFrame(CSV.File("data/data.csv"))
        select!(df, Not([:id]))
        for column in names(df)
            if column!="diagnosis"
                df[!,column]=binvec(df[!,column],5)
            end
        end
    end
    df=Matrix(df)
    df=[df[i,:] for i in 1:size(df,1)]
    result=StatsBase.sample(df, num; replace=false)
    return result
end

function model_pred_func(model_name::String, dataset::String)
    if model_name == "Flux_NN"
        if dataset == "mnist"
            logis = load_model("models/flux_NN_MNIST_new.bson")
        elseif dataset == "adult"
            logis = load_model("models/flux_NN_adult.bson")
        elseif dataset == "cancer"
            logis = load_model("models/flux_NN_cancer.bson")
        else error("unsupported dataset") end
    elseif model_name == "Flux_LR"
        if dataset == "mnist"
            logis = load_model("models/flux_LR_MNIST.bson")
        elseif dataset == "adult"
            logis = load_model("models/flux_LR_adult.bson")
        else error("unsupported dataset") end
    else error("unsupported model") end
    return logis
end

function sdp_exp(pc::ProbCircuit, logis, out_dir, id, exp, original, explanation, label ;sample_size=1000)
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    label_m=Matrix(label)
    exp_m=Matrix(exp)
    original_m=Matrix(original)
    explanation_m=Matrix(explanation)
    sdp=Vector{Float64}[]
    exp_3=0
    exp_5=0
    size_3=0
    size_5=0
    explanation_gpu=cu(explanation_m)
    S=ProbabilisticCircuits.sample(bpc, sample_size, explanation_gpu)
    S = Array{Int64}(S)
    for i in 1:size(original_m,1)
        pred=logis(original_m[i,:])
        pred = [if x < 0.5 0 else 1 end for x in pred];
        temp=0
        for j in 1:sample_size
            exp_pred=logis(S[j,i,:])
            exp_pred = [if x < 0.5 0 else 1 end for x in exp_pred];
            if exp_pred==pred
                temp+=1
            end
        end
        temp=temp/sample_size
        push!(sdp,[temp])
        if label_m[i,1]==1
            exp_3+=exp_m[i,1]
            size_3+=1
        else
            exp_5+=exp_m[i,1]
            size_5+=1
        end
    end
    ave=mean(sdp)
    push!(sdp,ave)
    s=std(sdp)
    push!(sdp,s)
    ave_3=exp_3/size_3
    ave_5=exp_5/size_5
    sdp=reduce(vcat,sdp')
    df=DataFrame(sdp,:auto)

    CSV.write(out_dir*"/"*"experiment_sdp"*"_"*id*".csv",df)
    println("ave exp for 3:",ave_3)
    println("ave exp for 5:",ave_5)
end


function get_exp(exp,label, out_dir, id)
    label_m=Matrix(label)
    exp_m=Matrix(exp)
    exp_3=Vector{Float64}[]
    exp_5=Vector{Float64}[]
    for i in 1:size(exp_m,1)
        if label_m[i,1]==1
            push!(exp_3,[exp_m[i,1]])
        else
            push!(exp_5,[exp_m[i,1]])
        end
    end
    ave_3=mean(exp_3)
    push!(exp_3,ave_3)
    std_3=std(exp_3)
    push!(exp_3,std_3)
    ave_5=mean(exp_5)
    push!(exp_5,ave_5)
    std_5=std(exp_5)
    push!(exp_5,std_5)
    exp_3=reduce(vcat,exp_3')
    df_3=DataFrame(exp_3,:auto)
    CSV.write(out_dir*"/"*"experiment_exp_1"*"_"*id*".csv",df_3)
    exp_5=reduce(vcat,exp_5')
    df_5=DataFrame(exp_5,:auto)
    CSV.write(out_dir*"/"*"experiment_exp_0"*"_"*id*".csv",df_5)
end


parsed_args = parse_cmd();
############

cuda_idx = parsed_args["cuda"] 
device!(collect(devices())[cuda_idx])

id = parsed_args["exp-id"]

dataset = parsed_args["dataset"]
classifier = parsed_args["classifier"]
logis = model_pred_func(classifier, dataset)

pc_path = parsed_args["circuit-path"]
pc = Base.read(pc_path, ProbCircuit)

num = parsed_args["num-output"]
beam_size = parsed_args["beam-size"]
k = parsed_args["features-k"]
num_sample = parsed_args["num-sample"]
output_dir = parsed_args["output-dir"]
select = parsed_args["feature-selection"]

rand_ins=rand_instance(num, dataset)
ins_output=reduce(vcat,rand_ins')
ins_df=DataFrame(ins_output,:auto)
CSV.write(output_dir*"/"*"experiment_original_ins"*"_"*id*".csv",ins_df[:, Not(:x1)])     
result=Vector{Union{Missing, Int64}}[]
Exp=Vector{Float64}[]
label=Vector{Int}[]
explanation_size=Vector{Int}[]
index_g=[]
n_g=100
if select
    index_g = vec(Matrix(DataFrame(CSV.File("data/ranking.csv"))))[1:n_g]
end

for ins in rand_ins       
    @time begin
        is_Max=true
        l=ins[1]
        if ins[1]==0
            is_Max=false
        end
        graph,exp,d=beam_search(pc,ins[2:end],logis,sample_size=num_sample,is_max=is_Max,g_acce=index_g,n=n_g, beam_size = beam_size, depth = k)
        push!(result,graph)
        push!(Exp,[exp])
        push!(label,[l])
        push!(explanation_size,[d])
        print("total time")
    end
        
end
result=reduce(vcat,result')
Exp=reduce(vcat,Exp')
Label=reduce(vcat,label')
Size=reduce(vcat,explanation_size')
df=DataFrame(result,:auto)
exp_df=DataFrame(Exp,:auto)
label_df=DataFrame(Label,:auto)
size_df=DataFrame(Size,:auto)
CSV.write(output_dir*"/"*"experiment_plot"*"_"*id*".csv",df)
CSV.write(output_dir*"/"*"experiment_exp"*"_"*id*".csv",exp_df)
CSV.write(output_dir*"/"*"experiment_label"*"_"*id*".csv",label_df)
CSV.write(output_dir*"/"*"experiment_size"*"_"*id*".csv",size_df)
sdp_exp(pc, logis, output_dir, id, exp_df,ins_df[:, Not(:x1)],df,label_df)
get_exp(exp_df,label_df, output_dir, id)