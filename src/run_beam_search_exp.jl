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

function rand_instance(num, dataset, logis, is_xgb, is_cnn)
    if dataset == "mnist"
        df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
    elseif dataset == "adult"
        df=DataFrame(CSV.File("data/adult/x_test_util.csv"))
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
    if is_xgb
        correct = x -> XGBoost.predict(logis, (x[2:end], [0]) ) == x[1]                         #LoadError: ArgumentError: DMatrix requires either an AbstractMatrix or table satisfying the Tables.jl interface
        incorrect = x -> XGBoost.predict(logis, (x[2:end], [0]) ) != x[1]
    elseif is_cnn
        correct = x -> let x_2d = permutedims(reshape(permutedims(x[2:end]), (28, 28, 1, 1)), (2,1,3,4))
                        round(logis(x_2d)[1]) == x[1]
                        end
        incorrect = x -> let x_2d = permutedims(reshape(permutedims(x[2:end]), (28, 28, 1, 1)), (2,1,3,4))
                        round(logis(x_2d)[1]) != x[1]
                        end
    else
        correct = x -> round(logis(x[2:end])[1]) == x[1]
        incorrect = x -> round(logis(x[2:end])[1]) != x[1]
    end
    filtered_correct = filter(correct, df)
    filtered_incorrect = filter(incorrect, df)
    if length(filtered_correct) < num
        selected_correct = filtered_correct
    else
        selected_correct = shuffle(filtered_correct)[1:num]
    end
    if length(filtered_incorrect) < num
        selected_incorrect = filtered_incorrect
    else
        selected_incorrect = shuffle(filtered_incorrect)[1:num]
    end
    @show size(selected_correct)
    @show size(selected_incorrect)
    result = vcat(selected_correct, selected_incorrect)
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
    elseif model_name == "XGBoost"
        if dataset == "adult"
            logis = load_model("models/XGBoost_adult.bson")
        else error("unsupported dataset") end
    elseif model_name == "Flux_CNN"
        if dataset == "mnist"
            logis = load_model("models/flux_CNN_MNIST.bson")
        else error("unsupported dataset") end
    else 
        try
            logis = load_model(model_name)
        catch
            error("could not find this model")
        end
    end
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
is_xgb = false
is_cnn = false
classifier = parsed_args["classifier"]
if classifier == "XGBoost"
    is_xgb = true
elseif classifier == "Flux_CNN"
    is_cnn = true
end
logis = model_pred_func(classifier, dataset)

pc_path = parsed_args["circuit-path"]
pc = Base.read(pc_path, ProbCircuit)

num = parsed_args["num-output"]
beam_size = parsed_args["beam-size"]
k = parsed_args["features-k"]
num_sample = parsed_args["num-sample"]
output_dir = parsed_args["output-dir"]
select = parsed_args["feature-selection"]

rand_ins=rand_instance(num, dataset, logis, is_xgb, is_cnn)
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

ins_num = 1

for ins in rand_ins       
    @time begin
        is_Max=true
        l=ins[1]
        if round(logis(ins[2:end])[1])==0
            is_Max=false             #choose based on prediction
        end
        graph,exp,d, history=beam_search(pc,ins[2:end],logis,sample_size=num_sample,is_max=is_Max,g_acce=index_g,n=n_g, beam_size = beam_size, depth = k, is_xgb = is_xgb, is_cnn = is_cnn)
        history_df=DataFrame(history,:auto)
        CSV.write(output_dir*"/"*"history_instance_$ins_num.csv",df)
        ins_num +=1
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