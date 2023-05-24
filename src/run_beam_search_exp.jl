using ArgParse
using Random
using StatsBase
include("./beam_search.jl");
include("regenerate.jl");
include("util.jl")

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

    ###############################temp solution#####################################

    #indices = [1, 255, 256, 257, 258, 259, 260, 261, 283, 284, 285, 286, 287, 288, 289, 311, 312, 313, 314, 315, 316, 317, 339, 340, 341, 342, 343, 344, 345, 367, 368, 369, 370, 371, 372, 373, 395, 396, 397, 398, 399, 400, 401, 423, 424, 425, 426, 427, 428, 429]
    
    df=Matrix(df)
    #df = df[:,indices]
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
        pred = round(logis(ins[2:end])[1])
        if pred==0
            is_Max=false             #choose based on prediction
        end
        graph,exp,d, history, exp_history=beam_search(pc,ins[2:end],logis,sample_size=num_sample,is_max=is_Max,g_acce=index_g,n=n_g, beam_size = beam_size, depth = k, is_xgb = is_xgb, is_cnn = is_cnn)
        history_df=DataFrame(history,:auto)
        exp_history = reduce(vcat, exp_history')
        exp_h = DataFrame(exp_history, :auto)
        CSV.write(output_dir*"/"*"exp_history_instance_$ins_num.csv",exp_h)
        CSV.write(output_dir*"/"*"history_instance_$ins_num.csv",history_df)
        sdp_h(pc, logis, output_dir, ins_num, pred, history_df)
        global ins_num +=1
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
df_r = regenerate(pc,output_dir*"/"*"experiment_plot"*"_"*id*".csv",2*num, sample_size = 9)
CSV.write(output_dir*"/"*"experiment_sampled"*"_"*id*".csv",df_r)