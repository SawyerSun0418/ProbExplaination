using ProbabilisticCircuits
using CUDA
using DataFrames
using CSV
using StatsBase: sample
using ScikitLearn
include("./models.jl")

function sdp_exp(pc::ProbCircuit, exp::String, original::String, explanation::String, label::String ;sample_size=1000,is_Flux=true)
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    label_m=Matrix(DataFrame(CSV.File(label)))
    exp_m=Matrix(DataFrame(CSV.File(exp)))
    original_m=Matrix(DataFrame(CSV.File(original)))
    explanation_m=Matrix(DataFrame(CSV.File(explanation)))
    if is_Flux
        logis=load_model("src/model/flux_NN_MNIST.bson")
    else 
        logis=train_LR()
    end
    sdp=Vector{Float64}[]
    exp_3=0
    exp_5=0
    size_3=0
    size_5=0
    explanation_gpu=cu(explanation_m)
    S=ProbabilisticCircuits.sample(bpc, sample_size, explanation_gpu)
    S = Array{Int64}(S)
    print(size(original_m,1))
    for i in 1:size(original_m,1)
        if is_Flux
            pred=logis(original_m[i,:])
        else
            pred = ScikitLearn.predict(logis, [original_m[i,:]])
        end
        pred = [if x < 0.5 0 else 1 end for x in pred];
        temp=0
        for j in 1:sample_size
            if is_Flux
                exp_pred=logis(S[j,i,:])
            else
                exp_pred = ScikitLearn.predict(logis, [S[j,i,:]])
            end
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
    ave_3=exp_3/size_3
    ave_5=exp_5/size_5
    sdp=reduce(vcat,sdp')
    df=DataFrame(sdp,:auto)
    CSV.write("experiment_sdp.csv",df)
    println("ave exp for 3:",ave_3)
    println("ave exp for 5:",ave_5)
end


function get_exp(exp::String,label::String)
    label_m=Matrix(DataFrame(CSV.File(label)))
    exp_m=Matrix(DataFrame(CSV.File(exp)))
    exp_3=Vector{Float64}[]
    exp_5=Vector{Float64}[]
    for i in 1:size(exp_m,1)
        if label_m[i,1]==1
            push!(exp_3,[exp_m[i,1]])
        else
            push!(exp_5,[exp_m[i,1]])
        end
    end
    exp_3=reduce(vcat,exp_3')
    df_3=DataFrame(exp_3,:auto)
    CSV.write("experiment_exp_3.csv",df_3)
    exp_5=reduce(vcat,exp_5')
    df_5=DataFrame(exp_5,:auto)
    CSV.write("experiment_exp_5.csv",df_5)
end

pc = Base.read("mnist35.jpc", ProbCircuit)
sdp_exp(pc,"experiment_exp_100.csv","experiment_original_ins_100.csv","experiment_plot_100.csv","experiment_label_100.csv")
get_exp("experiment_exp_100.csv","experiment_label_100.csv")
#pc = Base.read("trained_pc.jpc", ProbCircuit)
#sdp_exp(pc,"experiment_exp_c.csv","experiment_original_ins_c.csv","experiment_plot_c.csv","experiment_label_c.csv")
#get_exp("experiment_exp_c.csv","experiment_label_c.csv")