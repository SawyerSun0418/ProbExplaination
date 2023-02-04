using CUDA
using ProbabilisticCircuits
using DataFrames
using CSV
using StatsBase: sample
using XGBoost

function run_MNIST(num::Int)
    pc = Base.read("mnist35.jpc", ProbCircuit)
    rand_ins=rand_instance(num)
    ins_output=reduce(vcat,rand_ins')
    ins_df=DataFrame(ins_output,:auto)
    CSV.write("experiment_original_ins.csv",ins_df[:, Not(:x1)])     
    result=Vector{Union{Missing, Int64}}[]
    Exp=Vector{Float64}[]
    label=Vector{Int}[]
    size=Vector{Int}[]
    index_g=[]
    n_g=100
    for ins in rand_ins
        print("total time")
        @time begin
            is_Max=true
            l=ins[1]
            if ins[1]==0
                is_Max=false
            end
            #index_g = vec(Matrix(DataFrame(CSV.File("data/ranking.csv"))))[1:n_g]
            graph,exp,d=beam_search(pc,ins[2:end],sample_size=300,is_max=is_Max,g_acce=index_g,n=n_g)
            push!(result,graph)
            push!(Exp,[exp])
            push!(label,[l])
            push!(size,[d])
        end
        
    end
    result=reduce(vcat,result')
    Exp=reduce(vcat,Exp')
    Label=reduce(vcat,label')
    Size=reduce(vcat,size')
    df=DataFrame(result,:auto)
    exp_df=DataFrame(Exp,:auto)
    label_df=DataFrame(Label,:auto)
    size_df=DataFrame(Size,:auto)
    CSV.write("experiment_plot.csv",df)
    CSV.write("experiment_exp.csv",exp_df)
    CSV.write("experiment_label.csv",label_df)
    CSV.write("experiment_size.csv",size_df)
end



function run_cancer(num::Int)
    pc = Base.read("trained_pc.jpc", ProbCircuit)
    rand_ins=rand_instance_cancer(num)
    ins_output=reduce(vcat,rand_ins')
    ins_df=DataFrame(ins_output,:auto)
    CSV.write("experiment_original_ins_c.csv",ins_df[:, Not(:x1)])     
    result=Vector{Union{Missing, Int64}}[]
    Exp=Vector{Float64}[]
    label=Vector{Int64}[]
    size=Vector{Int}[]
    for ins in rand_ins
        @time begin
            is_Max=true
            l=ins[1]
            if l==0
                is_Max=false
            end
            graph,exp,d=beam_search(pc,ins[2:end],sample_size=300,is_max=is_Max,depth=11)
            push!(result,graph)
            push!(Exp,[exp])
            push!(label,[l])
            push!(size,[d])
        end
    end
    result=reduce(vcat,result')
    Exp=reduce(vcat,Exp')
    Label=reduce(vcat,label')
    Size=reduce(vcat,size')
    df=DataFrame(result,:auto)
    exp_df=DataFrame(Exp,:auto)
    label_df=DataFrame(Label,:auto)
    size_df=DataFrame(Size,:auto)
    CSV.write("experiment_plot_c.csv",df)
    CSV.write("experiment_exp_c.csv",exp_df)
    CSV.write("experiment_label_c.csv",label_df)
    CSV.write("experiment_size_c.csv",size_df)
end



function run_adult()
    pc = Base.read("adult.jpc", ProbCircuit)
    rand_ins=rand_instance_adult(300)
    ins_output=reduce(vcat,rand_ins')
    ins_df=DataFrame(ins_output,:auto)
    CSV.write("experiment_original_ins_c.csv",ins_df[:, Not(:x1)])     
    result=Vector{Union{Missing, Int64}}[]
    Exp=Vector{Float64}[]
    label=Vector{Int64}[]
    size=Vector{Int}[]
    for ins in rand_ins
        @time begin
            is_Max=true
            l=ins[1]
            if l==0
                is_Max=false
            end
            graph,exp,d=beam_search(pc,ins[2:end],sample_size=300,is_max=is_Max, depth=11)
            push!(result,graph)
            push!(Exp,[exp])
            push!(label,[l])
            push!(size,[d])
        end
    end
    result=reduce(vcat,result')
    Exp=reduce(vcat,Exp')
    Label=reduce(vcat,label')
    Size=reduce(vcat,size')
    df=DataFrame(result,:auto)
    exp_df=DataFrame(Exp,:auto)
    label_df=DataFrame(Label,:auto)
    size_df=DataFrame(Size,:auto)
    CSV.write("experiment_plot_c.csv",df)
    CSV.write("experiment_exp_c.csv",exp_df)
    CSV.write("experiment_label_c.csv",label_df)
    CSV.write("experiment_size_c.csv",size_df)
end
