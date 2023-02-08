
using ProbabilisticCircuits
using CUDA
using DataFrames
using CSV
using StatsBase: sample
using XGBoost
include("./models.jl")


function init_instance(instance::AbstractVector)

    n=size(instance)[1]
    m=Array{Union{Missing, Int64}}(missing, n, n)           # x1 _ ... _
    for i in 1:n                                            # _  x2 ... _
        m[i,i]=instance[i]                                 #_  _ ...  _
    end
    return m                                                # ..........xn
end

function init_instance_g(instance::AbstractVector,index::AbstractVector,n::Int64)
    s=size(instance, 1)
    m=Array{Union{Missing, Int64}}(missing, n, s)           
    for i in 1:n
        in=index[i]                                            
        m[i,in]=instance[in]                                 
    end
    return m                                                
end

function rand_instance_adult(num)
    df=Matrix(DataFrame(CSV.File("data/adult/x_test_oh_util.csv")))
    df=[df[i,:] for i in 1:size(df,1)]
    result=sample(df, num; replace=false)
    return result
end

function rand_instance_cancer(num)
    df=return_df()
    df=Matrix(df)
    df=[df[i,:] for i in 1:size(df,1)]
    result=sample(df, num; replace=false)
    return result
end


function rand_instance(num)
    df=return_MNIST_df_t()
    df=Matrix(df)
    df=[df[i,:] for i in 1:size(df,1)]
    result=sample(df, num; replace=false)
    return result
end

function expand_instance(instance::AbstractVector,m::AbstractMatrix)
    
    num_features = size(m)[2]
    ret = Array{Union{Missing, Int64}}(undef, 0, num_features) 
    num_k=size(m)[1]
    for i in 1:num_k
        for j in 1:num_features                     # a k*n array where each row is a possible candidate with missing feature
            if ismissing(m[i,j])
                newm = m[i,:]
                newm[j] = instance[j]
                ret=[ret;newm']         
            end
        end
    end
    ret=unique(ret,dims=1)
    return ret
end

function expand_instance_g(instance::AbstractVector,m::AbstractMatrix,index::AbstractVector)
    
    num_features = size(m)[2]
    ret = Array{Union{Missing, Int64}}(undef, 0, num_features) 
    num_k=size(m)[1]
    for i in 1:num_k
        for j in index                    
            if ismissing(m[i,j])
                newm = copy(m[i,:])
                newm[j] = instance[j]
                ret=[ret;newm']         
            end
        end
    end
    ret=unique(ret,dims=1)
    #print(size(ret))
    return ret
end

function model_pred_func(model_name::String, dataset::String)
    if model_name == "Flux_NN"
        if dataset == "mnist"
            logis = load_model("src/model/flux_NN_MNIST_new.bson")
        elseif dataset == "adult"
            logis = load_model("src/model/flux_NN_adult.bson")
        elseif dataset == "cancer"
            logis = load_model("src/model/flux_NN_cancer.bson")
        else error("unsupported dataset") end
    elseif model_name == "Flux_LR"
        if dataset == "mnist"
            logis = load_model("src/model/flux_LR_MNIST.bson")
        elseif dataset == "adult"
            logis = load_model("src/model/flux_LR_adult.bson")
        else error("unsupported dataset") end
    else error("unsupported model") end
    return logis
end
        


function beam_search(pc::ProbCircuit, instance, pred_func;is_max=true, th=1, k=3,depth=30,sample_size=100,g_acce=[],n=size(instance, 1))
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    instance = cu(instance)
    pred_func = gpu(pred_func)
    if g_acce!=[] 
        data=init_instance_g(instance,g_acce,n)
    else
        data=init_instance(instance)
    end
    data = collect(data)
    data_gpu = cu(data)
    
    new_data=[]
    for r in 1:depth
        S_gpu = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)  
        S2_gpu = convert(CuArray{Int64}, permutedims(S_gpu, [3, 1, 2]))::CuArray{Int64}
        top_k=[]
        print("predict time")
        CUDA.@time prediction = pred_func(S2_gpu)
        prediction_sum = dropdims(sum(prediction, dims=2), dims=2)
        cand_gpu = prediction_sum/sample_size
        cand = vec(cand_gpu)
        if is_max
            top_k=partialsortperm(cand, 1:k, rev=true)  
            #if  th!=1 && mean(cand[top_k])>float(th)
            #    #println(cand)
            #    first=top_k[1]
            #    result=data[first,:]
            #    exps=cand[first]                             
            #    #println(exps)
            #    return result,exps,r
            #end  
        else
            top_k=partialsortperm(cand, 1:k)
            #th=1-th
            #print(mean(cand[top_k]))
            #if th!=0 && mean(cand[top_k])<float(th)
                #println(cand)
            #    first=top_k[1]
            #    result=data[first,:]
            #    exps=cand[first]                      #issue: terminate too early
                #println(exps)
            #    return result,exps,r
            #end 
        end
            #display(cand[top_k])
        
        

        if r==depth
            #println(cand)
            first=top_k[1]
            result=data[first,:]
            exps=cand[first]
            #println(exps)
            return result,exps,r
        end

        new_data=data[top_k,:]
        if g_acce!=[] 
            print("expand time")
            @time data=expand_instance_g(instance,new_data,g_acce)
        else
            print("expand time")
            @time data=expand_instance(instance,new_data)
        end
        data_gpu=cu(data)
    end

end 



function run_MNIST(num::Int, model::String)
    pc = Base.read("mnist35.jpc", ProbCircuit)
    pred_func = model_pred_func(model, "mnist")
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
        @time begin
            is_Max=true
            l=ins[1]
            if ins[1]==0
                is_Max=false
            end
            #index_g = vec(Matrix(DataFrame(CSV.File("data/ranking.csv"))))[1:n_g]
            graph,exp,d=beam_search(pc,ins[2:end],pred_func,sample_size=300,is_max=is_Max,g_acce=index_g,n=n_g)
            push!(result,graph)
            push!(Exp,[exp])
            push!(label,[l])
            push!(size,[d])
            print("total time")
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



function run_cancer(num::Int, model::String)
    pc = Base.read("trained_pc.jpc", ProbCircuit)
    pred_func = model_pred_func(model, "cancer")
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
            graph,exp,d=beam_search(pc,ins[2:end],pred_func,sample_size=300,is_max=is_Max,depth=11)
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



function run_adult(num::Int, model::String)
    pc = Base.read("adult.jpc", ProbCircuit)
    pred_func = model_pred_func(model, "adult")
    rand_ins=rand_instance_adult(num)
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
            graph,exp,d=beam_search(pc,ins[2:end],pred_func,sample_size=300,is_max=is_Max, depth=11)
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
