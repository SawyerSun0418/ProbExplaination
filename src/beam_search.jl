
using ProbabilisticCircuits
using CUDA
using DataFrames
using CSV
using StatsBase: sample
include("./LR.jl")


#function kfoldperm(N,k)
#    n,r = divrem(N,k)
#    b = collect(1:n:N+1)
#    for i in 1:length(b)
#        b[i] += i > r ? r : i-1  
#    end
#    p = randperm(N)
#    return [p[r] for r in [b[i]:b[i+1]-1 for i=1:k]]
#end


function init_instance(instance::AbstractVector)

    n=size(instance)[1]
    m=Array{Union{Missing, Int64}}(missing, n, n)           # x1 _ ... _
    for i in 1:n                                            # _  x2 ... _
        m[i,i]=instance[i]                                 #_  _ ...  _
    end
    return m                                                # ..........xn
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

function expand_instance(instance::AbstractVector,m::AbstractMatrix; batch_size=64)
    
    num_features = size(m)[2]
    ret = Array{Union{Missing, Int64}}(undef, 0, num_features) 
    num_k=size(m)[1]
    for i in 1:num_k
        for j in 1:num_features                     # a k*n array where each row is a possible candidate with missing feature
            if ismissing(m[i,j])
                newm = copy(m[i,:])
                newm[j] = instance[j]
                ret=[ret;newm']         
            end
        end
    end
    ret=unique(ret,dims=1)
    #n=size(ret)[1]
    #parts=kfoldperm(n,batch_size)
    #result=[ret[p] for p in parts]
    return ret
end

#5 now
function beam_search(pc::ProbCircuit;is_max=true,is_Flux=true,th=1, k=3,depth=30,instance=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],sample_size=100 )
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    if is_Flux
        logis=load_model("src/model/flux_LR.bson")
        ins_prob = logis(instance)
    else
        logis=train_LR()
        ins_prob = ScikitLearn.predict_proba(logis, [instance])[:,2]
    end
    
    println("p(c|x) is ",ins_prob)
    data=init_instance(instance)
    data_gpu = cu(data)
    
    new_data=[]
    for r in 1:depth
        S = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)  
        S = Array{Int64}(S) # (sample_size, data_size, num_features)
        num_cand=size(data_gpu)[1]  #number of candidates in one step of beam search
        cand = Array{Float64}(undef, num_cand)
        top_k=[]
        for n in 1:num_cand
            prediction_sum=0
            if is_Flux
                prediction= logis(S[:,n,:]')
            else
                prediction = ScikitLearn.predict_proba(logis, S[:,n,:])[:,2]   ###This returns probability of prediction is 1 (aka 3)
            end
            prediction_sum=sum(prediction)
            exp=prediction_sum/sample_size
            cand[n]=exp
        end
        #deleteat!(cand, cand .== 1);
        if is_max
            top_k=partialsortperm(cand, 1:k, rev=true)  
            if mean(cand[top_k])>=th
                println(cand)
                first=top_k[1]
                result=data[first,:]
                exps=cand[first]
                println(exps)
                return result,exps,r
            end  
        else
            top_k=partialsortperm(cand, 1:k)
            th=1-th
            if mean(cand[top_k])<=th
                println(cand)
                first=top_k[1]
                result=data[first,:]
                exps=cand[first]
                println(exps)
                return result,exps,r
            end 
        end
            #display(cand[top_k])
        
        

        if r==depth
            println(cand)
            first=top_k[1]
            result=data[first,:]
            exps=cand[first]
            println(exps)
            return result,exps,r
        end

        new_data=data[top_k,:]
        
        #temp=cu(new_data)
        #temp_s=ProbabilisticCircuits.sample(bpc, sample_size, temp)
        #num_cand=size(new_data)[1]
        #for n in 1:num_cand
        #    prediction_sum=0
        #    prediction = ScikitLearn.predict_proba(logis, temp_s[:,n,:])[:,2]
        #    prediction_sum=sum(prediction)
        #    exps=prediction_sum/sample_size
        #    display(exps)
        #end
        #return
        
        data=expand_instance(instance,new_data)
        data_gpu=cu(data)
    end

end 



function run_MNIST()
    pc = Base.read("mnist35.jpc", ProbCircuit)
    rand_ins=rand_instance(100)
    ins_output=reduce(vcat,rand_ins')
    ins_df=DataFrame(ins_output,:auto)
    CSV.write("experiment_original_ins.csv",ins_df)     #temp: remember to delete first column
    result=Vector{Union{Missing, Int64}}[]
    Exp=Vector{Float64}[]
    label=Vector{Int}[]
    size=Vector{Int}[]
    for ins in rand_ins
        @time begin
            is_Max=true
            l=ins[1]
            if ins[1]==0
                is_Max=false
            end
            graph,exp,d=beam_search(pc,instance=ins[2:end],sample_size=300,is_max=is_Max)
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



function run_cancer()
    pc = Base.read("trained_pc.jpc", ProbCircuit)
    rand_ins=rand_instance_cancer(300)
    ins_output=reduce(vcat,rand_ins')
    ins_df=DataFrame(ins_output,:auto)
    CSV.write("experiment_original_ins_c.csv",ins_df[:, Not(:x1)])     #temp: remember to delete first column
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
            graph,exp,d=beam_search(pc,instance=ins[2:end],sample_size=300,is_max=is_Max,is_Flux=false,th=0.985)
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

