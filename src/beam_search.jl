
using ProbabilisticCircuits
using CUDA
#using DataFrames
#using CSV
#using StatsBase: sample
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


function expand_instance(instance::AbstractVector, m::AbstractMatrix)
    n, d = size(m)
    filled_matrix = Array{Union{Missing, Int64}}(undef, n * d, d)
    unique_rows = Set{Array{Union{Missing, Int64}}}()
    count = 0
    for i in 1:n
        r = m[i, :]
        missing_indexes = findall(ismissing.(r))
        for j in missing_indexes
            filled_vec = copy(r)
            filled_vec[j] = instance[j]
            row = filled_vec
            if row in unique_rows
                continue
            end
            count += 1
            filled_matrix[count, :] = filled_vec
            push!(unique_rows, row)
        end
    end
    return filled_matrix[1:count, :]
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
        


function beam_search(pc::ProbCircuit, instance, pred_func;is_max=true, th=1, beam_size=3,depth=30,sample_size=100,g_acce=[],n=size(instance, 1))
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
            top_k=partialsortperm(cand, 1:beam_size, rev=true)  
            #if  th!=1 && mean(cand[top_k])>float(th)
            #    #println(cand)
            #    first=top_k[1]
            #    result=data[first,:]
            #    exps=cand[first]                             
            #    #println(exps)
            #    return result,exps,r
            #end  
        else
            top_k=partialsortperm(cand, 1:beam_size)
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




