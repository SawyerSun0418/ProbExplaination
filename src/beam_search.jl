
using ProbabilisticCircuits
using CUDA
using XGBoost
include("./models.jl")


function init_instance(instance::Vector)
    n=size(instance)[1]
    m=Array{Union{Missing, Int64}}(missing, n, n)           
    for i in 1:n                                            
        m[i,i]=instance[i]                                 
    end
    return m                                                
end

function init_instance_g(instance::Vector, index::Vector, n::Int64)
    s=size(instance, 1)
    m=Array{Union{Missing, Int64}}(missing, n, s)           
    for i in 1:n
        in=index[i]                                            
        m[i,in]=instance[in]                                 
    end
    return m                                                
end


function expand_instance(instance::Vector, m::Matrix)
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
    return ret
end
        


function beam_search(pc::ProbCircuit, instance, pred_func; is_max=true, beam_size = 3,
                                                            depth = 30, sample_size = 100, g_acce = [], n = size(instance, 1))

    
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    instance = instance
    pred_func = gpu(pred_func)
    if g_acce!=[] 
        data=init_instance_g(instance,g_acce,n)
    else
        data=init_instance(instance)
    end
    data = collect(data)
    data_gpu = cu(data)
    
    new_data = []
    for r in 1:depth
        println("===Depth $(r)===")
        print("Sample time....")
        CUDA.@time begin
            S_gpu = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)                    # (num_samples, size(data_gpu, 1), size(data_gpu, 2))
        end

        print("Predict time...")
        CUDA.@time begin
            S2_gpu = permutedims(S_gpu, [3, 1, 2])                                              # (size(data_gpu, 2), num_samples, size(data_gpu, 1))
            # S2_gpu = unsafe_wrap(CuArray, CuPtr{Int64}(pointer(S2_gpu)), size(S2_gpu)) # not sure if correct
            # CUDA.@time S2_gpu = convert(CuArray{Int64}, S2_gpu) # too slow

            predictions = pred_func(S2_gpu)                                                     # (size(data_gpu, 1), num_samples)
            cand_gpu =  vec(mean(predictions, dims=2))                                          # (size(data_gpu, 1))
        end

        print("Expand time....")
        CUDA.@time begin
            cand = Array(cand_gpu)::Array # move to cpu

            top_k = partialsortperm(cand, 1:beam_size, rev=is_max)  
            if r == depth
                first = top_k[1]
                result = data[first,:]
                exps = cand[first]
                return result, exps, r
            end
    
            new_data = data[top_k,:]
            if g_acce != [] 
                data = expand_instance_g(instance,new_data,g_acce)
            else
                data = expand_instance(instance,new_data)
            end
            data_gpu = cu(data)
        end
    end
end 




