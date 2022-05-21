
using ProbabilisticCircuits
using CUDA
using DataFrames
include("./LR.jl")

function init_instance(instance::AbstractVector)

    n=size(instance)[1]
    m=Array{Union{Missing, Int64}}(missing, n, n)           # x1 _ ... _
    for i in 1:n                                            # _  x2 ... _
        m[i,i]=instance[i]                                 #_  _ ...  _
    end
    return m                                                # ..........xn
end

function expand_instance(instance::AbstractVector,m::AbstractMatrix)
    
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
    return ret
end

function beam_search(pc::ProbCircuit, k::Int,depth::Int;instance=[1, 1, 1, 1, 4, 2, 2, 2, 3, 5, 2, 2, 2, 1, 5, 2, 2, 4, 4, 3, 1, 1, 1, 1, 3, 1, 1, 2, 2, 3],sample_size=100 )
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    logis=train_LR()
    #instance=instance .-= 1
    data=init_instance(instance)
    data_gpu = cu(data)
    
    new_data=[]
    depth=depth+1
    for r in 1:depth
        S = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)  
        S = Array{Int64}(S) # (sample_size, data_size, num_features)
        num_cand=size(data)[1]
        cand=[]
        top_k=[]
        for n in 1:num_cand
            prediction_sum=0
            prediction = ScikitLearn.predict_proba(logis, S[:,n,:])[:,2]
            prediction_sum=sum(prediction)
            exp=prediction_sum/sample_size
            append!(cand,exp)
        end
        #println(cand)

        top_k=partialsortperm(cand, 1:k, rev=true)
        if r==depth
            first=top_k[1]
            result=data[first,:]
            display(result)
            return   
        end
        #println(top_k)

        new_data=data[top_k,:]
        #display(new_data)
        #return
        data=expand_instance(instance,new_data)
        #println(size(data))
        data_gpu=cu(data)
    end

    display(new_data)
end 
#instance=[1, 4, 1, 1, 5, 3, 2, 1, 3, 4, 3, 5, 3, 2, 5, 4, 3, 2, 5, 3, 1, 4, 1, 1, 5, 2, 1, 1, 4, 3]
#instance .-= 1
pc = Base.read("trained_pc.jpc", ProbCircuit)
beam_search(pc,5,7)

# beam_...(pv, instance; b=3, k=5, samples=100)