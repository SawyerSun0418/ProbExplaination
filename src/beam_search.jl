
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

function beam_search(pc::ProbCircuit, k::Int,instance::AbstractVector,depth::Int; )
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    logis=train_LR()
    num_features = size(instance)[1]
    data=init_instance(instance)
    data_gpu = cu(data)
    sample_size=3
    new_data=[]
    for r in 1:depth
        S = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)  
        S = Array{Int64}(S) # (sample_size, data_size, num_features)
        num_cand=size(data)[1]
        cand=[]
        top_k=[]
        for n in 1:num_cand
            prediction_sum=0
            for i in 1:sample_size
                S_df=DataFrame(radius_mean=Int[],texture_mean=Int[],perimeter_mean=Int[],area_mean=Int[],smoothness_mean=Int[],compactness_mean=Int[], 
                concavity_mean=Int[],concave_points_mean=Int[],symmetry_mean=Int[],fractal_dimension_mean=Int[],radius_se=Int[],texture_se=Int[], 
                perimeter_se=Int[],area_se=Int[],smoothness_se=Int[],compactness_se=Int[],concavity_se=Int[],concave_points_se=Int[],symmetry_se=Int[],fractal_dimension_se=Int[], 
                radius_worst=Int[],texture_worst=Int[],perimeter_worst=Int[],area_worst=Int[],smoothness_worst=Int[],compactness_worst=Int[],concavity_worst=Int[],concave_points_worst=Int[],symmetry_worst=Int[],fractal_dimension_worst=Int[])
                S_df=push!(S_df,S[i,num_cand,:])
                prediction = predict(logis, S_df)
                #prediction_class = [if x < 0.5 0 else 1 end for x in prediction]
                prediction_sum+=prediction[1]
            end
            exp=prediction_sum/sample_size
            append!(cand,exp)
        end
        top_k=partialsortperm(cand, 1:k, rev=true)   
        #println(top_k)
        new_data=data[top_k,:]
        #display(new_data)
        data=expand_instance(instance,new_data)
        #println(size(data))
        data_gpu=cu(data)
    end
    display(new_data)
end 
instance=[1, 4, 1, 1, 5, 3, 2, 1, 3, 4, 3, 5, 3, 2, 5, 4, 3, 2, 5, 3, 1, 4, 1, 1, 5, 2, 1, 1, 4, 3]
instance .-= 1
pc = Base.read("trained_pc.jpc", ProbCircuit)
beam_search(pc,3,instance,5)

# beam_...(pv, instance; b=3, k=5, samples=100)