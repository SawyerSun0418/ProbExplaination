
using ProbabilisticCircuits
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
    ret = []
    num_features = size(instance)[1]
    for i in 1:num_features
        for j in 1:num_features                     # a nk*n array where each row is a possible candidate with missing feature
            if ismissing(m[i,j])
                newm = copy(m[i])
                newm[j] = instance[j]
                append!(ret, newm)         #check if need to change to [newm]
            end
        end
    end
    return ret
end

function beam_search(pc::ProbCircuit, k::Int,instance::AbstractVector,depth::Int)
    logis=train_LR()
    num_features = size(instance)[1]
    data=init_instance(instance)
    #display(data)
    sample_size=3
    new_data=[]
    for r in 1:depth
        S=ProbabilisticCircuits.sample(pc,sample_size,data; batch_size = 1)  #TODO: add pc
        num_cand=size(data)[1]
        cand=[]
        top_k=[]
        for n in 1:num_cand
            prediction_sum=0
            for i in 1:sample_size
                prediction = predict(logis, S[i,num_cand,:])
                prediction_class = [if x < 0.5 0 else 1 end for x in prediction]
                prediction_sum+=prediction_class
            exp=prediction_sum/sample_size
            cand[n]=exp
            top_k=partialsortperm(cand, 1:k, rev=true)   #check return type
            new_data=data[top_k]
            data=expand_instance(instance,new_data)
            end
        end
    end
    println(new_data)
end 
instance=[1, 4, 1, 1, 5, 3, 2, 1, 3, 4, 3, 5, 3, 2, 5, 4, 3, 2, 5, 3, 1, 4, 1, 1, 5, 2, 1, 1, 4, 3]
pc = Base.read("trained_pc.jpc", ProbCircuit)
beam_search(pc,3,instance,5)