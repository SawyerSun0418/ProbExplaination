
using ProbabilisticCircuits
using CUDA
using DataFrames
using CSV
using StatsBase: sample
using XGBoost

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
    #print(size(m))
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
                newm = copy(m[i,:])
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



function beam_search(pc::ProbCircuit, instance;is_max=true,is_Flux=true,is_xgb=false, th=1, k=3,depth=30,sample_size=100,g_acce=[],n=size(instance, 1))
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    if is_Flux
        logis=load_model("src/model/flux_NN_MNIST.bson") |> gpu
        # instance = instance |> gpu
        # ins_prob = logis(instance)

    elseif is_xgb
        x_train = Matrix(DataFrame(CSV.File("data/adult/x_train_oh.csv")))     ###temperory solution for not able to load
        y_train = vec(Matrix(DataFrame(CSV.File("data/adult/y_train.csv"))))
        dtrain = DMatrix(x_train, label=y_train)
        logis = xgboost(dtrain, num_round = 6, max_depth = 6, eta = 0.5, eval_metric = "error", objective = "binary:logitraw")
        temp = reshape(instance, 1, :)
        ins_prob = XGBoost.predict(logis,temp)
    else
        logis=train_LR()
        ins_prob = ScikitLearn.predict_proba(logis, [instance])[:,2]
    end
    
    # println("p(c|x) is ",ins_prob)
    if g_acce!=[] 
        data=init_instance_g(instance,g_acce,n)
    else
        data=init_instance(instance)
    end
    data_gpu = cu(data)
    
    new_data=[]
    for r in 1:depth
        S_gpu = ProbabilisticCircuits.sample(bpc, sample_size, data_gpu)  
        S_gpu = Array{Int64}(S_gpu) # (sample_size, data_size, num_features)
        num_cand=size(data_gpu)[1]  #number of candidates in one step of beam search
        cand = Array{Float64}(undef, num_cand)
        top_k=[]
        print("predict time")
        @time begin
            for n in 1:num_cand
                prediction_sum=0
                if is_Flux
                    prediction= logis(S_gpu[:,n,:]')

                elseif is_xgb
                    prediction = XGBoost.predict(logis,S[:,n,:])
                else
                    prediction = ScikitLearn.predict_proba(logis, S[:,n,:])[:,2]   
                end
                prediction_sum=sum(prediction)
                exp=prediction_sum/sample_size
                cand[n]=exp
            end
        end
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
            @time begin
                data=expand_instance_g(instance,new_data,g_acce)
            end
        else
            @time begin
                data=expand_instance(instance,new_data)
            end
        end
        data_gpu=cu(data)
    end

end 



