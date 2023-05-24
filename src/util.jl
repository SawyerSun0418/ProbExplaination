using CSV
# using ROCAnalysis
using MLBase
#using Plots
# using Lathe
using DataFrames
using CategoricalArrays
using FreqTables
using Random

function binvec(x::AbstractVector, n::Int,
    rng::AbstractRNG=Random.default_rng())
    n > 0 || throw(ArgumentError("number of bins must be positive"))
    l = length(x)

    # find bin sizes
    d, r = divrem(l, n)
    lens = fill(d, n)
    lens[1:r] .+= 1
    # randomly decide which bins should be larger
    shuffle!(rng, lens)

    # ensure that we have data sorted by x, but ties are ordered randomly
    df = DataFrame(id=axes(x, 1), x=x, r=rand(rng, l))
    sort!(df, [:x, :r])

    # assign bin ids to rows
    binids = reduce(vcat, [fill(i, v) for (i, v) in enumerate(lens)])
    df.binids = binids

    # recover original row order
    sort!(df, :id)
    return df.binids
end

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function data_pre(df)
    train, test = splitdf(df, 0.6);
    train=Matrix(train)
    test=Matrix(test)
    y_train=permutedims(train[:,1])
    X_train=permutedims(train[:,2:end])
    y_test=test[:,1]
    X_test=test[:,2:end]
    train_data=[(X_train,y_train)]
    test_data=(X_test,y_test)
    return train_data, test_data
    
end



function return_df()    
    df = DataFrame(CSV.File("data/data.csv"))
    select!(df, Not([:id]))
    for column in names(df)
        if column!="diagnosis"
            df[!,column]=binvec(df[!,column],5)
        end
    end
    return df
end

function return_MNIST_df()    
    df = DataFrame(CSV.File("data/mnist_3_5_train.csv"))
    return df
end

function return_MNIST_df_t()    
    df = DataFrame(CSV.File("data/mnist_3_5_test.csv"))
    return df
end

function sdp_exp(pc::ProbCircuit, logis, out_dir, id, exp, original, explanation, label ;sample_size=1000)
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    label_m=Matrix(label)
    exp_m=Matrix(exp)
    original_m=Matrix(original)
    explanation_m=Matrix(explanation)
    sdp=Vector{Float64}[]
    exp_3=0
    exp_5=0
    size_3=0
    size_5=0
    explanation_gpu=cu(explanation_m)
    S=ProbabilisticCircuits.sample(bpc, sample_size, explanation_gpu)
    S = Array{Int64}(S)
    for i in 1:size(original_m,1)
        pred=logis(original_m[i,:])
        pred = [if x < 0.5 0 else 1 end for x in pred];
        temp=0
        for j in 1:sample_size
            exp_pred=logis(S[j,i,:])
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
    ave=mean(sdp)
    push!(sdp,ave)
    s=std(sdp)
    push!(sdp,s)
    ave_3=exp_3/size_3
    ave_5=exp_5/size_5
    sdp=reduce(vcat,sdp')
    df=DataFrame(sdp,:auto)

    CSV.write(out_dir*"/"*"experiment_sdp"*"_"*id*".csv",df)
    println("ave exp for 3:",ave_3)
    println("ave exp for 5:",ave_5)
end


function get_exp(exp,label, out_dir, id)
    label_m=Matrix(label)
    exp_m=Matrix(exp)
    exp_3=Vector{Float64}[]
    exp_5=Vector{Float64}[]
    for i in 1:size(exp_m,1)
        if label_m[i,1]==1
            push!(exp_3,[exp_m[i,1]])
        else
            push!(exp_5,[exp_m[i,1]])
        end
    end
    ave_3=mean(exp_3)
    push!(exp_3,ave_3)
    std_3=std(exp_3)
    push!(exp_3,std_3)
    ave_5=mean(exp_5)
    push!(exp_5,ave_5)
    std_5=std(exp_5)
    push!(exp_5,std_5)
    exp_3=reduce(vcat,exp_3')
    df_3=DataFrame(exp_3,:auto)
    CSV.write(out_dir*"/"*"experiment_exp_1"*"_"*id*".csv",df_3)
    exp_5=reduce(vcat,exp_5')
    df_5=DataFrame(exp_5,:auto)
    CSV.write(out_dir*"/"*"experiment_exp_0"*"_"*id*".csv",df_5)
end

function sdp_h(pc::ProbCircuit, logis, out_dir, id, pred, explanation ;sample_size=1000)
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    explanation_m=Matrix(explanation)
    sdp=Vector{Float64}[]
    exp_3=0
    exp_5=0
    explanation_gpu=cu(explanation_m)
    S=ProbabilisticCircuits.sample(bpc, sample_size, explanation_gpu)
    S = Array{Int64}(S)
    for i in 1:size(explanation,1)
        temp=0
        for j in 1:sample_size
            exp_pred=logis(S[j,i,:])
            exp_pred = [if x < 0.5 0 else 1 end for x in exp_pred];
            for e in exp_pred
                if e == pred
                    temp += 1
                end
            end
        end
        temp=temp/sample_size
        push!(sdp,[temp])
    end
    ave=mean(sdp)
    push!(sdp,ave)
    s=std(sdp)
    push!(sdp,s)
    sdp=reduce(vcat,sdp')
    df=DataFrame(sdp,:auto)

    CSV.write(out_dir*"/"*"sdp_history_instance"*"_"*"$id"*".csv",df)
end