include("./models.jl")
using Plots
using StatsBase: sample
function rand_instance(num)
    df=return_MNIST_df_t()
    df=Matrix(df)
    df=[df[i,:] for i in 1:size(df,1)]
    result=sample(df, num; replace=false)
    return result
end

function map()
    rand_ins=rand_instance(5)
    logis=load_model("src/model/flux_NN_MNIST.bson")
    i=0
    gr()
    for ins in rand_ins
        i=i+1
        x=ins[2:end]
        l=ins[1]
        gs=gradient(Flux.params(x,[l])) do 
            Flux.binarycrossentropy(logis(x), [l])
        end
        r=abs.(gs[x])
        m=reshape(r, 28, 28)
        original_m=reshape(x,28,28)
        p=heatmap(1:28,1:28,m,title="$l", size=(400,500))
        o_p=heatmap(1:28,1:28,original_m,title="$l", size=(400,500))
        png(p,"heatmap_label$l _$i")
        png(o_p,"original_label$l _$i")
    end
end
