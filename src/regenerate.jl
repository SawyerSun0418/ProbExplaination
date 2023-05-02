using ProbabilisticCircuits
using CUDA
using DataFrames
using CSV
include("./models.jl")

function regenerate(pc::ProbCircuit, explanation::String, ins_num; sample_size=1)
    CUDA.@time bpc = CuBitsProbCircuit(pc);
    explanation_m=Matrix(DataFrame(CSV.File(explanation)))
    regenerate=[]
    explanation_gpu=cu(explanation_m)
    S=ProbabilisticCircuits.sample(bpc, sample_size, explanation_gpu)
    S = Array{Int64}(S)
    for i in 1:ins_num
        for j in 1:sample_size
            push!(regenerate, S[j,i,:])
            
        end
    end
    regenerate=reduce(vcat,regenerate')
    df=DataFrame(regenerate,:auto)
    return df
end


#pc = Base.read("mnist35.jpc", ProbCircuit)
#sdp_exp(pc,"experiment_exp_c.csv","experiment_original_ins_c.csv","experiment_plot_c.csv","experiment_label_c.csv")
#get_exp("experiment_exp_c.csv","experiment_label_c.csv")
#pc = Base.read("circuits/mnist35.jpc", ProbCircuit)
#regenerate(pc,"experiments/MNIST/LR/10/experiment_plot_LR_t.csv",1)
