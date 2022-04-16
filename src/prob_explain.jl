using CUDA
using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihood, loglikelihoods, full_batch_em, mini_batch_em
using MLDatasets

include("./dataframe.jl")

function iris_cpu()
    df=return_df()
    train_cpu = Matrix(df)
    test_cpu = Matrix(df)
    train_cpu, test_cpu
end

function iris_gpu()
    cu.(iris_cpu())
end

function run(; batch_size = 512, num_epochs1 = 10, num_epochs2 = 10, num_epochs3 = 20, 
             pseudocount = 0.1, latents = 32, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.95)
    train, test = iris_cpu()
    train_gpu, test_gpu = iris_gpu()
    
    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(train, latents; num_cats = 5, pseudocount = 0.01, input_type = Categorical);
    init_parameters(pc; perturbation = 0.4);
    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)

    softness    = 0
    @time mini_batch_em(bpc, train_gpu, num_epochs1; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2, debug = false)

    ll1 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll1)")
    			 
    @time mini_batch_em(bpc, train_gpu, num_epochs2; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3)

    ll2 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll2)")
    
    @time full_batch_em(bpc, train_gpu, num_epochs3; batch_size, pseudocount, softness)

    ll3 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll3)")

    print("AAAAAA update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc)
    write("trained_pc.jlc", pc)
    pc
end


pc=run()
#Base.write("trained_pc.psdd",pc)
