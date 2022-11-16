using CUDA
using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihood, loglikelihoods, full_batch_em, mini_batch_em
using MLDatasets
include("./dataframe.jl")

function mnist_cpu()
    #train_cpu = return_MNIST_df()
    #test_cpu = return_MNIST_df_t()
    #train_cpu=Matrix(train_cpu)[:,2:end]
    #test_cpu=Matrix(test_cpu)[:,2:end]
    train_cpu = Matrix(DataFrame(CSV.File("data/adult/x_train_oh.csv")))
    test_cpu = Matrix(DataFrame(CSV.File("data/adult/x_test_oh.csv")))
    train_cpu, test_cpu
end

function mnist_gpu()
    cu.(mnist_cpu())
end

function truncate(data::Matrix; bits)
    data .÷ 2^bits
end

function run(; batch_size = 512, num_epochs1 = 100, num_epochs2 = 100, num_epochs3 = 20, 
             pseudocount = 0.1, latents = 32, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.95)
    train, test = mnist_cpu()
    train_gpu, test_gpu = mnist_gpu()
    
    trunc_train = cu(truncate(train; bits = 4))

    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(trunc_train[1:5000,:], latents; num_cats = 256, pseudocount = 0.1, input_type = Categorical);
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

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc)
    write("adult.jpc", pc)
    ll1, ll2, ll3, batch_size, pseudocount, latents
end


run()