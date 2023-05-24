using Flux
using Flux: params
include("./models.jl")

logis = load_model("models/flux_LR_MNIST_topright.bson")
# Create an empty DataFrame
df = DataFrame(Parameter = String[], Layer = String[], Index = String[], Value = Float64[])

# Access the weights and biases of the model and store them in the DataFrame
for (i, p) in enumerate(params(logis))
    layer, param_type = (i % 2 == 0) ? (div(i, 2), "Bias") : (div(i + 1, 2), "Weight")
    p_flat = vec(p) # Flatten the parameter array

    for (j, value) in enumerate(p_flat)
        push!(df, [param_type * " of Layer " * string(layer), param_type, string(j), value])
    end
end

# Write the DataFrame to a CSV file
CSV.write("model_weights_topright.csv", df)

