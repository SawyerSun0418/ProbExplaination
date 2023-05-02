include("./models.jl")
using Plots
using Flux
using Flux: params

function map()
    logis=load_model("models/flux_LR_MNIST.bson")
   # Extract the weights of the first layer
    first_layer_weights = abs.(params(logis.layers[1])[1])

    # Check if the input size matches 28x28
    if size(first_layer_weights, 2) != 28*28
        println(size(first_layer_weights))
        error("The input size of the first layer does not match 28x28.")
    end

    # Calculate the heatmap by reshaping the weights
    heatmap_data = reshape(first_layer_weights', 28, 28)

    # Plot the heatmap
    heatmap_plot = heatmap(heatmap_data, c=:viridis, aspect_ratio=1, legend=:none, title="28x28 Heatmap of First Layer Weights")

    savefig(heatmap_plot, "heatmap_output.png")
    return heatmap_data
end
