# ProbExplaination


## Installataion

- Always start Julia with project (similar to virtualenv in python)
    ```bash
    julia --project=.
    ```

- Inside Julia REPL. Update and install dependencies
    ```julia
    ] up
    ```

- Add `ProbabilisticCircuits` from master branch. Inside Julia REPL 
    ```julia
    ] add ProbabilisticCircuits#master
    ```

- [Optional] Build or precompile 

    ```julia
    ] build
    ```

    ```julia
    ] precompile
    ```

## Run

```julia
julia --project -i src/prob_explain.jl
```

## Run Beam Search

```bash
julia --project -i src/beam_search.jl then run_cancer() or run_MNIST()
```

## plot explanation

run plot/plot.ipynb with jupyter notebook

## experiment
experiment_exp.csv logs the expected predictions of the explanations
experiment_original_ins logs the original instances
experiment_plot logs the explanations
to plot the explanations, run plot/plot.ipynb with jupyter notebook, then run first cell

## Get SDP and EXP

```bash
julia --project -i src/beam_search.jl then run_cancer() or run_MNIST()
julia --project -i src/sdp_exp.jl
from the generated files: experiment_exp_3.csv logs the expected prediction of instances with label 1
                          experiment_exp_5.csv logs the expected prediction of instances with label 0
                          experiment_sdp.csv logs the SDP of instances
                          in these files the second last item is the average. the last item is the standard deviation
```

## Example of running run_beam_search_exp.jl

```
julia --project -i src/run_beam_search_exp.jl --dataset mnist --beam-size 3 --features-k 30 --num-sample 100 --classifier Flux_NN --circuit-path mnist35.jpc --num-output 3 --output-dir . --exp-id 123
```