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

```
julia run_beam_search_exp.jl --dataset mnist --beam_size 3 --features_k 20 --num_samples 100 --classifier model.bson/lr.xgb/... --output experiments/A/...
```