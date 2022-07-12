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
julia --project -i src/beam_search.jl
```

## plot explanation

run plot/plot.ipynb with jupyter notebook