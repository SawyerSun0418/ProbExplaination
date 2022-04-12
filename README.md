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