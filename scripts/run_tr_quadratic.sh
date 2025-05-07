#!/bin/bash

# Define arrays of models and solvers
models=("newton" "sr1" "bfgs" "dfp")
solvers=("cg" "cauchy")

# Run experiments for each combination
for model in "${models[@]}"; do
    for solver in "${solvers[@]}"; do
        exp_name="tr-${model}-${solver}-quadratic"
        echo "Running $exp_name..."
        
        python scripts/run.py algorithm=tr algorithm.model=${model} algorithm.solver=${solver} function=quadratic experiment.name=${exp_name}
        
        # Add a small delay to prevent system overload
        sleep 0.1
    done
done

echo "All quadratic experiments completed." 