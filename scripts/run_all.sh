#!/bin/bash

# Define arrays of models, solvers, and functions
models=("sr1" "newton" "bfgs" "dfp")
solvers=("cg" "cauchy")
functions=(
    "P1_quad_10_10"
    "P2_quad_10_1000"
    "P3_quad_1000_10"
    "P4_quad_1000_1000"
    "P5_quartic_1"
    "P6_quartic_2"
    "P7_rosenbrock_2"
    "P8_rosenbrock_100"
    "P10_exponential_10"
    "P11_exponential_1000"
    "P12_genhumps_5"
)

# Loop through all combinations
for model in "${models[@]}"; do
    for solver in "${solvers[@]}"; do
        for func in "${functions[@]}"; do
            # Create experiment name
            exp_name="tr-${model}-${solver}-${func}"
            
            # Run the experiment
            echo "Running experiment: ${exp_name}"
            python scripts/run.py algorithm=tr algorithm.model=${model} algorithm.solver=${solver} function=${func} experiment.name=${exp_name}
            
            # Add a small delay to prevent overwhelming the system
            sleep 1
        done
    done
done

echo "All experiments completed!" 