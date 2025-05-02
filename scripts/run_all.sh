#!/bin/bash

# Define arrays of algorithms and functions
algorithms=("gd" "newton" "bfgs" "lbfgs" "dfp" "tr")
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
for algo in "${algorithms[@]}"; do
    for func in "${functions[@]}"; do
        # Create experiment name
        exp_name="${algo}-${func}"
        
        # Run the experiment
        echo "Running experiment: ${exp_name}"
        python scripts/run.py function=${func} algorithm=${algo} experiment.name=${exp_name}
        
        # Add a small delay to prevent overwhelming the system
        sleep 1
    done
done

echo "All experiments completed!" 