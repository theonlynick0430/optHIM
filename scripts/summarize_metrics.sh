#!/bin/bash

# Create summary file
echo "Optimization Metrics Summary" > outputs/metrics_summary.txt
echo "===========================" >> outputs/metrics_summary.txt
echo "" >> outputs/metrics_summary.txt

# Define arrays of algorithms and functions
algorithms=("gd" "newton" "bfgs" "lbfgs" "dfp")
wolfe_algorithms=("gd-w" "newton-w" "bfgs-w" "lbfgs-w" "dfp-w")
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

# Process each regular algorithm
for algo in "${algorithms[@]}"; do
    echo "Algorithm: $algo" >> outputs/metrics_summary.txt
    echo "----------------" >> outputs/metrics_summary.txt
    
    # Process each problem
    for func in "${functions[@]}"; do
        exp_name="${algo}-${func}"
        metrics_file="outputs/${exp_name}/metrics.npy"
        
        echo "Problem: $func" >> outputs/metrics_summary.txt
        
        if [ -f "$metrics_file" ]; then
            # Use Python to read the numpy file and format the output
            python3 - <<EOF >> outputs/metrics_summary.txt
import numpy as np
metrics = np.load("$metrics_file", allow_pickle=True).item()
print(f"  Iterations: {metrics['iterations']}")
print(f"  Function evaluations: {metrics['function_evals']}")
print(f"  Gradient evaluations: {metrics['grad_evals']}")
print(f"  Time (seconds): {metrics['time']:.2f}")
print(f"  Converged: {'Yes' if metrics['converged'] else 'No'}")
EOF
        else
            echo "  No metrics found" >> outputs/metrics_summary.txt
        fi
        echo "" >> outputs/metrics_summary.txt
    done
    echo "" >> outputs/metrics_summary.txt
done

# Process each Wolfe variant algorithm
for algo in "${wolfe_algorithms[@]}"; do
    echo "Algorithm: $algo" >> outputs/metrics_summary.txt
    echo "----------------" >> outputs/metrics_summary.txt
    
    # Process each problem
    for func in "${functions[@]}"; do
        exp_name="${algo}-${func}"
        metrics_file="outputs/${exp_name}/metrics.npy"
        
        echo "Problem: $func" >> outputs/metrics_summary.txt
        
        if [ -f "$metrics_file" ]; then
            # Use Python to read the numpy file and format the output
            python3 - <<EOF >> outputs/metrics_summary.txt
import numpy as np
metrics = np.load("$metrics_file", allow_pickle=True).item()
print(f"  Iterations: {metrics['iterations']}")
print(f"  Function evaluations: {metrics['function_evals']}")
print(f"  Gradient evaluations: {metrics['grad_evals']}")
print(f"  Time (seconds): {metrics['time']:.2f}")
print(f"  Converged: {'Yes' if metrics['converged'] else 'No'}")
EOF
        else
            echo "  No metrics found" >> outputs/metrics_summary.txt
        fi
        echo "" >> outputs/metrics_summary.txt
    done
    echo "" >> outputs/metrics_summary.txt
done

echo "Metrics summary generated in outputs/metrics_summary.txt" 