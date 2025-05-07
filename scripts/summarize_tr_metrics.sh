#!/bin/bash

# Create summary file
echo "Trust Region Optimization Metrics Summary" > outputs/tr_metrics_summary.txt
echo "=======================================" >> outputs/tr_metrics_summary.txt
echo "" >> outputs/tr_metrics_summary.txt

# Define arrays of models and functions
models=("newton" "sr1" "bfgs" "dfp")
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

# Process each model
for model in "${models[@]}"; do
    echo "Model: $model" >> outputs/tr_metrics_summary.txt
    echo "----------------" >> outputs/tr_metrics_summary.txt
    
    # Process each problem
    for func in "${functions[@]}"; do
        echo "Problem: $func" >> outputs/tr_metrics_summary.txt
        
        # Get metrics for CG solver
        cg_exp_name="tr-${model}-cg-${func}"
        cg_metrics_file="outputs/${cg_exp_name}/metrics.npy"
        
        # Get metrics for Cauchy solver
        cauchy_exp_name="tr-${model}-cauchy-${func}"
        cauchy_metrics_file="outputs/${cauchy_exp_name}/metrics.npy"
        
        # Use Python to read and format the metrics
        python3 - <<EOF >> outputs/tr_metrics_summary.txt
import numpy as np

def get_metrics(file_path):
    try:
        return np.load(file_path, allow_pickle=True).item()
    except:
        return None

cg_metrics = get_metrics("$cg_metrics_file")
cauchy_metrics = get_metrics("$cauchy_metrics_file")

def format_metric(cg_val, cauchy_val, formatter=None):
    if formatter is None:
        formatter = lambda x: str(x)
    
    cg_str = formatter(cg_val) if cg_val is not None else "N/A"
    cauchy_str = formatter(cauchy_val) if cauchy_val is not None else "N/A"
    return f"{cg_str} | {cauchy_str}"

# Print metrics in the requested format
print(f"  Iterations: {format_metric(cg_metrics['iterations'] if cg_metrics else None, cauchy_metrics['iterations'] if cauchy_metrics else None)}")
print(f"  Function evaluations: {format_metric(cg_metrics['function_evals'] if cg_metrics else None, cauchy_metrics['function_evals'] if cauchy_metrics else None)}")
print(f"  Gradient evaluations: {format_metric(cg_metrics['grad_evals'] if cg_metrics else None, cauchy_metrics['grad_evals'] if cauchy_metrics else None)}")
print(f"  Time (seconds): {format_metric(cg_metrics['time'] if cg_metrics else None, cauchy_metrics['time'] if cauchy_metrics else None, lambda x: f'{x:.2f}')}")
print(f"  Converged: {format_metric(cg_metrics['converged'] if cg_metrics else None, cauchy_metrics['converged'] if cauchy_metrics else None, lambda x: 'Yes' if x else 'No')}")
EOF
        echo "" >> outputs/tr_metrics_summary.txt
    done
    echo "" >> outputs/tr_metrics_summary.txt
done

echo "Trust region metrics summary generated in outputs/tr_metrics_summary.txt" 