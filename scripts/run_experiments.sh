#!/bin/bash

# Create output directory
mkdir -p results

# Create log file
log_file="results/optimization_logs.txt"
echo "" > "$log_file"

# List of problems (excluding quadratic problems P1-P4)
problems=(
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

# List of algorithms and their configurations
declare -a algorithms=(
    "algorithm=gd algorithm.step_type=armijo"
    "algorithm=gd algorithm.step_type=wolfe"
    "algorithm=newton algorithm.step_type=armijo"
    "algorithm=newton algorithm.step_type=wolfe"
    "algorithm=tr algorithm.model=newton algorithm.solver=cg"
    "algorithm=tr algorithm.model=sr1 algorithm.solver=cg"
    "algorithm=bfgs algorithm.step_type=armijo"
    "algorithm=bfgs algorithm.step_type=wolfe"
    "algorithm=dfp algorithm.step_type=armijo"
    "algorithm=dfp algorithm.step_type=wolfe"
    "algorithm=lbfgs algorithm.step_type=armijo"
    "algorithm=lbfgs algorithm.step_type=wolfe"
)

# Run each problem with each algorithm
for problem in "${problems[@]}"; do
    for algo in "${algorithms[@]}"; do
        echo "Running ${algo} on ${problem}..."
        
        # Extract algorithm name for logging
        algo_name=$(echo "$algo" | cut -d' ' -f1 | cut -d'=' -f2)
        if [[ "$algo" == *"tr"* ]]; then
            model=$(echo "$algo" | grep -o "model=[^ ]*" | cut -d'=' -f2)
            solver=$(echo "$algo" | grep -o "solver=[^ ]*" | cut -d'=' -f2)
            algo_name="tr-${model}-${solver}"
        fi
        
        # Add separator and problem/algorithm info to log file
        echo "---" >> "$log_file"
        echo "$problem" >> "$log_file"
        echo "$algo_name" >> "$log_file"
        
        # Run the optimization and capture output
        python scripts/run.py function=${problem} ${algo} 2>&1 | tee -a "$log_file"
    done
done

echo "Experiments completed. Check $log_file for results." 