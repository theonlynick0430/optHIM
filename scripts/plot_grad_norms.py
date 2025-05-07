#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import optHIM.utils.plot as plot_utils

# Define algorithms and functions
algorithms = ["gd", "newton", "bfgs", "lbfgs", "dfp"]
functions = [
    "P4_quad_1000_1000",
    "P8_rosenbrock_100"
]

# Create plots directory
plots_dir = Path("outputs/plots")
plots_dir.mkdir(exist_ok=True)

# Process each problem
for func in functions:
    print(f"Processing {func}...")
    
    # Collect trajectories and algorithm names
    grad_norm_trajs = []
    algo_names = []
    
    for algo in algorithms:
        exp_name = f"{algo}-{func}"
        # Use Wolfe variants for specific problems
        if func == "P4_quad_1000_1000" and algo == "lbfgs":
            exp_name = "lbfgs-w-P4_quad_1000_1000"
        elif func == "P8_rosenbrock_100" and algo == "dfp":
            exp_name = "dfp-w-P8_rosenbrock_100"
            
        grad_norm_file = Path(f"outputs/{exp_name}/grad_norm_traj.npy")
        
        if grad_norm_file.exists():
            grad_norm_traj = np.load(grad_norm_file)
            grad_norm_trajs.append(grad_norm_traj)
            algo_names.append(algo)
    
    if grad_norm_trajs:
        # Create plot
        plt.figure(figsize=(5, 4))
        
        for (grad_norm_traj, algo_name) in zip(grad_norm_trajs, algo_names):
            # handle zero by adding a small epsilon
            grad_norm_traj += 1e-10
            # plot the gradient norm curve
            plt.plot(np.arange(len(grad_norm_traj)), grad_norm_traj, label=algo_name, linewidth=2)

        plt.xlabel('iterations')
        plt.ylabel('log grad norm')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.grid(True, which="minor", ls="--", alpha=0.1)
        
        # save plot
        plt.savefig(plots_dir / f"{func}.png")
        plt.close()
        print(f"Created plot for {func}")
    else:
        print(f"No gradient norm trajectories found for {func}")

print("All plots generated in outputs/plots/") 