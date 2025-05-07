#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define models and functions
models = ["newton", "sr1", "bfgs", "dfp"]
solvers = ["cg", "cauchy"]
functions = [
    "P1_quad_10_10",
    "P2_quad_10_1000",
    "P3_quad_1000_10",
    "P4_quad_1000_1000",
    "P5_quartic_1",
    "P6_quartic_2",
    "P7_rosenbrock_2",
    "P8_rosenbrock_100",
    "P10_exponential_10",
    "P11_exponential_1000",
    "P12_genhumps_5"
]

# Create plots directory
plots_dir = Path("outputs/plots")
plots_dir.mkdir(exist_ok=True)

# Define distinct colors for each combination
colors = {
    "newton-cg": "#8c564b",    # brown
    "newton-cauchy": "#e377c2", # pink
    "sr1-cg": "#1f77b4",      # blue
    "sr1-cauchy": "#ff7f0e",  # orange
    "bfgs-cg": "#2ca02c",     # green
    "bfgs-cauchy": "#d62728", # red
    "dfp-cg": "#9467bd",      # purple
    "dfp-cauchy": "#8c564b"   # brown
}

# Define display names for each combination
display_names = {
    "newton-cg": "TR-Newton-CG",
    "newton-cauchy": "TR-Newton-Cauchy",
    "sr1-cg": "TR-SR1-CG",
    "sr1-cauchy": "TR-SR1-Cauchy",
    "bfgs-cg": "TR-BFGS-CG",
    "bfgs-cauchy": "TR-BFGS-Cauchy",
    "dfp-cg": "TR-DFP-CG",
    "dfp-cauchy": "TR-DFP-Cauchy"
}

# Process each problem
for func in functions:
    print(f"Processing {func}...")
    
    # Collect trajectories and algorithm names
    grad_norm_trajs = []
    algo_names = []
    
    for model in models:
        for solver in solvers:
            exp_name = f"tr-{model}-{solver}-{func}"
            grad_norm_file = Path(f"outputs/{exp_name}/grad_norm_traj.npy")
            
            if grad_norm_file.exists():
                grad_norm_traj = np.load(grad_norm_file)
                grad_norm_trajs.append(grad_norm_traj)
                algo_names.append(f"{model}-{solver}")
    
    if grad_norm_trajs:
        # Create plot
        plt.figure(figsize=(8, 6))
        
        for (grad_norm_traj, algo_name) in zip(grad_norm_trajs, algo_names):
            # handle zero by adding a small epsilon
            grad_norm_traj += 1e-10
            # plot the gradient norm curve with appropriate color
            plt.plot(np.arange(len(grad_norm_traj)), grad_norm_traj, 
                    label=display_names[algo_name], 
                    color=colors[algo_name],
                    linewidth=2)

        plt.xlabel('iterations')
        plt.ylabel('log grad norm')
        plt.yscale('log')
        plt.legend(loc='upper right', framealpha=0.8)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.grid(True, which="minor", ls="--", alpha=0.1)
        plt.tight_layout()
        
        # save plot
        plt.savefig(plots_dir / f"tr_{func}.png", bbox_inches='tight')
        plt.close()
        print(f"Created plot for {func}")
    else:
        print(f"No gradient norm trajectories found for {func}")

print("All plots generated in outputs/plots/") 