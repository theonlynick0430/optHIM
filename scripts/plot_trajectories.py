#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import optHIM.utils.plot as plot_utils
import torch

# Define algorithms and functions
algorithms = ["gd", "newton", "bfgs", "lbfgs", "dfp"]
functions = [
    "quadratic",
    "P7_rosenbrock_2"
]

# Create plots directory
plots_dir = Path("outputs/plots")
plots_dir.mkdir(exist_ok=True)

# Define colors for each algorithm
colors = {
    "gd": "blue",
    "newton": "green",
    "bfgs": "red",
    "lbfgs": "purple",
    "dfp": "orange"
}

# Process each problem
for func in functions:
    print(f"Processing {func}...")
    
    # Collect trajectories and algorithm names
    trajs = []
    algo_names = []
    
    for algo in algorithms:
        exp_name = f"{algo}-{func}"
        # Use Wolfe variant for Rosenbrock
        if func == "P7_rosenbrock_2" and algo == "dfp":
            exp_name = "dfp-w-P7_rosenbrock_2"
            
        traj_file = Path(f"outputs/{exp_name}/traj.npy")
        
        if traj_file.exists():
            traj = np.load(traj_file)
            trajs.append(traj)
            algo_names.append(algo)
    
    if trajs:
        # Create plot
        plt.figure(figsize=(5, 4))
        
        # Create contours for the function
        if func == "quadratic":
            # Load quadratic function from file
            quadratic_data = np.load("data/quadratic/quadratic2.npy", allow_pickle=True).item()
            A = quadratic_data['A']  # Hessian matrix
            b = quadratic_data['b']  # Linear term
            c = quadratic_data['c']  # Constant term
            
            # Calculate the solution: x* = -A^(-1)b
            solution = -np.linalg.solve(A, b)
            print(f"Quadratic solution: {solution}")
            
            def quadratic(x):
                x_np = np.array([x[0], x[1]], dtype=np.float32)
                return 0.5 * x_np.T @ A @ x_np + b.T @ x_np + c
            
            # Create grid for contours with extended range
            x1 = np.linspace(-2, 2, 100)
            x2 = np.linspace(-2, 4, 100)  # Extended upper limit to 4
            X1, X2, Z = plot_utils.create_contours(quadratic, x1, x2)
            plt.contour(X1, X2, Z, levels=20, cmap='viridis')
            plt.colorbar(label='f(x)')
            
        elif func == "P7_rosenbrock_2":
            # Define Rosenbrock function
            def rosenbrock(x):
                return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
            
            # Create grid for contours
            x1 = np.linspace(-2, 2, 100)
            x2 = np.linspace(-1, 3, 100)
            X1, X2, Z = plot_utils.create_contours(rosenbrock, x1, x2)
            plt.contour(X1, X2, Z, levels=20, cmap='viridis')
            plt.colorbar(label='f(x)')
            solution = np.array([1, 1])  # Known solution for Rosenbrock
            
        for (traj, algo_name) in zip(trajs, algo_names):
            color = colors[algo_name]
            # plot the trajectory (line only)
            plt.plot(traj[:, 0], traj[:, 1], '-', linewidth=2.0, alpha=0.7, color=color, label=algo_name)
            # plot end point
            plt.plot(traj[-1, 0], traj[-1, 1], 'o', markersize=8, color=color, alpha=1.0)

        # Plot solution point last (not included in legend)
        plt.plot(solution[0], solution[1], 'ro', markersize=8)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        
        # save plot
        plt.savefig(plots_dir / f"{func}_traj.png")
        plt.close()
        print(f"Created trajectory plot for {func}")
    else:
        print(f"No trajectories found for {func}")

print("All trajectory plots generated in outputs/plots/") 