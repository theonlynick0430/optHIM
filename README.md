# OptHIM

A collection of **H**ybrid **I**terative **M**ethods for continuous optimization in PyTorch.


## Overview

This repository contains implementations of various optimization algorithms using PyTorch for automatic differentiation. The library is designed to be modular, configurable via Hydra, and easy to extend with new algorithms and functions. The repository is compatible with cpu, gpu and subclasses the native PyTorch optimizer for seamless integration.  


## Available Algorithms

### Line Search Methods
All line search methods support both Armijo backtracking and Wolfe conditions for step size selection.

- **Gradient Descent** (`gd.py`): A first-order optimization method that iteratively moves in the direction of steepest descent, scaled by a step size determined through line search.
- **Newton's Method** (`newton.py`): A second-order method that uses the Hessian matrix to compute exact Newton steps, providing quadratic convergence near optimal points.
- **BFGS** (`bfgs.py`): A quasi-Newton method that approximates the Hessian using rank-two updates, maintaining positive definiteness while avoiding explicit Hessian computation.
- **L-BFGS** (`lbfgs.py`): A memory-efficient variant of BFGS that stores only a limited history of updates, making it suitable for large-scale optimization problems.
- **DFP** (`dfp.py`): A quasi-Newton method that uses a different rank-two update formula to approximate the inverse Hessian, offering an alternative to BFGS.

### Trust Region Methods 
The trust region framework (`tr.py`) supports flexible combinations of models and subproblem solvers:

**Available Models:**
- **Newton** model: Uses exact second-order information
- **SR1** model: Symmetric Rank-One quasi-Newton approximation
- **BFGS** model: Broyden-Fletcher-Goldfarb-Shanno quasi-Newton approximation
- **DFP** model: Davidon-Fletcher-Powell quasi-Newton approximation

**Available Subproblem Solvers:**
- **Conjugate Gradient (CG)**: Iteratively solves the trust region subproblem
- **Cauchy Point**: Computes an approximate solution along the steepest descent direction



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/theonlynick0430/optHIM
   cd optHIM
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n optHIM python=3.10
   conda activate optHIM
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Basic Usage

Run optimization with default settings:
```bash
python scripts/run.py
```

### Configuration Options

The optimization can be configured through command line arguments. Here are the main categories of options:

1. **Algorithm Selection and Parameters:**
```bash
# Choose algorithm
python scripts/run.py algorithm=gd
python scripts/run.py algorithm=newton
python scripts/run.py algorithm=bfgs
python scripts/run.py algorithm=lbfgs
python scripts/run.py algorithm=dfp
python scripts/run.py algorithm=tr

# Configure algorithm parameters
python scripts/run.py algorithm=gd algorithm.step_size=0.01
python scripts/run.py algorithm=newton algorithm.step_type=armijo
python scripts/run.py algorithm=tr algorithm.model=newton algorithm.solver=cg
```

2. **Problem Selection:**
```bash
# Choose optimization problem
python scripts/run.py function=quadratic
python scripts/run.py function=rosenbrock
python scripts/run.py function=quartic
```

3. **Experiment Settings:**
```bash
# Configure experiment parameters
python scripts/run.py experiment.name=test_run
python scripts/run.py experiment.max_iter=200
python scripts/run.py experiment.tol=1e-8
```

4. **Combining Options:**
```bash
# Example: Run BFGS on Rosenbrock with custom settings
python scripts/run.py algorithm=bfgs function=rosenbrock experiment.name=bfgs_rosenbrock experiment.max_iter=500

# Example: Run Trust Region with Newton model and CG solver
python scripts/run.py algorithm=tr algorithm.model=newton algorithm.solver=cg experiment.tol=1e-6
```

### Configuration Files

The base configurations are defined in the `configs` directory:
- `config.yaml`: Main configuration file
- `function/*.yaml`: Function configurations (quadratic, rosenbrock, etc.)
- `algorithm/*.yaml`: Algorithm configurations (gd, newton, bfgs, etc.)

Command line arguments override the default values in these configuration files. The final configuration used in each run is saved in the output directory specified by `experiment.name`.


## Author

Nikhil Sridhar and Sajiv Shah with inspiration from Prof. Albert Berahas's MATH 562 class @ the University of Michigan. 


## License

MIT License