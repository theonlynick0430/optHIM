#!/usr/bin/env python3
"""
Run optimization with specified configuration.

This script reads a configuration file, creates the specified function and optimizer,
and runs the optimization until convergence or maximum iterations.
"""
import logging
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
import optHIM.utils.plot as plot_utils
import time
from torch.autograd.functional import hessian

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_initial_point(target, function_params=None):
    """
    Get initial point based on function type.
    
    Args:
        target (str): function target path
        function_params (dict, optional): function parameters including data file
        
    Returns:
        list or None: initial point coordinates
    """
    # First check if this is a quadratic function
    if 'quadratic' in target.lower():
        if function_params and 'data_file' in function_params:
            data_file = function_params['data_file']
            if 'P1_quad_10_10' in data_file:
                rng = np.random.default_rng(0)
                return (20 * rng.random(10) - 10).tolist()
            elif 'P2_quad_10_1000' in data_file:
                rng = np.random.default_rng(0)
                return (20 * rng.random(10) - 10).tolist()
            elif 'P3_quad_1000_10' in data_file:
                rng = np.random.default_rng(0)
                return (20 * rng.random(1000) - 10).tolist()
            elif 'P4_quad_1000_1000' in data_file:
                rng = np.random.default_rng(0)
                return (20 * rng.random(1000) - 10).tolist()
    
    # Handle other function types
    if 'quartic_1' in target or 'quartic_2' in target:
        return [np.cos(np.deg2rad(70)), np.sin(np.deg2rad(70)), 
                np.cos(np.deg2rad(70)), np.sin(np.deg2rad(70))]
    elif 'rosenbrock_2' in target:
        return [-1.2, 1.0]
    elif 'rosenbrock_100' in target:
        return [-1.2] + [1.0] * 99
    elif 'exponential_10' in target:
        return [1.0] + [0.0] * 9
    elif 'exponential_1000' in target:
        return [1.0] + [0.0] * 999
    elif 'genhumps_5' in target:
        return [-506.2] + [506.2] * 4
    return None

def create_function(function_config):
    """
    Create function from config.

    Args:
        function_config (dict): function config

    Returns:
        tuple: function instance and initial point
    """
    # extract function class and parameters
    function_cls = hydra.utils.get_class(function_config._target_)
    function_params = {
        k: torch.tensor(v, dtype=torch.float32) if isinstance(v, (list, ListConfig)) else v 
        for k, v in function_config.items() if not k.startswith('_')
    }
    print(f"function_params: {function_params}")
    
    # get initial point based on function type and parameters
    initial_point = get_initial_point(function_config._target_, function_params)
    
    # load function parameters from data file if specified
    if 'data_file' in function_params:
        data = np.load(function_params['data_file'], allow_pickle=True).item()
        for key, value in data.items():
            data[key] = torch.tensor(value, dtype=torch.float32)
        function_params.update(data)
        del function_params['data_file']
    
    # set the initial point if we determined one
    if initial_point is not None:
        function_params['initial_point'] = initial_point
    
    # remove initial_point from function parameters if present
    initial_point = function_params.pop('initial_point', None).requires_grad_(True)
    
    # create function instance
    return function_cls(**function_params), initial_point

def create_optimizer(algorithm_config, x):
    """
    Create optimizer from config.

    Args:
        algorithm_config (dict): algorithm config
        x (torch.Tensor): input to optimize

    Returns:
        optimizer instance
    """
    # extract optimizer class and parameters
    algorithm_cls = hydra.utils.get_class(algorithm_config._target_)
    algorithm_params = {k: v for k, v in algorithm_config.items() if not k.startswith('_')}
    # create optimizer instance
    optimizer = algorithm_cls(x, **algorithm_params)
    return optimizer

def eval_no_grad(function, x):
    with torch.no_grad():
        return function(x)

def run_optimization(function, x, optimizer, config):
    """
    Run optimization until convergence or maximum iterations.

    Args:
        function (torch.nn.Module): function to evaluate
        x (torch.Tensor): input to optimize
        optimizer (torch.optim.Optimizer): optimizer to use
        config (dict): configuration
    """
    # experiment settings
    max_iter = config.experiment.get('max_iter', 100)
    tol = config.experiment.get('tol', 1e-6)
    save_traj = config.logging.get('save_traj', True)
    save_loss = config.logging.get('save_loss', True)
    save_grad_norm = config.logging.get('save_grad_norm', True)
    plot_traj = config.logging.get('plot_traj', True)
    plot_loss = config.logging.get('plot_loss', True)
    plot_grad_norm = config.logging.get('plot_grad_norm', True)

    # Initialize evaluation counters
    fn_evals = 0
    grad_evals = 0
    hess_evals = 0

    x_star = function.x_soln()
    f_star = function.f_soln()
    if f_star is None:
        if x_star is not None:
            f_star = function(x_star)
        else:
            logger.warning("Function has no known solution, saving and plotting loss is disabled")
            save_loss = False
            plot_loss = False

    # initial metrics
    traj = []
    loss_traj = []
    grad_norm_traj = []
    f = function(x)
    fn_evals += 1
    f.backward()
    grad_evals += 1
    initial_grad_norm = torch.max(torch.abs(x.grad)).item()
    if save_traj or plot_traj:
        traj.append(x.clone().detach().cpu().numpy())
    if save_loss or plot_loss:
        loss_traj.append((f - f_star).item())    
    if save_grad_norm or plot_grad_norm:
        grad_norm_traj.append(initial_grad_norm)

    # compute convergence threshold
    conv_thresh = tol * max(initial_grad_norm, 1.0)
    
    # log initial state
    logger.info(f"Starting optimization")
    logger.info(f"Initial point: {x.tolist()}")
    logger.info(f"Initial gradient norm: {initial_grad_norm}")
    logger.info(f"Convergence threshold: {conv_thresh}")
    
    # Start timing the optimization loop
    start_time = time.time()
    
    # Define closures with counters
    def fn_closure(_x):
        nonlocal fn_evals
        fn_evals += 1
        return eval_no_grad(function, _x)
    
    def grad_closure(_x):
        nonlocal grad_evals
        grad_evals += 1
        optimizer.zero_grad()
        function(_x).backward()
    
    def hess_closure(_x):
        nonlocal hess_evals
        hess_evals += 1
        return hessian(function, _x)
    
    # optimization loop
    for i in range(max_iter):
        # step
        optimizer.step(
            fn_cls=fn_closure,
            grad_cls=grad_closure,
            hess_cls=hess_closure
        )
        optimizer.zero_grad()
        f = function(x)
        fn_evals += 1
        f.backward()
        grad_evals += 1

        # compute metrics
        loss = None
        if f_star is not None:
            loss = (f - f_star).item()
        grad_norm = torch.max(torch.abs(x.grad)).item()
        if save_traj or plot_traj:
            traj.append(x.clone().detach().cpu().numpy())
        if save_loss or plot_loss:
            loss_traj.append(loss)
        if save_grad_norm or plot_grad_norm:
            grad_norm_traj.append(grad_norm)
        
        # log progress
        if i % 10 == 0 or i == max_iter - 1:
            logger.info(f"Iteration {i+1}/{max_iter}, Grad norm: {grad_norm:.6f}")
        
        # check convergence
        if grad_norm <= conv_thresh:
            logger.info(f"Converged at iteration {i+1} with gradient norm {grad_norm:.6f} <= {conv_thresh:.6f}")
            break
    
    # End timing the optimization loop
    end_time = time.time()
    opt_time = end_time - start_time
    logger.info(f"Optimization loop time: {opt_time:.2f} seconds")
    
    # final results
    logger.info(f"Optimization completed")
    logger.info(f"Final point: {x.tolist()}")
    logger.info(f"Final gradient norm: {grad_norm:.6f}")
    logger.info(f"Function evaluations: {fn_evals}")
    logger.info(f"Gradient evaluations: {grad_evals}")
    logger.info(f"Hessian evaluations: {hess_evals}")
        
    # create log directory
    log_path = Path("outputs") / config.experiment.name
    log_path.mkdir(parents=True, exist_ok=True)
    # save configuration
    with open(log_path / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(config))
    # save metrics
    if save_traj:
        traj = np.array(traj)
        np.save(log_path / 'traj.npy', traj)
    if save_loss:
        loss_traj = np.array(loss_traj)
        np.save(log_path / 'loss_traj.npy', loss_traj)
    if save_grad_norm:
        grad_norm_traj = np.array(grad_norm_traj)
        np.save(log_path / 'grad_norm_traj.npy', grad_norm_traj)
        
    if plot_traj:
        if traj.shape[1] != 2:
            logger.warning("Trajectory is not 2D, skipping trajectory plot")
        else:
            # create contours
            x_min, x_max = traj[:, 0].min() - 1, traj[:, 0].max() + 1
            y_min, y_max = traj[:, 1].min() - 1, traj[:, 1].max() + 1
            x1 = np.linspace(x_min, x_max, 50)
            x2 = np.linspace(y_min, y_max, 50)
            contours = plot_utils.create_contours(function, x1, x2)
            plot_utils.plot_traj(
                traj, 
                "Trajectory Plot",
                contours=contours,
                log_dir=log_path
            )
    if plot_loss:
        plot_utils.plot_loss_trajs(
            [loss_traj], 
            [f"{config.algorithm._target_}"],
            "Loss Plot",
            log_dir=log_path
        )
    if plot_grad_norm:
        plot_utils.plot_loss_trajs(
            [grad_norm_traj],
            [f"{config.algorithm._target_}"],
            "Gradient Norm Plot",
            log_dir=log_path
        )
    
    logger.info(f"Results saved to {log_path}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # print the configuration
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    # create model
    function, x = create_function(cfg.function)
    # create optimizer
    optimizer = create_optimizer(cfg.algorithm, x)
    # run optimization
    run_optimization(function, x, optimizer, cfg)

if __name__ == "__main__":
    main() 