#!/usr/bin/env python3
"""
Run optimization with specified configuration.

This script reads a configuration file, creates the specified function and optimizer,
and runs the optimization until convergence or maximum iterations.
"""
import os
import sys
import argparse
import logging
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import plotting utilities
import optim.utils.plot as plots

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser(description='run optimization with config file')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    return parser.parse_args()


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
    function_params = {k: v for k, v in function_config.items() if not k.startswith('_')}
    # remove initial_point from function parameters if present
    initial_point = function_params.pop('initial_point', None)
    # create function instance
    return function_cls(**function_params), initial_point

def create_optimizer(algorithm_config, function, x):
    """
    Create optimizer from config.

    Args:
        algorithm_config (dict): algorithm config
        function (torch.nn.Module): function to evaluate
        x (torch.Tensor): input to optimize

    Returns:
        optimizer instance
    """
    # extract optimizer class and parameters
    algorithm_cls = hydra.utils.get_class(algorithm_config._target_)
    algorithm_params = {k: v for k, v in algorithm_config.items() if not k.startswith('_')}
    # create optimizer instance
    if algorithm_config._target_ == 'optim.algorithms.newton.Newton':
        optimizer = algorithm_cls([x], function, **algorithm_params)
    else:
        optimizer = algorithm_cls([x], **algorithm_params)
    return optimizer

def run_optimization(function, x, optimizer, config, log_dir=None):
    """
    Run optimization until convergence or maximum iterations.

    Args:
        function (torch.nn.Module): function to evaluate
        x (torch.Tensor): input to optimize
        optimizer (torch.optim.Optimizer): optimizer to use
        config (dict): configuration
        log_dir (str, optional): directory to save logs
    """
    # optimization settings
    max_iter = config.optimizer.get('max_iter', 100)
    tol = config.optimizer.get('tol', 1e-6)
    save_traj = config.optimizer.get('save_traj', True)
    plot_traj = config.optimizer.get('plot_traj', True)
    plot_loss = config.optimizer.get('plot_loss', True)

    # solution
    x_star = function.solution()
    f_star = function(x_star)
    
    traj = []
    loss_traj = []
    # store initial point
    if save_traj:
        traj.append(x.cpu().numpy())
    # compute initial loss
    initial_loss = torch.abs(f_star - function(x))
    loss_traj.append(initial_loss.item())

    # compute convergence threshold
    initial_grad_norm = torch.max(torch.abs(x.grad)).item()
    conv_thresh = tol * max(initial_grad_norm, 1.0)
    
    # log initial state
    logger.info(f"Starting optimization")
    logger.info(f"Initial point: {x.tolist()}")
    logger.info(f"Initial loss: {loss_traj[0]}")
    logger.info(f"Convergence threshold: {conv_thresh}")
    
    # optimization loop
    for i in range(max_iter):
        optimizer.zero_grad()
        f = function(x)
        f.backward()
        optimizer.step(fn_cls=lambda: function(x))
        
        # save metrics
        if save_traj:
            traj.append(x.cpu().numpy())
        loss = torch.abs(f_star - function(x))
        loss_traj.append(loss.item())

        grad_norm = torch.max(torch.abs(x.grad)).item()
        
        # log progress
        if i % 10 == 0 or i == max_iter - 1:
            logger.info(f"Iteration {i+1}/{max_iter}, Loss: {loss_traj[-1]:.6f}, Grad norm: {grad_norm:.6f}")
        
        # check convergence
        if grad_norm <= conv_thresh:
            logger.info(f"Converged at iteration {i+1} with gradient norm {grad_norm:.6f} <= {conv_thresh:.6f}")
            break
    
    # final results
    logger.info(f"Optimization completed")
    logger.info(f"Final point: {x.tolist()}")
    logger.info(f"Final loss: {loss_traj[-1]:.6f}")
    logger.info(f"Final gradient norm: {grad_norm:.6f}")
    
    # save results if log_dir is provided
    if log_dir is not None:
        # create log directory
        log_path = Path(log_dir) / config.experiment.name
        log_path.mkdir(parents=True, exist_ok=True)
        
        # save configuration
        with open(log_path / 'config.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(config))
        
        # save metrics
        np.savetxt(log_path / 'losses.txt', np.array(loss_traj))
        
        # save trajectory if requested
        if save_traj:
            trajectory_array = np.array(traj)
            np.save(log_path / 'trajectory.npy', trajectory_array)
        
        # create contours for 2D problems
        contours = None
        if save_traj and trajectory_array.shape[1] == 2:
            try:
                # create grid for contour plot
                x_min, x_max = trajectory_array[:, 0].min() - 1, trajectory_array[:, 0].max() + 1
                y_min, y_max = trajectory_array[:, 1].min() - 1, trajectory_array[:, 1].max() + 1
                
                x1 = np.linspace(x_min, x_max, 50)
                x2 = np.linspace(y_min, y_max, 50)
                
                # compute function values on grid
                contours = plots.create_contours(function, x1, x2)
            except Exception as e:
                logger.warning(f"Could not create contours: {e}")
        
        # plot trajectory if requested
        if plot_traj and save_traj:
            plots.plot_traj(
                trajectory_array, 
                f"{config.algorithm._target_} on {config.function._target_}",
                contours=contours,
                log_dir=log_path
            )
        
        # plot loss if requested
        if plot_loss:
            plots.plot_losses(
                [loss_traj], 
                [f"{config.algorithm._target_}"],
                f"Loss for {config.function._target_}",
                log_dir=log_path
            )
        
        logger.info(f"Results saved to {log_path}")
    
    return loss_traj, traj, grad_norm


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """main function to run optimization with the specified configuration."""
    # print the configuration
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # create model and optimizer
    function, x = create_function(cfg.function)
    optimizer = create_optimizer(cfg.algorithm, function, x)
    
    # run optimization
    log_dir = cfg.logging.get('save_dir', None) if hasattr(cfg, 'logging') else None
    loss_traj, traj, grad_norm = run_optimization(function, x, optimizer, cfg, log_dir)
    
    return loss_traj, traj, grad_norm


if __name__ == "__main__":
    args = parse_args()
    
    # override hydra's config_path with the provided config file
    config_path = os.path.dirname(args.config)
    config_name = os.path.basename(args.config).split('.')[0]
    
    # update sys.argv for hydra
    sys.argv = [sys.argv[0], f"--config-path={config_path}", f"--config-name={config_name}"]
    
    main() 