#!/usr/bin/env python3
"""
Run ML optimization with specified configuration.

This script reads a configuration file, loads the specified dataset,
and runs the optimization until the gradient evaluation budget is reached.
"""
import logging
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(function_config):
    """
    Load dataset from function config.
    
    Args:
        function_config (dict): function config containing dataset info
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, f_star, initial_weights)
    """
    # Load dataset
    data = np.load(function_config.data_file, allow_pickle=True).item()
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    
    # Get optimal value and initial weights
    function_file = function_config._target_.split('.')[-2]  # get 'lr' or 'ls' from path
    f_star = data[f"f_star_{function_file}"]
    initial_weights = torch.tensor(data['initial_point'], dtype=torch.float32, requires_grad=True)
    
    return X_train, X_test, y_train, y_test, f_star, initial_weights

def create_function(function_config, initial_weights):
    """
    Create function from config.
    
    Args:
        function_config (dict): function config
        initial_weights (torch.Tensor): initial weights for the function
        
    Returns:
        function instance
    """
    # extract function class and parameters
    function_cls = hydra.utils.get_class(function_config._target_)
    function_params = {k: v for k, v in function_config.items() if not k.startswith('_')}
    del function_params['data_file']
    
    # add initial weights to function parameters
    function_params['w'] = initial_weights
    
    # create function instance
    function = function_cls(**function_params)
    
    return function

def create_optimizer(algorithm_config, function, w):
    """
    Create optimizer from config.
    
    Args:
        algorithm_config (dict): algorithm config
        function (torch.nn.Module): function to evaluate
        w (torch.Tensor): weights to optimize
        
    Returns:
        optimizer instance
    """
    # extract optimizer class and parameters
    algorithm_cls = hydra.utils.get_class(algorithm_config._target_)
    algorithm_params = {k: v for k, v in algorithm_config.items() if not k.startswith('_')}
    
    # create optimizer instance
    if algorithm_config._target_ == 'optim.algorithms.newton.Newton':
        optimizer = algorithm_cls([w], function, **algorithm_params)
    else:
        optimizer = algorithm_cls([w], **algorithm_params)
    return optimizer

def run_optimization(function, optimizer, X_train, X_test, y_train, y_test, f_star, config):
    """
    Run optimization until gradient evaluation budget is reached.
    
    Args:
        function (torch.nn.Module): function to evaluate
        optimizer (torch.optim.Optimizer): optimizer to use
        X_train (torch.Tensor): training features
        X_test (torch.Tensor): test features
        y_train (torch.Tensor): training labels
        y_test (torch.Tensor): test labels
        f_star (float): optimal function value
        config (dict): configuration
    """
    # experiment settings
    batch_size = config.experiment.batch_size
    n = len(X_train)  # number of training examples
    max_grad_evals = 20 * n  # budget: 20n gradient evaluations
    grad_evals = 0
    
    # metrics to track
    metrics = defaultdict(list)
    
    # initial evaluation
    f = function(X_train, y_train)
    f.backward()
    grad_evals += batch_size
    
    # store initial metrics
    metrics['train_loss'].append(torch.abs(f_star - f).item())
    metrics['test_loss'].append(torch.abs(f_star - function(X_test, y_test)).item())
    metrics['train_acc'].append(function.accuracy(X_train, y_train))
    metrics['test_acc'].append(function.accuracy(X_test, y_test))
    metrics['grad_evals'].append(grad_evals)
    
    # log initial state
    logger.info(f"Starting optimization")
    logger.info(f"Initial train loss: {metrics['train_loss'][-1]}")
    logger.info(f"Initial test loss: {metrics['test_loss'][-1]}")
    logger.info(f"Initial train accuracy: {metrics['train_acc'][-1]}")
    logger.info(f"Initial test accuracy: {metrics['test_acc'][-1]}")
    
    # optimization loop
    while grad_evals < max_grad_evals:
        # get random batch
        indices = torch.randperm(n)[:batch_size]
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        
        # step with batch data
        optimizer.step(fn_cls=lambda: function(X_batch, y_batch))
        optimizer.zero_grad()
        
        # evaluate on full training set
        f = function(X_train, y_train)
        f.backward()
        grad_evals += batch_size
        
        # store metrics
        metrics['train_loss'].append(torch.abs(f_star - f).item())
        metrics['test_loss'].append(torch.abs(f_star - function(X_test, y_test)).item())
        metrics['train_acc'].append(function.accuracy(X_train, y_train))
        metrics['test_acc'].append(function.accuracy(X_test, y_test))
        metrics['grad_evals'].append(grad_evals)
        
        # log progress
        if grad_evals % (n // 10) == 0 or grad_evals >= max_grad_evals:
            logger.info(f"Gradient evals: {grad_evals}/{max_grad_evals}")
            logger.info(f"Train loss: {metrics['train_loss'][-1]}")
            logger.info(f"Test loss: {metrics['test_loss'][-1]}")
            logger.info(f"Train accuracy: {metrics['train_acc'][-1]}")
            logger.info(f"Test accuracy: {metrics['test_acc'][-1]}")
    
    # final results
    logger.info(f"Optimization completed")
    logger.info(f"Final train loss: {metrics['train_loss'][-1]}")
    logger.info(f"Final test loss: {metrics['test_loss'][-1]}")
    logger.info(f"Final train accuracy: {metrics['train_acc'][-1]}")
    logger.info(f"Final test accuracy: {metrics['test_acc'][-1]}")
    
    # create log directory
    log_path = Path(config.experiment.name)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # save configuration
    with open(log_path / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(config))
    
    # save metrics
    np.save(log_path / 'metrics.npy', dict(metrics))
    
    # plot results
    plt.figure(figsize=(12, 8))
    
    # plot losses
    plt.subplot(2, 1, 1)
    plt.semilogy(metrics['grad_evals'], metrics['train_loss'], 'b-', label='Train Loss')
    plt.semilogy(metrics['grad_evals'], metrics['test_loss'], 'r-', label='Test Loss')
    plt.xlabel('Gradient Evaluations')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss vs Gradient Evaluations')
    plt.legend()
    plt.grid(True)
    
    # plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(metrics['grad_evals'], metrics['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(metrics['grad_evals'], metrics['test_acc'], 'r-', label='Test Accuracy')
    plt.xlabel('Gradient Evaluations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Gradient Evaluations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(log_path / 'results.png')
    plt.close()
    
    logger.info(f"Results saved to {log_path}")

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # print the configuration
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # load dataset from function config
    X_train, X_test, y_train, y_test, f_star, w = load_dataset(cfg.function)
    
    # create model
    function = create_function(cfg.function, w)
    
    # create optimizer
    optimizer = create_optimizer(cfg.algorithm, function, w)
    
    # run optimization
    run_optimization(function, optimizer, X_train, X_test, y_train, y_test, f_star, cfg)

if __name__ == "__main__":
    main() 