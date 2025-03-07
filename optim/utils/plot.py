import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_contours(model, x1, x2):
    """
    Create contours to visualize the function.

    Args:
        model (torch.nn.Module): model to create contours for
        x1 (np.ndarray): contour x-coordinates of shape (n,)
        x2 (np.ndarray): contour y-coordinates of shape (n,)

    Returns:
        X1 (np.ndarray): contour x-coordinates of shape (n, n)
        X2 (np.ndarray): contour y-coordinates of shape (n, n)
        Z (np.ndarray): function values at the contour points of shape (n, n)
    """
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = torch.tensor([x1[i], x2[j]], dtype=torch.float32)
            Z[j,i] = model(x)
    return X1, X2, Z

def plot_traj(traj, title, contours=None, log_dir=None):
    """
    Plot the trajectory of the optimization algorithm.

    Args:
        traj (numpy.ndarray): trajectory of shape (n, 2)
        title (str): title of the plot
        contours (tuple, optional): tuple of (X1, X2, Z) from create_contours
        log_dir (str, optional): directory to save the plot
    """
    plt.figure(figsize=(5, 4))
    # plot contours
    if contours is not None:
        X1, X2, Z = contours
        plt.contour(X1, X2, Z, levels=20, cmap='viridis')
        plt.colorbar(label='f(x)')
    # plot trajectory
    plt.plot(traj[:,0], traj[:,1], 'o-', markersize=4, linewidth=1.5, alpha=0.7)
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # save plot
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        plt.savefig(os.path.join(log_dir, f'{title}.png'))
        plt.close()
    else:
        plt.show()

def plot_loss_trajs(loss_trajs, fn_names, title, log_dir=None):
    """
    Plot multiple loss curves on a log scale.

    Args:
        loss_trajs (tuple): tuple of loss trajectories
        fn_names (tuple): tuple of function names
        title (str): title of the plot
        log_dir (str, optional): directory to save the plot
    """
    plt.figure(figsize=(5, 4))
    
    for (loss_traj, fn_name) in zip(loss_trajs, fn_names):
        # handle zero by adding a small epsilon
        loss_traj += 1e-10
        # plot the loss curve
        plt.plot(np.arange(len(loss_traj)), loss_traj, label=fn_name, linewidth=2)

    plt.xlabel('iterations')
    plt.ylabel('log loss')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="minor", ls="--", alpha=0.1)
    
    # save plot
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        plt.savefig(os.path.join(log_dir, f'{title}.png'))
        plt.close()
    else:
        plt.show()