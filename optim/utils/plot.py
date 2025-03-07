import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_contours(model, x1, x2):
    """
    Create contours to visualize the function.

    Args:
        model (torch.nn.Module): model to create contours for
        x1 (torch.Tensor): contour x-coordinates of shape (n,)
        x2 (torch.Tensor): contour y-coordinates of shape (n,)

    Returns:
        X1 (torch.Tensor): contour x-coordinates of shape (n, n)
        X2 (torch.Tensor): contour y-coordinates of shape (n, n)
        Z (torch.Tensor): function values at the contour points of shape (n, n)
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
        traj (torch.Tensor): trajectory of shape (n, 2)
        title (str): title of the plot
        contours (tuple, optional): contours to plot
        log_dir (str, optional): directory to save the plot
    """
    plt.figure()
    # plot contours
    if contours is not None:
        X1, X2, Z = contours
        plt.contour(X1, X2, Z, levels=20)
        plt.colorbar(label='f(x)')
    # plot trajectory
    plt.plot(traj[:,0], traj[:,1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.grid(True)
    if log_dir is not None:
        plt.savefig(os.path.join(log_dir, f'{title}.png'))
    else:
        plt.show()