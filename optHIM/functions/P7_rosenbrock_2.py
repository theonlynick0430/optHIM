import torch.nn as nn
import torch
    

class Rosenbrock2(nn.Module):
    def __init__(self):
        """
        Initialize the Rosenbrock function:
        f(x) = (1-x1)^2 + 100*(x2-x1^2)^2
        """
        super(Rosenbrock2, self).__init__()

    def forward(self, x):
        """
        Compute the function value.

        Args:
            x (torch.Tensor): vector of shape (2,)
        Returns:
            f (float): scalar
        """
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    
    def x_soln(self):
        """
        Returns the optimal solution.

        Returns:
            x_star (torch.Tensor): optimal solution of shape (2,)
        """
        return None
    
    def f_soln(self):
        return None