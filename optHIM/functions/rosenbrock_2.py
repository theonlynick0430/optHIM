import torch.nn as nn
import torch
    

class Rosenbrock2(nn.Module):
    def __init__(self):
        """
        Initialize the Rosenbrock function:
        f(x) = (1-x1)^2 + 100*(x2-x1^2)^2
        """
        super(Rosenbrock, self).__init__()
        self.register_buffer('x_star', torch.tensor([1.0, 1.0], dtype=torch.float32))

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
        return self.x_star
    
    def f_soln(self):
        return None