import torch.nn as nn
import torch
    

class Rosenbrock(nn.Module):
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
            x: vector of shape (2,)
        Returns:
            f: scalar
        """
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    
    def solution(self):
        """
        Returns the solution x* to the function.
        """
        return self.x_star