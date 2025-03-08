import torch.nn as nn
import torch


class Func3(nn.Module):
    def __init__(self, n=2):
        """
        Initialize the function:
        f(x) = (exp(z1) - 1)/(exp(z1) + 1) + 0.1*exp(-z1) + sum_{i=2...n} (zi - 1)^4
        where x = [z1, z2, ..., zn]^T
        """
        super(Func3, self).__init__()
        self.register_buffer('x_star', torch.tensor([-1.25] + [0.0] * n, dtype=torch.float32))

    def forward(self, x):
        """
        Compute the function value.
        
        Args:
            x (torch.Tensor): vector of shape (n,)
        Returns:
            f (float): scalar
        """
        z1 = x[0]
        exp_z1 = torch.exp(z1)
        term1 = (exp_z1 - 1) / (exp_z1 + 1)
        term2 = 0.1 / exp_z1
        term3 = torch.sum((x[1:] - 1)**4)
        return term1 + term2 + term3 
    
    def solution(self):
        """
        Returns the solution x* of shape (n,).
        """
        return self.x_star