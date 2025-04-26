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
    
    def x_soln(self):
        return None
    
    def f_soln(self):
        return None