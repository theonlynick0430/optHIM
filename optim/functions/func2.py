import torch.nn as nn
import torch


class Func2(nn.Module):
    def __init__(self):
        """
        Initialize the function:
        f(x) = sum_{i=1...3} (y_i - w(1-z^i))^2
        where x = [w, z]^T and y = [1.5, 2.25, 2.625]^T
        """
        super(Func2, self).__init__()
        self.y = torch.tensor([1.5, 2.25, 2.625], dtype=torch.float32)
    
    def forward(self, x):
        """
        Compute the function value.
        
        Args:
            x: vector of shape (2,)
        Returns:
            f: scalar
        """
        w = x[0]
        z = x[1]
        z_pow = torch.stack([1-z, 1-z**2, 1-z**3])
        term = self.y - w * z_pow
        return torch.sum(term**2)