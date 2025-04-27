import torch
import torch.nn as nn

class Exponential1000(nn.Module):
    def __init__(self):
        """
        Initialize the 100-dimensional exponential function:
          f(x) = (exp(x[0]) - 1)/(exp(x[0]) + 1)
                 + 0.1 * exp(-x[0])
                 + sum_{i=1..99} (x[i] - 1)^4
        """
        super(Exponential1000, self).__init__()
        self.n = 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x).

        Args:
            x (torch.Tensor): 1D tensor of length 100.

        Returns:
            torch.Tensor: scalar f(x).
        """
        # assume x has correct length
        x0 = x[0]
        # (exp(x0)-1)/(exp(x0)+1)
        ex0 = torch.exp(x0)
        term1 = (ex0 - 1.0) / (ex0 + 1.0)
        # 0.1 * exp(-x0)
        term2 = 0.1 * torch.exp(-x0)
        # sum of fourth powers for remaining components
        rest = torch.sum((x[1:] - 1.0) ** 4)
        return term1 + term2 + rest

    def x_soln(self):
        """
        (Optional) placeholder for known minimizer. Returns None.
        """
        return None

    def f_soln(self):
        """
        (Optional) placeholder for known optimal value. Returns None.
        """
        return None