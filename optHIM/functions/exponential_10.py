import torch
import torch.nn as nn

class Exponential10(nn.Module):
    def __init__(self):
        """
        Initialize the 10-dimensional exponential function:
          f(x) = (exp(x[0]) - 1)/(exp(x[0]) + 1)
                 + 0.1*exp(-x[0])
                 + sum_{i=1..9}(x[i] - 1)^4
        """
        super(Exponential10, self).__init__()
        # no buffers needed, no known closed-form minimizer provided

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x).

        Args:
            x (torch.Tensor): 1D tensor of length 10.

        Returns:
            torch.Tensor: scalar f(x).
        """
        # first component terms
        x0 = x[0]
        # (exp(x0)-1)/(exp(x0)+1)
        num = torch.exp(x0) - 1.0
        den = torch.exp(x0) + 1.0
        term1 = num / den
        # 0.1 * exp(-x0)
        term2 = 0.1 * torch.exp(-x0)
        # sum_{i=2..10} (x[i] - 1)^4
        rest = torch.sum((x[1:] - 1.0) ** 4)
        return term1 + term2 + rest

    def x_soln(self):
        """
        (Optional) placeholder for known minimizer. Returns None.
        """
        return None

    def f_soln(self):
        return None
