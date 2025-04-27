import torch
import torch.nn as nn

class Genhumps5(nn.Module):
    def __init__(self):
        """
        Initialize the Genhumps-5 function:
          f(x) = sum_{i=1..4} [ sin(2*x[i-1])^2 * sin(2*x[i])^2
                              + 0.05 * (x[i-1]^2 + x[i]^2 ) ]
        for x in R^5.
        """
        super(Genhumps5, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x).

        Args:
            x (torch.Tensor): 1D tensor of length 5.

        Returns:
            torch.Tensor: scalar f(x).
        """
        # assume x has length 5
        total = 0.0
        # iterate i from 0 to 3 (pairs x[i], x[i+1])
        for i in range(4):
            xi   = x[i]
            xip1 = x[i+1]
            term_sin = (torch.sin(2.0 * xi) ** 2) * (torch.sin(2.0 * xip1) ** 2)
            term_quad = 0.05 * (xi ** 2 + xip1 ** 2)
            total = total + term_sin + term_quad
        return total

    def x_soln(self):
        """
        (Optional) placeholder for known minimizer. Returns None.
        """
        return None

    def f_soln(self):
        """
        (Optional) placeholder for known minimal value. Returns None.
        """
        return None
