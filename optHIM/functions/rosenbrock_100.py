import torch
import torch.nn as nn

class Rosenbrock100(nn.Module):
    def __init__(self):
        """
        Initialize the 100-dimensional Rosenbrock function:
            f(x) = sum_{i=1..99} [ (1 - x[i-1])^2 + 100*(x[i] - x[i-1]^2)^2 ]
        """
        super(Rosenbrock100, self).__init__()
        self.n = 100
        # the true minimizer is x* = [1,1,...,1] of length 100
        self.register_buffer('x_star', torch.ones(self.n, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Rosenbrock function value at x.

        Args:
            x (torch.Tensor): 1D tensor of length 100.

        Returns:
            torch.Tensor: scalar f(x).
        """
        # assume x has correct dimensionality
        x_prev = x[:-1]
        x_next = x[1:]
        term1 = (1.0 - x_prev) ** 2
        term2 = 100.0 * (x_next - x_prev ** 2) ** 2
        return torch.sum(term1 + term2)

    def x_soln(self) -> torch.Tensor:
        """
        Returns the known minimizer x* = [1,...,1].
        """
        return self.x_star

    def f_soln(self):
        return None