import torch
import torch.nn as nn

class Quartic2(nn.Module):
    def __init__(self):
        """
        Initialize the quartic function:
          f(x) = ½ x^T x + (σ/4) (x^T Q x)^2,
        where Q ∈ ℝ^{4×4}, σ = 1e4.
        """
        super(Quartic2, self).__init__()
        Q = torch.tensor([
            [5.0, 1.0, 0.0, 0.5],
            [1.0, 4.0, 0.5, 0.0],
            [0.0, 0.5, 3.0, 0.0],
            [0.5, 0.0, 0.0, 2.0]
        ], dtype=torch.float32)
        self.register_buffer('Q', Q)
        # quartic coefficient
        self.sigma = 1e4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x).

        Args:
            x (torch.Tensor): 1D tensor of length 4.
        Returns:
            torch.Tensor: scalar f(x).
        """
        quadratic = 0.5 * x.dot(x)
        inner = x.dot(self.Q.matmul(x))
        quartic = (self.sigma / 4.0) * inner * inner
        return quadratic + quartic

    def x_soln(self) -> torch.Tensor:
        """
        Returns the minimizer x* = 0-vector.
        """
        return torch.zeros(4, dtype=torch.float32)

    def f_soln(self):
        """
        (Optional) placeholder method. Returns None.
        """
        return None