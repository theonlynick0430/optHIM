import torch
import torch.nn as nn


class QP2(nn.Module):
    def __init__(self, nu=1.0):
        """
        Initialize the QP2 function:
        f(x) = e^(x1x2x3x4x5) - 0.5*(x1^3 + x2^3 + 1)^2 + nu * ||c(x)||^2

        Args:
            nu (float): penalty parameter
        """
        super(QP2, self).__init__()
        self.nu = nu
        self.register_buffer('x_star', torch.tensor([-1.71, 1.59, 1.82, -0.763, -0.763], dtype=torch.float32))

    def forward(self, x):
        """
        Compute the function value.

        Args:
            x (torch.Tensor): vector of shape (5,)
        Returns:
            f (float): scalar
        """
        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
        objective = torch.exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (x1**3 + x2**3 + 1)**2
        constraints = self.eval_constraints(x)
        penalty = self.nu * torch.sum(constraints**2)
        return objective + penalty
    
    def x_soln(self):
        """
        Returns the optimal solution.

        Returns:
            x_star (torch.Tensor): optimal solution of shape (5,)
        """
        return self.x_star
    
    def f_soln(self):
        return None

    def eval_constraints(self, x):
        """
        Compute the constraint values:
        1. x1^2 + x2^2 + x3^2 + x4^2 + x5^2 - 10 = 0
        2. x2x3 - 5x4x5 = 0
        3. x1^3 + x2^3 + 1 = 0

        Args:
            x (torch.Tensor): vector of shape (5,)
        Returns:
            constraints (torch.Tensor): constraint vector of shape (3,)
        """
        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
        c1 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10
        c2 = x2 * x3 - 5 * x4 * x5
        c3 = x1**3 + x2**3 + 1
        return torch.tensor([c1, c2, c3], device=x.device)
