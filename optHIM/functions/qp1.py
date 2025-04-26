import torch
import torch.nn as nn


class QP1(nn.Module):
    def __init__(self, nu=1.0):
        """
        Initialize the QP1 function:
        f(x) = x1 + x2 + nu * (x1^2 + x2^2 - 2)^2

        Args:
            nu (float): penalty parameter
        """
        super(QP1, self).__init__()
        self.nu = nu
        self.register_buffer('x_star', torch.tensor([-1.0, -1.0], dtype=torch.float32))

    def forward(self, x):
        """
        Compute the function value.

        Args:
            x (torch.Tensor): vector of shape (2,)
        Returns:
            f (float): scalar
        """
        x1, x2 = x[0], x[1]
        objective = x1 + x2
        constraint = self.eval_constraint(x)
        penalty = self.nu * (constraint**2)
        return objective + penalty
    
    def x_soln(self):
        """
        Returns the optimal solution.

        Returns:
            x_star (torch.Tensor): optimal solution of shape (2,)
        """
        return self.x_star
    
    def f_soln(self):
        return None

    def eval_constraint(self, x):
        """
        Compute the constraint value: x1^2 + x2^2 - 2

        Args:
            x (torch.Tensor): vector of shape (2,)
        Returns:
            constraint (float): constraint value
        """
        x1, x2 = x[0], x[1]
        return x1**2 + x2**2 - 2

