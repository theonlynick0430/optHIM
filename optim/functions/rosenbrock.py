import torch.nn as nn
import torch


class RosenbrockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Compute the forward pass of the Rosenbrock function.

        Args:
            x: vector of shape (2,1)
        Returns:
            f: scalar
        """
        ctx.save_for_backward(x)
        return (1 - x[0,0]) ** 2 + 100 * (x[1,0] - x[0,0] ** 2) ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of the Rosenbrock function.

        Args:
            ctx: context object
            grad_output: scalar incoming gradient
        Returns:
            grad_x: vector of shape (2,1)
        """
        x = ctx.saved_tensors[0]
        grad_x = torch.zeros_like(x)
        grad_x[0,0] = -2 * (1 - x[0,0]) - 400 * (x[1,0] - x[0,0] ** 2) * x[0,0]
        grad_x[1,0] = 200 * (x[1,0] - x[0,0] ** 2)
        grad_x *= grad_output
        return grad_x
    

class Rosenbrock(nn.Module):
    def __init__(self):
        """
        Initialize the Rosenbrock function:
        f(x) = (1-x1)^2 + 100*(x2-x1^2)^2
        """
        super(Rosenbrock, self).__init__()

    def forward(self, x):
        """
        Compute the function value.

        Args:
            x: vector of shape (2,1)
        Returns:
            f: scalar
        """
        return RosenbrockFunction.apply(x)