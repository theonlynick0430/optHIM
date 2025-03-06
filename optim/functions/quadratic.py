import torch.nn as nn
import torch


class QuadraticFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b, c, x):
        """
        Compute the forward pass of the quadratic function.

        Args:
            A: symmetric matrix of shape (n,n)
            b: vector of shape (n,1)
            c: scalar
            x: vector of shape (n,1)
        Returns:
            f: scalar
        """
        ctx.save_for_backward(A, b, x)
        f = 0.5 * x.T @ A @ x + b.T @ x + c
        return f[0, 0]
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of the quadratic function.

        Args:
            ctx: context object
            grad_output: scalar incoming gradient
        Returns:
            grad_A: None
            grad_b: None
            grad_c: None
            grad_x: vector of shape (n,1)
        """
        A, b, x = ctx.saved_tensors
        grad_x = A @ x + b
        grad_x *= grad_output
        return None, None, None, grad_x


class Quadratic(nn.Module):
    def __init__(self, A, b, c):
        """
        Initialize the Quadratic function:
        f(x) = 0.5 * x^T A x + b^T x + c

        Args:
            A: symmetric matrix of shape (n,n)
            b: vector of shape (n,1)
            c: scalar
        """
        super(Quadratic, self).__init__()
        self.A = A
        self.b = b
        self.c = c  

    def forward(self, x):
        """
        Compute the function value.

        Args:
            x: vector of shape (n,1)
        Returns:
            f: scalar
        """
        return QuadraticFunction.apply(self.A, self.b, self.c, x)