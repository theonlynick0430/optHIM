import torch.nn as nn
import torch


class QuadraticFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b, c, x):
        """
        Compute the forward pass of the quadratic function.

        Args:
            A (torch.Tensor): symmetric matrix of shape (n, n)
            b (torch.Tensor): vector of shape (n,)
            c (float): scalar
            x (torch.Tensor): vector of shape (n,)
        Returns:
            f (float): scalar
        """
        ctx.save_for_backward(A, b, x)
        return 0.5 * torch.einsum('i,ij,j->', x, A, x) + torch.einsum('i,i->', b, x) + c
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of the quadratic function.

        Args:
            ctx (torch.autograd.Function): context object
            grad_output (float): scalar incoming gradient
        Returns:
            grad_A (torch.Tensor): None
            grad_b (torch.Tensor): None
            grad_c (float): None
            grad_x (torch.Tensor): vector of shape (n,)
        """
        # custom gradient for symmetric A
        A, b, x = ctx.saved_tensors
        grad_x = torch.einsum('ij,j->i', A, x) + b
        grad_x *= grad_output
        return None, None, None, grad_x


class Quadratic(nn.Module):
    def __init__(self, A, b, c):
        """
        Initialize the Quadratic function:
        f(x) = 0.5 * x^T A x + b^T x + c

        Args:
            A (torch.Tensor): symmetric matrix of shape (n, n)
            b (torch.Tensor): vector of shape (n,)
            c (float): scalar
        """
        super(Quadratic, self).__init__()
        self.register_buffer('A', A)
        self.register_buffer('b', b)
        self.register_buffer('c', c)  

    def forward(self, x):
        """
        Compute the function value.

        Args:
            x (torch.Tensor): vector of shape (n,)
        Returns:
            f (float): scalar
        """
        return QuadraticFunction.apply(self.A, self.b, self.c, x)
    
    def solution(self):
        """
        Returns the solution x* of shape (n,).
        """
        return -torch.linalg.pinv(self.A) @ self.b