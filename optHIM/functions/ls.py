import torch.nn as nn
import torch


class LeastSquares(nn.Module):
    def __init__(self, w):
        """
        Initialize the Linear Least Squares function:
        f(w) = (1/2n) * ||Xw - y||^2

        Args:
            w (torch.Tensor): weight vector of shape (d,)
        """
        super(LeastSquares, self).__init__()
        self.register_buffer('w', w)

    def forward(self, X, y):
        """
        Compute the function value using the formula:
        f(w) = (1/2n) * ||Xw - y||^2

        Args:
            X (torch.Tensor): data matrix of shape (n, d)
            y (torch.Tensor): target vector of shape (n,)
        Returns:
            f (float): scalar loss value
        """
        return 0.5 * torch.mean((X @ self.w - y)**2)

    def accuracy(self, X, y, threshold=0.0):
        """
        Compute prediction accuracy using a threshold.

        Args:
            X (torch.Tensor): data matrix of shape (n, d)
            y (torch.Tensor): target vector of shape (n,)
            threshold (float, optional): threshold for binary classification
        Returns:
            acc (float): prediction accuracy between 0 and 1
        """
        with torch.no_grad():
            predictions = torch.sign(X @ self.w - threshold)
            return (predictions == torch.sign(y)).float().mean().item()
