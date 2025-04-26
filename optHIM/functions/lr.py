import torch.nn as nn
import torch


class LogisticRegression(nn.Module):
    def __init__(self, w):
        """
        Initialize the Logistic Regression function:
        f(w) = (1/n) * sum(log(1 + exp(-y_i * (X_i * w))))

        Args:
            w (torch.Tensor): weight vector of shape (d,)
        """
        super(LogisticRegression, self).__init__()
        self.register_buffer('w', w)

    def forward(self, X, y):
        """
        Compute the function value using the formula:
        f(w) = (1/n) * sum(log(1 + exp(-y_i * (X_i * w))))

        Args:
            X (torch.Tensor): data matrix of shape (n, d)
            y (torch.Tensor): target vector of shape (n,) with values in {-1, 1}
        Returns:
            f (float): scalar loss value
        """
        return torch.mean(torch.log(1 + torch.exp(-y * (X @ self.w))))

    def accuracy(self, X, y):
        """
        Compute prediction accuracy.

        Args:
            X (torch.Tensor): data matrix of shape (n, d)
            y (torch.Tensor): target vector of shape (n,) with values in {-1, 1}
        Returns:
            acc (float): prediction accuracy between 0 and 1
        """
        with torch.no_grad():
            predictions = torch.sign(X @ self.w)
            return (predictions == y).float().mean().item()