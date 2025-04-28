from math import sqrt
import torch


def quadratic_2D(a, b, c):
    """
    Solve a quadratic equation of the form ax^2 + bx + c = 0.

    Args:
        a (float): coefficient of x^2
        b (float): coefficient of x
        c (float): constant term

    Returns:
        x_soln (tuple): solutions to the quadratic equation
    """
    determinant = b**2 - 4*a*c
    if determinant < 0:
        raise ValueError("No real solutions")
    x1 = (-b + sqrt(determinant)) / (2*a)
    x2 = (-b - sqrt(determinant)) / (2*a)
    return x1, x2


def correct_hess(H, beta=1e-6, max_iter=1e2):
    """
    Corrects the Hessian to be positive definite.

    Args:
        H (torch.Tensor): Hessian matrix of shape (n, n)
        beta (float): positive scalar for Hessian modification
        max_iter (int, optional): maximum number of iterations

    Returns:
        H_hat (torch.Tensor): corrected Hessian matrix of shape (n, n)
    """
    # initialize eta
    min_diag = torch.min(torch.diag(H))
    if min_diag > 0:
        eta = 0.0
    else:
        eta = -min_diag + beta

    I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    
    for _ in range(int(max_iter)):
        try:
            # attempt Cholesky factorization
            H_hat = H + eta * I
            torch.linalg.cholesky(H_hat)
            # break if successful
            break
        except RuntimeError:
            # if factorization fails, increase eta
            if eta >= beta:
                break
            else:
                eta = max(2 * eta, beta)
    
    return H_hat