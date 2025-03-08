import torch
from torch.optim.optimizer import Optimizer
from torch.autograd.functional import hessian
import optim.algorithms.ls as ls


class Newton(Optimizer):
    def __init__(self, params, function, beta=1e-6, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """
        Implements Newton's method with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            function (nn.Module): function to compute Hessian of
            beta (float): positive scalar for Hessian modification
            step_type (str): type of step size to use ('constant' or 'armijo')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
        """
        if step_type not in ['constant', 'armijo']:
            raise ValueError(f"step_type must be 'constant' or 'armijo', got {step_type}")
        defaults = dict(beta=beta, step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(Newton, self).__init__(params, defaults)
        self.function = function

    def step(self, fn_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function.
                Required for backtracking line search.
        """
        for group in self.param_groups:
            step_type = group['step_type']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue
                
                p = param.data
                d_p = param.grad.data
                # compute Hessian
                H = hessian(self.function, p)
                # ensure PD => descent direction
                H = self.correct_hess(H, beta)
                # compute search direction
                d = -torch.linalg.pinv(H) @ d_p
                
                if step_type == 'constant':
                    alpha = group['step_size']
                    p += alpha * d
                
                elif step_type == 'armijo':
                    if fn_cls is None:
                        raise ValueError("fn_cls must be provided for backtracking line search")
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    ls.armijo(param, d, fn_cls, alpha, tau, c1) 

    def correct_hess(self, H, beta=1e-6, max_iter=1e2):
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