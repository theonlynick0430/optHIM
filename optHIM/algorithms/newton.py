import torch
from torch.optim.optimizer import Optimizer
from torch.autograd.functional import hessian
import optHIM.algorithms.ls as ls


class Newton(Optimizer):
    def __init__(self, x, function, beta=1e-6, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4, c2=0.9, alpha_high=1000.0, alpha_low=0.0, c=0.5):
        """
        Implements Newton's method with constant step size or backtracking line search.
        
        Args:
            x (torch.Tensor): parameter to optimize
            function (nn.Module): function to compute Hessian of
            beta (float): positive scalar for Hessian modification
            step_type (str): type of step size to use ('constant', 'armijo', or 'wolfe')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
            c2 (float, optional): curvature condition parameter for 'wolfe' step type
            alpha_high (float, optional): upper bound for step size for 'wolfe' step type
            alpha_low (float, optional): lower bound for step size for 'wolfe' step type
            c (float, optional): interpolation parameter for 'wolfe' step type
        """
        if step_type not in ['constant', 'armijo', 'wolfe']:
            raise ValueError(f"step_type must be 'constant', 'armijo', or 'wolfe', got {step_type}")
        defaults = dict(beta=beta, step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1, c2=c2, alpha_high=alpha_high, alpha_low=alpha_low, c=c)
        super(Newton, self).__init__([x], defaults)
        self.x = x
        self.function = function

    def step(self, fn_cls=None, grad_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function.
                Required for backtracking line search.
            grad_cls (callable, optional): closure that recomputes the gradients.
                Required for Wolfe line search.
        """
        if self.x.grad is None:
            return
            
        # x_k
        x = self.x.data
        # grad x_k
        grad_x = self.x.grad.data
        # hess x_k
        hess_x = hessian(self.function, x)
        # ensure PD => descent direction
        hess_x = self.correct_hess(hess_x, self.param_groups[0]['beta'])
        # compute search direction
        d = -torch.linalg.pinv(hess_x) @ grad_x

        # line search
        if self.param_groups[0]['step_type'] == 'constant':
            alpha = self.param_groups[0]['step_size']
            x += alpha * d
        elif self.param_groups[0]['step_type'] == 'armijo':
            if fn_cls is None:
                raise ValueError("fn_cls must be provided for backtracking line search")
            ls.armijo(self.x, d, fn_cls, self.param_groups[0]['alpha'], 
                     self.param_groups[0]['tau'], self.param_groups[0]['c1']) 
        elif self.param_groups[0]['step_type'] == 'wolfe':
            if fn_cls is None:
                raise ValueError("fn_cls must be provided for Wolfe line search")
            if grad_cls is None:
                raise ValueError("grad_cls must be provided for Wolfe line search")
            ls.wolfe(self.x, d, fn_cls, grad_cls, self.param_groups[0]['alpha'], 
                     self.param_groups[0]['alpha_high'], self.param_groups[0]['alpha_low'], 
                     self.param_groups[0]['c'], self.param_groups[0]['c1'], self.param_groups[0]['c2'])

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