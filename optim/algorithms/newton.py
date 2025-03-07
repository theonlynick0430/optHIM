import torch
from torch.optim.optimizer import Optimizer
from torch.autograd.functional import hessian
import optim.algorithms.ls as ls


class Newton(Optimizer):
    def __init__(self, params, model, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """Implements Newton's method with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            model (nn.Module): model used to compute Hessian
            step_type (str): type of step size to use ('constant' or 'armijo')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
        
        Example:
            >>> optimizer = Newton(model.parameters(), step_type='constant', step_size=1.0)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step(hess_cl=lambda: compute_hessian(model, input))
        """
        if step_type not in ['constant', 'armijo']:
            raise ValueError(f"step_type must be 'constant' or 'armijo', got {step_type}")
        defaults = dict(step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        self.model = model
        super(Newton, self).__init__(params, defaults)

    def step(self, loss_cl=None):
        """Performs a single optimization step.
        
        Args:
            loss_cl (callable, optional): closure that reevaluates the model
                and returns the loss. Required for backtracking line search.
        """
        for group in self.param_groups:
            step_type = group['step_type']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # set search direction
                d_p = p.grad.data
                H = hessian(self.model, p.data)
                # ensure descent search direction
                H = self.correct_hess(H)
                d = -torch.linalg.pinv(H) @ d_p
                
                if step_type == 'constant':
                    alpha = group['step_size']
                    p.data += alpha * d
                
                elif step_type == 'armijo':
                    if loss_cl is None:
                        raise ValueError("loss_cl must be provided for backtracking line search")
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    ls.armijo(p, d, loss_cl, alpha, tau, c1) 

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