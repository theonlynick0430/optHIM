import torch
from torch.optim.optimizer import Optimizer
import optHIM.algorithms.ls as ls


class BFGS(Optimizer):
    def __init__(self, params, eps_sy=1e-6, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """
        Implements BFGS quasi-Newton method with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            eps_sy (float, optional): skip update if s^T y <= eps_sy ||s|| ||y||
            step_type (str): type of step size to use ('constant' or 'armijo')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
        """
        if step_type not in ['constant', 'armijo']:
            raise ValueError(f"step_type must be 'constant' or 'armijo', got {step_type}")
        defaults = dict(eps_sy=eps_sy, step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(BFGS, self).__init__(params, defaults)

    def step(self, fn_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function.
                Required for backtracking line search.
        """
        for group in self.param_groups:
            eps_sy = group['eps_sy']
            step_type = group['step_type']
            
            for param in group['params']:
                if param.grad is None:
                    continue

                p = param.data
                d_p = param.grad.data
                n = p.numel()
                I = torch.eye(n, device=p.device, dtype=p.dtype)
                H = I # H_0 = I

                if param in self.state:
                    # retrieve history for this param
                    p_prev = self.state[param]['p_prev']
                    d_p_prev = self.state[param]['d_p_prev']
                    H_prev = self.state[param]['H_prev']
                    # compute BFGS update
                    s = p - p_prev
                    y = d_p - d_p_prev
                    curv_cond = y @ s
                    if curv_cond > eps_sy * torch.norm(s) * torch.norm(y):
                        rho = 1.0 / curv_cond
                        H = (I - rho * torch.einsum('i,j->ij', s, y)) @ H_prev @ (I - rho * torch.einsum('i,j->ij', y, s))
                        H = H + rho * torch.einsum('i,j->ij', s, s)

                # compute search direction
                d = -H @ d_p

                if step_type == 'constant':
                    alpha = group['step_size']
                    p += alpha * d
                
                elif step_type == 'armijo':
                    if fn_cls is None:
                        raise ValueError("fn_cls must be provided for armijo line search")
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    ls.armijo(param, d, fn_cls, alpha, tau, c1)    
                
                # update history
                self.state[param]['p_prev'] = p.clone()
                self.state[param]['d_p_prev'] = d_p.clone()
                self.state[param]['H_prev'] = H
                
