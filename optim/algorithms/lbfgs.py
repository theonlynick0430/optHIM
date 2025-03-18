import torch
from torch.optim.optimizer import Optimizer
import optim.algorithms.ls as ls
from collections import deque


class LBFGS(Optimizer):
    def __init__(self, params, m=5, eps_sy=1e-6, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """
        Implements L-BFGS quasi-Newton method with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            m (int): memory of s, y buffers
            eps_sy (float, optional): skip update if s^T y <= eps_sy ||s|| ||y||
            step_type (str): type of step size to use ('constant' or 'armijo')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
        """
        if step_type not in ['constant', 'armijo']:
            raise ValueError(f"step_type must be 'constant' or 'armijo', got {step_type}")
        defaults = dict(m=m, eps_sy=eps_sy, step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(LBFGS, self).__init__(params, defaults)
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.state[param]['S'] = deque()
                    self.state[param]['Y'] = deque()

    def two_loop_recursion(self, d_p, H_0, S, Y):
        """
        Performs two-loop recursion to compute the search direction.
        
        Args:
            d_p (torch.Tensor): gradient of the objective function    
            H_0 (torch.Tensor): initial Hessian matrix
            S (deque): deque of s vectors
            Y (deque): deque of y vectors
        """
        q = d_p.clone()
        m = len(S)
        alphas = []
        for i in range(m):
            s = S[i]
            y = Y[i]
            alpha = (s @ q) / (s @ y)
            alphas.append(alpha)
            q = q - alpha * y
        r = H_0 @ q
        for i in range(m):
            s = S[m - i - 1]
            y = Y[m - i - 1]
            beta = (y @ r) / (s @ y)
            r = r + s * (alphas[m - i - 1] - beta)
        return r

    def step(self, fn_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function.
                Required for backtracking line search.
        """
        for group in self.param_groups:
            m = group['m']
            eps_sy = group['eps_sy']
            step_type = group['step_type']
            
            for param in group['params']:
                if param.grad is None:
                    continue

                p = param.data
                d_p = param.grad.data
                n = p.numel()
                I = torch.eye(n, device=p.device, dtype=p.dtype)
                H_0 = I

                if 'p_prev' in self.state[param] and 'd_p_prev' in self.state[param]:
                    # retrieve history for this param
                    p_prev = self.state[param]['p_prev']
                    d_p_prev = self.state[param]['d_p_prev']
                    s = p - p_prev
                    y = d_p - d_p_prev
                    curv_cond = y @ s
                    if curv_cond > eps_sy * torch.norm(s) * torch.norm(y):
                        # add to s, y buffers
                        self.state[param]['S'].appendleft(s)
                        self.state[param]['Y'].appendleft(y)
                        if len(self.state[param]['S']) > m:
                            # flush out old information
                            self.state[param]['S'].pop()
                            self.state[param]['Y'].pop()

                # compute search direction
                d = -self.two_loop_recursion(d_p, H_0, self.state[param]['S'], self.state[param]['Y'])

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
                
