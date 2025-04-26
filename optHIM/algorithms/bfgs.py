import torch
from torch.optim.optimizer import Optimizer
import optHIM.algorithms.ls as ls


class BFGS(Optimizer):
    def __init__(self, x, eps_sy=1e-6, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4, c2=0.9, alpha_high=1000.0, alpha_low=0.0, c=0.5):
        """
        Implements BFGS quasi-Newton method with constant step size or backtracking line search.
        
        Args:
            x (torch.Tensor): parameter to optimize
            eps_sy (float, optional): skip update if s^T y <= eps_sy ||s|| ||y||
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
        defaults = dict(eps_sy=eps_sy, step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1, c2=c2, alpha_high=alpha_high, alpha_low=alpha_low, c=c)
        super(BFGS, self).__init__([x], defaults)
        self.x = x
        # initialize state
        self.state = {
            'x_prev': None,
            'grad_x_prev': None,
            'hess_x_prev': None
        }

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
        n = x.numel()
        I = torch.eye(n, device=x.device, dtype=x.dtype)
        # hess x_k
        hess_x = I # H_0 = I

        if self.state['x_prev'] is not None and self.state['grad_x_prev'] is not None:
            # retrieve history
            # x_{k-1}
            x_prev = self.state['x_prev']
            # grad x_{k-1}
            grad_x_prev = self.state['grad_x_prev']
            # hess x_{k-1}
            hess_x_prev = self.state['hess_x_prev']
            # compute BFGS update
            # s_{k-1} = x_k - x_{k-1}
            s = x - x_prev
            # y_{k-1} = grad x_k - grad x_{k-1}
            y = grad_x - grad_x_prev
            # curvature condition
            curv_cond = y @ s
            if curv_cond > self.param_groups[0]['eps_sy'] * torch.norm(s) * torch.norm(y):
                # update hess x_k
                rho = 1.0 / curv_cond
                hess_x = (I - rho * torch.einsum('i,j->ij', s, y)) @ hess_x_prev @ (I - rho * torch.einsum('i,j->ij', y, s))
                hess_x = hess_x + rho * torch.einsum('i,j->ij', s, s)

        # compute search direction
        d = -hess_x @ grad_x

        # update history
        self.state['x_prev'] = x.clone()
        self.state['grad_x_prev'] = grad_x.clone()
        self.state['hess_x_prev'] = hess_x

        # line search
        if self.param_groups[0]['step_type'] == 'constant':
            alpha = self.param_groups[0]['step_size']
            x += alpha * d
        elif self.param_groups[0]['step_type'] == 'armijo':
            if fn_cls is None:
                raise ValueError("fn_cls must be provided for armijo line search")
            ls.armijo(self.x, d, fn_cls, self.param_groups[0]['alpha'], 
                     self.param_groups[0]['tau'], self.param_groups[0]['c1'])
        elif self.param_groups[0]['step_type'] == 'wolfe':
            if fn_cls is None:
                raise ValueError("fn_cls must be provided for wolfe line search")
            if grad_cls is None:
                raise ValueError("grad_cls must be provided for wolfe line search")
            ls.wolfe(self.x, d, fn_cls, grad_cls, self.param_groups[0]['alpha'], 
                     self.param_groups[0]['alpha_high'], self.param_groups[0]['alpha_low'], 
                     self.param_groups[0]['c'], self.param_groups[0]['c1'], self.param_groups[0]['c2'])
                
