import torch
import optHIM.algorithms.ls as ls
from optHIM.algorithms.base import BaseOptimizer


def dfp_update(x, grad_x, hess_x_prev, eps_sy, x_prev=None, grad_x_prev=None):
    """
    Performs a DFP update of the Hessian approximation.
    
    Args:
        x (torch.Tensor): current parameter 
        grad_x (torch.Tensor): current gradient
        hess_x_prev (torch.Tensor): previous Hessian approximation
        eps_sy (float): skip update if s^T y <= eps_sy ||s|| ||y||
        x_prev (torch.Tensor, optional): previous parameter
        grad_x_prev (torch.Tensor, optional): previous gradient

    Returns:
        hess_x (torch.Tensor): updated Hessian approximation
    """
    # if DFP update is skipped, use previous approximation
    hess_x = hess_x_prev
    if x_prev is not None and grad_x_prev is not None:
        s = x - x_prev
        y = grad_x - grad_x_prev
        # curvature condition
        curv_cond = y @ s
        if curv_cond > eps_sy * torch.norm(s) * torch.norm(y):
            # compute DFP update
            I = torch.eye(x.numel(), device=x.device, dtype=x.dtype)
            rho = 1.0 / curv_cond
            term1 = (I - rho * torch.outer(y, s)) @ hess_x_prev @ (I - rho * torch.outer(s, y))
            term2 = rho * torch.outer(y, y)
            hess_x = term1 + term2
    return hess_x

def dfp_update_inv(x, grad_x, inv_hess_x_prev, eps_sy, x_prev=None, grad_x_prev=None):
    """
    Computes DFP update for inverse Hessian approximation.
    
    Args:
        x (torch.Tensor): current parameter
        grad_x (torch.Tensor): current gradient
        inv_hess_x_prev (torch.Tensor): previous inverse Hessian approximation
        eps_sy (float): skip update if s^T y <= eps_sy ||s|| ||y||
        x_prev (torch.Tensor, optional): previous parameter
        grad_x_prev (torch.Tensor, optional): previous gradient

    Returns:
        inv_hess_x (torch.Tensor): updated inverse Hessian approximation    
    """
    # if DFP update is skipped, use previous approximation
    inv_hess_x = inv_hess_x_prev
    if x_prev is not None and grad_x_prev is not None:
        s = x - x_prev
        y = grad_x - grad_x_prev
        # curvature condition
        curv_cond = y @ s
        if curv_cond > eps_sy * torch.norm(s) * torch.norm(y):
            # compute DFP update
            term1 = - inv_hess_x_prev @ torch.outer(y, y) @ inv_hess_x_prev / (y @ inv_hess_x_prev @ y)
            term2 = torch.outer(s, s) / curv_cond
            inv_hess_x = inv_hess_x_prev + term1 + term2
    return inv_hess_x


class DFP(BaseOptimizer):
    def __init__(self, x, eps_sy=1e-6, step_type='constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4, c2=0.9, alpha_high=1000.0, alpha_low=0.0, c=0.5):
        """
        Implements DFP quasi-Newton method with constant step size or backtracking line search.
        
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
        super(DFP, self).__init__([x], defaults)
        self.x = x
        # initialize state
        self.state = {
            'x_prev': None,
            'grad_x_prev': None,
            # start approximation of inverse Hessian as identity matrix
            'inv_hess_x_prev': torch.eye(x.numel(), device=x.device, dtype=x.dtype)
        }

    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that returns the function evaluated at given point. 
                Required for backtracking line search.
            grad_cls (callable, optional): closure (void) that updates the gradient at given point.
                Required for Wolfe line search.
            hess_cls (callable, optional): not required for this optimizer
        """
        x = self.x.data
        grad_x = self.x.grad.data

        # DFP update
        inv_hess_x = dfp_update_inv(
            x, grad_x, self.state['inv_hess_x_prev'], self.param_groups[0]['eps_sy'],
            self.state['x_prev'], self.state['grad_x_prev']
        )

        # compute search direction
        d = -inv_hess_x @ grad_x

        # update history
        self.state['x_prev'] = x.clone()
        self.state['grad_x_prev'] = grad_x.clone()
        self.state['inv_hess_x_prev'] = inv_hess_x.clone()

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
