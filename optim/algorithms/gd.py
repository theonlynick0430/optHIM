import torch
from torch.optim.optimizer import Optimizer
from optim.algorithms.ls import backtracking_ls

class GD(Optimizer):
    def __init__(self, params, step_type='Constant', step_size=1e-3, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """Implements gradient descent with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            step_type (str): type of step size to use ('Constant' or 'Backtracking')
            step_size (float, optional): constant step size for 'Constant' step type
            alpha (float, optional): initial step size for 'Backtracking' step type
            tau (float, optional): step size reduction factor for 'Backtracking' step type
            c1 (float, optional): sufficient decrease parameter for 'Backtracking' step type
        
        Example:
            >>> optimizer = GD(model.parameters(), step_type='Constant', step_size=0.1)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()
        """
        if step_type not in ['Constant', 'Backtracking']:
            raise ValueError(f"step_type must be 'Constant' or 'Backtracking', got {step_type}")
        defaults = dict(step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(GD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

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
                d = -d_p

                if step_type == 'Constant':
                    alpha = group['step_size']
                    p.data += alpha * d
                
                elif step_type == 'Backtracking':
                    if loss_cl is None:
                        raise ValueError("loss_cl must be provided for backtracking line search")
                    
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    backtracking_ls(p, d, loss_cl, alpha, tau, c1)