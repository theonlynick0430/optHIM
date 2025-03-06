import torch
from torch.optim.optimizer import Optimizer

class GD(Optimizer):
    def __init__(self, params, step_type='Constant', step_size=1e-3, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """Implements gradient descent with constant step size or backtracking line search.
        
        Inputs:
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
            >>> optimizer.step(closure=lambda: loss_fn(model(input), target))
        """
        if step_type not in ['Constant', 'Backtracking']:
            raise ValueError(f"step_type must be 'Constant' or 'Backtracking', got {step_type}")
        defaults = dict(step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(GD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Inputs:
            closure (callable, optional): A closure that reevaluates the model
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
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    
                    # initial parameter and gradient
                    p0 = p.data.clone()
                    d_p0 = d_p.clone()
                    
                    if closure is not None:
                        # compute initial loss if closure is provided
                        loss0 = closure()
                    else:
                        # if no closure provided, we can't do backtracking line search
                        p.data += alpha * d
                        continue
                    
                    # backtracking line search
                    while True:
                        # update parameter with current step size
                        p.data = p0 + alpha * d
                                        
                        # check armijo condition
                        armijo_rhs = loss0 + c1 * alpha * torch.dot(d_p0.view(-1), d.view(-1))
                        loss = closure()
                        if loss <= armijo_rhs:
                            break
                        
                        # reduce step size
                        alpha = tau * alpha
                        
                        # safety check to prevent infinite loop
                        if alpha < 1e-10:
                            p.data = p0  # revert to original parameters
                            break