import torch
from torch.optim.optimizer import Optimizer
import torch.autograd.functional as F
import numpy as np

class Newton(Optimizer):
    def __init__(self, params, step_type='Constant', step_size=1.0, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """Implements Newton's method with constant step size or backtracking line search.
        
        Inputs:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            step_type (str): type of step size to use ('Constant' or 'Backtracking')
            step_size (float, optional): constant step size for 'Constant' step type
            alpha (float, optional): initial step size for 'Backtracking' step type
            tau (float, optional): step size reduction factor for 'Backtracking' step type
            c1 (float, optional): sufficient decrease parameter for 'Backtracking' step type
        
        Example:
            >>> optimizer = Newton(model.parameters(), step_type='Constant', step_size=1.0)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step(closure=lambda: loss_fn(model(input), target))
        """
        if step_type not in ['Constant', 'Backtracking']:
            raise ValueError(f"step_type must be 'Constant' or 'Backtracking', got {step_type}")
        defaults = dict(step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(Newton, self).__init__(params, defaults)
        
        # Store model reference for Hessian computation
        self.model = None

    def __setstate__(self, state):
        super(Newton, self).__setstate__(state)

    def set_model(self, model):
        """Set the model for Hessian computation.
        
        Inputs:
            model: PyTorch model
        """
        self.model = model

    def compute_hessian(self, p):
        """Compute the Hessian matrix at point p using autograd.
        
        Inputs:
            p: Parameter tensor
            
        Outputs:
            H: Hessian matrix
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model first.")
            
        # Use PyTorch's functional Hessian
        def func(x):
            return self.model(x.view_as(p))
        
        return F.hessian(func, p.data.flatten()).view(p.numel(), p.numel())

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
                
                # Get gradient
                d_p = p.grad.data
                
                # Compute Hessian if model is set
                if self.model is not None:
                    try:
                        # Compute Hessian
                        H = self.compute_hessian(p)
                        
                        # Compute Newton direction by solving H*d = -g
                        d = -torch.linalg.solve(H, d_p.flatten()).view_as(p)
                    except:
                        # Fallback to gradient descent if Hessian is singular
                        d = -d_p
                else:
                    # Fallback to gradient descent if model is not set
                    d = -d_p
                
                # Determine step size based on step_type
                if step_type == 'Constant':
                    alpha = group['step_size']
                    p.data.add_(alpha, d)
                
                elif step_type == 'Backtracking':
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    
                    # Initial parameter and gradient
                    p0 = p.data.clone()
                    d_p0 = d_p.clone()
                    
                    if closure is not None:
                        # Compute initial loss if closure is provided
                        loss0 = closure()
                    else:
                        # If no closure provided, we can't do backtracking line search
                        p.data.add_(alpha, d)
                        continue
                    
                    # Backtracking line search
                    while True:
                        # Update parameter with current step size
                        p.data = p0 + alpha * d
                                                
                        # Check Armijo condition
                        armijo_rhs = loss0 + c1 * alpha * torch.dot(d_p0.view(-1), d.view(-1))
                        loss = closure()
                        if loss <= armijo_rhs:
                            break
                        
                        # Reduce step size
                        alpha = tau * alpha
                        
                        # Safety check to prevent infinite loop
                        if alpha < 1e-10:
                            p.data = p0  # Revert to original parameters
                            break 