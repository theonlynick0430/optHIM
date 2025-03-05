import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd.functional as F
import numpy as np

class Newton(Optimizer):
    """Implements Newton's method with constant step size or backtracking line search.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        step_type (str): type of step size to use ('Constant' or 'Backtracking')
        constant_step_size (float, optional): constant step size for 'Constant' step type
        alpha_bar (float, optional): initial step size for 'Backtracking' step type
        tau (float, optional): step size reduction factor for 'Backtracking' step type
        c1 (float, optional): sufficient decrease parameter for 'Backtracking' step type
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    
    Example:
        >>> optimizer = Newton(model.parameters(), step_type='Constant', constant_step_size=1.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step(closure=lambda: loss_fn(model(input), target))
    """

    def __init__(self, params, step_type='Constant', constant_step_size=1.0, 
                 alpha_bar=1.0, tau=0.5, c1=1e-4, weight_decay=0):
        if step_type not in ['Constant', 'Backtracking']:
            raise ValueError(f"step_type must be 'Constant' or 'Backtracking', got {step_type}")
        
        defaults = dict(step_type=step_type, constant_step_size=constant_step_size,
                        alpha_bar=alpha_bar, tau=tau, c1=c1, weight_decay=weight_decay)
        super(Newton, self).__init__(params, defaults)
        
        # Store model reference for Hessian computation
        self.model = None
        self.loss_fn = None

    def __setstate__(self, state):
        super(Newton, self).__setstate__(state)

    def set_model_and_loss(self, model, loss_fn):
        """Set the model and loss function for Hessian computation.
        
        Args:
            model: PyTorch model
            loss_fn: Loss function
        """
        self.model = model
        self.loss_fn = loss_fn

    def compute_hessian(self, x):
        """Compute the Hessian matrix at point x using autograd.
        
        Args:
            x: Input tensor
            
        Returns:
            H: Hessian matrix
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model_and_loss first.")
            
        # Use PyTorch's functional Hessian
        return F.hessian(lambda x: self.model(x), x)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Required for backtracking line search.
        
        Returns:
            loss: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            step_type = group['step_type']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                d_p = p.grad.data
                
                # Apply weight decay if specified
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Compute Hessian if model is set
                if self.model is not None:
                    # This is a simplified approach - in practice, you'd need to handle
                    # the Hessian computation for all parameters together
                    H = self.compute_hessian(p.data)
                    
                    # Compute Newton direction by solving H*d = -g
                    try:
                        d = -torch.linalg.solve(H, d_p.unsqueeze(1)).squeeze(1)
                    except:
                        # Fallback to gradient descent if Hessian is singular
                        d = -d_p
                else:
                    # Fallback to gradient descent if model is not set
                    d = -d_p
                
                # Determine step size based on step_type
                if step_type == 'Constant':
                    alpha = group['constant_step_size']
                    p.data.add_(d, alpha=alpha)
                
                elif step_type == 'Backtracking':
                    alpha = group['alpha_bar']
                    tau = group['tau']
                    c1 = group['c1']
                    
                    # Store original parameter value and gradient
                    original_p = p.data.clone()
                    grad = d_p.clone()
                    
                    # Compute current function value if closure is provided
                    if closure is not None:
                        original_loss = closure()
                    else:
                        # If no closure provided, we can't do backtracking line search
                        p.data.add_(d, alpha=alpha)
                        continue
                    
                    # Backtracking line search
                    while True:
                        # Update parameter with current step size
                        p.data = original_p + alpha * d
                        
                        # Compute new function value
                        new_loss = closure()
                        
                        # Check Armijo condition
                        armijo_rhs = original_loss + c1 * alpha * torch.sum(grad * d)
                        
                        if new_loss <= armijo_rhs:
                            break
                        
                        # Reduce step size
                        alpha = tau * alpha
                        
                        # Safety check to prevent infinite loop
                        if alpha < 1e-10:
                            p.data = original_p  # Revert to original parameters
                            break
                
                else:
                    raise ValueError(f"Unknown step type: {step_type}")

        return loss 