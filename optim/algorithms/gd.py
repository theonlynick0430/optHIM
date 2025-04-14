from torch.optim.optimizer import Optimizer
import optim.algorithms.ls as ls


class GD(Optimizer):
    def __init__(self, params, step_type='constant', step_size=1e-3, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """
        Implements gradient descent with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            step_type (str): type of step size to use ('constant', 'diminish', or 'armijo')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
        """
        if step_type not in ['constant', 'diminish', 'armijo']:
            raise ValueError(f"step_type must be 'constant' or 'armijo', got {step_type}")
        defaults = dict(step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1)
        super(GD, self).__init__(params, defaults)
        if step_type == 'diminish':
            self.k = 1

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, fn_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function.
                Required for backtracking line search.
        """
        for group in self.param_groups:
            step_type = group['step_type']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                p = param.data
                d_p = param.grad.data
                # compute search direction
                d = -d_p

                if step_type == 'constant':
                    alpha = group['step_size']
                    p += alpha * d

                elif step_type == 'diminish':
                    alpha_k = group['step_size']/self.k
                    p += alpha_k * d
                    self.k += 1
                
                elif step_type == 'armijo':
                    if fn_cls is None:
                        raise ValueError("fn_cls must be provided for armijo line search")
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    ls.armijo(param, d, fn_cls, alpha, tau, c1)