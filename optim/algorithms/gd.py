from torch.optim.optimizer import Optimizer
import optim.algorithms.ls as ls


class GD(Optimizer):
    def __init__(self, params, step_type='constant', step_size=1e-3, 
                 alpha=1.0, tau=0.5, c1=1e-4):
        """Implements gradient descent with constant step size or backtracking line search.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            step_type (str): type of step size to use ('constant' or 'armijo')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
        
        Example:
            >>> optimizer = GD(model.parameters(), step_type='constant', step_size=0.1)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()
        """
        if step_type not in ['constant', 'armijo']:
            raise ValueError(f"step_type must be 'constant' or 'armijo', got {step_type}")
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

                if step_type == 'constant':
                    alpha = group['step_size']
                    p.data += alpha * d
                
                elif step_type == 'armijo':
                    if loss_cl is None:
                        raise ValueError("loss_cl must be provided for armijo line search")
                    alpha = group['alpha']
                    tau = group['tau']
                    c1 = group['c1']
                    ls.armijo(p, d, loss_cl, alpha, tau, c1)