from optHIM.algorithms.base import BaseOptimizer
import optHIM.algorithms.ls as ls


class GD(BaseOptimizer):
    def __init__(self, x, step_type='constant', step_size=1e-3, 
                 alpha=1.0, tau=0.5, c1=1e-4, c2=0.9, alpha_high=1000.0, 
                 alpha_low=0.0, c=0.5):
        """
        Implements gradient descent with constant step size or backtracking line search.
        
        Args:
            x (torch.Tensor): parameter to optimize
            step_type (str): type of step size to use ('constant', 'diminish', 'armijo', or 'wolfe')
            step_size (float, optional): constant step size for 'constant' step type
            alpha (float, optional): initial step size for 'armijo' step type
            tau (float, optional): step size reduction factor for 'armijo' step type
            c1 (float, optional): sufficient decrease parameter for 'armijo' step type
            c2 (float, optional): curvature condition parameter for 'wolfe' step type
            alpha_high (float, optional): upper bound for step size for 'wolfe' step type
            alpha_low (float, optional): lower bound for step size for 'wolfe' step type
            c (float, optional): interpolation parameter for 'wolfe' step type
        """
        if step_type not in ['constant', 'diminish', 'armijo', 'wolfe']:
            raise ValueError(f"step_type must be 'constant', 'diminish', 'armijo', or 'wolfe', got {step_type}")
        defaults = dict(step_type=step_type, step_size=step_size,
                        alpha=alpha, tau=tau, c1=c1, c2=c2, alpha_high=alpha_high,
                        alpha_low=alpha_low, c=c)
        super(GD, self).__init__([x], defaults)
        self.x = x
        if step_type == 'diminish':
            self.k = 1

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function.
                Required for backtracking line search.
            grad_cls (callable, optional): closure that recomputes the gradients.
                Required for Wolfe line search.
            hess_cls (callable, optional): Not required for this optimizer.
        """
        if self.x.grad is None:
            return
            
        # x_k
        x = self.x.data
        # grad x_k
        grad_x = self.x.grad.data
        # compute search direction
        d = -grad_x

        # line search
        if self.param_groups[0]['step_type'] == 'constant':
            alpha = self.param_groups[0]['step_size']
            x += alpha * d
        elif self.param_groups[0]['step_type'] == 'diminish':
            alpha_k = self.param_groups[0]['step_size']/self.k
            x += alpha_k * d
            self.k += 1
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