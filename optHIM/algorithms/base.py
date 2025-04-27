from torch.optim.optimizer import Optimizer


class BaseOptimizer(Optimizer):

    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that returns the function evaluated at given point
            grad_cls (callable, optional): closure (void) that updates the gradient at given point
            hess_cls (callable, optional): closure that returns the Hessian evaluated at given point
        """
        raise NotImplementedError("Derived optimizers must implement this method")
