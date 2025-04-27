from optHIM.algorithms.base import BaseOptimizer


class TrustRegion(BaseOptimizer):
    def __init__(self, x, delta0=1.0, c1=0.25, c2=0.75):
        """
        Implements trust region method.
        
        Args:
            x (torch.Tensor): parameter to optimize
            delta0 (float, optional): initial trust region radius
            c1 (float, optional): lower bound for acceptance ratio (0 < c1 < c2 < 1)
            c2 (float, optional): upper bound for acceptance ratio (0 < c1 < c2 < 1)
        """
        if not (0 < c1 < c2 < 1):
            raise ValueError("c1 and c2 must satisfy 0 < c1 < c2 < 1")
        defaults = dict(c1=c1, c2=c2)
        super(TrustRegion, self).__init__([x], defaults)
        self.x = x
        # trust region radius
        self.delta = delta0

    def solve_subproblem(self):
        pass

    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function
            grad_cls (callable, optional): closure that recomputes the gradients
            hess_cls (callable, optional): closure that recomputes the Hessian
        """ 
        # TODO: solve subproblem using model
        self.solve_subproblem() 
        f = fn_cls()
        f_trial = None
        m0 = None
        md = None

        # evaluate accuracy of the model
        rho = (f - f_trial) / (m0 - md)
        
        # update trust region radius
        if rho > self.param_groups[0]['c1']:
            # TODO: accept step
            if rho > self.param_groups[0]['c2']:
                # increase trust
                self.delta *= 2.0
        else:
            # TODO: reject step
            # decrease trust
            self.delta *= 0.5
