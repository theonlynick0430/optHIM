from optHIM.algorithms.base import BaseOptimizer
import torch


class TrustRegion(BaseOptimizer):
    def __init__(self, x, model="sr1", solver="cg", delta0=1.0, c1=0.25, c2=0.75, c3=1e-6):
        """
        Implements trust region method.
        
        Args:
            x (torch.Tensor): parameter to optimize
            model (str, optional): type of model to approximate function
            solver (str, optional): type of solver for trust region subproblem
            delta0 (float, optional): initial trust region radius
            c1 (float, optional): lower bound for acceptance ratio (0 < c1 < c2 < 1)
            c2 (float, optional): upper bound for acceptance ratio (0 < c1 < c2 < 1)
            c3 (float, optional): skip sr1 update unless ||(y-Bs)^T s|| >= c3 ||y-Bs|| ||s||
        """
        if not (0 < c1 < c2 < 1):
            raise ValueError("c1 and c2 must satisfy 0 < c1 < c2 < 1")
        defaults = dict(c1=c1, c2=c2, c3=c3)
        super(TrustRegion, self).__init__([x], defaults)
        self.x = x
        self.model = model
        self.solver = solver
        # trust region radius
        self.delta = delta0
        # initialize state
        if self.model == "sr1":
            self.state = {
                'x_prev': None,
                'grad_x_prev': None,
                # start approximation of Hessian as identity matrix
                'hess_x_prev': torch.eye(x.numel(), device=x.device, dtype=x.dtype)
            }

    def solve_subproblem(self):
        # TODO: solve subproblem using model
        pass

    def sr1_update(self):
        # x_k
        x = self.x.data
        # grad x_k
        grad_x = self.x.grad.data
        # if SR1 update is skipped, use previous approximation
        # hess x_k
        hess_x = self.state['hess_x_prev']

        if self.state['x_prev'] is not None:
            # retrieve history
            # x_{k-1}
            x_prev = self.state['x_prev']
            # grad x_{k-1}
            grad_x_prev = self.state['grad_x_prev']
            # hess x_{k-1}
            hess_x_prev = self.state['hess_x_prev']
            # compute SR1 update
            # s_{k-1} = x_k - x_{k-1}
            s = x - x_prev
            # y_{k-1} = grad x_k - grad x_{k-1}
            y = grad_x - grad_x_prev
            vec = y - hess_x_prev @ s
            if torch.norm(vec @ s) >= self.param_groups[0]['c3'] * torch.norm(vec) * torch.norm(s):
                # update hess_x if ||(y-Bs)^T s|| >= c3 ||y-Bs|| ||s||
                hess_x = hess_x_prev + torch.outer(vec, vec) / (vec @ s)

        # update history
        self.state['x_prev'] = x.clone()
        self.state['grad_x_prev'] = grad_x.clone()
        self.state['hess_x_prev'] = hess_x.clone()
        
        return hess_x
    
    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function
            grad_cls (callable, optional): closure that recomputes the gradients
            hess_cls (callable, optional): closure that recomputes the Hessian
        """ 
        # build quadratic model of function
        grad_x = self.x.grad.data
        if self.model == "newton":
            hess_x = hess_cls()
        elif self.model == "sr1":
            hess_x = self.sr1_update()
        else:
            raise ValueError(f"Invalid model: {self.model}")
        

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
