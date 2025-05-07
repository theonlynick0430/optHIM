from optHIM.algorithms.base import BaseOptimizer
from optHIM.utils.solvers import quadratic_2D
from optHIM.algorithms.bfgs import bfgs_update
from optHIM.algorithms.dfp import dfp_update
import torch


class TrustRegion(BaseOptimizer):
    def __init__(self, x, model="sr1", solver="cg", delta0=1.0, min_delta=1e-6, max_delta=1e2, c1=0.25, c2=0.75, c3=1e-6, tol=1e-6, max_iter=1e2):
        """
        Implements trust region method.
        
        Args:
            x (torch.Tensor): parameter to optimize
            model (str, optional): type of model to approximate function ('newton', 'sr1', 'bfgs', or 'dfp')
            solver (str, optional): type of solver for trust region subproblem ('cg' or 'cauchy')
            delta0 (float, optional): initial trust region radius
            min_delta (float, optional): minimum trust region radius
            max_delta (float, optional): maximum trust region radius
            c1 (float, optional): lower bound for acceptance ratio (0 < c1 < c2 < 1)
            c2 (float, optional): upper bound for acceptance ratio (0 < c1 < c2 < 1)
            c3 (float, optional): skip sr1 update unless ||(y-Bs)^T s|| >= c3 ||y-Bs|| ||s||
            tol (float, optional): tolerance for CG convergence
            max_iter (int, optional): maximum number of CG iterations 
        """
        if not (0 < c1 < c2 < 1):
            raise ValueError("c1 and c2 must satisfy 0 < c1 < c2 < 1")
        defaults = dict(min_delta=min_delta, max_delta=max_delta, c1=c1, c2=c2, c3=c3, tol=tol, max_iter=max_iter)
        super(TrustRegion, self).__init__([x], defaults)
        self.x = x
        self.model = model
        self.solver = solver
        # trust region radius
        self.delta = delta0
        if self.model == "sr1" or self.model == "bfgs" or self.model == "dfp":
            # initialize state for any model with memory
            self.state = {
                'x_prev': None,
                'grad_x_prev': None,
                # start approximation of Hessian as identity matrix
                'hess_x_prev': torch.eye(x.numel(), device=x.device, dtype=x.dtype)
            }

    def cg_solve(self, grad_x, hess_x, delta, tol, max_iter):
        """
        Solve the trust region subproblem using CG Steihaug method.

        Args:
            grad_x (torch.Tensor): gradient of function
            hess_x (torch.Tensor): Hessian of function
            delta (float): trust region radius
            tol (float): tolerance for CG convergence
            max_iter (int): maximum number of CG iterations

        Returns:
            d (torch.Tensor): solution step
        """
        z = torch.zeros_like(grad_x)
        r = grad_x
        p = -r

        if torch.norm(r) < tol:
            return z
        
        for _ in range(max_iter):
            # check for negative curvature
            if p @ hess_x @ p <= 0:
                # go to boundary of trust region
                # find tau s.t. ||z + tau p|| = delta using quadratic formula                
                _a = p @ p 
                _b = 2 * p @ z
                _c = z @ z - delta ** 2
                tau1, tau2 = quadratic_2D(_a, _b, _c)
                # return positive solution
                return z + max(tau1, tau2) * p
            
            # compute optimal step size for quadratic model in direction p
            alpha = (r @ r) / (p @ hess_x @ p)
            z_next = z + alpha * p

            # check if we are outside the trust region
            if torch.norm(z_next) > delta:
                # project onto trust region boundary
                # find tau s.t. ||z + tau p|| = delta using quadratic formula
                _a = p @ p 
                _b = 2 * p @ z
                _c = z @ z - delta ** 2
                tau1, tau2 = quadratic_2D(_a, _b, _c)
                # return positive solution
                return z + max(tau1, tau2) * p
            
            # CG residual
            r_next =  r + alpha * hess_x @ p

            if torch.norm(r_next) < tol:
                return z_next

            # CG direction update
            beta = (r_next @ r_next) / (r @ r)
            p = -r_next + beta * p

            r = r_next
            z = z_next

        return z

    def sr1_update(self, x, grad_x, hess_x_prev, c3, x_prev=None, grad_x_prev=None):
        """
        Update the Hessian approximation using SR1 method.

        Args:
            x (torch.Tensor): current parameter
            grad_x (torch.Tensor): gradient of function
            hess_x_prev (torch.Tensor): previous Hessian approximation
            c3 (float): skip SR1 update unless ||(y-Bs)^T s|| >= c3 ||y-Bs|| ||s||
            x_prev (torch.Tensor, optional): previous parameter
            grad_x_prev (torch.Tensor, optional): previous gradient

        Returns:
            hess_x (torch.Tensor): updated Hessian approximation
        """
        # if SR1 update is skipped, use previous approximation
        # hess x_k
        hess_x = hess_x_prev

        if x_prev is not None:
            # compute SR1 update
            # s_{k-1} = x_k - x_{k-1}
            s = x - x_prev
            # y_{k-1} = grad x_k - grad x_{k-1}
            y = grad_x - grad_x_prev
            vec = y - hess_x_prev @ s
            if vec @ s >= c3 * torch.norm(vec) * torch.norm(s):
                # update hess_x if |(y-Bs)^T s| >= c3 ||y-Bs|| ||s||
                hess_x = hess_x_prev + torch.outer(vec, vec) / (vec @ s)
        
        return hess_x
    
    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        """
        Performs a single optimization step.
        
        Args:
            fn_cls (callable, optional): closure that reevaluates the function
            grad_cls (callable, optional): closure that recomputes the gradients
            hess_cls (callable, optional): closure that recomputes the Hessian
        """ 
        x = self.x.data
        f = fn_cls(x) # disable gradient computation

        # build quadratic model of function
        grad_x = self.x.grad.data
        if self.model == "newton":
            hess_x = hess_cls(x)
        elif self.model == "sr1":
            hess_x = self.sr1_update(x, grad_x, self.state['hess_x_prev'], self.param_groups[0]['c3'], self.state['x_prev'], self.state['grad_x_prev'])
        elif self.model == "bfgs":
            hess_x = bfgs_update(x, grad_x, self.state['hess_x_prev'], self.param_groups[0]['c3'], self.state['x_prev'], self.state['grad_x_prev'])
        elif self.model == "dfp":
            hess_x = dfp_update(x, grad_x, self.state['hess_x_prev'], self.param_groups[0]['c3'], self.state['x_prev'], self.state['grad_x_prev'])
        else:
            raise ValueError(f"Invalid model: {self.model}")
        
        # solve subproblem using model
        if self.solver == "cg":
            d = self.cg_solve(grad_x, hess_x, self.delta, self.param_groups[0]['tol'], self.param_groups[0]['max_iter'])
        elif self.solver == "cauchy":
            # cauchy point is the first step of CG
            d = self.cg_solve(grad_x, hess_x, self.delta, self.param_groups[0]['tol'], 1)
        else:
            raise ValueError(f"Invalid solver: {self.solver}")
        
        # evaluate accuracy of the model
        f_trial = fn_cls(x + d) # disable gradient computation
        md = f + grad_x @ d + 0.5 * d @ hess_x @ d
        # safe guard for division by zero
        rho = self.param_groups[0]['c2'] + 1.0 if f - md < 1e-6 else (f - f_trial) / (f - md)

        # update trust region radius
        if rho > self.param_groups[0]['c1']:
            # update state for hessian approximators with memory
            if self.model == "sr1" or self.model == "bfgs" or self.model == "dfp":
                self.state['x_prev'] = x.clone()
                self.state['grad_x_prev'] = grad_x.clone()
                self.state['hess_x_prev'] = hess_x.clone()

            # accept step
            x += d

            if rho > self.param_groups[0]['c2']:
                # increase trust
                self.delta *= 2.0
        else:
            # reject step (do nothing)
            # decrease trust
            self.delta *= 0.5

        # clip trust region radius
        self.delta = max(self.param_groups[0]['min_delta'], min(self.param_groups[0]['max_delta'], self.delta))