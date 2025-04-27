from torch.optim.optimizer import Optimizer


class BaseOptimizer(Optimizer):

    def step(self, fn_cls=None, grad_cls=None, hess_cls=None):
        raise NotImplementedError("Derived optimizers must implement this method")
