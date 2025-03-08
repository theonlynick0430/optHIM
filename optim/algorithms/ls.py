import torch


def armijo(param, d, fn_cls, alpha=1.0, tau=0.5, c1=1e-4, max_iter=1e2):
    """
    Performs armijo backtracking line search to find an acceptable step size.
    
    Args:
        param (nn.Parameter): parameter to be updated
        d (torch.Tensor): search direction
        fn_cls (callable): closure that reevaluates the function
        alpha (float, optional): initial step size
        tau (float, optional): step size reduction factor
        c1 (float, optional): sufficient decrease parameter
        max_iter (int, optional): maximum number of iterations
    
    Returns:
        alpha (float): selected step size
    """
    p0 = param.data.clone()
    d_p0 = param.grad.data
    f0 = fn_cls()
    
    # backtracking line search
    for _ in range(int(max_iter)):
        # update parameter with current step size
        param.data = p0 + alpha * d
                        
        # check armijo condition
        armijo_rhs = f0 + c1 * alpha * d_p0 @ d
        f = fn_cls()
        if f <= armijo_rhs:
            return alpha
        
        # reduce step size
        alpha = tau * alpha

    return alpha
