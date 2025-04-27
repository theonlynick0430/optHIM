

def armijo(x, d, fn_cls, alpha=1.0, tau=0.5, c1=1e-4, max_iter=1e2):
    """
    Performs armijo backtracking line search to find an acceptable step size.
    
    Args:
        x (torch.Tensor): parameter to be optimized
        d (torch.Tensor): search direction
        fn_cls (callable): closure that returns the function evaluated at given point
        alpha (float, optional): initial step size
        tau (float, optional): step size reduction factor
        c1 (float, optional): sufficient decrease parameter
        max_iter (int, optional): maximum number of iterations
    
    Returns:
        alpha (float): selected step size
    """
    x0 = x.data.clone()
    grad_x0 = x.grad.data.clone()
    f0 = fn_cls(x0) # disable gradient computation
    
    # backtracking line search
    for _ in range(int(max_iter)):
        # update parameter with current step size
        x.data = x0 + alpha * d
                        
        # check armijo condition
        armijo_rhs = f0 + c1 * alpha * grad_x0 @ d
        f = fn_cls(x.data) # disable gradient computation
        if f <= armijo_rhs:
            return alpha
        
        # reduce step size
        alpha = tau * alpha

    return alpha

def wolfe(x, d, fn_cls, grad_cls, alpha=1.0, alpha_high=1000.0, alpha_low=0.0, c=0.5, c1=1e-4, c2=0.9, max_iter=1e2):
    """
    Performs Wolfe line search to find an acceptable step size.
    
    Args:
        x (torch.Tensor): parameter to be optimized
        d (torch.Tensor): search direction
        fn_cls (callable): closure that returns the function evaluated at given point
        grad_cls (callable): closure (void) that updates the gradient at given point
        alpha (float, optional): initial step size
        alpha_high (float, optional): upper bound for step size
        alpha_low (float, optional): lower bound for step size
        c (float, optional): interpolation parameter
        c1 (float, optional): sufficient decrease parameter (armijo condition)
        c2 (float, optional): curvature condition parameter
        max_iter (int, optional): maximum number of iterations
    
    Returns:
        alpha (float): selected step size
    """
    x0 = x.data.clone()
    grad_x0 = x.grad.data.clone()
    f0 = fn_cls(x0) # disable gradient computation
        
    for _ in range(int(max_iter)):
        # update parameter with current step size
        x.data = x0 + alpha * d
        
        # check armijo condition
        armijo_rhs = f0 + c1 * alpha * grad_x0 @ d
        f = fn_cls(x.data) # disable gradient computation
        if f <= armijo_rhs:
            grad_cls(x) # enable gradient computation
            grad_x = x.grad.data
            # check curvature condition
            if grad_x @ d >= c2 * grad_x0 @ d:
                return alpha
            
        # update bounds
        if f <= armijo_rhs:
            alpha_low = alpha
        else:
            alpha_high = alpha
        
        # interpolate to get new alpha
        alpha = c * alpha_low + (1 - c) * alpha_high
    
    return alpha