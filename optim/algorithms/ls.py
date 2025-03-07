import torch


def armijo(p, d, loss_cl, alpha=1.0, tau=0.5, c1=1e-4, max_iter=1e2):
    """
    Performs armijo backtracking line search to find an acceptable step size.
    
    Args:
        p (torch.Tensor): parameter tensor to be updated
        d (torch.Tensor): search direction
        loss_cl (callable): closure that reevaluates the model and returns the loss
        alpha (float, optional): initial step size
        tau (float, optional): step size reduction factor
        c1 (float, optional): sufficient decrease parameter
        max_iter (int, optional): maximum number of iterations
    
    Returns:
        alpha (float): selected step size
    """
    # compute initial param, gradient, and loss
    p0 = p.data.clone()
    d_p0 = p.grad.data.clone()
    loss0 = loss_cl()
    
    # backtracking line search
    for _ in range(int(max_iter)):
        # update parameter with current step size
        p.data = p0 + alpha * d
                        
        # check armijo condition
        armijo_rhs = loss0 + c1 * alpha * torch.dot(d_p0.view(-1), d.view(-1))
        loss = loss_cl()
        if loss <= armijo_rhs:
            return alpha
        
        # reduce step size
        alpha = tau * alpha

    return alpha
