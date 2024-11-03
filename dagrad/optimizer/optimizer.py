import numpy as np
from scipy.optimize import OptimizeResult
import scipy.linalg as sla
from ..utils.general_utils import get_variables_from_optimizer_settings


# This is the optimizer for the dagma function

def optimizer_dagma(
        fun,
        x0,
        args=(),
        options = None,
        **kwargs
):  
    if options is None: 
        options = {}
    
    d = x0.shape[0]
    W = np.array(x0, copy=True)

    obj_prev = 1e16
    obj_new, g = fun(W,*args)

    # get options for printing
    verbose = options.get('verbose', False)
    vprint = print if verbose else lambda *a, **k: None

    # get options for all optimizer
    num_steps, check_iterate, tol, lr_decay = get_variables_from_optimizer_settings(options)
    lr = options['lr']
    s = options.get('s', 1.0)
    
    Id = np.eye(d)

    if options.get('name')=='adam':
        m = np.zeros_like(W)
        v = np.zeros_like(W)    
        beta1, beta2 = options.get('betas', (0.99, 0.999))
        eps = options.get('eps', 1e-8)
    elif options.get('name')=='lbfgs':
        # get options for lbfgs optimizer
        # lbfgs is not proper for this problem
        raise NotImplementedError("L-BFGS optimizer is not implemented yet.")
    elif options.get('name')=='gd':
        pass
    vprint(f"\n\nMinimize with -- mu:{options['mu']} -- lr: {lr} -- s: {options['s']} for {num_steps} max iterations")
    mask_exc = np.ones((d, d))
    if options['exc_c'] is not None:
        mask_exc[options['exc_r'], options['exc_c']] = 0.
    for iter in range(1,num_steps+1):
        M = sla.inv(s * Id - W * W) + 1e-16
        while np.any(M < 0):
            if iter == 1 or s <= 0.9:
                vprint(f'W went out of domain for s={s} at iteration {iter}')
                return OptimizeResult(x = W, fun = 1e16, success = False)
            else:
                W += lr * grad 
                lr *= .5
                obj_new, g = fun(W,*args)
                if lr < 1e-16:
                    return OptimizeResult(x = W, fun = obj_new, success = True)
                W -= lr * grad
                M = sla.inv(s * Id - W * W) + 1e-16
                vprint(f'Learning rate decreased to lr: {lr}')

        obj_new, g = fun(W,*args)
        # then update the gradient 
        if options.get('name')=='adam':
            # pack this to a function
            m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(iter))  # bias correction.
            vhat = v / (1 - beta2**(iter))
            grad = mhat / (np.sqrt(vhat) + eps)
        elif options.get('name')=='lbfgs':
            pass
        elif options.get('name')=='gd':
            grad = g
            

        W = W - lr * grad
        W *= mask_exc

        if  iter % check_iterate == 0 or iter == num_steps:
            obj_new, g = fun(W,*args)
            if abs(obj_prev - obj_new) / max(abs(obj_prev),abs(obj_new),1) <= tol:
                break
            obj_prev = obj_new
                                                                                                                                                                                                                           
    return OptimizeResult(x=W, fun=obj_new, success=True)

def adam(
    fun,
    x0,
    args=(),
    bnds = [],
    options = None,
    **kwargs
):
    if options is None: 
        options = {}
    
    num_steps = options.get('num_steps',50000)
    lr = options.get('lr', 0.005)
    beta1,beta2 = options.get('betas', (0.95, 0.999))
    eps = options.get('eps', 1e-8)
    check_iterate = options.get('check_iterate', 1000)
    tol = options.get('tol', 1e-6)
    x = np.array(x0, copy=True)
    x[bnds] = 0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    obj_prev = 1e16
    for i in range(num_steps):
        obj, g = fun(x,*args)
        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        x[bnds] = 0
        if  i % check_iterate == 0 or i == (num_steps-1):
            obj_new, g = fun(x,*args)
            if abs(obj_prev - obj_new) / max(abs(obj_prev),abs(obj_new),1) < tol:
                break
            obj_prev = obj_new

    return OptimizeResult(x=x, fun=obj, success=True)
