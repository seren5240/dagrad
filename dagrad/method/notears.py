import numpy as np
import scipy.optimize as sopt
import time
import torch
from ..utils.configure import h_functions, loss_functions, reg_functions, nl_loss_functions, optimizer_functions, allowed_method_options
from ..optimizer.optimizer import adam
from ..utils.NNstructure import NotearsMLP
from ..utils.general_utils import set_functions,print_options, set_options, get_variables_from_general_options,get_variables_from_optimizer_settings, check_device, demean, process_data
__all__ = ['notears_linear_numpy', 'notears_linear_torch', 'notears_nonlinear']

def notears_linear_numpy(X, 
                         loss_fn = 'l2', 
                         h_fn = 'h_exp_sq',
                         reg = 'l1', 
                         optimizer = 'lbfgs',
                         main_iter=100, 
                         h_tol=1e-8, 
                         rho = 1.0,
                         rho_max=1e+16,
                         dtype = np.float64,
                         verbose = False, 
                         **options):
    
    
    """Learn DAG structure using NOTEARS algorithm with **linear** structural equation and **numpy**

    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    loss_fn : str, optional
        Loss function, by default ``'l2'``.
    h_fn : str, optional
        h function, by default ``'h_exp_sq'``.
    reg : str, optional
        Regularization function, by default ``'l1'``.
    optimizer : str, optional
        Optimizer, by default ``'lbfgs'``.
    main_iter : int, optional
        Number of main iterations, by default ``100``.
    h_tol : float, optional
        Tolerance for h, by default ``1e-8``.
    rho : float, optional
        hyperparameter for lbfgs, by default ``1.0``.
    rho_max : float, optional
        maximum value for rho, by default ``1e+16``.
    dtype : np.float64, optional
        data type, by default ``np.float64``.
    verbose : bool, optional
        print out the information, by default ``False``.
    options : dict, optional
        Additional general/optimizer options, by default {}.
    
    Returns
    -------
    np.ndarray
        Estimated DAG matrix with shape :math:`(p,p)`.

    """
    vprint = print if verbose else lambda *a, **k: None
    # get options
    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'linear', compute_lib = 'numpy', method= 'notears', 
                                                                                        optimizer = optimizer, reg = reg)
    
    method_options = allowed_method_options['linear']['notears'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)

    optimizer_options = {**optimizer_options_config, **optimizer_options_settings}
    _, _, _, w_threshold, initialization, _ = get_variables_from_general_options(general_options)
    compute_lib, device = 'numpy', 'cpu'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None



    X = demean(X) if loss_fn == 'l2' else X

    _loss = set_functions(loss_fn,loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    
    
    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return w.reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W = W,X = X, **general_options)
        h, G_h = _h(W, **general_options)
        reg, G_reg = _reg(W, **general_options)
        obj = loss + 0.5 * rho * h * h + alpha * h +  reg
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = (G_smooth + G_reg).flatten()
        return obj, g_obj

    n, d = X.shape
    if initialization is None:
        # default lbfgs only takes 1d array
        w_est, alpha, h = np.zeros(d * d), 0.0, np.inf  
    else:
        assert initialization.shape == (d,d) or initialization.shape == (d*d,), "initialization must be a square matrix or a vector"
        if initialization.shape == (d,d):
            w_est = initialization.flatten()
        else:
            w_est = initialization
        alpha, h = 0.0, np.inf
    
    time_start = time.time()
    for _ in range(main_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            if optimizer == 'lbfgs': # use scipy implementation
                bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
                sol = sopt.minimize(fun = _func, x0 =  w_est, method = 'L-BFGS-B', jac = True, bounds = bnds, options=optimizer_options)
                w_new = sol.x
            elif optimizer == 'adam': 
                # self implemented adam
                bnds = [i*d+i for i in range(d)] # diagonal elements
                sol = adam(fun = _func, x0 = w_est, bnds = bnds, options = optimizer_options)
                # sol = sopt.minimize(fun = _func, x0 = w_est, method = adam, jac = True, options = optimizer_options)
                w_new = sol.x
            h_new, __ = _h(_adj(w_new),**general_options)
            vprint(f"current iteration: {_}, current rho: {rho}, current h: {h_new}")
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    time_end = time.time()
    vprint(f"Total Time: {time_end - time_start}")
    return W_est


def notears_linear_torch(X, 
                        loss_fn = 'l2', 
                        h_fn = 'h_exp_sq',
                        reg = 'l1', 
                        optimizer = 'lbfgs',
                        rho = 1.0,
                        main_iter=100, 
                        h_tol=1e-8, 
                        rho_max=1e+16,
                        dtype = torch.float64,
                        verbose = False,
                        **options):
    
    
    """Learn DAG structure using NOTEARS algorithm with **linear** structural equation and **torch**

    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    loss_fn : str, optional
        Loss function, by default ``'l2'``.
    h_fn : str, optional
        h function, by default ``'h_exp_sq'``.
    reg : str, optional
        Regularization function, by default ``'l1'``.
    optimizer : str, optional
        Optimizer, by default ``'lbfgs'``.
    main_iter : int, optional
        Number of main iterations, by default ``100``.
    h_tol : float, optional
        Tolerance for h, by default ``1e-8``.
    rho : float, optional
        hyperparameter for lbfgs, by default ``1.0``.
    rho_max : float, optional
        maximum value for rho, by default ``1e+16``.
    dtype : torch.float64, optional
        data type, by default ``torch.float64``.
    verbose : bool, optional
        print out the information, by default ``False``.
    options : dict, optional
        Additional general/optimizer options, by default {}.

    Returns
    -------
    np.ndarray
        Estimated DAG matrix with shape :math:`(p,p)`.
    """
    
    torch.set_default_dtype(dtype)
    vprint = print if verbose else lambda *a, **k: None
    device = options.get('device', 'cpu')
    device = check_device(device)
    vprint(f"Using device: {device}")
    X = demean(X) if loss_fn == 'l2' else X
    X = process_data(X = X, dtype = dtype, device = device)
    

    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'linear', compute_lib = 'torch', method= 'notears', 
                                                                                        optimizer = optimizer,reg = reg)
    method_options = allowed_method_options['linear']['notears'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)

    _, _, _, w_threshold, initialization,_ = get_variables_from_general_options(general_options)

    compute_lib = 'torch'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None


    _loss = set_functions(loss_fn,loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    _opt = set_functions(optimizer, optimizer_functions)


    n, d = X.shape
    if initialization is None:
        W_est = torch.zeros(d, d, dtype=dtype, requires_grad=True, device=device)
    else:
        assert initialization.shape == (d,d) or initialization.shape == (d*d,), "initialization must be a square matrix or a vector"
        if initialization.shape == (d,d):
            W_est = torch.tensor(initialization, dtype=dtype, requires_grad=True, device=device)
        else:
            W_est = torch.tensor(initialization.reshape(d,d), dtype=dtype, requires_grad=True, device=device)

    rho, alpha, h = 1.0, 0.0, float('inf')
    time_start = time.time()

    def minimize(W_est, rho, alpha):
        num_steps, check_iterate, tol, lr_decay = get_variables_from_optimizer_settings(optimizer_options_settings)
        
        opt = _opt([W_est],**optimizer_options_config)
        def closure():
            opt.zero_grad()
            loss = _loss(W = W_est, X = X, **general_options)
            h = _h(W_est,**general_options)
            penalty = 0.5 * rho * h * h + alpha * h
            total_loss = loss + penalty + _reg(W_est,**general_options)
            total_loss.backward()
            return total_loss
        obj_prev = float('inf')
        for i in range(1+num_steps):
            loss = opt.step(closure)
            obj_new = loss.item()
            vprint(f"iter: {i}, loss: {obj_new}")
            if (i % check_iterate == 0 or i == num_steps):
                diff = abs(obj_prev - obj_new)
                denom = max(abs(obj_prev),abs(obj_new),1)
                if diff / denom <= tol:
                    break
                obj_prev = obj_new
        
            
    for _ in range(main_iter):
        vprint("working on iteration: ", _)
        
        while rho < rho_max:
            ######### 
            # create a new W_est_backup to store the new W_est,
            # to avoid the failure of minimize_ function such that h is not significantly reduced
            W_est_backup = W_est.clone()
            time_start = time.time()
            minimize(W_est = W_est,
                      rho = rho, alpha = alpha)
            time_end = time.time()
            with torch.no_grad():
                h_new = _h(W_est,**general_options).item()
            vprint(f"current iteration: {_}, current rho: {rho}, current h: {h_new}")
            if h_new > 0.25 * h:
                rho *= 10
                with torch.no_grad():
                    W_est.copy_(W_est_backup).requires_grad_(True)
            else:
                break
        h = h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est_np = W_est.detach().cpu().numpy()
    W_est_np[np.abs(W_est_np) < w_threshold] = 0
    time_end = time.time()
    vprint(f"Total Time: {time_end - time_start}")
    return W_est_np

def notears_nonlinear(X, 
                    loss_fn = 'l2', 
                    h_fn = 'h_exp_sq',
                    reg = 'l1', 
                    optimizer = 'lbfgs', 
                    bias = True, 
                    activation = 'sigmoid',
                    main_iter = 100,
                    rho_max = 1e+16, 
                    h_tol = 1e-8,
                    dims = None,
                    dtype = torch.float64,
                    verbose = False,
                    **options):
    
    """Learn DAG structure using NOTEARS algorithm with **nonlinear** structural equation and **torch**

    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    loss_fn : str, optional
        Loss function, by default ``'l2'``.
    h_fn : str, optional
        h function, by default ``'h_exp_sq'``.
    reg : str, optional
        Regularization function, by default ``'l1'``.
    optimizer : str, optional
        Optimizer, by default ``'lbfgs'``.
    bias : bool, optional
        whether to include bias, by default ``True``.
    activation : str, optional
        activation function, by default ``'sigmoid'``.
    main_iter : int, optional
        Number of main iterations, by default ``100``.
    rho_max : float, optional
        maximum value for rho, by default ``1e+16``.
    h_tol : float, optional
        Tolerance for h, by default ``1e-8``.
    dims : list, optional
        dimensions for the neural network, by default ``None``.
    dtype : torch.float64, optional
        data type, by default ``torch.float64``.
    verbose : bool, optional
        print out the information, by default ``False``.
    options : dict, optional
        Additional general/optimizer options, by default {}.

    Returns
    -------
    np.ndarray
        Estimated DAG matrix with shape :math:`(p,p)`.
    """

    torch.set_default_dtype(dtype)
    vprint = print if verbose else lambda *a, **k: None
    device = options.get('device', 'cpu')
    device = check_device(device)
    vprint(f"Using device: {device}")
    X = process_data(X = X, dtype = dtype, device = device)
    n,d = X.shape
    if dims is None:
        dims = [d,40,1]

    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'nonlinear', compute_lib = 'torch', method= 'notears', 
                                                                                        optimizer = optimizer, reg = reg)
    method_options = allowed_method_options['nonlinear']['notears'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)
    compute_lib = 'torch'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None

    lambda1, lambda2, gamma, w_threshold, initialization, _ = get_variables_from_general_options(general_options)

    _loss = set_functions(loss_fn,nl_loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    _opt = set_functions(optimizer, optimizer_functions)

    
    def minimize(model, alpha, rho):
            num_steps, check_iterate, tol, lr_decay = get_variables_from_optimizer_settings(optimizer_options_settings)
            opt = _opt(model.parameters(),**optimizer_options_config)
            # scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)
            def closure():
                opt.zero_grad()
                X_hat = model(X)
                loss = _loss(X_hat,X, **general_options)
                h_val = _h(model.adj(), **general_options)
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * model.l2_reg()

                reg = lambda1 * model.fc1_l1_reg()
                # reg =  _reg(model.adj(),**general_options)

                total_loss = loss + penalty + l2_reg + reg
                total_loss.backward()
                return total_loss

            prev_loss = float('inf')
            for i in range(1,num_steps+1):
                loss = opt.step(closure)
                current_loss = loss.item()
                # scheduler.step()
                # print(f"iter: {i}, loss: {current_loss}")

                if (i % check_iterate == 0 or i == num_steps):
                    diff = abs(prev_loss - current_loss)
                    denom = max(abs(prev_loss),abs(current_loss),1)
                    if diff / denom <= tol:
                        break
                    prev_loss = current_loss
    
    
    def dual_ascent_step(model, rho, alpha, h):
        # model_copy = deepcopy(model).to(device)
        while rho < rho_max:
            minimize(model = model,alpha = alpha, rho = rho)
            with torch.no_grad():
                h_new = _h(model.adj(),**general_options).item() 
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break

        alpha += rho * h_new
        return rho, alpha, h_new


    rho, alpha, h = 1.0, 0.0, np.inf
    if initialization is None:
        model = NotearsMLP(dims = dims, bias = bias, activation = activation, dtype = dtype).to(device)
    else:
        # check initialization and model are consistent 
        model = NotearsMLP(dims = dims, bias = bias, activation = activation, dtype = dtype).to(device)
        try:
            model.load_state_dict(initialization)
        except:
            raise ValueError("initialization must be a state_dict of the mode, and the model must be consistent with the initialization")
    time_start = time.time()
    for _ in range(main_iter):
        rho, alpha, h = dual_ascent_step(model = model,
                                         rho = rho, alpha = alpha, h = h)       
        vprint(f"current {_}, current rho: {rho}, h: {h}")
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    time_end = time.time()
    vprint(f"Total Time: {time_end - time_start}")
    return W_est



