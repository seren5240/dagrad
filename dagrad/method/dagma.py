import numpy as np
import typing
import copy
import time
import torch, torch.optim as optim
from ..utils.configure import h_functions, loss_functions,reg_functions,nl_loss_functions, optimizer_functions, allowed_method_options
from ..optimizer.optimizer import optimizer_dagma
from ..utils.NNstructure import DagmaMLP
from ..utils.general_utils import set_functions, print_options, set_options, get_variables_from_general_options, get_variables_from_optimizer_settings, demean, process_data, check_device  
__all__ = ["dagma_linear_numpy", "dagma_linear_torch",'dagma_nonlinear']

def dagma_linear_numpy(X:np.ndarray, 
                 loss_fn = 'l2',
                 h_fn = 'h_logdet_sq',
                 reg = 'l1', 
                 optimizer = 'adam',
                 T: int = 5, 
                 mu_init: float = 1.0, 
                 mu_factor: float = 0.1, 
                 s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6],
                 warm_iter: int = 3e4, 
                 main_iter: int = 6e4, 
                 dtype: type = np.float64,
                 exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None, 
                 include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
                 verbose: bool = False,
                   **options):
    """Learn DAG structure using DAGMA algorithm with **linear** structural equation and **numpy**.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.

    loss_fn : str, optional
        Loss function, by default ``'l2'``.

    h_fn : str, optional
        H function, by default ``'h_logdet_sq'``.

    reg : str, optional
        Regularization function, by default ``'l1'``.

    optimizer : str, optional
        Optimizer, by default ``'adam'``.

    T : int, optional
        Number of iterations, by default ``5``.

    mu_init : float, optional
        Initial value of :math:`\mu`, by default ``1.0``.

    mu_factor : float, optional
        Factor to increase :math:`\mu`, by default ``0.1``.

    s : typing.Union[typing.List[float], float], optional
        Controls the domain of M-matrices. Defaults to ``[1.0, .9, .8, .7, .6]``.

    warm_iter : int, optional
        Number of iterations for :math:`t < T`. Defaults to ``3e4``.

    main_iter : int, optional
        Number of iterations for :math:`t = T`. Defaults to ``6e4``.

    dtype : type, optional
        Data type, by default ``np.float64``.
    
    exclude_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
        Tuple of edges that should be excluded from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.

    include_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
        Tuple of edges that should be included from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.

    verbose : bool, optional
        Print information, by default ``False``.

    options : dict, optional
        Additional general/optimizer options, by default {}.

    Returns
    -------
    np.ndarray
        Estimated adjacency matrix with shape :math:`(p,p)`.
    
    
    """
    vprint = print if verbose else lambda *a, **k: None

    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'linear', compute_lib = 'numpy', method= 'dagma', 
                                                                                        optimizer = optimizer, reg = reg)
    method_options = allowed_method_options['linear']['dagma'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)
    optimizer_options = {**optimizer_options_config, **optimizer_options_settings}
    _, _, _, w_threshold, initialization,_ = get_variables_from_general_options(general_options)

    compute_lib, device = 'numpy', 'cpu'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device ,method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None
    
    X = demean(X) if loss_fn == 'l2' else X
    
    _loss = set_functions(loss_fn,loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)


    def _func(W: np.ndarray, mu: float, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        r"""
        Evaluate value of the penalized objective function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        mu : float
            Weight of the score function.
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            Objective value, and gradient of the objective
        """
        score, grad_score = _loss(W = W, X = X,**general_options)
        general_options['s'] = s
        h, grad_h = _h(W, **general_options)
        reg, grad_reg = _reg(W,**general_options)

        obj = mu * (score + reg) + h 
        grad = mu * (grad_score + grad_reg) + grad_h
        np.fill_diagonal(grad, 0)
        return obj, grad


    n, d = X.shape

    exc_r, exc_c = None, None
    inc_r, inc_c = None, None

    if exclude_edges is not None:
        if type(exclude_edges) is tuple and type(exclude_edges[0]) is tuple and np.all(np.array([len(e) for e in exclude_edges]) == 2):
            exc_r, exc_c = zip(*exclude_edges)
        else:
            ValueError("blacklist should be a tuple of edges, e.g., ((1,2), (2,3))")
    if include_edges is not None:
        if type(include_edges) is tuple and type(include_edges[0]) is tuple and np.all(np.array([len(e) for e in include_edges]) == 2):
            inc_r, inc_c = zip(*include_edges)
        else:
            ValueError("whitelist should be a tuple of edges, e.g., ((1,2), (2,3))")

    # we need optimizer know include and exclude edges
    optimizer_options['exc_r'] = exc_r
    optimizer_options['exc_c'] = exc_c
    optimizer_options['inc_r'] = inc_r
    optimizer_options['inc_c'] = inc_c

    if initialization is None:
        W_est = np.zeros((d,d)).astype(dtype) # init W0 at zero matrix
    else:
        W_est = np.array(initialization).astype(dtype)

    mu = mu_init
    if type(s) == list:
        if len(s) < T: 
            vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
            s = s + (T - len(s)) * [s[-1]]
    elif type(s) in [int, float]:
        s = T * [s]
    else:
        ValueError("s should be a list, int, or float.")

    optimizer_options['name'] = optimizer # we need to know which optimizer is used
    optimizer_options['verbose'] = verbose
    lr = optimizer_options['lr']
    time_start = time.time()

    for i in range(int(T)):
        vprint(f'\nIteration -- {i+1}:')
        success = False
        inner_iters = int(main_iter) if i == T - 1 else int(warm_iter)
        # update optimizer options
        optimizer_options['lr'] = lr # reset learning rate
        optimizer_options['num_steps'] = inner_iters
        optimizer_options['mu'] = mu
        while success is False:
            optimizer_options['s'] = s[i]
            sol = optimizer_dagma(_func, W_est, args = (mu,s[i]), options = optimizer_options)
            W_temp, success= sol.x, sol.success
            if success is False:
                vprint(f'Retrying with larger s')
                optimizer_options['lr'] *= 0.5
                s[i] += 0.1
        W_est = W_temp
        mu *= mu_factor
    time_end = time.time()
    W_est[np.abs(W_est) < w_threshold] = 0
    vprint(f"Total Time: {time_end - time_start}")
    return W_est


def dagma_linear_torch(X:np.ndarray, 
                 loss_fn = 'l2',
                 h_fn = 'h_logdet_sq',
                 reg = 'l1', 
                 optimizer = 'adam',
                 T: int = 5, 
                 mu_init: float = 1.0, 
                 mu_factor: float = 0.1, 
                 s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6],
                 warm_iter: int = 3e4, 
                 main_iter: int = 6e4, 
                 dtype: type = torch.float64,
                 exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None, 
                 include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
                 verbose: bool = False,
                   **options):
    """Learn DAG structure using DAGMA algorithm with **linear** structural equation with **torch**.

    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    loss_fn : str, optional
        Loss function, by default ``'l2'``.
    h_fn : str, optional
        H function, by default ``'h_logdet_sq'``.
    reg : str, optional
        Regularization function, by default ``'l1'``.
    optimizer : str, optional
        Optimizer, by default ``'adam'``.
    T : int, optional
        Number of iterations, by default ``5``.
    mu_init : float, optional
        Initial value of :math:`\mu`, by default ``1.0``.
    mu_factor : float, optional
        Factor to increase :math:`\mu`, by default ``0.1``.
    s : typing.Union[typing.List[float], float], optional
        Controls the domain of M-matrices. Defaults to ``[1.0, .9, .8, .7, .6]``.
    warm_iter : int, optional
        Number of iterations for :math:`t < T`. Defaults to ``3e4``.
    main_iter : int, optional
        Number of iterations for :math:`t = T`. Defaults to ``6e4``.
    dtype : type, optional
        Data type, by default ``np.float64``.
    exclude_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
        Tuple of edges that should be excluded from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
    include_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
        Tuple of edges that should be included from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
    verbose : bool, optional
        Print information, by default``False``.
    options : dict, optional
        Additional general/optimizer options, by default {}.
    Returns
    -------
    np.ndarray
        Estimated adjacency matrix with shape :math:`(p,p)`.

    """

    vprint = print if verbose else lambda *a, **k: None
    torch.set_default_dtype(dtype)
    device = options.get('device', 'cpu')
    device = check_device(device) 
    vprint(f'Using device: {device}')
    X = demean(X) if loss_fn == 'l2' else X
    X = process_data(X = X,dtype =dtype, device = device)

    # general options
    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'linear', compute_lib = 'torch', method= 'dagma', 
                                                                                        optimizer = optimizer, reg = reg)
    method_options = allowed_method_options['linear']['dagma'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)

    _ , _, _, w_threshold, initialization, _ = get_variables_from_general_options(general_options)
    
    lr = optimizer_options_config.get('lr', 0.0003)

    compute_lib = 'torch'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None

    _loss = set_functions(loss_fn,loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    _opt = set_functions(optimizer, optimizer_functions)

    def _func(W, mu: float, s: float = 1.0):
        r"""
        Evaluate value of the penalized objective function.

        Parameters
        ----------
        W : np.ndarray
            :math:`(d,d)` adjacency matrix
        mu : float
            Weight of the score function.
        s : float, optional
            Controls the domain of M-matrices. Defaults to 1.0.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            Objective value, and gradient of the objective
        """
        score = _loss(W = W, X = X,**general_options)
        general_options['s'] = s
        h = _h(W, **general_options)
        reg = _reg(W,**general_options)
        obj = mu * (score + reg) + h 
        return obj


    n,d = X.shape
    Id = torch.eye(d,dtype = dtype, device=device)

    
    exc_r, exc_c = None, None
    inc_r, inc_c = None, None

    if exclude_edges is not None:
        if type(exclude_edges) is tuple and type(exclude_edges[0]) is tuple and np.all(np.array([len(e) for e in exclude_edges]) == 2):
            exc_r, exc_c = zip(*exclude_edges)
        else:
            ValueError("blacklist should be a tuple of edges, e.g., ((1,2), (2,3))")
    if include_edges is not None:
        if type(include_edges) is tuple and type(include_edges[0]) is tuple and np.all(np.array([len(e) for e in include_edges]) == 2):
            inc_r, inc_c = zip(*include_edges)
        else:
            ValueError("whitelist should be a tuple of edges, e.g., ((1,2), (2,3))")



    
    if type(s) == list:
        if len(s) < T: 
            vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
            s = s + (T - len(s)) * [s[-1]]
    elif type(s) in [int, float]:
        s = T * [s]
    else:
        ValueError("s should be a list, int, or float.")
    if initialization is None:
        W_est = torch.zeros(d, d, dtype = dtype, requires_grad = True, device = device)
    else:
        W_est = torch.tensor(initialization, dtype = dtype, requires_grad = True, device = device)
    mu = mu_init
    
    def minimize(W_est,optimizer_options_config,optimizer_options_settings, mu, s):
        num_steps, check_iterate, tol, lr_decay = get_variables_from_optimizer_settings(optimizer_options_settings)
        lr = optimizer_options_config.get('lr', 0.0003)  
        
        vprint(f"\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} for {num_steps} max iterations")
        
        opt = _opt([W_est], **optimizer_options_config)

        def closure():
            opt.zero_grad()
            loss = _func(W_est,mu,s)
            loss.backward()
            return loss
            
        obj_prev = 1e16
        
        for i in range(1,num_steps+1):
            W_est_backup = W_est.clone()
            opt_state_dict = opt.state_dict()
            with torch.no_grad():
                # M = torch.linalg.inv(s * Id - W_est @ W_est) + 1e-6
                general_options['s'] = s
                h = _h(W_est, **general_options)
            # while torch.any(M <0):
            while h.item() < 0:
                if i == 1 or s <= 0.9:
                    vprint(f'W went out of domain for s={s} at iteration {i}')
                    return W_est, False
                else:
                    with torch.no_grad():
                        W_est.copy_(W_est_backup).requires_grad_(True).to(device)
                    lr *= 0.5
                    optimizer_options_config['lr'] = lr
                    if lr < 1e-16:
                        return W_est, True
                    opt = _opt([W_est], **optimizer_options_config)
                    opt.load_state_dict(opt_state_dict)
                    opt.param_groups[0]['lr'] = lr
                    opt.step(closure)
                    with torch.no_grad():
                        general_options['s'] = s
                        h = _h(W_est, **general_options)
                    vprint(f'Learning rate decreased to lr: {lr}')
            loss = opt.step(closure)
            obj_new = loss.item()

            if (i % check_iterate == 0 or i == num_steps):
                diff = abs(obj_prev - obj_new)
                denom = max(abs(obj_prev),abs(obj_new),1)
                if diff / denom <= tol:
                    break
                obj_prev = obj_new
        return W_est, True
    time_start = time.time()
    for i in range(int(T)):
        vprint(f'\nIteration -- {i+1}:')
        success = False
        inner_iters = int(main_iter) if i == T - 1 else int(warm_iter)
        ## initialize optimizer 
        optimizer_options_config['lr'] = lr # reset learning rate
        optimizer_options_settings['num_steps'] = inner_iters
        W_est_backup = W_est.clone()
        while success is False:
            W_temp, success = minimize(W_est,
                               optimizer_options_config = optimizer_options_config, 
                               optimizer_options_settings= optimizer_options_settings,
                                mu = mu ,s = s[i])
            if success is False:
                vprint(f'Retrying with larger s')
                optimizer_options_config['lr'] *= 0.5
                s[i]+=0.1
                with torch.no_grad():
                    W_est.copy_(W_est_backup).requires_grad_(True)
        W_est = W_temp
        mu *= mu_factor
    W_est_np = W_est.detach().cpu().numpy()
    W_est_np[np.abs(W_est_np) < w_threshold] = 0
    time_end = time.time()
    vprint(f"Total Time: {time_end - time_start}")
    return W_est_np

def dagma_nonlinear(X,
                loss_fn = 'log_l2',
                 h_fn = 'h_logdet_sq',
                 reg = 'l1', 
                 optimizer = 'adam',
                 activation = 'sigmoid',
                 dims = None,
                 T: int = 4, 
                 mu_init: float = 0.1, 
                 mu_factor: float = 0.1, 
                 s: float = 1.0,
                 warm_iter: int = 5e4, 
                 main_iter: int = 8e4, 
                 dtype: torch.dtype = torch.double,
                 verbose: bool = False,
                 bias = True,
                **options):
    """Learn DAG structure using DAGMA algorithm with **nonlinear** structural equation with **torch**.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    loss_fn : str, optional
        Loss function, by default ``'log_l2'``.
    h_fn : str, optional
        H function, by default ``'h_logdet_sq'``.
    reg : str, optional
        Regularization function, by default ``'l1'``.
    optimizer : str, optional
        Optimizer, by default ``'adam'``
    activation : str, optional
        Activation function, by default ``'sigmoid'``.
    dims : list, optional
        Number of neurons in hidden layers of each MLP representing each structural equation. Defaults to :math:`[d, 40, 1]`.
    T : int, optional
        Number of iterations, by default ``4``.
    mu_init : float, optional
        Initial value of :math:`\mu`, by default ``0.1``.
    mu_factor : float, optional
        Factor to increase :math:`\mu`, by default ``0.1``.
    s : float, optional
        Controls the domain of M-matrices. Defaults to ``1.0``.
    warm_iter : int, optional
        Number of iterations for :math:`t < T`. Defaults to ``5e4``.
    main_iter : int, optional
        Number of iterations for :math:`t = T`. Defaults to ``8e4``.
    dtype : torch.dtype, optional
        Data type, by default ``torch.double``.
    verbose : bool, optional
        Print information, by default ``False``.
    bias : bool, optional
        Include bias in the model, by default ``True``.
    options : dict, optional
        Additional general/optimizer options, by default {}.

    Returns
    -------
    np.ndarray
        Estimated adjacency matrix with shape :math:`(p,p)`.

    """
    torch.set_default_dtype(dtype)
    vprint = print if verbose else lambda *a, **k: None
    device = options.get('device', 'cpu')
    device = check_device(device)
    vprint(f'Using device: {device}')
    X = process_data(X = X, dtype = dtype, device = device)
    n,d = X.shape

    if dims is None:
        dims = [d,40,1]

    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'nonlinear', compute_lib = 'torch', method= 'dagma', 
                                                                                        optimizer = optimizer, reg = reg)
    method_options = allowed_method_options['nonlinear']['dagma'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)
    compute_lib = 'torch'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None

    lambda1, lambda2, gamma, w_threshold, initialization, _ = get_variables_from_general_options(general_options)

    lr = optimizer_options_config.get('lr', 0.0002)

    if type(s) == list:
        if len(s) < T: 
            vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
            s = s + (T - len(s)) * [s[-1]]
    elif type(s) in [int, float]:
        s = T * [s]
    else:
        ValueError("s should be a list, int, or float.")

    _loss = set_functions(loss_fn,nl_loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    _opt = set_functions(optimizer, optimizer_functions)
    
    if initialization is None:
        model = DagmaMLP(dims = dims, activation = activation, bias= bias, dtype=dtype).to(device)
    else:
        model = DagmaMLP(dims = dims, activation = activation, bias= bias, dtype=dtype).to(device)
        try:
            model.load_state_dict(initialization)
        except:
            ValueError("Initialization must be a valid state_dict for the model and must be consistent with dims and activation.")
    mu = mu_init
    time_start = time.time()
    
    def minimize(model,
                optimizer_options_config, 
                optimizer_options_settings, 
                mu, 
                s
        ) -> bool:
        num_steps, check_iterate, tol, lr_decay = get_variables_from_optimizer_settings(optimizer_options_settings)
        vprint(f'\nMinimize s={s} -- lr={lr}')
        # optimizer_options_config['weight_decay'] = lambda2*mu
        # opt = optim.Adam(model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        opt = _opt(model.parameters(),**optimizer_options_config)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        obj_prev = 1e16
        for i in range(1+num_steps):
            opt.zero_grad()
            general_options['s'] = s
            h_val = _h(model.adj(), **general_options) 
            if h_val.item() < 0:
                vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            X_hat = model(X)
            score = _loss(X_hat,X,**general_options)
            reg = lambda1 * model.fc1_l1_reg()
            # reg =  _reg(model.adj(),**general_options)
            l2_reg = lambda2 * model.l2_reg()
            obj = mu * (score + reg + l2_reg) + h_val
            obj.backward()
            opt.step()
            if lr_decay and i % check_iterate == 0:
                scheduler.step()
            if i % check_iterate == 0 or i == num_steps:
                obj_new = obj.item()
                vprint(f"\nInner iteration {i}")
                vprint(f'\th(W(model)): {h_val.item()}')
                vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    break
                obj_prev = obj_new
        return True

    for i in range(int(T)):
        vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
        success, s_cur = False, s[i]
        inner_iters = int(main_iter) if i == T - 1 else int(warm_iter)
        # optimizer_options_config['lr'] = lr # reset learning rate
        optimizer_options_settings['num_steps'] = inner_iters
        optimizer_options_settings['lr_decay'] = False
        optimizer_options_config['lr'] = lr 
        model_copy = copy.deepcopy(model)

        while success is False:
            success = minimize(model, 
                               optimizer_options_config = optimizer_options_config, 
                               optimizer_options_settings= optimizer_options_settings, 
                               mu = mu, 
                               s = s_cur)
            if success is False:
                model.load_state_dict(model_copy.state_dict().copy()).to(device)
                optimizer_options_config['lr']  *= 0.5 
                optimizer_options_settings['lr_decay'] = True
                if optimizer_options_config['lr'] < 1e-10:
                    break # lr is too small
                s_cur = 1
        mu *= mu_factor
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    time_end = time.time()
    vprint(f"Total Time: {time_end - time_start}")
    return W_est 