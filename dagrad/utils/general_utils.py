from ..utils.configure import allowed_general_options,allowed_method_options,allowed_optimizer_options
import torch
import numpy as np
from warnings import warn
def validate_options(options, allowed_options):
    """
    Validate user-provided options against allowed options.
    
    Parameters:
    - options: dict, user-provided options.
    - allowed_options: dict, keys are option names and values are the expected type.
    
    Raises ValueError if an option is not allowed or if a value has the wrong type.
    """
    for key, value in options.items():
        if key not in allowed_options:
            raise ValueError(f"Invalid option: {key}. Allowed options are: {list(allowed_options.keys())}.")
        
        if not isinstance(value, allowed_options[key]):
            expected_type = allowed_options[key].__name__
            raise ValueError(f"Option '{key}' should be of type {expected_type}, but got {type(value).__name__}.")


def merge_dicts_if_needed(dict0):
    # Check if the dictionary contains nested dictionaries
    if all(isinstance(value, dict) for value in dict0.values()):
        # Merge all nested dictionaries into one
        merged_dict = {}
        for subdict in dict0.values():
            merged_dict.update(subdict)
        return merged_dict
    else:
        # If not nested or contains mixed types, return the original dictionary
        return dict0


def set_functions(selection, function_map):
    return function_map[selection]

def set_tuning_method_from_options(options, reg):

    general_options = {k:options[k] for k in allowed_general_options if k in options}
    tuning_method = general_options.get('tuning_method', None)
    K = general_options.get('K', 5)
    initialization = options.get('initialization', None)
    if reg =='l1' or reg =='l2':
        reg_paras = options.get('reg_paras', [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    elif reg == 'mcp':
        reg_paras = options.get('reg_paras', [[0, 0.05, 0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5, 1]])
    return tuning_method, K, reg_paras, initialization
    


def set_options(options, model, compute_lib, method, optimizer, reg):
    
    general_options = {k:options[k] for k in allowed_general_options if k in options}
    optimizer_options_config = {k:options[k] for k in allowed_optimizer_options[compute_lib][optimizer]['opt_config'] if k in options}
    optimizer_options_settings = {k:options[k] for k in allowed_optimizer_options[compute_lib][optimizer]['opt_settings'] if k in options}

    w_threshold = options.get('w_threshold', 0.3)
    gamma = options.get('gamma', 1.0)
    initialization = options.get('initialization', None)

    tuning_method = options.get('tuning_method', None)
    K = options.get('K', 5)
    if reg =='l1' or reg =='l2':
        reg_paras = options.get('reg_paras', [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    elif reg == 'mcp':
        reg_paras = options.get('reg_paras', [[0, 0.05, 0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5, 1]])
    else:
        reg_paras = options.get('reg_paras', None)
    user_params = options.get('user_params', None)

    if model == 'linear':
        lambda1 = options.get('lambda1', 0.1)
        lambda2 = options.get('lambda2', 0)
    elif model == 'nonlinear':
        if method == 'dagma':
            lambda1 = options.get('lambda1', 0.02)
            lambda2 = options.get('lambda2', 0.005)
        elif method == 'topo':
            lambda1 = options.get('lambda1', 0.1)
            lambda2 = options.get('lambda2', 0.01)
        else:
            lambda1 = options.get('lambda1', 0.01)
            lambda2 = options.get('lambda2', 0.01)
        

    general_options['gamma'] = gamma
    general_options['w_threshold'] = w_threshold
    general_options['lambda1'] = lambda1
    general_options['lambda2'] = lambda2
    general_options['initialization'] = initialization
    general_options['tuning_method'] = tuning_method
    general_options['K'] = K
    general_options['reg_paras'] = reg_paras
    general_options['user_params'] = user_params

    

    if optimizer == 'adam' and (compute_lib in ['numpy','torch']):
        # set learning rate, betas, eps
        if method == 'dagma' and model == 'nonlinear':
            lr = options.get('lr', 0.0002)
        elif method == 'topo' and model == 'linear':
            lr = options.get('lr', 0.01)
        else:
            lr = options.get('lr', 0.0003)
        betas = options.get('betas', (0.99, 0.999))
        eps = options.get('eps', 1e-8)

        # set num_steps, tol, check_iterate
        if method == 'dagma' and model == 'nonlinear':
            num_steps = options.get('num_steps',10000)
            tol = options.get('tol', 1e-3)
            check_iterate = options.get('check_iterate', 500)
        else:
            num_steps = options.get('num_steps',15000)
            tol = options.get('tol', 1e-6)
            check_iterate = options.get('check_iterate', 1000)
        if compute_lib == 'torch':
            lr_decay = options.get('lr_decay', False)

        optimizer_options_config['lr'] = lr
        optimizer_options_config['betas'] = betas
        optimizer_options_config['eps'] = eps

        optimizer_options_settings['num_steps'] = num_steps
        optimizer_options_settings['check_iterate'] = check_iterate
        optimizer_options_settings['tol'] = tol
        if compute_lib == 'torch':
            optimizer_options_settings['lr_decay'] = lr_decay

    if optimizer == 'lbfgs' and compute_lib == 'numpy':
        pass
        # disp = options.get('disp', None)
        # maxls = options.get('maxls', 20)
        # maxcor = options.get('maxcor', 10)
        
        # factr = options.get('factr', 1e7)
        # pgtol = options.get('pgtol', 1e-5)
        # epsilon = options.get('epsilon', 1e-8)
        # iprint = options.get('iprint', -1)
        # maxfun = options.get('maxfun', 15000)
        # max_iter = options.get('max_iter', 15000)
        
        # optimizer_options_config['m'] = m
        # optimizer_options_config['factr'] = factr
        # optimizer_options_config['pgtol'] = pgtol
        # optimizer_options_config['epsilon'] = epsilon
        # optimizer_options_config['iprint'] = iprint
        # optimizer_options_config['maxfun'] = maxfun
        # optimizer_options_config['max_iter'] = max_iter
        # optimizer_options_config['disp'] = disp
        # optimizer_options_config['maxls'] = maxls


    if optimizer == 'lbfgs' and compute_lib == 'torch':
        
        num_steps = options.get('num_steps', 30)
        check_iterate = options.get('check_iterate', 5)
        if method == 'topo':
            tol = options.get('tol', 0.005)
        tol = options.get('tol', 1e-5)
        lr_decay = options.get('lr_decay', False)

        optimizer_options_settings['num_steps'] = num_steps
        optimizer_options_settings['check_iterate'] = check_iterate
        optimizer_options_settings['tol'] = tol
        optimizer_options_settings['lr_decay'] = lr_decay

        lr = options.get('lr',1)
        max_iter = options.get('max_iter', 20)
        max_eval = options.get('max_eval', None)
        tolerance_grad = options.get('tolerance_grad', 1e-7)
        tolerance_change = options.get('tolerance_change', 1e-9)
        history_size  = options.get('history_size', 100)
        line_search_fn = options.get('line_search_fn', 'strong_wolfe')

        optimizer_options_config['lr'] = lr
        optimizer_options_config['max_iter'] = max_iter
        optimizer_options_config['max_eval'] = max_eval
        optimizer_options_config['tolerance_grad'] = tolerance_grad
        optimizer_options_config['tolerance_change'] = tolerance_change
        optimizer_options_config['history_size'] = history_size
        optimizer_options_config['line_search_fn'] = line_search_fn

    return general_options, optimizer_options_config, optimizer_options_settings


def get_variables_from_general_options(general_options):
    lambda1 = general_options.get('lambda1', 0.1)
    lambda2 = general_options.get('lambda2', 0)
    gamma = general_options.get('gamma', 1.0)
    w_threshold = general_options.get('w_threshold', 0.3)
    initialization = general_options.get('initialization', None)
    user_params = general_options.get('user_params', None)
    return lambda1, lambda2, gamma, w_threshold, initialization, user_params

def get_variables_from_optimizer_settings(optimizer_options_settings):
    num_steps = optimizer_options_settings.get('num_steps', 20000)
    check_iterate = optimizer_options_settings.get('check_iterate', 1000)
    tol = optimizer_options_settings.get('tol', 1e-5)
    lr_decay = optimizer_options_settings.get('lr_decay', False)
    return num_steps, check_iterate, tol, lr_decay

def check_data(X):
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    if X.shape[0] < X.shape[1]:
        warn("Number of samples is less than number of variables.")
        warn("We are assuming first dimension is number of samples and second dimension is number of nodes.")
        warn("If this is not the case, please transpose the input matrix.")

def check_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA is not available. Using CPU instead.')
    return device

def process_data(X, dtype, device):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(dtype=dtype, device=device)
    elif torch.is_tensor(X):
        X = X.to(dtype=dtype, device=device)
    else:
        raise ValueError("X must be numpy array or tensor")
    return X

def demean(X):
    if isinstance(X, torch.Tensor):
        mean = torch.mean(X, dim=0, keepdim=True)
        result = X - mean
    elif isinstance(X, np.ndarray):
        mean = np.mean(X, axis=0, keepdims=True)
        result = X - mean
    return result

def get_default(model, method, loss_fn, h_fn, reg, optimizer, compute_lib, device):
    if model is None:
        model = 'linear'

    if model == 'linear':
        if method == 'notears':
            _loss_fn, _h_fn, _reg, _optimizer, _compute_lib, _device = 'l2', 'h_exp_sq', 'l1', 'lbfgs', 'numpy', 'cpu'
        elif method == 'dagma':
            _loss_fn, _h_fn, _reg, _optimizer, _compute_lib, _device = 'l2', 'h_logdet_sq', 'l1', 'adam', 'numpy', 'cpu'
        elif method == 'topo':
            _loss_fn, _h_fn, _reg, _optimizer, _compute_lib, _device = 'l2', 'h_logdet_topo', 'none', 'sklearn', 'numpy', 'cpu'
    elif model == 'nonlinear':
        if method == 'notears':
            _loss_fn, _h_fn, _reg, _optimizer, _compute_lib, _device = 'l2', 'h_exp_sq', 'l1', 'lbfgs', 'torch', 'cpu'
        elif method == 'dagma':
            _loss_fn, _h_fn, _reg, _optimizer, _compute_lib, _device = 'logll', 'h_logdet_sq', 'l1', 'adam', 'torch', 'cpu'
        elif method == 'topo':
            _loss_fn, _h_fn, _reg, _optimizer, _compute_lib, _device = 'l2', 'h_logdet_topo', 'l1', 'lbfgs', 'torch', 'cpu'
    
    set_opt = lambda string, default: string.lower() if string is not None else default
    
    loss_fn = set_opt(loss_fn, _loss_fn)
    h_fn = set_opt(h_fn, _h_fn)
    reg = set_opt(reg, _reg)
    optimizer = set_opt(optimizer, _optimizer)
    compute_lib = set_opt(compute_lib, _compute_lib)
    device = set_opt(device, _device)
    
    return model,loss_fn, h_fn, reg, optimizer, compute_lib, device

def print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings):
    print("-"*50)
    print("Loss Function:", loss_fn)
    print("H Function:", h_fn)
    print("Regularizer:", reg)
    print("Optimizer:", optimizer)
    print('Computation Library:', compute_lib)
    print('Device:', device)
    print("-"*50)
    print("Method Options:")
    for key, value in method_options.items():
        print(f"{key}: {value}")
    print("-"*50)
    print("General Options:")
    for key, value in general_options.items():
        print(f"{key}: {value}")
    print("-"*50)
    print("\nOptimizer Config:")
    for key, value in optimizer_options_config.items():
        print(f"{key}: {value}")
    print("-"*50)
    print("\nOptimizer Settings:")
    for key, value in optimizer_options_settings.items():
        print(f"{key}: {value}")
    print("-"*50)

def get_method_options(method,model):
    method_options = allowed_method_options[model][method].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)

    return method_options