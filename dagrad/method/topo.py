import numpy as np
import scipy.optimize as sopt
import torch
import torch.optim as optim
import time
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from ..utils.topo_utils import threshold_W, create_Z, create_new_topo, create_new_topo_greedy,find_idx_set_updated, set_sizes_linear,set_sizes_nonlinear
from ..utils.configure import h_functions, loss_functions,reg_functions, nl_loss_functions,optimizer_functions,allowed_method_options
from ..utils.NNstructure import TopoMLP
from ..utils.general_utils import set_functions, set_options, get_variables_from_general_options, get_variables_from_optimizer_settings,demean,process_data,print_options
from ..optimizer.optimizer import adam
from ..utils.utils import is_dag, threshold_till_dag, find_topo

__all__ = ['topo_linear', 'topo_nonlinear'] 
def topo_linear(X,
                topo = None,
                loss_fn = 'l2',
                h_fn = 'h_logdet_topo', 
                reg = 'none',  
                optimizer = 'sklearn',
                no_large_search = -1, 
                size_small = -1, 
                size_large = -1, 
                dtype = np.float64,
                verbose = False,
                **options):
    """ LearnLearn DAG structure using TOPO algorithm with **linear** structural equation and **numpy**

    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    topo : list, optional
        Initial topological order. e.g. ``[0,1,...,p-1]`` If not provided, a random order is used.
    loss_fn : str, optional
        Loss function. Default is ``'l2'``.
    h_fn : str, optional
        H function. Default is ``'h_logdet_topo'``.
    reg : str, optional
        Regularizer. Default is ``'none'``.
    optimizer : str, optional
        Optimizer. Default is ``'sklearn'``.
    no_large_search : int, optional
        Number of times to search in large space. Default is ``-1``, it will be automatically set up.
    size_small : int, optional
        Size of small search space. Default is ``-1``, it will be automatically set up.
    size_large : int, optional
        Size of large search space. Default is ``-1``, it will be automatically set up.
    dtype : np.dtype, optional
        Data type. Default is ``np.float64``.
    verbose : bool, optional
        Verbose mode. Default is ``False``.

    Returns
    -------
    np.ndarray
        Estimated adjacency with shape :math:`(p,p)`.

    """
    n, d = X.shape
    X = demean(X) if loss_fn == 'l2' else X
    vprint = print if verbose else lambda *a, **k: None
    # set size_small and size_large and no_large_search
    size_small, size_large, no_large_search = set_sizes_linear(d, size_small, size_large, no_large_search)
    print(f"Parameter is automatically set up.\n size_small: {size_small}, size_large: {size_large}, no_large_search: {no_large_search}")


    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'linear', compute_lib = 'numpy', method= 'topo', 
                                                                                        optimizer = optimizer, reg = reg)
    
    method_options = allowed_method_options['linear']['topo'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)

    optimizer_options = {**optimizer_options_config, **optimizer_options_settings}
    lambda1, lambda2, gamma, w_threshold, initialization, _ = get_variables_from_general_options(general_options)
    compute_lib, device = 'numpy', 'cpu'
    print_options(loss_fn, h_fn, reg, optimizer,compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None


    if topo is None:
        if initialization is not None:
            if not is_dag(initialization):
                initialization = threshold_till_dag(initialization)
            topo = find_topo(initialization)
        else:
            topo = list(np.random.permutation(range(d)))

    _loss = set_functions(loss_fn,loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    
    def _score(X,W):
        val1, grad1 = _loss(X = X,W = W, **general_options)
        val2, grad2 = _reg(W,**general_options)
        return  val1 + val2, grad1 + grad2 

    # set optimizer for regression
    # basically, this part is set _regress function
    if optimizer == 'sklearn':
        if loss_fn == 'l2':
            if  reg == 'none':
                def _regress(X,y):
                    regression = LinearRegression(fit_intercept=False)
                    regression.fit(X = X, y = y)
                    return regression.coef_
            elif reg == 'l1':
                def _regress(X,y):
                    regression = Lasso(fit_intercept=False, alpha = lambda1)
                    regression.fit(X = X, y = y)
                    return regression.coef_
            elif reg == 'l2':
                def _regress(X,y):
                    regression = Ridge(fit_intercept=False, alpha = lambda2)
                    regression.fit(X = X, y = y)
                    return regression.coef_
            elif reg == 'mcp':
                raise NotImplementedError(f'Sklearn does not support MCP regularizer. Please use other optimizers.')
            else:
                raise ValueError(f'Unknown regularizer: {reg}')
        elif loss_fn == 'logistic':
            if reg == 'none':
                def _regress(X,y):
                    regression = LogisticRegression(fit_intercept=False)
                    regression.fit(X = X, y = y)
                    return regression.coef_
            elif reg == 'l1':
                def _regress(X,y):
                    regression = LogisticRegression(fit_intercept=False, penalty = 'l1', C = 1/lambda1)
                    regression.fit(X = X, y = y)
                    return regression.coef_
            elif reg == 'l2':
                def _regress(X,y):
                    regression = LogisticRegression(fit_intercept=False, penalty = 'l2', C = 1/lambda2)
                    regression.fit(X = X, y = y)
                    return regression.coef_
            elif reg == 'mcp':
                raise NotImplementedError(f'Sklearn does not support MCP regularizer. Please use other optimizers.')
            else:
                raise ValueError(f'Unknown regularizer: {reg}')
        else:
            raise ValueError(f'Unknown combination of loss_fn and reg: {loss_fn}, {reg}. This combination is not supported. You can use other optimizers.')
    else:
        def _regress(X,y):
            y = y.reshape(-1,1)
            d1 = X.shape[1]
            X_0 = np.zeros((n, d - d1 -1))
            X_bar = np.concatenate((y, X, X_0),axis = 1)

            def _func(w):
                W = np.zeros((d,d))
                W[1:(d1+1),0] = w
                loss, G_loss = _loss(W = W,X = X_bar,**general_options)
                reg, G_reg = _reg(W, **general_options)
                obj = loss + reg
                g_obj = G_loss + G_reg
                g_obj = g_obj[1:(d1+1),0]
                return obj, g_obj
            w_est = np.zeros(d1)
            if optimizer == 'lbfgs':
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, options = optimizer_options)
            elif optimizer == 'adam':
                sol = sopt.minimize(_func, w_est, method=adam, jac=True, bounds = [], options = optimizer_options)
            return sol.x

    
    def _init_W_slice(idx_y, idx_x):
        y = X[:, idx_y]
        x = X[:, idx_x]
        w = _regress(X=x, y=y)
        return w
    
    def _init_W(Z):
        d = Z.shape[0]
        W = np.zeros((d, d))
        for j in range(d):
            if (~Z[:, j]).any():
                W[~Z[:, j], j] = _regress(X=X[:, ~Z[:, j]], y=X[:, j])
            else:
                W[:, j] = 0
        return W
    
    def _update_topo_linear(W, topo, idx, opt=1):

        topo0 = topo.copy()
        W0 = np.zeros_like(W)
        i, j = idx
        i_pos, j_pos = topo.index(i), topo.index(j)

        W0[:, topo[:j_pos]] = W[:, topo[:j_pos]]
        W0[:, topo[(i_pos + 1):]] = W[:, topo[(i_pos + 1):]]
        topo0 = create_new_topo(topo=topo0, idx=idx, opt=opt)
        for k in range(j_pos, i_pos + 1):
            if len(topo0[:k]) != 0:
                W0[topo0[:k], topo0[k]] = _init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])
            else:
                W0[:, topo0[k]] = 0
        return W0, topo0
    
    def _fit(topo: list):
        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        Z = create_Z(topo)
        W = _init_W(Z)
        loss, G_loss = _score(X=X, W=W)
        h, G_h = _h(W=W,**general_options)
        vprint(f"Initial loss: {loss}")
        idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=Z, size_small=size_small,
                                                    size_large=size_large)
        idx_set = list(idx_set_small)
        while bool(idx_set):
            idx_len = len(idx_set)
            loss_collections = np.zeros(idx_len)
            for i in range(idx_len):
                W_c, topo_c = _update_topo_linear(W = W,topo = topo,
                                                  idx = idx_set[i])
                loss_c,_ = _score(X = X, W = W_c)
                loss_collections[i] = loss_c
            if np.any(loss > np.min(loss_collections)):
                vprint(f"current loss : {loss} and find better loss in small space")
                topo = create_new_topo_greedy(topo,loss_collections,idx_set,loss)
            else:
                if large_space_used < no_large_search:
                    vprint(f"current loss : {loss} and cannot find better loss in small space")
                    vprint(f"Using larger search space for {large_space_used+1} times")
                    idx_set = list(set(idx_set_large) - set(idx_set_small))
                    idx_len = len(idx_set)
                    loss_collections = np.zeros(idx_len)
                    for i in range(idx_len):
                        W_c, topo_c = _update_topo_linear(W=W, topo=topo, idx=idx_set[i])
                        loss_c, _ = _score(X=X, W=W_c)
                        loss_collections[i] = loss_c

                    if np.any(loss > loss_collections):
                        large_space_used += 1
                        topo = create_new_topo_greedy(topo, loss_collections, idx_set, loss)
                        vprint(f"current loss : {loss} and find better loss in large space")
                    else:
                        vprint("Using larger search space, but we cannot find better loss")
                        break


                else:
                    vprint("We reach the number of chances to search large space, it is {}".format(
                        no_large_search))
                    break
            Z = create_Z(topo)
            W = _init_W(Z)
            loss, G_loss = _score(X=X, W=W)
            h, G_h = _h(W=W, **general_options)
            idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=Z, size_small=size_small,
                                                        size_large=size_large)
            idx_set = list(idx_set_small)

            iter_count += 1

        return W, topo, Z, loss
    time_start = time.time()
    W, topo, Z, loss = _fit(topo = topo)
    W = threshold_W(W, w_threshold)
    time_end =time.time()
    print(f"Total Time: {time_end - time_start}")
    return W


def topo_nonlinear(X,
                   topo = None,
                   loss_fn = 'l2',
                   h_fn = 'h_logdet_topo',
                   reg = 'none',
                   optimizer = 'lbfgs',
                    dims = None,
                    bias = True,
                    activation = 'sigmoid',
                   no_large_search = -1,
                   size_small = -1,
                   size_large = -1,
                   verbose = False,
                   dtype = torch.float,
                   **options):
    """
    Learn DAG structure using TOPO algorithm with **nonlinear** structural equation and **torch**
    
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with shape :math:`(n,p)`.
    topo : list, optional
        Initial topological order. e.g. ``[0,1,...,p-1]`` If not provided, a random order is used.
    loss_fn : str, optional
        Loss function. Default is ``'l2'``.
    h_fn : str, optional
        H function. Default is ``'h_logdet_topo'``.
    reg : str, optional
        Regularizer. Default is ``'none'``.
    optimizer : str, optional
        Optimizer. Default is ``'lbfgs'``.
    dims : list, optional
        Dimension of neural network. Default is ``None``.
    bias : bool, optional
        Bias in neural network. Default is ``True``.
    activation : str, optional
        Activation function in neural network. Default is ``'sigmoid'``.
    no_large_search : int, optional
        Number of times to search in large space. Default is ``-1``, it will be automatically set up.
    size_small : int, optional
        Size of small search space. Default is ``-1``, it will be automatically set up.
    size_large : int, optional
        Size of large search space. Default is ``-1``, it will be automatically set up.
    verbose : bool, optional
        Verbose mode. Default is ``False``.
    dtype : torch.dtype, optional
        Data type. Default is ``torch.float``.

    Returns
    -------
    np.ndarray
        Estimated adjacency with shape :math:`(p,p)`.

    """

    torch.set_default_dtype(dtype)
    vprint = print if verbose else lambda *a, **k: None
    X = process_data(X, dtype, device = 'cpu')
    n,d = X.shape

    if topo is None:
        topo = list(np.random.permutation(range(d)))
    if dims is None:
        dims = [d,40,1]

    # set size_small and size_large and no_large_search
    size_small, size_large, no_large_search = set_sizes_nonlinear(d, size_small, size_large, no_large_search)
    vprint(f"Parameter is automatically set up.\n size_small: {size_small}, size_large: {size_large}, no_large_search: {no_large_search}")


    general_options, optimizer_options_config, optimizer_options_settings = set_options(options = options, 
                                                                                        model = 'nonlinear', compute_lib = 'torch', method= 'topo', 
                                                                                        optimizer = optimizer, reg = reg)
    method_options = allowed_method_options['nonlinear']['topo'].copy()
    for key, value in method_options.items():
        method_options[key] = eval(key)
    lambda1, lambda2, gamma, w_threshold, _, _ = get_variables_from_general_options(general_options)
    compute_lib, device = 'torch', 'cpu'
    print_options(loss_fn, h_fn, reg, optimizer, compute_lib, device, method_options,
        general_options,optimizer_options_config,optimizer_options_settings) if verbose else None


    _loss = set_functions(loss_fn,nl_loss_functions)
    _h = set_functions(h_fn, h_functions)
    _reg = set_functions(reg,reg_functions)
    _opt = set_functions(optimizer, optimizer_functions)

    model = TopoMLP(dims = dims, bias = bias, activation = activation, dtype = dtype)
    
    def _train(model):
        num_steps, check_iterate, tol, lr_decay = get_variables_from_optimizer_settings(optimizer_options_settings)
        opt = _opt(model.parameters(),**optimizer_options_config)
        
        if lr_decay:
            scheduler = optim.lr_scheduler.StepLR(opt, gamma=0.9)

        def closure():
            opt.zero_grad()
            X_hat = model(X)
            loss = _loss(X_hat, X, **general_options)
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            reg = lambda1 * model.layer0_l1_reg()
            # reg =  _reg(model.adj(),**general_options)
            total_loss = loss + l2_reg + reg
            total_loss.backward()
            return total_loss
        
        prev_loss = float('inf')
        for i in range(1,num_steps+1):
            loss = opt.step(closure)
            current_loss = loss.item()
            if i % check_iterate == 0 and lr_decay:
                scheduler.step()
            if abs((prev_loss - current_loss)/max(prev_loss,current_loss,1)) < tol:
                break
            prev_loss = current_loss

        return current_loss

    def _copy_model(model, model_clone):
        model_clone.load_state_dict(model.state_dict())
        for param_target, (name_source, param_source) in zip(model_clone.parameters(), model.named_parameters()):
            
            param_target.requires_grad = param_source.requires_grad

        return model_clone


    def _fit(topo,model):
        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        
        Z = create_Z(topo)
        model.reset_by_topo(topo = topo)
        loss = _train(model = model)
        vprint(f"The initial model, current loss {round(loss,5)}")
        W_adj = model.layer0_to_adj()
        h, G_h = _h(W=W_adj, **general_options)
        G_loss = np.ones((d,d))
        idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=Z, size_small=size_small,
                                        size_large=size_large)
        idx_set = list(idx_set_small)

        while bool(idx_set):
            idx_len = len(idx_set)
            indicator_improve = False
           
            model_clone = type(model)(dims = model.dims, bias = model.bias,
                                      activation = activation,
                                      dtype = model.dtype)
            
            for i in range(idx_len):
                model_clone = _copy_model(model = model, model_clone = model_clone)
                topo_clone = create_new_topo(topo, idx_set[i], opt=1)
                model_clone.update_nn_by_topo(topo = topo, index = idx_set[i])
            
                loss_clone = _train(model = model_clone)

                vprint(f"working with topological sort:{topo_clone}, current loss {round(loss_clone,5)}")

                model_clone.reset_by_topo(topo = topo_clone)

                if loss_clone< loss:
                    indicator_improve = True
                    # model_clone is successful, and we get copy of it
                    vprint(f"better loss found, topological sort: {topo_clone}, and loss: {round(loss_clone,5)}")
                    model = _copy_model(model = model_clone, model_clone = model)
                    topo = topo_clone
                    Z = create_Z(topo_clone)
                    loss = loss_clone
                    W_adj = model.layer0_to_adj()
                    h, G_h = _h(W=W_adj, **general_options)
                    break

            if not indicator_improve:
                if large_space_used < no_large_search:
                    indicator_improve_large = False
                    # print('++++++++++++++++++++++++++++++++++++++++++++')
                    vprint(f"start to use large search space for {large_space_used + 1} times")
                    # print('++++++++++++++++++++++++++++++++++++++++++++')
                    idx_set = list(set(idx_set_large) - set(idx_set_small))
                    idx_len = len(idx_set)
                    
                    for i in range(idx_len):
                        model_clone = _copy_model(model=model, model_clone=model_clone)
                        topo_clone = create_new_topo(topo, idx_set[i], opt=1)
                        model_clone.update_nn_by_topo(topo=topo, index=idx_set[i])
                        
                        loss_clone = _train(model=model_clone)
                        vprint(f"working with topological sort:{topo_clone}, current loss {loss_clone}")
                        model_clone.reset_by_topo(topo=topo_clone)


                        if loss_clone<loss:
                            indicator_improve_large = True
                            model = _copy_model(model=model_clone, model_clone=model)
                            topo = topo_clone
                            Z = create_Z(topo_clone)
                            loss = loss_clone
                            W_adj = model.layer0_to_adj()
                            h, G_h = _h(W=W_adj,**general_options)
                            vprint(f"better loss found, topological sort: {topo_clone}, and loss: {loss_clone}")
                             
                            break
                    if not indicator_improve_large:
                        vprint("Using larger search space, but we cannot find better loss")
                        break
                    large_space_used =large_space_used+ 1 
                else:
                    vprint("We reach the number of chances to search large space, it is {}".format(
                        no_large_search))
                    break

             
            idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=Z,
                                                        size_small=size_small,
                                                        size_large=size_large)
            idx_set = list(idx_set_small)

            iter_count += 1
        
        return W_adj, topo, loss, model
    time_start = time.time()
    W, topo, loss, model = _fit(topo = topo, model = model)
    W = threshold_W(W, w_threshold)
    time_end = time.time()
    vprint(f"Total Time: {time_end - time_start}")
    return W


    