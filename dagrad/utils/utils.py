import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import typing

def find_topo(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    topo = G.topological_sorting()
    return topo

def generate_sem_data(n,d,s0,graph_type = 'ER', sem_type = 'linear', noise_type = 'gauss', error_var = 'eq', seed = None):
    '''
    Generate data from a SEM model

    n: number of samples
    d: number of variables
    s0: expected number of edges
    graph_type: type of graph, one of ['ER', 'SF']
    sem_type: type of sem, one of ['linear', 'mlp', 'mim', 'gp', 'gp-add']
    noise_type: type of noise, one of ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']
    error_var: variance of the noise, one of ['eq','random']
    '''
    if seed is not None:
            set_random_seed(seed=seed)
    else:
        set_random_seed(seed = random.randint(0, 100000))
    
    B_true = simulate_dag(d, s0, graph_type)
    if sem_type == 'linear':
        W_true = simulate_parameter(B_true)
        if error_var == 'eq':
            X = simulate_linear_sem(W_true, n, noise_type)
        elif error_var == 'random':
            X = simulate_linear_sem(W_true, n, noise_type, noise_scale = np.random.uniform(0.5,1.0,d))
        else:
            raise ValueError('error_var must be one of eq or random')
    else:
        W_true = None
        if error_var == 'eq':
            X = simulate_nonlinear_sem(B_true, n, sem_type, noise_type)
        elif error_var == 'random':
            X = simulate_nonlinear_sem(B_true, n, sem_type, noise_type, noise_scale = np.random.uniform(0.5,1.0,d))
        else:
            raise ValueError('error_var must be one of eq or random')
    return X, W_true, B_true

def generate_linear_data(n,d,s0,graph_type = 'ER', noise_type = 'gauss', error_var = 'eq', seed = None):
    '''
    Wrapper for generate_sem_data
    Generate data from a linear SEM model

    n: number of samples
    d: number of variables
    s0: expected number of edges
    graph_type: type of graph, one of ['ER', 'SF']
    noise_type: type of noise, one of ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']
    error_var: variance of the noise, one of ['eq','random']
    seed: random seed

    Returns:

    X: data matrix
    W_true: true parameter matrix
    B_true: true adjacency matrix, 0 for no edge, 1 for edge
    '''
        
    X, W_true, B_true = generate_sem_data(n,d,s0,graph_type, 'linear', noise_type, error_var, seed)
    return X, W_true, B_true

def generate_nonlinear_data(n,d,s0,graph_type = 'ER', sem_type = 'mlp', noise_type = 'gauss', error_var = 'eq', seed = None):
    '''
    Wrapper for generate_sem_data
    Generate data from a nonlinear SEM model

    n: number of samples
    d: number of variables
    s0: expected number of edges
    graph_type: type of graph, one of ['ER', 'SF']
    sem_type: type of sem, one of ['mlp', 'mim', 'gp', 'gp-add']
    noise_type: type of noise, one of ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']
    error_var: variance of the noise, one of ['eq','random']
    seed: random seed

    Returns:

    X: data matrix
    W_true: None, as the true parameter matrix is not available for nonlinear SEM
    B_true: true adjacency matrix, 0 for no edge, 1 for edge
    '''
    X, W_true, B_true = generate_sem_data(n,d,s0,graph_type, sem_type, noise_type, error_var, seed)
    return X, W_true, B_true

def threshold_W(W, threshold=0.3):
    """
    :param W: adjacent matrix
    :param threshold:
    :return: a threshed adjacent matrix
    """
    W_new = np.zeros_like(W)
    W_new[:] = W
    W_new[np.abs(W_new) < threshold] = 0
    return W_new


# function adapted from "https://github.com/ignavierng/golem"
def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W: np.ndarray) -> bool:
    """
    Returns ``True`` if ``W`` is a DAG, ``False`` otherwise.
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d: int, s0: int, graph_type: str) -> np.ndarray:
    r"""
    Simulate random DAG with some expected number of edges.

    Parameters
    ----------
    d : int
        num of nodes
    s0 : int
        expected num of edges
    graph_type : str
        One of ``["ER", "SF", "BP"]``
    
    Returns
    -------
    numpy.ndarray
        :math:`(d, d)` binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == 'Fully':
        B = np.triu(np.ones((d,d)), 1)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B: np.ndarray, 
                       w_ranges: typing.List[typing.Tuple[float,float]]=((-2.0, -0.5), (0.5, 2.0)),
                       ) -> np.ndarray:
    r"""
    Simulate SEM parameters for a DAG.

    Parameters
    ----------
    B : np.ndarray
        :math:`[d, d]` binary adj matrix of DAG.
    w_ranges : typing.List[typing.Tuple[float,float]], optional
        disjoint weight ranges, by default :math:`((-2.0, -0.5), (0.5, 2.0))`.

    Returns
    -------
    np.ndarray
        :math:`[d, d]` weighted adj matrix of DAG.
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W: np.ndarray, 
                        n: int, 
                        sem_type: str, 
                        noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                        ) -> np.ndarray:
    r"""
    Simulate samples from linear SEM with specified type of noise.
    For ``uniform``, noise :math:`z \sim \mathrm{uniform}(-a, a)`, where :math:`a` is the ``noise_scale``.
    
    Parameters
    ----------
    W : np.ndarray
        :math:`[d, d]` weighted adj matrix of DAG.
    n : int
        num of samples. When ``n=inf`` mimics the population risk, only for Gaussian noise.
    sem_type : str
        ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
    noise_scale : typing.Optional[typing.Union[float,typing.List[float]]], optional
        scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.

    Returns
    -------
    np.ndarray
        :math:`[n, d]` sample matrix, :math:`[d, d]` if ``n=inf``.
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or have length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B:np.ndarray, 
                           n: int, 
                           sem_type: str = 'mlp', 
                           noise_type: str = 'gauss',
                           noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                           ) -> np.ndarray:
    r"""
    Simulate samples from nonlinear SEM.

    Parameters
    ----------
    B : np.ndarray
        :math:`[d, d]` binary adj matrix of DAG.
    n : int
        num of samples
    sem_type : str
        ``mlp``, ``mim``, ``gp``, ``gp-add``
    noise_type: str
        ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
    noise_scale : typing.Optional[typing.Union[float,typing.List[float]]], optional
        scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.

    Returns
    -------
    np.ndarray
        :math:`[n, d]` sample matrix.
    """
    def generate_noise(n, scale, noise_type):
        if noise_type == 'gauss':
            return np.random.normal(scale=scale, size=n)
        elif noise_type == 'exp':
            return np.random.exponential(scale=scale, size=n)
        elif noise_type == 'gumbel':
            return np.random.gumbel(scale=scale, size=n)
        elif noise_type == 'uniform':
            return np.random.uniform(low=-scale, high=scale, size=n)
        else:
            raise ValueError('Unknown noise type')

    def compute_x(X, z, sem_type):
        pa_size = X.shape[1]
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(0.5, 2.0, size=(pa_size, hidden))
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(0.5, 2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            return sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            weights = [np.random.uniform(0.5, 2.0, size=pa_size) for _ in range(3)]
            for w in weights:
                w[np.random.rand(pa_size) < 0.5] *= -1
            return (np.tanh(X @ weights[0]) + np.cos(X @ weights[1]) +
                    np.sin(X @ weights[2]) + z)
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            return gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            return sum(gp.sample_y(X[:, i:i+1], random_state=None).flatten()
                    for i in range(pa_size)) + z
        else:
            raise ValueError('Unknown SEM type')


    def _simulate_single_equation(X, scale):
        n, pa_size = X.shape
        if pa_size == 0:
            # No parents
            if noise_type == 'logistic':
                return np.random.binomial(1, 0.5, size=n).astype(float)
            elif noise_type == 'poisson':
                return np.random.poisson(scale, size=n).astype(float)
            else:
                return generate_noise(n, scale, noise_type)
        else:
            if noise_type in ['logistic', 'poisson']:
                z = 0
                x = compute_x(X, z, sem_type)
                if noise_type == 'logistic':
                    return np.random.binomial(1, sigmoid(x)).astype(float)
                else:  # noise_type == 'poisson'
                    return np.random.poisson(np.exp(x)).astype(float)
            else:
                z = generate_noise(n, scale, noise_type)
                return compute_x(X, z, sem_type)

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale is not None else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    r"""
    Compute various accuracy metrics for B_est.

    | true positive = predicted association exists in condition in correct direction
    | reverse = predicted association exists in condition in opposite direction
    | false positive = predicted association does not exist in condition
    
    Parameters
    ----------
    B_true : np.ndarray
        :math:`[d, d]` ground truth graph, :math:`\{0, 1\}`.
    B_est : np.ndarray
        :math:`[d, d]` estimate, :math:`\{0, 1, -1\}`, -1 is undirected edge in CPDAG.

    Returns
    -------
    dict
        | fdr: (reverse + false positive) / prediction positive
        | tpr: (true positive) / condition positive
        | fpr: (reverse + false positive) / condition negative
        | shd: undirected extra + undirected missing + reverse
        | nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}