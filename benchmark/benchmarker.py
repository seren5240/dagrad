from typing import Callable, Optional
import joblib
from numpy import ndarray
import numpy as np
from joblib import Parallel, delayed
from dagrad.utils import utils


def create_one_dataset(
    n: int,
    d: int,
    edges: int,
    sem_type: str,
    noise_type: str,
    error_var: str,
    linearity: str,
    graph_type: str,
):
    B_true = utils.simulate_dag(d, edges, graph_type)
    if error_var == "eq":
        noise_scale = None
    elif error_var == "random":
        noise_scale = np.random.uniform(0.5, 1.0, d)
    else:
        raise ValueError(f"Unknown error_var: {error_var}")

    if linearity == "linear":
        dataset = utils.simulate_linear_sem(
            B_true, n, sem_type=noise_type, noise_scale=noise_scale
        )
    elif linearity == "nonlinear":
        dataset = utils.simulate_nonlinear_sem(
            B_true, n, sem_type=sem_type, noise_type=noise_type, noise_scale=noise_scale
        )
    else:
        raise ValueError(f"Unknown linearity: {linearity}")
    return dataset, B_true


def run_one_trial(
    d: int,
    dataset: ndarray,
    B_true: ndarray,
    benchmark_fn: Callable[[ndarray], ndarray],
    metric: Optional[tuple[str, Callable[[ndarray, ndarray], int]]] = None,
):
    W_est = benchmark_fn(dataset)
    try:
        if metric is None:
            acc = utils.count_accuracy(B_true, W_est != 0)
            val = acc["shd"] / d
        else:
            val = metric[1](B_true, W_est)
    except ValueError as e:
        print(f"Error in counting accuracy: {e}")
        return None
    return val


def run_benchmarks(
    n: int,
    sizes: list[tuple[int, int]],
    noise_types: list[str],
    error_vars: list[str],
    linearities: list[str],
    graph_types: list[str],
    benchmark_fns: dict[str, Callable[[ndarray], ndarray]],
    trials: int,
    output_filename: str,
    sem_type: str = "mlp",
    metric: Optional[tuple[str, Callable[[ndarray, ndarray], int]]] = None,
):
    """
    Run benchmarks on multiple vertex/edge combinations and benchmark functions.

    Parameters
    ----------
    n: int
        Number of samples
    sizes: list[tuple[int, int]]
        List of node/edge combinations
    noise_type: list[str]
        list of ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
    error_var: str
        list of ``eq``, ``random``
    linearities: str
        list of ``linear``, ``nonlinear``
    graph_types: str
        list of ``ER``, ``SF``, ``BP``
    benchmark_fns: dict[str, Callable[[ndarray], ndarray]]
        Dictionary of benchmark functions to run. Keys are method names, values are functions that take a dataset and return an estimated adjacency matrix.
    trials: int
        Number of trials to run for each combination of parameters.
    output_filename: str
        Name of the output file to save the results.
    sem_type: str
        ``mlp``, ``mim``, ``gp``, ``gp-add``. Only applicable for nonlinear models.
    metric: Optional[tuple[str, Callable[[ndarray, ndarray], int]]]
        An optional tuple containing the name of the metric and a function that:
            - Accepts parameters ``(B_true, W_est)``
            - Returns a numeric metric value
        If not provided, normalized SHD will be used.
    """
    num_cores = joblib.cpu_count()
    print(f"Detected {num_cores} CPU cores. Running benchmarks in parallel.")

    tasks = []
    keys = []
    for d, edges in sizes:
        for noise_type in noise_types:
            for error_var in error_vars:
                for linearity in linearities:
                    for graph_type in graph_types:
                        for _ in range(trials):
                            dataset, B_true = create_one_dataset(
                                n,
                                d,
                                edges,
                                sem_type,
                                noise_type,
                                error_var,
                                linearity,
                                graph_type,
                            )
                            for name, benchmark_fn in benchmark_fns.items():
                                tasks.append(
                                    delayed(run_one_trial)(
                                        d,
                                        dataset,
                                        B_true,
                                        benchmark_fn,
                                        metric,
                                    )
                                )
                                keys.append(
                                    (
                                        name,
                                        n,
                                        d,
                                        edges,
                                        noise_type,
                                        error_var,
                                        linearity,
                                        graph_type,
                                    )
                                )

    results = Parallel(n_jobs=-1, backend="loky")(tasks)

    aggregated = {}
    for key, res in zip(keys, results):
        if key not in aggregated:
            aggregated[key] = []
        if res is not None:
            aggregated[key].append(res)

    with open(output_filename, "w") as f:
        f.write(
            f"method,n,d,edges,noise_type,error_var,linearity,graph_type,{'mean_normalized_shd' if metric is None else metric[0]}\n"
        )
        for key, vals in aggregated.items():
            method, n, d, edges, noise_type, error_var, linearity, graph_type = key
            mean_metric = np.mean(vals)
            f.write(
                f"{method},{n},{d},{edges},{noise_type},{error_var},{linearity},{graph_type},{mean_metric}\n"
            )
