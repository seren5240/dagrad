from typing import Callable
from numpy import ndarray
import numpy as np
from dagrad.utils import utils


def run_one_trial(
    n: int,
    d: int,
    edges: int,
    sem_type: str,
    noise_type: str,
    error_var: str,
    linear: bool,
    graph_type: str,
    benchmark_fns: dict[str, Callable[[ndarray], ndarray]],
    results: dict[str, list[float]],
):
    B_true = utils.simulate_dag(d, edges, graph_type)
    if error_var == "eq":
        noise_scale = None
    elif error_var == "random":
        noise_scale = np.random.uniform(0.5, 1.0, d)
    else:
        raise ValueError(f"Unknown error_var: {error_var}")
    dataset = (
        utils.simulate_linear_sem(
            B_true, n, sem_type=sem_type, noise_type=noise_type, noise_scale=noise_scale
        )
        if linear
        else utils.simulate_nonlinear_sem(
            B_true, n, sem_type=sem_type, noise_type=noise_type, noise_scale=noise_scale
        )
    )

    for name, benchmark_fn in benchmark_fns.items():
        W_est = benchmark_fn(dataset)
        acc = utils.count_accuracy(B_true, W_est != 0)
        results[name].append(acc["shd"])


def run_one_benchmark(
    n: int,
    d: int,
    edges: int,
    sem_type: str,
    noise_type: str,
    error_var: str,
    linear: bool,
    graph_type: str,
    benchmark_fns: dict[str, Callable[[ndarray], ndarray]],
    trials: int,
    output_filename: str,
):
    results = {name: [] for name in benchmark_fns.keys()}
    for _ in range(trials):
        run_one_trial(
            n,
            d,
            edges,
            sem_type,
            noise_type,
            error_var,
            linear,
            graph_type,
            benchmark_fns,
            results,
        )
    with open(output_filename, "a") as f:
        for method in results:
            mean = np.mean(results[method])
            f.write(
                f"{method},{n},{d},{edges},{noise_type},{error_var},{'linear' if linear else 'nonlinear'},{graph_type},{mean}\n"
            )


def run_benchmarks(
    n: int,
    sizes: list[tuple[int, int]],
    sem_type: str,
    noise_type: str,
    error_var: str,
    linear: bool,
    graph_type: str,
    benchmark_fns: dict[str, Callable[[ndarray], ndarray]],
    trials: int,
    output_filename: str,
):
    """

    Parameters
    ----------
    n: int
        Number of samples
    sizes: list[tuple[int, int]]
        List of node/edge combinations
    sem_type: str
        ``mlp``, ``mim``, ``gp``, ``gp-add``
    noise_type: str
        ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
    error_var: str
        ``eq``, ``random``
    linear: bool
        True for a linear SEM, False for a nonlinear SEM
    graph_type: str
        One of ``["ER", "SF", "BP"]``
    """
    with open(output_filename, "w") as f:
        f.write(
            "method,n,d,edges,noise_type,error_var,linearity,graph_type,mean_normalized_shd\n"
        )
