from typing import Callable
import joblib
from numpy import ndarray
import numpy as np
from joblib import Parallel, delayed
from dagrad.utils import utils


def run_one_trial(
    n: int,
    d: int,
    edges: int,
    sem_type: str,
    noise_type: str,
    error_var: str,
    linearity: str,
    graph_type: str,
    benchmark_fns: dict[str, Callable[[ndarray], ndarray]],
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

    results = {}
    for name, benchmark_fn in benchmark_fns.items():
        W_est = benchmark_fn(dataset)
        acc = utils.count_accuracy(B_true, W_est != 0)
        results[name] = acc["shd"] / d
    return results


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
):
    """
    Run benchmarks on multiple vertex/edge combinations and benchmark functions,
    flattening the nested parallelism so that each trial is scheduled as an individual task.
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
                            tasks.append(
                                delayed(run_one_trial)(
                                    n,
                                    d,
                                    edges,
                                    sem_type,
                                    noise_type,
                                    error_var,
                                    linearity,
                                    graph_type,
                                    benchmark_fns,
                                )
                            )
                            keys.append(
                                (
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
            aggregated[key] = {method: [] for method in benchmark_fns.keys()}
        for method, value in res.items():
            aggregated[key][method].append(value)

    with open(output_filename, "w") as f:
        f.write(
            "method,n,d,edges,noise_type,error_var,linearity,graph_type,mean_normalized_shd\n"
        )
        for key, method_vals in aggregated.items():
            n, d, edges, noise_type, error_var, linearity, graph_type = key
            for method, values in method_vals.items():
                mean_normalized_shd = np.mean(values)
                f.write(
                    f"{method},{n},{d},{edges},{noise_type},{error_var},{linearity},{graph_type},{mean_normalized_shd}\n"
                )
