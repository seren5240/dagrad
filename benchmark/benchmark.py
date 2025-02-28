from numpy import ndarray
import numpy as np
from dagrad.utils import utils


def run_one_trial(
    n: int,
    d: int,
    edges: int,
    noise_type: str,
    error_var: str,
    linear: bool,
    graph_type: str,
    benchmark_fns: dict[str, callable[[ndarray], ndarray]],
    results: dict[str, list[float]],
):
    B_true = utils.simulate_dag(d, edges, graph_type)
    noise_scale = None if error_var == "eq" else np.random.uniform(0.5, 1.0, d)
    dataset = (
        utils.simulate_linear_sem(
            B_true, n, sem_type=noise_type, noise_scale=noise_scale
        )
        if linear
        else utils.simulate_nonlinear_sem(
            B_true, n, sem_type=noise_type, noise_scale=noise_scale
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
    noise_type: str,
    error_var: str,
    linear: bool,
    graph_type: str,
    benchmark_fns: dict[str, callable[[ndarray], ndarray]],
    trials: int,
):
    results = {name: [] for name in benchmark_fns.keys()}
    for _ in range(trials):
        run_one_trial(
            n,
            d,
            edges,
            noise_type,
            error_var,
            linear,
            graph_type,
            benchmark_fns,
            results,
        )
    output_filename = "benchmark.txt"
    with open(output_filename, "w") as f:
        f.write(
            "method,n,d,edges,noise_type,error_var,linearity,graph_type,mean_normalized_shd\n"
        )
        for method in results:
            mean = np.mean(results[method])
            f.write(
                f"{method},{n},{d},{edges},{noise_type},{error_var},{'linear' if linear else 'nonlinear'},{graph_type},{mean}\n"
            )
