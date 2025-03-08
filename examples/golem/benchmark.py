import os
os.environ["R_LIBS_USER"] = "~/Rlibs"
from dagrad import dagrad
from dagrad import generate_linear_data, count_accuracy, threshold_till_dag
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed
import joblib

def format_ratio(ratio):
    return 0.5 if ratio == 0.5 else int(ratio)

def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def notears(n, d, s0, graph_type, noise_type, error_var, seed=None):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
    X = torch.from_numpy(X).float()
    model = 'linear' # Define the model
    W_notears = dagrad(
        X,
        model = model,
        method = 'notears',
        compute_lib='torch',
    ) # Learn the structure of the DAG using Golem
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, nodes: {d}, edges: {s0}, error_var: {error_var}, sem type: {noise_type}")

    W_processed = postprocess(W_notears)
    acc_notears = count_accuracy(B_true, W_processed != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of NOTEARS:', acc_notears)

    return acc_notears

def dagma(n, d, s0, graph_type, noise_type, error_var, seed=None):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
    X = torch.from_numpy(X).float()
    model = 'linear' # Define the model
    W_notears = dagrad(
        X,
        model = model,
        method = 'dagma',
        compute_lib='torch',
    ) # Learn the structure of the DAG using Golem
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, nodes: {d}, edges: {s0}, error_var: {error_var}, sem type: {noise_type}")

    W_processed = postprocess(W_notears)
    acc_dagma = count_accuracy(B_true, W_processed != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of DAGMA:', acc_dagma)

    return acc_dagma

def golem_ev(n, d, s0, graph_type, noise_type, error_var, seed=None):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
    X = torch.from_numpy(X).float()
    model = 'linear' # Define the model
    W_golem = dagrad(
        X,
        model = model,
        method = 'notears',
        compute_lib='torch',
        loss_fn='user_loss',
        reg='user_reg',
        h_fn='user_h',
        general_options={'user_params': {
            'equal_variances': True,
        }}
    ) # Learn the structure of the DAG using Golem
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, nodes: {d}, edges: {s0}, error_var: {error_var}, sem type: {noise_type}")

    W_processed = postprocess(W_golem)
    acc_golem = count_accuracy(B_true, W_processed != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of Golem:', acc_golem)

    return acc_golem

def golem_nv(n, d, s0, graph_type, noise_type, error_var, seed=None):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
    X = torch.from_numpy(X).float()
    model = 'linear' # Define the model
    W_ev = dagrad(
        X,
        model = model,
        method = 'notears',
        compute_lib='torch',
        loss_fn='user_loss',
        reg='user_reg',
        h_fn='user_h',
        general_options={'user_params': {
            'equal_variances': True,
        }}
    ) # Learn the structure of the DAG using Golem
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, nodes: {d}, edges: {s0}, error_var: {error_var}, sem type: {noise_type}")

    W_ev_processed = postprocess(W_ev)
    acc_ev = count_accuracy(B_true, W_ev_processed != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of Golem after EV stage:', acc_ev)

    W_nv = dagrad(
        X,
        model = model,
        method = 'notears',
        compute_lib='torch',
        loss_fn='user_loss',
        reg='user_reg',
        h_fn='user_h',
        general_options={'user_params': {
            'equal_variances': False,
        },
        'initialization': W_ev}
    ) 

    W_processed = postprocess(W_nv)
    acc_nv = count_accuracy(B_true, W_processed != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of Golem after NV stage:', acc_nv)

    return acc_nv

def run_one_experiment(trials, n, s0_ratio, noise_type, error_var):
    num_nodes = [5, 10, 50, 100] if s0_ratio <= 2 else [10, 50, 100]
    methods = ["GOLEM-NOTEARS-EV", "GOLEM-NOTEARS-NV", "NOTEARS", "DAGMA"]
    shd_results = {method: {d: [] for d in num_nodes} for method in methods}
    sid_results = {method: {d: [] for d in num_nodes} for method in methods}

    def run_trial(d, s0, i):
        print(f"Running trial {i} for {d} nodes")
        try:
            ev_result = golem_ev(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type)
            nv_result = golem_nv(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type)
            notears_result = notears(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type)
            dagma_result = dagma(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type)

            return (d, ev_result, nv_result, notears_result, dagma_result)
        except Exception as e:
            print(e)
            print(f'Trial {i} with {d} nodes and {noise_type} noise and s0_ratio {s0_ratio} skipped due to error')
            return (d, None, None)

    num_cores = joblib.cpu_count()
    print(f"Detected {num_cores} CPU cores. Running trials in parallel.")

    for d in num_nodes:
        s0 = int(s0_ratio * d)
        trial_results = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_trial)(d, s0, i) for i in range(trials)
        )

        for d, ev_result, nv_result, notears_result, dagma_result in trial_results:
            if ev_result is not None:
                shd_results["GOLEM-NOTEARS-EV"][d].append(ev_result["shd"] / d)
                sid_results["GOLEM-NOTEARS-EV"][d].append(ev_result["sid"] / d)

            if nv_result is not None:
                shd_results["GOLEM-NOTEARS-NV"][d].append(nv_result["shd"] / d)
                sid_results["GOLEM-NOTEARS-NV"][d].append(nv_result["sid"] / d)

            if notears_result is not None:
                shd_results["NOTEARS"][d].append(notears_result['shd'] / d)
                sid_results["NOTEARS"][d].append(notears_result['sid'] / d)

            if dagma_result is not None:
                shd_results["DAGMA"][d].append(dagma_result["shd"] / d)
                sid_results["DAGMA"][d].append(dagma_result["sid"] / d)

    make_one_plot(s0_ratio, noise_type, methods, num_nodes, trials, n, error_var, shd_results, "shd")
    make_one_plot(s0_ratio, noise_type, methods, num_nodes, trials, n, error_var, sid_results, "sid")

def make_one_plot(s0_ratio, noise_type, methods, num_nodes, trials, n, error_var, results, metric: str):
    plt.figure(figsize=(8, 6))

    for method in methods:
        means = [
            np.mean(results[method][d]) if results[method][d] else None
            for d in num_nodes
        ]
        plt.plot(num_nodes, means, marker="o", label=method)

    noise_names = {
        "gauss": "Gaussian",
        "exp": "Exponential",
        "gumbel": "Gumbel"
    }

    plt.title(f"{noise_names[noise_type]} Noise, ER{s0_ratio}\n(n={n}, trials={trials}, error_var={error_var})")
    plt.xlabel("d (Number of Nodes)")
    plt.ylabel(f"Normalized {metric.upper()}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"golem_{metric}_ER{format_ratio(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.png")

    output_filename = f"golem_{metric}_ER{format_ratio(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.txt"
    with open(output_filename, "w") as f:
        f.write(f"method,d,mean_normalized_{metric}\n")
        for method in methods:
            for d in num_nodes:
                mean_metric = np.mean(results[method][d]) if results[method][d] else None
                if mean_metric is not None:
                    f.write(f"{method},{d},{mean_metric}\n")


nTrials = int(sys.argv[1])
nSamples = int(sys.argv[2])
s0_ratio = float(sys.argv[3])
noise_type = sys.argv[4]
error_var = sys.argv[5]
run_one_experiment(nTrials, nSamples, s0_ratio, noise_type, error_var)
