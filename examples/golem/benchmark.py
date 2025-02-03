import os
os.environ["R_LIBS_USER"] = "~/Rlibs"
from dagrad import dagrad
from dagrad import generate_linear_data, count_accuracy, threshold_till_dag
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

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

def golem_ev(n, d, s0, graph_type, noise_type, error_var, seed=None):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
    X = torch.from_numpy(X).float()
    model = 'linear' # Define the model
    W_golem = dagrad(
        X,
        model = model,
        method = 'dagma',
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
        method = 'dagma',
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
        method = 'dagma',
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
    num_nodes = [5, 10, 50, 100]
    methods = ["GOLEM-EV", "GOLEM-NV"]
    shd_results = {method: {d: [] for d in num_nodes} for method in methods}
    sid_results = {method: {d: [] for d in num_nodes} for method in methods}

    for d in num_nodes:
        s0 = int(s0_ratio * d)

        for i in range(trials):
            print(f"Running trial {i} for {d} nodes")
            try:
                ev_result = golem_ev(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type)
                shd_results["GOLEM-EV"][d].append(ev_result["shd"] / s0)
                sid_results["GOLEM-EV"][d].append(ev_result["sid"] / s0)

                nv_result = golem_nv(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type)
                shd_results["GOLEM-NV"][d].append(nv_result["shd"] / s0)
                sid_results["GOLEM-NV"][d].append(nv_result["sid"] / s0)
            except Exception as e:
                print(e)
                print(f'trial with {d} nodes and {noise_type} noise and s0_ratio {s0_ratio} skipped due to error')

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
    plt.savefig(f"golem_{metric}_ER{int(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.png")

    output_filename = f"golem_{metric}_ER{int(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.txt"
    with open(output_filename, "w") as f:
        f.write(f"method,d,mean_normalized_{metric}\n")
        for method in methods:
            for d in num_nodes:
                mean_metric = np.mean(results[method][d]) if results[method][d] else None
                if mean_metric is not None:
                    f.write(f"{method},{d},{mean_metric}\n")


def run_experiment(trials, error_var):
    """
    Parameters:
        trials (int): Number of trials to run for each configuration.
    
    Returns:
        None (Generates and saves plots).
    """
    n = 1000
    num_nodes = [5, 10, 50, 100]
    s0_ratios = [1, 2, 4]
    noise_types = ["gauss", "exp", "gumbel"]
    methods = ["GOLEM-EV", "GOLEM-NV"]

    shd_results = {method: {sem: {s0: {d: [] for d in num_nodes} for s0 in s0_ratios} for sem in noise_types} for method in methods}
    sid_results = {method: {sem: {s0: {d: [] for d in num_nodes} for s0 in s0_ratios} for sem in noise_types} for method in methods}

    for d in num_nodes:
        for sem_type in noise_types:
            for s0_ratio in s0_ratios:
                s0 = int(s0_ratio * d)

                for _ in range(trials):
                    try:
                        ev_result = golem_ev(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=sem_type)
                        shd_results["GOLEM-EV"][sem_type][s0_ratio][d].append(ev_result["shd"] / s0)
                        sid_results["GOLEM-EV"][sem_type][s0_ratio][d].append(ev_result["sid"] / s0)

                        nv_result = golem_nv(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=sem_type)
                        shd_results["GOLEM-NV"][sem_type][s0_ratio][d].append(nv_result["shd"] / s0)
                        sid_results["GOLEM-NV"][sem_type][s0_ratio][d].append(nv_result["sid"] / s0)
                    except Exception as e:
                        print(e)
                        print(f'trial with {d} nodes and {sem_type} noise and s0_ratio {s0_ratio} skipped due to error')

    make_subplots(s0_ratios, noise_types, methods, num_nodes, trials, n, error_var, shd_results, "shd")
    make_subplots(s0_ratios, noise_types, methods, num_nodes, trials, n, error_var, sid_results, "sid")

def make_subplots(s0_ratios, noise_types, methods, num_nodes, trials, n, error_var, results, metric: str):
    num_rows = len(s0_ratios)
    num_cols = len(noise_types)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, sharey=True)

    for i, s0_ratio in enumerate(s0_ratios):
        for j, noise in enumerate(noise_types):
            ax = axes[i, j] if num_rows > 1 else axes[j]
            
            for method in methods:
                means = [
                    np.mean(results[method][noise][s0_ratio][d])
                    for d in num_nodes
                ]
                ax.plot(num_nodes, means, marker="o", label=method)

            noise_names = {
                "gauss": "Gaussian",
                "exp": "Exponential",
                "gumbel": "Gumbel"
            }

            ax.set_title(f"{noise_names[noise]} noise, ER{s0_ratio}")
            ax.set_xlabel("d (Number of Nodes)")
            if j == 0:
                ax.set_ylabel(f"Normalized {metric.upper()}")
            ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    plt.title(f"Trials={trials}, error_var={error_var}")
    fig.legend(handles, labels, loc="upper center", ncol=len(methods))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"normalized_{metric}_n={n}_var={error_var}_trials={trials}.png")

if len(sys.argv) < 3:
    run_experiment(10, 'eq')
    run_experiment(10, 'random')
    sys.exit(1)
else:
    nTrials = int(sys.argv[1])
    nSamples = int(sys.argv[2])
    s0_ratio = float(sys.argv[3])
    noise_type = sys.argv[4]
    error_var = sys.argv[5]
    run_one_experiment(nTrials, nSamples, s0_ratio, noise_type, error_var)
