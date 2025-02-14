import os

from dagrad.hfunction.h_functions import SCCPowerIteration
os.environ["R_LIBS_USER"] = "~/Rlibs"
from dagrad import dagrad
from dagrad import generate_linear_data, count_accuracy, threshold_till_dag
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    # B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def sdcd_ev(n, d, s0, graph_type, noise_type, error_var, seed=None, mu_factor=0.1):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
    X = torch.from_numpy(X).float()
    model = 'linear' # Define the model
    W_sdcd = dagrad(
        X,
        model = model,
        method = 'dagma',
        compute_lib='torch',
        h_fn='user_h',
        general_options={
            'user_params': {
                'is_prescreen': False,
                'power_grad': SCCPowerIteration(
                    torch.zeros(d, d, dtype = torch.double, requires_grad = True, device = 'cpu'),
                    d,
                )
            }
        },
        method_options={
            'mu_factor': mu_factor,
        }
    ) # Learn the structure of the DAG using SDCD
    W_sdcd = postprocess(W_sdcd)
    print(f"Linear Model")
    print(f"data size: {n}, graph type: {graph_type}, sem type: {noise_type}")
    acc_sdcd = count_accuracy(B_true, W_sdcd != 0) # Measure the accuracy of the learned structure using SDCD
    print('Accuracy of SDCD:', acc_sdcd)

    return acc_sdcd

# def golem_nv(n, d, s0, graph_type, noise_type, error_var, seed=None):
#     X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,noise_type,error_var,seed)
#     X = torch.from_numpy(X).float()
#     model = 'linear' # Define the model
#     W_ev = dagrad(
#         X,
#         model = model,
#         method = 'dagma',
#         compute_lib='torch',
#         loss_fn='user_loss',
#         reg='user_reg',
#         h_fn='user_h',
#         general_options={'user_params': {
#             'equal_variances': True,
#         }}
#     ) # Learn the structure of the DAG using Golem
#     print(f"Linear Model")
#     print(f"data size: {n}, graph type: {graph_type}, nodes: {d}, edges: {s0}, error_var: {error_var}, sem type: {noise_type}")

#     W_ev_processed = postprocess(W_ev)
#     acc_ev = count_accuracy(B_true, W_ev_processed != 0) # Measure the accuracy of the learned structure using Golem
#     print('Accuracy of Golem after EV stage:', acc_ev)

#     W_nv = dagrad(
#         X,
#         model = model,
#         method = 'dagma',
#         compute_lib='torch',
#         loss_fn='user_loss',
#         reg='user_reg',
#         h_fn='user_h',
#         general_options={'user_params': {
#             'equal_variances': False,
#         },
#         'initialization': W_ev}
#     ) 

#     W_processed = postprocess(W_nv)
#     acc_nv = count_accuracy(B_true, W_processed != 0) # Measure the accuracy of the learned structure using Golem
#     print('Accuracy of Golem after NV stage:', acc_nv)

#     return acc_nv

def run_one_experiment(trials, n, s0_ratio, noise_type, error_var):
    num_nodes = [5, 10, 50, 100] if s0_ratio <= 2 else [10, 50, 100]
    methods = ["SDCD-HIGH", "SDCD-LOW"]
    shd_results = {method: {d: [] for d in num_nodes} for method in methods}
    sid_results = {method: {d: [] for d in num_nodes} for method in methods}

    for d in num_nodes:
        s0 = int(s0_ratio * d)

        for i in range(trials):
            print(f"Running trial {i} for {d} nodes")
            try:
                low_factor_result = sdcd_ev(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type, mu_factor=0.1)
                shd_results["SDCD-LOW"][d].append(low_factor_result["shd"] / d)
                sid_results["SDCD-LOW"][d].append(low_factor_result["sid"] / d)

                high_factor_result = sdcd_ev(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=noise_type, mu_factor=0.9)
                shd_results["SDCD-HIGH"][d].append(high_factor_result["shd"] / d)
                sid_results["SDCD-HIGH"][d].append(high_factor_result["sid"] / d)
                
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
    plt.savefig(f"sdcd_{metric}_ER{format_ratio(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.png")

    output_filename = f"sdcd_{metric}_ER{format_ratio(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.txt"
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
