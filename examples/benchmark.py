from dagrad import dagrad
from dagrad import generate_linear_data, count_accuracy
import torch
import numpy as np
import matplotlib.pyplot as plt 

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

    acc_golem = count_accuracy(B_true, W_golem != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of Golem:', acc_golem)

    return acc_golem

def golem_nv(n, d, s0, graph_type, sem_type, error_var, seed=None, intermediate_accuracy=True):
    X, W_true, B_true = generate_linear_data(n,d,s0,graph_type,sem_type,error_var,seed)
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
    print(f"data size: {n}, graph type: {graph_type}, nodes: {d}, edges: {s0}, error_var: {error_var}, sem type: {sem_type}")

    if intermediate_accuracy:
        acc_ev = count_accuracy(B_true, W_ev != 0) # Measure the accuracy of the learned structure using Golem
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

    acc_nv = count_accuracy(B_true, W_nv != 0) # Measure the accuracy of the learned structure using Golem
    print('Accuracy of Golem after NV stage:', acc_nv)

    return acc_nv

def run_experiment(trials, error_var):
    """
    Parameters:
        trials (int): Number of trials to run for each configuration.
    
    Returns:
        None (Generates and saves plots).
    """
    n = 1000
    num_nodes = [5, 10, 50, 100]
    s0_ratios = [0.5, 1, 2]
    noise_types = ["gauss", "exp", "gumbel"]
    methods = ["GOLEM-EV", "GOLEM-NV"]

    results = {method: {sem: {d: [] for d in num_nodes} for sem in noise_types} for method in methods}

    for d in num_nodes:
        for sem_type in noise_types:
            for s0_ratio in s0_ratios:
                s0 = int(s0_ratio * d)

                for _ in range(trials):
                    ev_result = golem_ev(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, noise_type=sem_type)
                    results["GOLEM-EV"][sem_type][d].append(ev_result["shd"] / s0)

                    nv_result = golem_nv(n=n, d=d, s0=s0, graph_type="ER", error_var=error_var, sem_type=sem_type)
                    results["GOLEM-NV"][sem_type][d].append(nv_result["shd"] / s0)

    num_rows = len(s0_ratios)
    num_cols = len(noise_types)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True, sharey=True)

    for i, s0_ratio in enumerate(s0_ratios):
        for j, noise in enumerate(noise_types):
            ax = axes[i, j] if num_rows > 1 else axes[j]
            
            for method in methods:
                means = [
                    np.mean(results[method][noise][d])
                    for d in num_nodes
                ]
                ax.plot(num_nodes, means, marker="o", label=method)

            noise_names = {
                "gauss": "Gaussian",
                "exp": "Exponential",
                "gumbel": "Gumbel"
            }
            s0_ratio_to_ER = {
                0.5: '1',
                1: '2',
                2: '4'
            }

            ax.set_title(f"{noise_names[noise]} noise, ER{s0_ratio_to_ER[s0_ratio]}")
            ax.set_xlabel("d (Number of Nodes)")
            if j == 0:
                ax.set_ylabel("Normalized SHD")
            ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"normalized_shd_n={n}_var={error_var}.png")

run_experiment(1, 'eq')
