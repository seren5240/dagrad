import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
from dagrad import flex
from dagrad.flex.prune import cam_pruning
from dagrad.utils import utils

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
    B, _ = utils.threshold_till_dag(B)

    return B

def grandag_aug_lagrangian(n, d, s0, num_layers=2, noise_type="gauss"):
    graph_type, sem_type = "ER", "mlp"
    nonlinear_B_true = utils.simulate_dag(d, s0, graph_type)
    nonlinear_dataset = utils.simulate_nonlinear_sem(nonlinear_B_true, n, sem_type, noise_type=noise_type)
    # print(f'b_true is {nonlinear_B_true}')

    train_samples = int(nonlinear_dataset.shape[0] * 0.8)
    train_dataset = nonlinear_dataset[:train_samples, :]
    test_dataset = nonlinear_dataset[train_samples:, :]

    # Nonlinear model
    model = flex.MLP(dims=[d, 2, d], num_layers=num_layers, hid_dim=10, activation="sigmoid", bias=True)

    # Use AML to solve the constrained problem
    cons_solver = flex.AugmentedLagrangian(
        num_iter=100000,
        num_steps=[1,1],
        # l1_coeff=0.01,
        # weight_decay=0.01,
        rho_init=1e-3,
    )

    # Use Adam to solve the unconstrained problem
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.RMSprop(model.parameters(), lr=1e-3),
    )

    # Use MSE loss
    loss_fn = flex.MSELoss()

    # Use Trace of matrix exponential as DAG function
    dag_fn = flex.Grandag_h()

    # Learn the DAG
    W_est = flex.struct_learn(
        dataset=train_dataset,
        model=model,
        constrained_solver=cons_solver,
        unconstrained_solver=uncons_solver,
        loss_fn=loss_fn,
        dag_fn=dag_fn,
        w_threshold=0.0,
    )

    W_est = postprocess(W_est)

    acc = utils.count_accuracy(nonlinear_B_true, W_est != 0)
    print(f'Results before CAM pruning: {acc}')

    to_keep = (torch.from_numpy(W_est) > 0).type(torch.Tensor)
    B_est = model.adjacency * to_keep
    # print(f'B_est is {B_est}')

    opt = {
        'cam_pruning_cutoff': np.logspace(-6, 0, 10),
        'exp_path': 'cam_pruning',
    }
    try:
        cam_pruning_cutoff = [float(i) for i in opt['cam_pruning_cutoff']]
    except:
        cam_pruning_cutoff = [float(opt['cam_pruning_cutoff'])]
    for cutoff in cam_pruning_cutoff:
        B_est = cam_pruning(B_est, train_dataset, test_dataset, opt, cutoff=cutoff)
    acc = utils.count_accuracy(nonlinear_B_true, B_est.detach().cpu().numpy() != 0)
    print("Results: ", acc)
    return acc

def run_one_experiment(trials, n, s0_ratio, noise_type, error_var):
    num_nodes = [5, 10, 20]#[5, 10, 50, 100]
    methods = ["GRAN-DAG"]
    shd_results = {method: {d: [] for d in num_nodes} for method in methods}
    sid_results = {method: {d: [] for d in num_nodes} for method in methods}

    for d in num_nodes:
        s0 = int(s0_ratio * d)

        for i in range(trials):
            print(f"Running trial {i} for {d} nodes")
            try:
                results = grandag_aug_lagrangian(n=n, d=d, s0=s0, num_layers=2, noise_type=noise_type)
                shd_results["GRAN-DAG"][d].append(results["shd"] / d)
                sid_results["GRAN-DAG"][d].append(results["sid"] / d)
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
