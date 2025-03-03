import numpy as np
import torch
from benchmark.benchmarker import run_benchmarks
from dagrad import flex
from dagrad.core import dagrad
from dagrad.flex.prune import cam_pruning
from dagrad.utils import utils


def notears(dataset):
    d = dataset.shape[1]
    model = flex.LinearModel(d)

    cons_solver = flex.AugmentedLagrangian(
        num_iter=10,
        num_steps=[3e4, 6e4],
        l1_coeff=0.03,
    )
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
    )
    loss_fn = flex.MSELoss()
    dag_fn = flex.Exp()
    W_est = flex.struct_learn(
        dataset=dataset,
        model=model,
        constrained_solver=cons_solver,
        unconstrained_solver=uncons_solver,
        loss_fn=loss_fn,
        dag_fn=dag_fn,
        w_threshold=0.3,
    )
    return W_est


def dagma(dataset):
    return dagrad(dataset, model="linear", method="dagma")


def golem_like(dataset):
    return dagrad(
        dataset,
        model="linear",
        method="dagma",
        reg="l1",
        h_fn="h_exp_sq",
        general_options={
            "lambda1": 2e-3,
        },
    )


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
    B[np.abs(B) <= graph_thres] = 0  # Thresholding
    B, _ = utils.threshold_till_dag(B)

    return B


def grandag(dataset):
    d = dataset.shape[1]
    train_samples = int(dataset.shape[0] * 0.8)
    train_dataset = dataset[:train_samples, :]
    test_dataset = dataset[train_samples:, :]

    # Nonlinear model
    model = flex.GrandagMLP(
        dims=[d, 2, d], num_layers=2, hid_dim=10, activation="sigmoid", bias=True
    )

    # Use AML to solve the constrained problem
    cons_solver = flex.GrandagAugmentedLagrangian(
        num_iter=100000,
        num_steps=[1, 1],
        rho_init=1e-3,
    )

    # Use Adam to solve the unconstrained problem
    uncons_solver = flex.GrandagSolver(
        optimizer=torch.optim.RMSprop(model.parameters(), lr=1e-3),
    )

    # Use MSE loss
    loss_fn = None

    # Use Trace of matrix exponential as DAG function
    dag_fn = flex.TrExp()

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

    to_keep = (torch.from_numpy(W_est) > 0).type(torch.Tensor)
    B_est = model.adjacency * to_keep

    opt = {
        "cam_pruning_cutoff": np.logspace(-6, 0, 10),
        "exp_path": "cam_pruning",
    }
    try:
        cam_pruning_cutoff = [float(i) for i in opt["cam_pruning_cutoff"]]
    except:
        cam_pruning_cutoff = [float(opt["cam_pruning_cutoff"])]
    for cutoff in cam_pruning_cutoff:
        B_est = cam_pruning(B_est, train_dataset, test_dataset, opt, cutoff=cutoff)
    return B_est.detach().cpu().numpy()


benchmark_fns = {
    "GRAN-DAG": grandag,
    "NOTEARS": notears,
    "DAGMA": dagma,
    "GOLEM": golem_like,
}
run_benchmarks(
    500,
    [[5, 5], [5, 10]],
    ["gauss", "gumbel"],
    ["eq", "random"],
    ["linear"],
    ["ER"],
    benchmark_fns,
    2,
    "benchmark.txt",
)
