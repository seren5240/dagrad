import sys
from benchmark.cpdag_shd import shd_cpdag
import numpy as np
import torch
from benchmark.benchmarker import run_benchmarks
from dagrad import flex
from dagrad.core import dagrad
from dagrad.flex.prune import cam_pruning
from dagrad.utils import utils

# first command line argument is lmd, second is gamma
# g = float(sys.argv[1])
# a = float(sys.argv[2])
# rho_init = float(sys.argv[3])
# lmd = g
# gamma = a * g
# print(f"using lmd: {lmd}, gamma: {gamma}, rho_init: {rho_init}")


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


def notears_mcp_base(dataset):
    general_options = {"gamma": 0.4, "lambda1": 0.1}
    optimizer_options = {
        "lr": 0.01,
        "num_steps": 5000,
        "check_iterate": 500,
        "tol": 1e-5,
    }
    return dagrad(
        dataset,
        model="linear",
        method="notears",
        reg="mcp",
        optimizer="adam",
        compute_lib="numpy",
        general_options=general_options,
        optimizer_options=optimizer_options,
    )


def notears_mcp_flex(dataset):
    d = dataset.shape[1]
    # general_options = {'gamma':0.4, 'lambda1':0.1} # Define the general options
    model = flex.LinearModelMCP(d, lmd=0.1, gamma=0.4)

    # method_options = {'verbose': False, 'rho':0.1}
    cons_solver = flex.AugmentedLagrangian(
        num_iter=10,
        num_steps=[3e4, 6e4],
        l1_coeff=0.03,
    )
    # optimizer_options = {'lr':0.01,'num_steps':5000, 'check_iterate':500, 'tol':1e-5} # Define the optimizer options
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
        tol=1e-5,
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


def notears_nonlinear(dataset):
    d = dataset.shape[1]
    model = flex.MLP(dims=[d, 10, 1], activation="sigmoid", bias=True)

    # Use AML to solve the constrained problem
    cons_solver = flex.AugmentedLagrangian(
        num_iter=10,
        num_steps=[4e4, 6e4],
        l1_coeff=0.01,
        weight_decay=0.01,
    )

    # Use Adam to solve the unconstrained problem
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
    )

    # Use MSE loss
    loss_fn = flex.MSELoss()

    # Use Trace of matrix exponential as DAG function
    dag_fn = flex.Exp()

    # Learn the DAG
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


def notears_nonlinear_mcp_flex(dataset):
    d = dataset.shape[1]
    model = flex.MLPMCP(dims=[d, 10, 1], activation="sigmoid", bias=True)

    # Use AML to solve the constrained problem
    cons_solver = flex.AugmentedLagrangian(
        num_iter=10,
        num_steps=[4e4, 6e4],
        l1_coeff=0.01,
    )

    # Use Adam to solve the unconstrained problem
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
    )

    # Use MSE loss
    loss_fn = flex.MSELoss()

    # Use Trace of matrix exponential as DAG function
    dag_fn = flex.Exp()

    # Learn the DAG
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


def flex_dagma(dataset):
    d = dataset.shape[1]
    model = flex.LinearModel(d)

    # Use path following to solve the constrained problem
    cons_solver = flex.PathFollowing(
        num_iter=5,
        mu_init=1.0,
        mu_scale=0.1,
        logdet_coeff=[1.0, 0.9, 0.8, 0.7, 0.6],
        num_steps=[3e4, 6e4],
        l1_coeff=0.03,
    )

    # use Adam to solve the unconstrained problem
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.99, 0.999))
    )

    # Use MSE loss
    loss_fn = flex.MSELoss()

    # Use LogDet as DAG function
    dag_fn = flex.LogDet()

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


def dagma_mcp_flex(dataset):
    d = dataset.shape[1]
    model = flex.LinearModelMCP(d)

    # Use path following to solve the constrained problem
    cons_solver = flex.PathFollowing(
        num_iter=5,
        mu_init=1.0,
        mu_scale=0.1,
        logdet_coeff=[1.0, 0.9, 0.8, 0.7, 0.6],
        num_steps=[3e4, 6e4],
        l1_coeff=0.03,
    )

    # use Adam to solve the unconstrained problem
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.99, 0.999))
    )

    # Use MSE loss
    loss_fn = flex.MSELoss()

    # Use LogDet as DAG function
    dag_fn = flex.LogDet()

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


def flex_dagma_nonlinear(dataset):
    d = dataset.shape[1]
    model = flex.MLP(dims=[d, 10, 1], activation="sigmoid", bias=True)

    # Use path following to solve the constrained problem
    cons_solver = flex.PathFollowing(
        num_iter=4,
        mu_init=0.1,
        mu_scale=0.1,
        logdet_coeff=1.0,
        num_steps=[5e4, 8e4],
        weight_decay=0.02,
        l1_coeff=0.005,
    )

    # use Adam to solve the unconstrained problem
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.99, 0.999))
    )

    # Use NLL loss
    loss_fn = flex.NLLLoss()

    # Use LogDet as DAG function
    dag_fn = flex.LogDet()

    # Learn the DAG
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


# def dagma_nonlinear_mcp_flex(dataset):
#     d = dataset.shape[1]
#     model = flex.MLPMCP(
#         dims=[d, 10, 1], activation="sigmoid", bias=True, lmd=lmd, gamma=gamma
#     )

#     # Use path following to solve the constrained problem
#     cons_solver = flex.PathFollowing(
#         num_iter=4,
#         mu_init=0.1,
#         mu_scale=0.1,
#         logdet_coeff=1.0,
#         num_steps=[5e4, 8e4],
#         weight_decay=0.02,
#         l1_coeff=0.005,
#     )

#     # use Adam to solve the unconstrained problem
#     # optimizer_options = {'lr':0.01,'num_steps':5000, 'check_iterate':500, 'tol':1e-5} # Define the optimizer options
#     uncons_solver = flex.GradientBasedSolver(
#         optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.99, 0.999)),
#         tol=1e-5,
#     )

#     # Use NLL loss
#     loss_fn = flex.NLLLoss()

#     # Use LogDet as DAG function
#     dag_fn = flex.LogDet()

#     # Learn the DAG
#     W_est = flex.struct_learn(
#         dataset=dataset,
#         model=model,
#         constrained_solver=cons_solver,
#         unconstrained_solver=uncons_solver,
#         loss_fn=loss_fn,
#         dag_fn=dag_fn,
#         w_threshold=0.3,
#     )
#     return W_est


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

    # opt = {
    #     "cam_pruning_cutoff": np.logspace(-6, 0, 10),
    #     "exp_path": "cam_pruning",
    # }
    # try:
    #     cam_pruning_cutoff = [float(i) for i in opt["cam_pruning_cutoff"]]
    # except:
    #     cam_pruning_cutoff = [float(opt["cam_pruning_cutoff"])]
    # for cutoff in cam_pruning_cutoff:
    #     B_est = cam_pruning(B_est, train_dataset, test_dataset, opt, cutoff=cutoff)
    return B_est.detach().cpu().numpy()


def cpdag_metric(B_true, W_est):
    d = B_true.shape[0]
    cpdag = shd_cpdag(B_true, W_est)
    return cpdag / d


benchmark_fns = {
    # "GRAN-DAG": grandag,
    # "NOTEARS": notears,
    # "DAGMA": flex_dagma_nonlinear,
    # "NOTEARS-MCP": notears_mcp_flex,
    # "DAGMA-MCP": flex_dagma_nonlinear_mcp,
    # "GOLEM": golem_like,
    "NOTEARS-BASE": notears_mcp_base,
    "NOTEARS-FLEX": notears_mcp_flex,
}

large_sizes = [
    [5, 5],
    [5, 10],
    [10, 10],
    [10, 20],
    [10, 40],
    [20, 20],
    [20, 40],
    [20, 80],
    [50, 50],
    [50, 100],
    [50, 200],
    [100, 100],
    [100, 200],
    [100, 400],
]

# parallelize, tell user how many cores initialized
run_benchmarks(
    1000,
    large_sizes,
    ["gauss", "exp", "gumbel"],
    ["eq", "random"],
    ["linear"],
    ["ER"],
    benchmark_fns,
    10,
    "benchmark_notears_mcp_flex_vs_base.txt",
)
