import torch
from benchmark.benchmarker import run_benchmarks
from dagrad import flex
from dagrad.core import dagrad


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


benchmark_fns = {
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
