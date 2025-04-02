def notears_mcp(dataset):
    d = dataset.shape[1]
    model = flex.LinearModelMCP(d)

    cons_solver = flex.AugmentedLagrangian(
        num_iter=10,
        num_steps=[3e4, 6e4],
        l1_coeff=0.01,
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