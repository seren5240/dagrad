def notears_nonlinear_mcp(dataset):
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
