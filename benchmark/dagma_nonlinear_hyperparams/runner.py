def flex_dagma_nonlinear_mcp(dataset):
    d = dataset.shape[1]
    model = flex.MLPMCP(dims=[d, 10, 1], activation="sigmoid", bias=True, gamma=0.4)

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
    # optimizer_options = {'lr':0.01,'num_steps':5000, 'check_iterate':500, 'tol':1e-5} # Define the optimizer options
    uncons_solver = flex.GradientBasedSolver(
        optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.99, 0.999)),
        tol=1e-5,
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
