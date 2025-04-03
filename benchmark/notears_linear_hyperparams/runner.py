def notears_mcp(dataset):
    d = dataset.shape[1]
    # general_options = {'gamma':0.4, 'lambda1':0.1} # Define the general options
    model = flex.LinearModelMCP(d, gamma=0.4)

    # method_options = {'verbose': False, 'rho':0.1}
    cons_solver = flex.AugmentedLagrangian(
        num_iter=10, num_steps=[3e4, 5e3], l1_coeff=0.01, rho_init=0.1
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
