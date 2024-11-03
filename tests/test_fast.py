import dagrad

dagrad.set_random_seed(2)
n, d, s0, graph_type, sem_type, model = 1000, 10, 10, 'ER', 'gauss', 'linear'
X, W_true, B_true = dagrad.generate_linear_data(n,d,s0,graph_type,sem_type) # Generate the data

W_notears = dagrad.dagrad(X, model = model, method = 'notears') # Learn the structure of the DAG using Notears
W_dagma = dagrad.dagrad(X, model = model, method = 'dagma') # Learn the structure of the DAG using Dagma
W_topo = dagrad.dagrad(X, model = model, method = 'topo') # Learn the structure of the DAG using Topo

acc_notears = dagrad.count_accuracy(B_true, W_notears != 0) # Measure the accuracy of the learned structure using Notears
acc_dagma = dagrad.count_accuracy(B_true, W_dagma != 0) # Measure the accuracy of the learned structure using Dagma
acc_topo = dagrad.count_accuracy(B_true, W_topo != 0) # Measure the accuracy of the learned structure using Topo

print(f"Linear Model")
print(f"data size: {n}, graph type: {graph_type}, sem type: {sem_type}")
print('Accuracy of Notears:', acc_notears)
print('Accuracy of Dagma:', acc_dagma)
print('Accuracy of Topo:', acc_topo)