import os
import uuid

import numpy as np
from cdt.utils.R import RPackages, launch_R_script
import pandas as pd
import torch

def np_to_csv(array, save_path):
    """
    Convert np array to .csv

    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def cam_pruning_(model_adj, train_data, test_data, cutoff, save_path, verbose=False):
    # convert numpy data to csv, so R can access them
    data_np = np.concatenate([train_data, test_data], 0)
    data_csv_path = np_to_csv(data_np, save_path)
    dag_csv_path = np_to_csv(model_adj, save_path)

    #dag_pruned = cam_pruning(path_data, path_dag, cutoff, verbose)
    if not RPackages.CAM:
        raise ImportError("R Package CAM is not available.")

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{CUTOFF}'] = str(cutoff)

    if verbose:
        arguments['{VERBOSE}'] = "TRUE"
    else:
        arguments['{VERBOSE}'] = "FALSE"

    def retrieve_result():
        return pd.read_csv(arguments['{PATH_RESULTS}']).values

    dag_pruned = launch_R_script("{}/../utils/cam_pruning.R".format(os.path.dirname(os.path.realpath(__file__))),
                                     arguments, output_function=retrieve_result, verbose=False)

    # remove the temporary csv files
    os.remove(data_csv_path)
    os.remove(dag_csv_path)

    return dag_pruned

def cam_pruning(model_adj, train_data, test_data, opt, cutoff=0.001, verbose=False):
    """Execute CAM pruning on a given model and datasets"""
    # time0 = time.time()
    # model.eval()

    # Prepare path for saving results
    stage_name = "cam-pruning/cutoff_%.6f" % cutoff
    save_path = os.path.join(opt['exp_path'], stage_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # Check if already computed
    # if os.path.exists(os.path.join(save_path, "DAG.npy")):
    #     print(stage_name, "already computed. Loading result from disk.")
    #     return load(save_path, "model.pkl")

    # model_adj = model.adjacency.detach().cpu().numpy()

    dag_pruned = cam_pruning_(model_adj, train_data, test_data, cutoff, save_path, verbose)

    # set new adjacency matrix to model
    # model.adjacency.copy_(torch.as_tensor(dag_pruned).type(torch.Tensor))

    # evaluate on validation set
    # x, _ = test_data.sample(test_data.num_samples)
    # weights, biases, extra_params = model.get_parameters(mode="wbx")
    # nll_val = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params)).item()

    # # Compute SHD and SID metrics
    # pred_adj_ = model.adjacency.detach().cpu().numpy()
    # # train_adj_ = train_data.adjacency.detach().cpu().numpy()
    # sid = float(cdt.metrics.SID(target=train_adj_, pred=pred_adj_))
    # acc = count_accuracy(pred_adj_, B_true)
    # shd = acc['shd']
    # shd_cpdag = float(cdt.metrics.SHD_CPDAG(target=train_adj_, pred=pred_adj_))
    # # fn, fp, rev = edge_errors(pred_adj_, train_adj_)
    # del train_adj_, pred_adj_

    # metrics_callback(stage=stage_name, step=0,
    #                  metrics={"nll_val": nll_val, "sid": sid, "shd": shd, "shd_cpdag": shd_cpdag, "fn": fn, "fp": fp,
    #                           "rev": rev}, throttle=False)

    # timing = time.time() - time0

    # save
    # dump(model, save_path, 'model')
    # dump(opt, save_path, 'opt')
    # dump(timing, save_path, 'timing')
    # dump(nll_val, save_path, "nll-val", txt=True)
    # np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())

    # plot adjacency
    # plot_adjacency(model.adjacency.detach().cpu().numpy(), train_data.adjacency.detach().cpu().numpy(), save_path)

    return torch.as_tensor(dag_pruned).type(torch.Tensor)
