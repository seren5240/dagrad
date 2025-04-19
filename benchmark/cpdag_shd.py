from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
import numpy as np
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag


def shd_cpdag(W0, W1):
    def create_graph_from_W(W):
        d = W.shape[0]
        nodes = []
        for k in range(d):
            nodes.append(GraphNode(f"X{int(k)+1}"))
        dag = Dag(nodes)
        nonzero_indices = np.where(W != 0)
        for i, j in zip(*nonzero_indices):
            dag.add_directed_edge(nodes[i], nodes[j])
        return dag

    if isinstance(W0, np.ndarray):
        W0 = create_graph_from_W(W0)
        W0 = dag2cpdag(W0)
    if isinstance(W1, np.ndarray):
        W1 = create_graph_from_W(W1)
        W1 = dag2cpdag(W1)

    shd = SHD(W0, W1).get_shd()
    return shd
