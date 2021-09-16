#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.qaoa as qaoa


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    dev = qml.device('default.qubit', wires=NODES)
    cost, mixer = qaoa.max_independent_set(graph)
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost)
        qaoa.mixer_layer(alpha, mixer)
    def quant_func(params):
        qml.layer(qaoa_layer, 10, params[0], params[1])
        return qml.probs(wires=graph.nodes)
    circuit = qml.QNode(quant_func, dev)
    probs = circuit(params)
    best_basis = np.argmax(probs)

    for i in range(NODES):
        exp = 2**(NODES-1-i)
        if (best_basis >= exp):
            max_ind_set.append(i)
            best_basis -= exp
        
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = '{"directed": false, "multigraph": false, "graph": [], "nodes": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}], "adjacency": [[{"id": 2}, {"id": 3}], [{"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}], [{"id": 0}, {"id": 1}], [{"id": 0}, {"id": 1}, {"id": 5}], [{"id": 1}, {"id": 5}], [{"id": 1}, {"id": 3}, {"id": 4}]], "params": [[0.007251195968545696, 0.0008594067983382697, 6.757788685303326e-05, -0.002960663108058259, -3.990107834181506e-05, 0.0008459551312108134, -0.001277886948034289, 0.0013917849664037338, 0.0050844175570855495, 0.0004046899603893801], [0.056410689341680226, 0.07332551003086096, 0.11573770360937545, 0.1471947383136579, 0.16520379259436668, 0.16107577283133256, 0.13346398582028385, 0.08533445938002224, 0.06505979609496622, 0.047259251575614024]]}'
    #sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
