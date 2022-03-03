# used for baseline (BL) graph coarsen method

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_networkx

from sparsenet.util.util import summary, fix_seed, random_pygeo_graph, timefunc

INFINITY = 1e8


@timefunc
def sample_N2Nlandmarks(G, N, weight_key='edge_weight', reproducible=True):
    '''
    Node to nearest landmarks sampling.
    Selected a number of landmarks, then every node is collapsed to its nearest landmark
    :param G: The input networkx Graph or pygeo graph. Required to be CONNECTED. The input graph is by default
            DIRECTED.
    :param N: Number of nodes (to be sampled) in the sampled graph.
    :param weight_key: The key name(in the dictionary) for the weight information.
    :return: The sampled graph G_prime, and the correspondence dictionary Assignment. The sampled graph is relabeled
    to (0 - N-1). The assigment is the a dict where key is 0-num_nodes_sml and value is a set
    '''
    if reproducible: fix_seed()

    if isinstance(G, torch_geometric.data.data.Data):
        G = to_networkx(G, edge_attrs=[weight_key])

    assert (nx.is_directed(G) and nx.is_strongly_connected(
        G)), f'Input graph must be connected. {nx.number_strongly_connected_components(G)}' \
        ' components detected, with sizes {[len(c) for c in nx.strongly_connected_components(G)]}'
    V_length = G.number_of_nodes()
    assert (V_length >= N), f'graph has fewer nodes than input sample size {N}'
    V = list(G.nodes)
    assert (isinstance(V[0], int)), 'the node id should be integers'
    landmarks = [V[i] for i in np.random.choice(V_length, N, replace=False).tolist()]
    nearest_neighbor = {x: x for x in V}
    shortest_path_distance = {x: INFINITY for x in V}
    for landmark in landmarks:
        shortest_path_lengths = nx.single_source_shortest_path_length(G, landmark)
        for key, value in shortest_path_lengths.items():
            if value < shortest_path_distance[key]:
                shortest_path_distance[key] = value
                nearest_neighbor[key] = landmark

    # new ids for those landmarks are 0-N-1 in G', build a new sparsified graph G' here
    G_prime = nx.Graph()
    G_prime.add_nodes_from([i for i in range(N)])
    Assignment, map_landmarkGid2Gpid = {}, {}
    for i, id in enumerate(landmarks):
        map_landmarkGid2Gpid[id] = i
    for key, value in nearest_neighbor.items():
        id = map_landmarkGid2Gpid[value]
        Assignment[id] = [key] if id not in Assignment else Assignment[id] + [key]
    for key, value in Assignment.items():
        Assignment[key] = set(value)

    # build edge in the sparsified graph
    g_prime_edges = {}
    for u, v, feature in G.edges.data():
        i, j, weight = map_landmarkGid2Gpid[nearest_neighbor[u]], map_landmarkGid2Gpid[
            nearest_neighbor[v]], feature.get(weight_key, 1)
        if i != j:
            if i > j:
                i, j = j, i
            g_prime_edges[(i, j)] = weight if (i, j) not in g_prime_edges else g_prime_edges[(i, j)] + weight

    # divided by 2 to make sure in the limit (no compression), the resulting graph is the same as original graph
    g_prime_edges = [(i, j, weight / 2.0) for (i, j), weight in g_prime_edges.items()]
    G_prime.add_weighted_edges_from(g_prime_edges, weight=weight_key)
    # todo: shall we make G_prime undirected?
    return G_prime, Assignment


if __name__ == '__main__':
    fix_seed()
    n_node, n_edge, n_sample = 320, 5000, 100
    nfeat_dim = 42
    efeat_dim = 20
    G = random_pygeo_graph(n_node, nfeat_dim, n_edge, efeat_dim, device='cpu')
    G.edge_weight = torch.rand(G.edge_index.size(1), device=G.edge_index.device)
    summary(G, 'G')

    G_prime, Assignment = sample_N2Nlandmarks(G, n_sample, weight_key='edge_weight')
    print(nx.info(G_prime))
