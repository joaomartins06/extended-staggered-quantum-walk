import numpy as np
from .graph import build_hgraph


def build_walk_operator(G) -> dict:
    #get the info from the new clique insterted graph
    hgraph = build_hgraph(G)
    clique_map = hgraph['clique_map']
    node_index = hgraph['node_index']
    dim        = len(hgraph['node_list'])
    directed   = hgraph['directed']

    I = np.eye(dim)

    #W_alpha
    P_alpha = np.zeros((dim, dim))
    for v, clique_nodes in clique_map.items():
        if not clique_nodes:
            continue
        weights = np.array(
            [np.sqrt(G[v][u].get('weight', 1.0)) for (_, u) in clique_nodes],
            dtype=float,
        )
        norm = np.linalg.norm(weights)
        if norm == 0:
            continue
        weights = weights / norm
        state = np.zeros(dim)
        for amp, node in zip(weights, clique_nodes):
            state[node_index[node]] = amp
        P_alpha += np.outer(state, state)
    R_alpha = I - 2.0 * P_alpha

    #W_beta
    P_beta = np.zeros((dim, dim))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    seen = set()
    for v, u in G.edges():
        key = frozenset((v, u))
        if key in seen:
            continue
        seen.add(key)
        state = np.zeros(dim)
        state[node_index[(v, u)]] = inv_sqrt2
        state[node_index[(u, v)]] = inv_sqrt2
        P_beta += np.outer(state, state)
    R_beta = I - 2.0 * P_beta

    W = R_beta @ R_alpha

    return {**hgraph, 'W': W, 'R_alpha': R_alpha, 'R_beta': R_beta}
