"""
QWalkVec node embedding via the staggered quantum walk.

Reproduces the QWalkVec construction (Sakamoto et al.) on top of the sqw
directed walk operator. For each source vertex v0 the graph is reweighted with
node2vec-style p/q biases relative to v0's BFS distances, a directed walk
operator is built, and the per-vertex probability trajectory is accumulated:

    Phi = sum_{v0} trajectory(v0)   ->   (N, steps) embedding tensor

The directional reweighting (w_ij may differ from w_ji) is exactly what the
DiGraph support in sqw.build_walk_operator provides.
"""

import numpy as np
import networkx as nx

from sqw.operators import build_walk_operator


def compute_qwalkvec_weights(G: nx.Graph, v0, w_p: float, w_q: float) -> nx.DiGraph:
    """
    Build the directed, reweighted graph for source vertex v0.

    Weight of directed edge (i, j) depends on BFS distances l(i), l(j) from v0:
        l_i == 0            -> 1
        l_i  > l_j          -> 1/w_p   (stepping back toward v0)
        l_i <= l_j          -> 1/w_q   (same level or outward)
    """
    dist = nx.single_source_shortest_path_length(G, v0)
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(G.nodes())

    for u, v in G.edges():
        for i, j in [(u, v), (v, u)]:
            l_i, l_j = dist[i], dist[j]
            if l_i == 0:
                w = 1.0
            elif l_i > l_j:
                w = 1.0 / w_p
            else:
                w = 1.0 / w_q
            G_dir.add_edge(i, j, weight=w)

    return G_dir


def _uniform_alpha_state(walk_result: dict, G_dir: nx.DiGraph) -> np.ndarray:
    """
    Uniform superposition over the alpha states of every clique:
        psi0 = 1/sqrt(N) sum_i |alpha_i>
        |alpha_i> = 1/sqrt(W_i) sum_{u} sqrt(w_iu) |(i,u)>
    """
    node_index = walk_result['node_index']
    clique_map = walk_result['clique_map']
    dim = len(walk_result['node_list'])
    N = len(clique_map)

    psi0 = np.zeros(dim, dtype=complex)
    for i_node, clique_nodes in clique_map.items():
        if not clique_nodes:
            continue
        ws = np.array([G_dir[i_node][u].get('weight', 1.0)
                       for (_, u) in clique_nodes], dtype=float)
        W_i = ws.sum()
        if W_i == 0:
            continue
        amps = np.sqrt(ws / W_i)
        for amp, n in zip(amps, clique_nodes):
            psi0[node_index[n]] = amp
    psi0 /= np.sqrt(N)
    return psi0


def compute_node_probs(walk_result: dict, initial_state: np.ndarray,
                       steps: int) -> np.ndarray:
    """
    Per-vertex probability trajectory (N, steps): at each step the vertex
    probability is the summed probability of the directed edges in its clique.
    """
    W = walk_result['W']
    clique_map = walk_result['clique_map']
    node_index = walk_result['node_index']
    vertices = list(clique_map.keys())
    N = len(vertices)

    trajectory = np.zeros((N, steps), dtype=float)
    state = np.asarray(initial_state, dtype=complex).copy()
    for k in range(steps):
        state = W @ state
        for i, v in enumerate(vertices):
            trajectory[i, k] = sum(abs(state[node_index[n]])**2
                                   for n in clique_map[v])
    return trajectory


def qwalkvec_embedding(G: nx.Graph, steps: int, w_p: float, w_q: float) -> np.ndarray:
    """
    Compute the QWalkVec embedding tensor Phi of shape (N, steps).

    Parameters
    ----------
    G     : undirected graph
    steps : number of walk steps
    w_p   : return parameter (node2vec-style)
    w_q   : in-out parameter

    Returns
    -------
    Phi : (N, steps) float array, Phi = sum over source vertices of the
          per-vertex probability trajectory.
    """
    vertices = sorted(G.nodes(), key=lambda x: (str(type(x)), str(x)))
    N = len(vertices)
    Phi = np.zeros((N, steps), dtype=float)

    for v0 in vertices:
        G_dir = compute_qwalkvec_weights(G, v0, w_p, w_q)
        wr = build_walk_operator(G_dir)
        psi0 = _uniform_alpha_state(wr, G_dir)
        Phi += compute_node_probs(wr, psi0, steps)

    return Phi
