"""
Community detection via the staggered quantum walk.

Implements the walk-based community detection procedure: run the walk from the
high-degree seed set, form the Cesaro-averaged edge-probability vector Pi, then
apply the classical path-weight procedure to extract communities.

The walk operator now comes from sqw.build_walk_operator (verified identical to
the previous standalone implementation on unweighted graphs).
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sqw.operators import build_walk_operator


def create_walk_operator(G):
    """
    Build the walk operator for community detection.

    Thin wrapper over sqw.build_walk_operator that also attaches `G` and `dim`
    for the downstream routines.
    """
    result = build_walk_operator(G)
    result['G'] = G
    result['dim'] = len(result['node_list'])
    return result


def _build_initial_state(walk_result, V_max):
    """Uniform superposition over the directed edges of the seed cliques V_max."""
    clique_map = walk_result['clique_map']
    node_index = walk_result['node_index']
    dim = walk_result['dim']

    init_nodes = []
    for i in V_max:
        init_nodes.extend(clique_map[i])
    psi0 = np.zeros(dim, dtype=complex)
    amp = 1.0 / np.sqrt(len(init_nodes))
    for n in init_nodes:
        psi0[node_index[n]] = amp
    return psi0


def run_walk(walk_result, V_max, epsilon=1e-4, max_steps=1000,
             min_steps=50, verbose=False):
    """
    Run the walk and return the Cesaro-averaged edge-probability vector Pi.

    Pi^T = (1/(T+1)) sum_{t=0}^{T} P(t), where P(t)_e is the probability on
    original edge e at step t. Stops when ||Pi^T - Pi^(T-1)|| < epsilon.
    """
    G = walk_result['G']
    W = walk_result['W']
    node_index = walk_result['node_index']

    edges = [(u, v) for u, v in G.edges()]
    m = len(edges)
    edge_index = {frozenset(e): k for k, e in enumerate(edges)}

    def _edge_prob_vector(state):
        P = np.zeros(m)
        amps_sq = np.abs(state) ** 2
        for k, (u, v) in enumerate(edges):
            P[k] = amps_sq[node_index[(u, v)]] + amps_sq[node_index[(v, u)]]
        return P

    psi = _build_initial_state(walk_result, V_max)

    P_sum = _edge_prob_vector(psi).copy()
    Pi_prev = P_sum.copy()
    diff_history = []

    converged = False
    T_final = 0
    for t in range(1, max_steps + 1):
        psi = W @ psi
        P_sum += _edge_prob_vector(psi)
        Pi_curr = P_sum / (t + 1)
        diff = np.linalg.norm(Pi_curr - Pi_prev)
        diff_history.append(diff)
        if verbose and (t < 10 or t % 100 == 0):
            print("t=%5d   ||Pi^T - Pi^(T-1)|| = %.6e" % (t, diff))
        if diff < epsilon and t >= min_steps:
            T_final = t
            converged = True
            Pi_prev = Pi_curr
            break
        Pi_prev = Pi_curr
        T_final = t

    return {
        'Pi': Pi_prev,
        'edges': edges,
        'edge_index': edge_index,
        'steps_run': T_final,
        'converged': converged,
        'diff_history': diff_history,
        'psi_final': psi,
    }


def path_weight(G, Pi, edge_index, u, v):
    """Path weight between u and v along the shortest path, scaled by deg(v)."""
    if u == v:
        return 0.0
    try:
        path = nx.shortest_path(G, source=u, target=v)
    except nx.NetworkXNoPath:
        return 0.0
    prod = 1.0
    for a, b in zip(path[:-1], path[1:]):
        prod *= Pi[edge_index[frozenset({a, b})]]
    return prod / G.degree(v)


def procedure(G, Pi, edge_index, q, refine=True):
    """Classical community-extraction procedure using the edge-probability Pi."""
    remaining = set(G.nodes())
    communities = {}

    def sort_key(x):
        return (G.degree(x), str(type(x)), str(x))

    for _ in range(G.number_of_nodes() + 1):
        if not remaining:
            break

        v_i = max(remaining, key=sort_key)
        Nbd = set(G.neighbors(v_i)) & remaining

        candidates = (remaining - Nbd) - {v_i}
        for v_j in candidates:
            Nj = set(G.neighbors(v_j))
            if len(Nj) == 0:
                continue
            if len(Nj & Nbd) >= len(Nj) / 2:
                Nbd.add(v_j)

        C = {v_i}
        for v_j in Nbd:
            if path_weight(G, Pi, edge_index, v_i, v_j) < q:
                C.add(v_j)

        if refine:
            to_remove = set()
            for v_j in C:
                if v_j == v_i:
                    continue
                internal_deg = sum(1 for nb in G.neighbors(v_j) if nb in C)
                if internal_deg <= G.degree(v_j) / 2:
                    to_remove.add(v_j)
            C -= to_remove

        communities[v_i] = sorted(C, key=lambda x: (str(type(x)), str(x)))
        remaining -= C

    return communities


def detect_communities(G, V_max, q=None, epsilon=1e-4, max_steps=5000,
                       min_steps=50, refine=True, verbose=False):
    """Full pipeline: build operator, run walk, extract communities."""
    if q is None:
        q = 1.0 / max(G.number_of_edges(), 1)

    walk_operator = create_walk_operator(G)
    walk_run = run_walk(walk_operator, V_max, epsilon=epsilon,
                        max_steps=max_steps, min_steps=min_steps, verbose=verbose)

    Pi = walk_run['Pi']
    edge_index = walk_run['edge_index']
    comms = procedure(G, Pi, edge_index, q, refine=refine)

    return {
        'communities': comms,
        'Pi': Pi,
        'edges': walk_run['edges'],
        'edge_index': edge_index,
        'steps_run': walk_run['steps_run'],
        'converged': walk_run['converged'],
        'q': q,
        'walk_operator': walk_operator,
        'walk_run': walk_run,
    }


def visualize_communities(G, result, seed=42, show_edge_probs=True):
    """Draw the detected communities on G, optionally with the Pi histogram."""
    comms = result['communities']
    Pi = result['Pi']
    edges = result['edges']

    non_singleton = [rep for rep, m in comms.items() if len(m) > 1]
    cmap = plt.get_cmap('tab20')
    rep_color = {rep: cmap(i % 20) for i, rep in enumerate(non_singleton)}

    node_color = []
    for v in G.nodes():
        owner = None
        for rep, members in comms.items():
            if v in members:
                owner = rep
                break
        node_color.append(rep_color.get(owner, (0.7, 0.7, 0.7, 1.0)))

    pos = nx.spring_layout(G, seed=seed)

    n_panels = 1 + int(show_edge_probs)
    fig, axes = plt.subplots(1, n_panels, figsize=(16, 6))
    if n_panels == 1:
        axes = [axes]
    ax_iter = iter(axes)

    ax = next(ax_iter)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, edge_color='lightgray')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                           node_size=350, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title("Communities on G")
    ax.axis('off')

    if show_edge_probs:
        ax = next(ax_iter)
        order = np.argsort(-Pi)
        labels = ["%s-%s" % (edges[i][0], edges[i][1]) for i in order]
        ax.bar(range(len(Pi)), Pi[order], color='steelblue')
        ax.set_xticks(range(len(Pi)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("p(e)")
        ax.set_title("Edge-probability vector $\\Pi$  (T = %d, %s)"
                     % (result['steps_run'],
                        'converged' if result['converged'] else 'max_steps reached'))

    plt.tight_layout()
    plt.show()


def vertices_of_max_degree(G):
    """Vertices attaining the maximum degree."""
    if G.number_of_nodes() == 0:
        return []
    d_max = max(dict(G.degree()).values())
    return [v for v, d in G.degree() if d == d_max]


def top_k_degree_vertices(G, k):
    """Top-k vertices by degree."""
    ranked = sorted(G.nodes(), key=lambda v: (-G.degree(v), str(type(v)), str(v)))
    return ranked[:k]


def modularity(G, communities):
    """Newman modularity of a community partition (dict rep -> members)."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    Q = 0.0
    for members in communities.values():
        S = set(members)
        intra_edges = sum(1 for u, v in G.edges() if u in S and v in S)
        sum_deg = sum(deg[v] for v in S)
        Q += (2.0 * intra_edges) / two_m - (sum_deg / two_m) ** 2
    return Q
