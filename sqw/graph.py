import networkx as nx


def _neighbours(G, v):
    #aux function to get the sorted neighbours 
    it = G.successors(v) if G.is_directed() else G.neighbors(v)
    return sorted(it, key=lambda x: (str(type(x)), str(x)))


def build_hgraph(G) -> dict:
    #creates the new graph, applying the clique insertion operator
    directed = G.is_directed()
    H = nx.Graph()
    clique_map: dict = {}
    internal_edges: list = []
    cross_edges: list = []

    for v in G.nodes():
        #each vertex is replaced by a clique of size equal to its degree
        clique_nodes = [(v, u) for u in _neighbours(G, v)]
        H.add_nodes_from(clique_nodes)
        clique_map[v] = clique_nodes
        for i in range(len(clique_nodes)):
            for j in range(i + 1, len(clique_nodes)):
                H.add_edge(clique_nodes[i], clique_nodes[j])
                internal_edges.append((clique_nodes[i], clique_nodes[j]))

    for v, u in G.edges():
        #add edges between the cliques, one for each original edge
        if directed and not (str(v) < str(u)):
            continue
        w = G[v][u].get('weight', 1.0)
        H.add_edge((v, u), (u, v), weight=w)
        cross_edges.append(((v, u), (u, v)))

    node_list = list(H.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}

    return {
        'H':              H,
        'clique_map':     clique_map,
        'node_list':      node_list,
        'node_index':     node_index,
        'internal_edges': internal_edges,
        'cross_edges':    cross_edges,
        'directed':       directed,
    }
