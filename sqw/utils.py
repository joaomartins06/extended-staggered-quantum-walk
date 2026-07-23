import networkx as nx
import matplotlib.pyplot as plt


def visualise_graph(G: nx.Graph, after_clique_insertion: bool = False,
                    ax=None, seed: int = 42):
    #just a simple visualisation of the graph G, or the clique-inserted graph H
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    pos = nx.spring_layout(G, seed=seed)

    if not after_clique_insertion:
        nx.draw_networkx_nodes(G, pos, node_color='lightyellow',
                               node_size=700, edgecolors='black', ax=ax)
        nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.5, ax=ax)
        edge_labels = {(u, v): d.get('weight', '')
                       for u, v, d in G.edges(data=True) if 'weight' in d}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                         font_color='black', ax=ax)
        ax.set_title('Graph G')
        ax.axis('off')
        return

    internal_edges = [(a, b) for a, b in G.edges() if a != (b[1], b[0])]
    cross_edges    = [(a, b) for a, b in G.edges() if a == (b[1], b[0])]

    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=900, ax=ax)
    label_map = {n: f'{n[0]}_{n[1]}' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=label_map,
                            font_weight='bold', font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges,
                           edge_color='red', width=2.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges,
                           edge_color='blue', width=2.0, ax=ax)
    cross_weight_labels = {
        (a, b): G[a][b]['weight']
        for a, b in cross_edges if 'weight' in G[a][b]
    }
    if cross_weight_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=cross_weight_labels,
                                     font_color='blue', font_size=9, ax=ax)
    ax.set_title('Clique-inserted graph H\n'
                 'red = alpha tessellation   blue = beta tessellation')
    ax.axis('off')


def visualise_circuit(G: nx.Graph, output: str = 'mpl'):
    #draws the circuit using qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate
    from .circuit import build_gate_list

    layout = build_gate_list(G)
    n = layout['num_qubits']
    qc = QuantumCircuit(n)

    for name, target, param, aux in layout['gates']:
        if name == 'X':
            qc.x(target)
        elif name == 'Z':
            qc.z(target)
        elif name == 'H':
            qc.h(target)
        elif name == 'RY':
            qc.ry(param, target)
        elif name == 'CRY':
            qc.cry(param, aux, target)
        elif name == 'CX':
            qc.cx(aux, target)
        elif name == 'MCX':
            controls = list(aux)
            if len(controls) == 1:
                qc.cx(controls[0], target)
            elif len(controls) == 2:
                qc.ccx(controls[0], controls[1], target)
            else:
                qc.append(MCXGate(num_ctrl_qubits=len(controls)),
                          controls + [target])
        elif name == 'SWAP':
            qc.swap(target, aux)
        elif name == 'BARRIER':
            qc.barrier(label=aux)

    if output == 'mpl':
        qc.draw('mpl')
        plt.show()
    else:
        print(qc.draw(output))
    return qc