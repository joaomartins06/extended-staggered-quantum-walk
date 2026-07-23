import numpy as np
import matplotlib.pyplot as plt


def _plot_probs(trajectory_probs: dict, steps: int, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, series in trajectory_probs.items():
        ax.plot(range(steps + 1), series, marker='o', label=str(label))
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _clique_probs_from_amps(get_amp, clique_map, node_index, cliques):
    return {v: sum(abs(get_amp(n))**2 for n in clique_map[v]) for v in cliques}


def run_walk_numpy(W: np.ndarray, clique_map: dict, node_index: dict,
                   initial_state: np.ndarray, steps: int,
                   plot_cliques: bool = True, plot_nodes: bool = False,
                   items: list | None = None):
    
    #this one runs using simply numpy
    state = np.asarray(initial_state, dtype=complex).copy()
    trajectory = [state.copy()]
    for _ in range(steps):
        state = W @ state
        trajectory.append(state.copy())

    if plot_cliques:
        cliques = list(clique_map.keys()) if items is None else items
        probs = {v: [] for v in cliques}
        for s in trajectory:
            for v in cliques:
                probs[v].append(sum(abs(s[node_index[n]])**2 for n in clique_map[v]))
        _plot_probs(probs, steps, 'Probability', 'Clique probability evolution')

    if plot_nodes:
        nodes = list(node_index.keys()) if items is None else items
        probs = {n: [] for n in nodes}
        for s in trajectory:
            for n in nodes:
                probs[n].append(abs(s[node_index[n]])**2)
        labels = {n: (f"{n[0]}_{n[1]}" if isinstance(n, tuple) else str(n))
                  for n in nodes}
        _plot_probs({labels[n]: probs[n] for n in nodes}, steps,
                    'Probability', 'Node probability evolution')

    return state, trajectory


def run_walk_quimb(gates: list, num_qubits: int, index_of: dict,
                   clique_map: dict, initial_state: dict, steps: int,
                   plot_cliques: bool = True, plot_nodes: bool = False,
                   items: list | None = None):
    
    #this one runs using quimb, which is a tensor network library
    import quimb.tensor as qtn

    qubit_of_index = {idx: (num_qubits - 1 - idx.bit_length() + 1)
                      for idx in index_of.values()}

    def basis_mps(active_qubit):
        arrays = []
        for qb in range(num_qubits):
            vec = np.array([0.0, 1.0]) if qb == active_qubit else np.array([1.0, 0.0])
            arrays.append(vec.reshape(1, 1, 2) if 0 < qb < num_qubits - 1
                          else vec.reshape(1, 2))
        return qtn.MatrixProductState(arrays)

    psi0 = None
    for node, amp in initial_state.items():
        active = index_of[node].bit_length() - 1
        active_qubit = num_qubits - 1 - active
        term = basis_mps(active_qubit) * amp
        psi0 = term if psi0 is None else (psi0 + term)
    psi0.compress()

    circ = qtn.CircuitMPS(num_qubits, psi0=psi0)

    label_of = {idx: node for node, idx in index_of.items()}

    def snapshot():
        # amplitude extraction per populated edge (no full dense vector)
        out = {}
        for node, idx in index_of.items():
            bitstring = format(idx, f'0{num_qubits}b')
            out[node] = complex(circ.amplitude(bitstring))
        return out

    trajectory = [snapshot()]
    from .circuit import apply_gates
    for _ in range(steps):
        apply_gates(circ, gates)
        trajectory.append(snapshot())
    final_amps = trajectory[-1]

    if plot_cliques:
        cliques = list(clique_map.keys()) if items is None else items
        probs = {v: [] for v in cliques}
        for snap in trajectory:
            for v in cliques:
                probs[v].append(sum(abs(snap[n])**2 for n in clique_map[v]))
        _plot_probs(probs, steps, 'Probability',
                    'Clique probability evolution (quimb)')

    if plot_nodes:
        nodes = list(index_of.keys()) if items is None else items
        probs = {n: [] for n in nodes}
        for snap in trajectory:
            for n in nodes:
                probs[n].append(abs(snap[n])**2)
        labels = {n: (f"{n[0]}_{n[1]}" if isinstance(n, tuple) else str(n))
                  for n in nodes}
        _plot_probs({labels[n]: probs[n] for n in nodes}, steps,
                    'Probability', 'Directed-edge probability evolution (quimb)')

    return final_amps, trajectory
