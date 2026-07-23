import numpy as np
import quimb.tensor as qtn

from .graph import _neighbours


def _compute_layout(G):
    
    directed = G.is_directed()
    vertices = sorted(G.nodes(), key=lambda x: (str(type(x)), str(x)))
    neighbours_of = {v: _neighbours(G, v) for v in vertices}

    qubit_of: dict = {}
    clique_qubits: dict = {}
    q = 0
    for v in vertices:
        qs = []
        for u in neighbours_of[v]:
            qubit_of[(v, u)] = q
            qs.append(q)
            q += 1
        clique_qubits[v] = qs
    num_qubits = q

    # cascade rotation angles per clique (weighted one-hot state preparation)
    angles_of: dict = {}
    for v in vertices:
        nbrs = neighbours_of[v]
        d = len(nbrs)
        if d <= 1:
            angles_of[v] = []
            continue
        weights = np.array([G[v][u].get('weight', 1.0) for u in nbrs], dtype=float)
        a = np.sqrt(weights / weights.sum())
        thetas = []
        running = 1.0
        for k in range(d - 1):
            tk = 2.0 * np.arccos(a[k] / running)
            thetas.append(tk)
            running *= np.sin(tk / 2.0)
        angles_of[v] = thetas

    # cross-edge pairs for W_beta (once per unordered original edge)
    cross_pairs = []
    seen = set()
    for v, u in G.edges():
        key = frozenset((v, u))
        if key in seen:
            continue
        seen.add(key)
        cross_pairs.append(((v, u), (u, v)))

    return (vertices, qubit_of, clique_qubits, num_qubits, angles_of,
            cross_pairs, directed)


def build_gate_list(G) -> dict:

    (vertices, qubit_of, clique_qubits, num_qubits, angles_of,
     cross_pairs, _directed) = _compute_layout(G)
    gates: list = []

    def apply_U_alpha(v, dagger):
        qs = clique_qubits[v]
        d = len(qs)
        if d <= 1:
            return
        thetas = angles_of[v]
        if not dagger:
            for qb in qs:
                gates.append(('X', qb, None, None))
            for k, tk in enumerate(thetas):
                if k == 0:
                    gates.append(('RY', qs[0], tk, None))
                else:
                    gates.append(('CRY', qs[k], tk, qs[k - 1]))
            for k in range(d - 1, 0, -1):
                gates.append(('CX', qs[k], None, qs[k - 1]))
            gates.append(('X', qs[0], None, None))
        else:
            gates.append(('X', qs[0], None, None))
            for k in range(1, d):
                gates.append(('CX', qs[k], None, qs[k - 1]))
            for k in reversed(range(len(thetas))):
                tk = thetas[k]
                if k == 0:
                    gates.append(('RY', qs[0], -tk, None))
                else:
                    gates.append(('CRY', qs[k], -tk, qs[k - 1]))
            for qb in qs:
                gates.append(('X', qb, None, None))

    def apply_clique_reflection(v):
        qs = clique_qubits[v]
        d = len(qs)
        if d == 0:
            return
        if d == 1:
            gates.append(('Z', qs[0], None, None))
            return
        target = qs[-1]
        controls = tuple(qs[:-1])
        gates.append(('H', target, None, None))
        gates.append(('MCX', target, None, controls))
        gates.append(('H', target, None, None))

    # W_alpha = U_alpha . C^(k-1)Z . U_alpha^dag
    gates.append(('BARRIER', None, None, 'U_a_dag'))
    for v in vertices:
        apply_U_alpha(v, dagger=True)
    gates.append(('BARRIER', None, None, 'C^kZ'))
    for v in vertices:
        apply_clique_reflection(v)
    gates.append(('BARRIER', None, None, 'U_a'))
    for v in vertices:
        apply_U_alpha(v, dagger=False)

    # W_beta: a SWAP on each directed-edge pair.  On the one-hot subspace this
    # equals the beta reflection I - 2 P_beta up to a global -1 per step.
    gates.append(('BARRIER', None, None, 'W_beta'))
    for (a, b) in cross_pairs:
        gates.append(('SWAP', qubit_of[a], None, qubit_of[b]))

    # big-endian statevector indices (quimb convention: qubit 0 is MSB)
    index_of = {edge: 1 << (num_qubits - 1 - qb) for edge, qb in qubit_of.items()}
    label_of = {idx: edge for edge, idx in index_of.items()}

    return {
        'gates':         gates,
        'qubit_of':      qubit_of,
        'clique_qubits': clique_qubits,
        'num_qubits':    num_qubits,
        'index_of':      index_of,
        'label_of':      label_of,
    }


def apply_gates(circ, gates: list):
    for name, target, param, aux in gates:
        if name == 'X':
            circ.apply_gate('X', target)
        elif name == 'Z':
            circ.apply_gate('Z', target)
        elif name == 'H':
            circ.apply_gate('H', target)
        elif name == 'RY':
            circ.apply_gate('RY', param, target)
        elif name == 'CRY':
            circ.apply_gate('CRY', param, aux, target)
        elif name == 'CX':
            circ.apply_gate('CX', aux, target)
        elif name == 'MCX':
            circ.apply_gate('X', target, controls=list(aux))
        elif name == 'SWAP':
            circ.apply_gate('SWAP', target, aux)
        elif name == 'BARRIER':
            continue   #drawing only
        else:
            raise ValueError(f'unknown gate {name}')


def build_walk_circuit(G, mps: bool = True) -> dict:

    layout = build_gate_list(G)
    CircuitCls = qtn.CircuitMPS if mps else qtn.Circuit
    circ = CircuitCls(layout['num_qubits'])
    apply_gates(circ, layout['gates'])
    return {'circuit': circ, **layout}