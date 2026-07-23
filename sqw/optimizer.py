import numpy as np
import networkx as nx

from .circuit import _compute_layout


_W_BETA_LOCAL = np.array([[1, 0, 0, 0],
                          [0, 0, -1, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=complex)


def w_beta_gate(theta: float) -> np.ndarray:
    #this could be done using the same apporach as W_alpha, but for the code I am just going to do like this
    return (np.cos(theta) * np.eye(4, dtype=complex)
            - 1j * np.sin(theta) * _W_BETA_LOCAL)


def build_parameterized_gate_list(G: nx.Graph, theta1: float, theta2: float) -> dict:

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
            gates.append(('P', qs[0], 2.0 * theta1, None))
            return
        target = qs[-1]
        controls = tuple(qs[:-1])
        gates.append(('MCP', target, 2.0 * theta1, controls))

    gates.append(('BARRIER', None, None, 'U_a_dag'))
    for v in vertices:
        apply_U_alpha(v, dagger=True)
    gates.append(('BARRIER', None, None, 'C^kZ_theta'))
    for v in vertices:
        apply_clique_reflection(v)
    gates.append(('BARRIER', None, None, 'U_a'))
    for v in vertices:
        apply_U_alpha(v, dagger=False)

    gates.append(('BARRIER', None, None, 'W_beta_theta'))
    for (a, b) in cross_pairs:
        gates.append(('WBETA_T', qubit_of[a], theta2, qubit_of[b]))

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


def apply_parameterized_gates(circ, gates: list):
    for name, target, param, aux in gates:
        if name == 'X':
            circ.apply_gate('X', target)
        elif name == 'H':
            circ.apply_gate('H', target)
        elif name == 'RY':
            circ.apply_gate('RY', param, target)
        elif name == 'CRY':
            circ.apply_gate('CRY', param, aux, target)
        elif name == 'CX':
            circ.apply_gate('CX', aux, target)
        elif name == 'P':
            circ.apply_gate('PHASE', param, target)
        elif name == 'MCP':
            circ.apply_gate('PHASE', param, target, controls=list(aux))
        elif name == 'WBETA_T':
            circ.apply_gate_raw(w_beta_gate(param), [target, aux])
        elif name == 'BARRIER':
            continue   # drawing only
        else:
            raise ValueError(f'unknown gate {name}')


def build_parameterized_operator(G: nx.Graph, theta1: float, theta2: float) -> np.ndarray:

    from .graph import build_hgraph
    hg = build_hgraph(G)
    node_index = hg['node_index']
    clique_map = hg['clique_map']
    dim = len(hg['node_list'])
    I = np.eye(dim, dtype=complex)

    P_alpha = np.zeros((dim, dim), dtype=complex)
    for v, clique_nodes in clique_map.items():
        if not clique_nodes:
            continue
        w = np.array([np.sqrt(G[v][u].get('weight', 1.0)) for (_, u) in clique_nodes])
        nrm = np.linalg.norm(w)
        if nrm == 0:
            continue
        w = w / nrm
        st = np.zeros(dim, dtype=complex)
        for a, n in zip(w, clique_nodes):
            st[node_index[n]] = a
        P_alpha += np.outer(st, st.conj())
    W_alpha = I - 2.0 * P_alpha          # reflection, W_alpha^2 = I
    R_alpha = np.cos(theta1) * I - 1j * np.sin(theta1) * W_alpha

    P_beta = np.zeros((dim, dim), dtype=complex)
    seen = set()
    for v, u in G.edges():
        key = frozenset((v, u))
        if key in seen:
            continue
        seen.add(key)
        st = np.zeros(dim, dtype=complex)
        st[node_index[(v, u)]] = 1 / np.sqrt(2)
        st[node_index[(u, v)]] = 1 / np.sqrt(2)
        P_beta += np.outer(st, st.conj())
    W_beta = I - 2.0 * P_beta            # reflection, W_beta^2 = I
    R_beta = np.cos(theta2) * I - 1j * np.sin(theta2) * W_beta

    return R_beta @ R_alpha


def optimize_angles(objective, x0=None, method: str = 'COBYLA',
                    n_restarts: int = 1, seed: int = 0, **minimize_kwargs) -> dict:
    
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    history = []

    def wrapped(x):
        val = float(objective(x[0], x[1]))
        history.append((x[0], x[1], val))
        return val

    best = None
    for r in range(n_restarts):
        start = x0 if (x0 is not None and r == 0) else rng.uniform(0, 2 * np.pi, size=2)
        res = minimize(wrapped, start, method=method, **minimize_kwargs)
        if best is None or res.fun < best.fun:
            best = res

    return {
        'theta1':  best.x[0],
        'theta2':  best.x[1],
        'value':   best.fun,
        'history': history,
        'result':  best,
    }