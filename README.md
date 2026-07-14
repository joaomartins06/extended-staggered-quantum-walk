# Extended Staggered Quantum Walk

Source code accompanying the paper *"..."*.

## Structure

    sqw/                  # Core library
      graph.py            # Clique insertion, H-graph construction
      operators.py        # R_alpha, R_beta, W (NumPy)
      circuit.py          # Qiskit circuit implementation
      walk.py             # Statevector evolution
      optimizer.py        # Parameterized walk U = exp(iθ1 W1) exp(iθ2 W2)
      utils.py            # Visualisation helpers

    applications/
      community_detection.py
      qwalkvec.py

    notebooks/
      staggered_networkX.ipynb
      staggered_qiskit.ipynb
      QWalkVec.ipynb
      community_detection.ipynb

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT