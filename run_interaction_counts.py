"""Algorithmic complexity of Barnes-Hut via accepted-interaction counts.
Produces exp3_counts.npz used by make_fig_highN.py for the O(N log N) evidence
(total interactions ~ N^1.12; per-particle interactions ~ 27 ln N)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import numpy as np
from bh_numba import build_tree, count_interactions

DENSITY = 400.0
THETA = 0.5

def make(N, seed=0):
    L = np.sqrt(N / DENSITY)
    return np.ascontiguousarray(np.random.default_rng(seed).random((N, 2)) * L), L

# JIT warm-up
Xw, Lw = make(2000)
count_interactions(Xw, Lw, THETA, *build_tree(Xw, Lw, 4 * 2000 + 64, 28))

rows = []
for N in [1000, 3000, 10000, 30000, 100000, 300000, 1000000]:
    X, L = make(N)
    tree = build_tree(X, L, 4 * N + 64, 28)
    c = count_interactions(X, L, THETA, *tree)
    tot = int(c.sum()); per = tot / N
    rows.append((N, tot, per))
    print(f"N={N:>8}: total interactions={tot:>12}, per particle={per:7.1f}", flush=True)

Ns = np.array([r[0] for r in rows])
tot = np.array([r[1] for r in rows])
per = np.array([r[2] for r in rows])
mask = Ns >= 10000
b = np.polyfit(np.log(Ns[mask]), np.log(tot[mask]), 1)[0]
A = np.polyfit(np.log(Ns), per, 1)
print(f"total-interaction exponent (N>=1e4) = {b:.3f}   (N logN ref ~ 1.09)")
print(f"per-particle interactions ~ {A[0]:.1f} ln N + {A[1]:.1f}")

np.savez(os.path.join(RESULTS, "exp3_counts.npz"), Ns=Ns, tot=tot, per=per, exp=b)
print("saved exp3_counts.npz")
