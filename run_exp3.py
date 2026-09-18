"""Experiment 3: algorithmic scaling (split into parts, saves incrementally)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import sys, time, json, os
import numpy as np
from hybrid_model import (short_range_direct_vec, short_range_celllist,
                          long_range_barnes_hut)

PART = sys.argv[1] if len(sys.argv) > 1 else "all"
DENSITY = 400.0
r_c = 0.06
eps_s, eps_l, reg, theta = 1.0, 1.0, 0.02, 0.5
RES = os.path.join(RESULTS, "exp3_results.json")

def load():
    return json.load(open(RES)) if os.path.exists(RES) else {"direct":{}, "cell":{}, "bh":{}, "hybrid":{}}
def save(r):
    json.dump(r, open(RES, "w"))

def timeit(fn, X, L, reps=3):
    fn(X, L)
    t0 = time.time()
    for _ in range(reps):
        fn(X, L)
    return (time.time() - t0) / reps

def make(N, seed=0):
    L = np.sqrt(N / DENSITY)
    return np.random.default_rng(seed).random((N, 2)) * L, L

r = load()

if PART in ("direct", "all"):
    for N in [500, 1000, 2000, 4000, 8000]:
        X, L = make(N)
        r["direct"][str(N)] = timeit(lambda X, L: short_range_direct_vec(X, L, eps_s, r_c), X, L)
        print(f"direct  N={N:>7}: {r['direct'][str(N)]*1e3:8.2f} ms", flush=True); save(r)

if PART in ("cell", "all"):
    for N in [1000, 4000, 16000, 64000, 200000]:
        X, L = make(N)
        reps = 3 if N <= 64000 else 1
        r["cell"][str(N)] = timeit(lambda X, L: short_range_celllist(X, L, eps_s, r_c), X, L, reps)
        print(f"cell    N={N:>7}: {r['cell'][str(N)]*1e3:8.2f} ms", flush=True); save(r)

if PART in ("bh", "all"):
    for N in [1000, 2000, 4000, 8000]:
        X, L = make(N)
        r["bh"][str(N)] = timeit(lambda X, L: long_range_barnes_hut(X, L, eps_l, reg, theta), X, L, 1)
        print(f"bh      N={N:>7}: {r['bh'][str(N)]*1e3:8.2f} ms", flush=True); save(r)

if PART in ("hybrid", "all"):
    for N in [1000, 2000, 4000, 8000]:
        X, L = make(N)
        def hyb(X, L):
            short_range_celllist(X, L, eps_s, r_c)
            long_range_barnes_hut(X, L, eps_l, reg, theta)
        r["hybrid"][str(N)] = timeit(hyb, X, L, 1)
        print(f"hybrid  N={N:>7}: {r['hybrid'][str(N)]*1e3:8.2f} ms", flush=True); save(r)
# ---- fitted exponents, so the numbers quoted in Table 6 are in the file ----
def _fit(dd):
    if not dd or len(dd) < 2:
        return None
    N = np.array(sorted(int(k) for k in dd), float)
    T = np.array([dd[str(int(n))] for n in N], float)
    return float(np.polyfit(np.log(N), np.log(T), 1)[0])

fits = {k: _fit(r.get(k, {})) for k in ("direct", "cell", "bh", "hybrid")}
# N log N reference exponent over the same range as the tree timings
if r.get("bh"):
    Nb = np.array(sorted(int(k) for k in r["bh"]), float)
    fits["nlogn"] = float(np.polyfit(np.log(Nb), np.log(Nb * np.log(Nb)), 1)[0])
# accepted-interaction exponent, if run_interaction_counts.py has been run
cpath = os.path.join(RESULTS, "exp3_counts.npz")
if os.path.exists(cpath):
    z = np.load(cpath)
    key_n = "Ns" if "Ns" in z else ("N" if "N" in z else None)
    key_c = "tot" if "tot" in z else ("total" if "total" in z else None)
    if key_n and key_c:
        Nc, Cc = np.asarray(z[key_n], float), np.asarray(z[key_c], float)
        if Nc.size >= 2:
            fits["counts"] = float(np.polyfit(np.log(Nc), np.log(Cc), 1)[0])
r["fits"] = fits
save(r)
print("fitted exponents:", {k: (None if v is None else round(v, 3))
                            for k, v in fits.items()}, flush=True)
print("part", PART, "done", flush=True)
