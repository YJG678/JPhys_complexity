"""Force-error validation metrics E_2 and E_infty.
Cell-list vs exact direct (short range); Barnes-Hut vs exact direct (long range),
across opening parameter theta and particle number N."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import json
import numpy as np
from hybrid_model import (short_range_celllist, short_range_direct_vec,
                          long_range_direct_vec, long_range_barnes_hut)

EPS = 1e-30

def E2(Fa, Fd):
    num = np.sum(np.sum((Fa - Fd)**2, axis=1))
    den = np.sum(np.sum(Fd**2, axis=1)) + EPS
    return np.sqrt(num / den)

def Einf(Fa, Fd):
    na = np.sqrt(np.sum((Fa - Fd)**2, axis=1))
    nd = np.sqrt(np.sum(Fd**2, axis=1))
    return np.max(na) / (np.max(nd) + EPS)

def clustered(N, L, seed=0):
    rng = np.random.default_rng(seed)
    n = N // 2
    a = rng.normal([0.3*L, 0.4*L], 0.05*L, (n, 2))
    b = rng.normal([0.7*L, 0.6*L], 0.05*L, (N-n, 2))
    return np.clip(np.vstack([a, b]), 0, L-1e-9)

L = 1.0
# ---- cell-list exactness ----
cell = {}
for N in [1000, 4000, 8000]:
    X = np.random.default_rng(1).random((N, 2)) * L
    Fd = short_range_direct_vec(X, L, 1.0, 0.06)
    Fc = short_range_celllist(X, L, 1.0, 0.06)
    cell[N] = {"E2": float(E2(Fc, Fd)), "Einf": float(Einf(Fc, Fd))}
    print(f"cell-list N={N:>6}: E2={cell[N]['E2']:.2e} Einf={cell[N]['Einf']:.2e}")

# ---- Barnes-Hut error vs theta (clustered, N=5000) ----
thetas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
X = clustered(5000, L, seed=2)
Fd = long_range_direct_vec(X, L, 1.0, 0.02)
bh_theta = {}
for th in thetas:
    Fb = long_range_barnes_hut(X, L, 1.0, 0.02, th)
    bh_theta[th] = {"E2": float(E2(Fb, Fd)), "Einf": float(Einf(Fb, Fd))}
    print(f"BH theta={th}: E2={bh_theta[th]['E2']:.3e} Einf={bh_theta[th]['Einf']:.3e}")
# fit slope E2 ~ theta^p
tt = np.array(thetas); e2 = np.array([bh_theta[t]["E2"] for t in thetas])
slope = np.polyfit(np.log(tt), np.log(e2), 1)[0]
print(f"BH E2 vs theta fitted slope p = {slope:.3f}  (expected ~2)")

# ---- BH error vs N at fixed theta=0.5 ----
bh_N = {}
for N in [1000, 3000, 6000]:
    Xc = clustered(N, L, seed=3)
    Fd2 = long_range_direct_vec(Xc, L, 1.0, 0.02)
    Fb2 = long_range_barnes_hut(Xc, L, 1.0, 0.02, 0.5)
    bh_N[N] = {"E2": float(E2(Fb2, Fd2)), "Einf": float(Einf(Fb2, Fd2))}
    print(f"BH N={N:>6} theta=0.5: E2={bh_N[N]['E2']:.3e}")

json.dump({"cell": cell, "bh_theta": bh_theta, "bh_slope": slope, "bh_N": bh_N,
           "thetas": thetas}, open(os.path.join(RESULTS, "forceerr.json"), "w"), indent=0)
print("saved forceerr.json")

# ---- isolated-cluster monopole error: the clean per-cell O((s/d)^2) law ----
def monopole_test(reg=1e-6):
    rng = np.random.default_rng(7)
    n, s = 40, 0.02          # compact cluster of side s
    base = np.array([0.5, 0.5])
    cluster = base + (rng.random((n, 2)) - 0.5) * s
    c = cluster.mean(0)
    ds = np.array([0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2])
    rel = []
    for d in ds:
        x = c + np.array([d, 0.0])           # target at distance d
        # exact cluster force at x (open space, no periodicity, unit weights /1)
        zz = cluster - x
        rr = np.sqrt((zz**2).sum(1))
        Fex = np.sum(zz / (rr[:, None]**2 + reg**2), axis=0)
        zc = c - x
        Fmono = n * zc / ((zc**2).sum() + reg**2)
        rel.append(np.linalg.norm(Fex - Fmono) / (np.linalg.norm(Fmono) + 1e-30))
    rel = np.array(rel)
    sd = s / ds
    slope = np.polyfit(np.log(sd), np.log(rel), 1)[0]
    return sd.tolist(), rel.tolist(), float(slope)

sd, rel, mslope = monopole_test()
print(f"isolated-cluster monopole error vs (s/d): fitted slope = {mslope:.3f} (expected 2)")
d = json.load(open(os.path.join(RESULTS, "forceerr.json")))
d["monopole"] = {"sd": sd, "rel": rel, "slope": mslope}
json.dump(d, open(os.path.join(RESULTS, "forceerr.json"), "w"), indent=0)
print("updated forceerr.json")
