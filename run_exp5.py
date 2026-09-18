"""Experiment 5: 2D versus 3D dynamics and computational efficiency."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import time
import numpy as np
from model3d import (HybridSimND, R_c, clustering_index,
                     long_range_octree, long_range_direct3d)

# ---- matched dynamics -------------------------------------------------
N, NSTEPS, REC = 800, 800, 20
par = dict(v0=0.12, lam0=5.0, lam1=4.5, kappa=20, tau_m=0.25, dt=0.05,
           eps_s=12.0, r_c=0.06)

out = {}
for d in (2, 3):
    s = HybridSimND(N=N, d=d, seed=11, **par)
    r0 = R_c(s)
    hist = s.run(NSTEPS, record_every=REC)
    out[d] = dict(t=np.array(hist["t"]), Rc=np.array(hist["Rc"]),
                  Rc0=r0, Rcf=R_c(s), clust=clustering_index(s), X=s.X.copy())
    # migration rate: initial slope of R_c(t) over first quarter
    tt, rr = out[d]["t"], out[d]["Rc"]
    k = max(2, len(tt) // 4)
    slope = np.polyfit(tt[:k], rr[:k], 1)[0]
    out[d]["rate"] = -slope
    print(f"d={d}: Rc {r0:.3f}->{R_c(s):.3f}, clustering={out[d]['clust']:.2f}, "
          f"migration rate={-slope:.4f}/time", flush=True)

# ---- 3D octree timing (fixed number density: L ~ N^{1/3}) -------------
DENSITY3 = 400.0
eps_l, reg, theta = 1.0, 0.02, 0.5
def make3(N, seed=0):
    L = (N / DENSITY3) ** (1.0 / 3.0)
    return np.random.default_rng(seed).random((N, 3)) * L, L

tim = {"oct": {}, "direct": {}}
for N3 in [500, 1000, 2000, 4000]:
    X, L = make3(N3)
    long_range_octree(X, L, eps_l, reg, theta)
    t0 = time.time(); long_range_octree(X, L, eps_l, reg, theta)
    tim["oct"][N3] = time.time() - t0
    if N3 <= 4000:
        t0 = time.time(); long_range_direct3d(X, L, eps_l, reg)
        tim["direct"][N3] = time.time() - t0
    print(f"3D N={N3}: octree {tim['oct'][N3]*1e3:.1f} ms, "
          f"direct {tim['direct'].get(N3, float('nan'))*1e3:.1f} ms", flush=True)

Noct = np.array(sorted(tim["oct"])); Toct = np.array([tim["oct"][n] for n in Noct])
bo = np.polyfit(np.log(Noct), np.log(Toct), 1)[0]
Nd = np.array(sorted(tim["direct"])); Td = np.array([tim["direct"][n] for n in Nd])
bd = np.polyfit(np.log(Nd), np.log(Td), 1)[0]
print(f"3D octree fitted exponent {bo:.3f}; 3D direct fitted exponent {bd:.3f}", flush=True)

np.savez(os.path.join(RESULTS, "exp5.npz"),
         t2=out[2]["t"], Rc2=out[2]["Rc"], Rcf2=out[2]["Rcf"], clust2=out[2]["clust"],
         rate2=out[2]["rate"], X2=out[2]["X"],
         t3=out[3]["t"], Rc3=out[3]["Rc"], Rcf3=out[3]["Rcf"], clust3=out[3]["clust"],
         rate3=out[3]["rate"], X3=out[3]["X"],
         Noct=Noct, Toct=Toct, Nd=Nd, Td=Td, exp_oct=bo, exp_direct=bd)
import json as _json
_json.dump({"oct_exp": float(bo), "direct_exp": float(bd),
            "N_oct": [int(n) for n in Noct], "T_oct": [float(t) for t in Toct],
            "N_direct": [int(n) for n in Nd], "T_direct": [float(t) for t in Td],
            "migration_rate_2d": float(out[2]["rate"]),
            "migration_rate_3d": float(out[3]["rate"])},
           open(os.path.join(RESULTS, "exp5_results.json"), "w"), indent=1)
print("saved exp5.npz and exp5_results.json")
