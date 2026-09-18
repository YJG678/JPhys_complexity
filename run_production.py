"""Production runs on the COUPLED model: retuned ablation + uptake (beta) sweep.

Produces results/production.json, which make_fig_uptake.py turns into the
uptake figure and which supplies Tables 4 and 5 of the paper.

Usage:   python run_production.py            (full: ~12 min)
         python run_production.py --quick    (reduced: ~2 min, for smoke-testing)
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

import numpy as np
from coupled_model import (CoupledSim, R_c, R_unif, aggregation_index, min_image)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="reduced N/steps/seeds")
args = ap.parse_args()

if args.quick:
    N, NSTEP, SEEDS1, SEEDS2 = 150, 400, (1, 2), (1, 2)
    EPS_L_GRID, BETA_GRID = (0.05, 0.10, 0.20), (0.1, 0.5, 2.0)
else:
    N, NSTEP, SEEDS1, SEEDS2 = 300, 1200, (1, 2, 3), (1, 2, 3, 4)
    EPS_L_GRID, BETA_GRID = (0.05, 0.10, 0.15, 0.20, 0.30), (0.1, 0.25, 0.5, 1.0, 2.0)

Ru = R_unif(1.0)
BASE = dict(M=64, dt=0.02, v0=0.12, tau_m=0.25, lam0=5.0, kappa_s=20.0,
            r_c=0.06, reg=0.03, D_rho=0.02, alpha=0.5, S=1.0,
            source="gaussian", source_width=0.12)


def clusters(X, L, eps_db=0.12, min_pts=5):
    """Periodic DBSCAN-style clustering (union-find on the eps-graph)."""
    n = X.shape[0]
    z = min_image(X[:, None, :] - X[None, :, :], L)
    d = np.sqrt((z * z).sum(2))
    core = (d <= eps_db).sum(1) - 1 >= min_pts
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.where((d <= eps_db) & core[:, None] & core[None, :])
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    lab = np.array([find(a) if core[a] else -1 for a in range(n)])
    sizes = {}
    for a in range(n):
        if lab[a] >= 0:
            sizes[lab[a]] = sizes.get(lab[a], 0) + 1
    if not sizes:
        return 0, 0.0, 0.0
    szs = sorted(sizes.values(), reverse=True)
    biggest = max(sizes, key=sizes.get)
    mem = X[lab == biggest]
    dd = min_image(mem - mem[0], L)
    lagg = float(np.sqrt((((dd - dd.mean(0)) ** 2).sum(1)).mean()))
    return len(szs), szs[0] / n, lagg


def one(seed, eps_l, beta, eps_s=12.0, lam1=4.5):
    s = CoupledSim(N=N, seed=seed, eps_s=eps_s, eps_l=eps_l, lam1=lam1,
                   beta=beta, **BASE)
    s.run(NSTEP)
    nc, f1, lagg = clusters(s.X, s.L)
    return dict(Rc=R_c(s) / Ru, A=aggregation_index(s), Nc=nc, f1=f1,
                lagg=lagg, rho=float(s.field.rho.mean()))


def agg(rows):
    return {k: [float(np.mean([r[k] for r in rows])),
                float(np.std([r[k] for r in rows]))] for k in rows[0]}


t0 = time.time()
RES = {"config": dict(N=N, nsteps=NSTEP, dt=BASE["dt"], quick=args.quick)}

print("STAGE 1  retune A_ell  (beta = 1.0)", flush=True)
print(f"{'A_ell':>7} | {'Rhat_c':>15} | {'A':>15} | {'Nc':>5} | {'f1':>5}", flush=True)
stage1 = {}
for el in EPS_L_GRID:
    a = agg([one(sd, el, 1.0) for sd in SEEDS1])
    stage1[el] = a
    print(f"{el:>7.2f} | {a['Rc'][0]:>7.3f}+-{a['Rc'][1]:<6.3f} | "
          f"{a['A'][0]:>7.2f}+-{a['A'][1]:<6.2f} | {a['Nc'][0]:>5.1f} | {a['f1'][0]:>5.2f}",
          flush=True)
RES["stage1_eps_l"] = {str(k): v for k, v in stage1.items()}
EL = min(stage1, key=lambda k: abs(stage1[k]["A"][0] - 10.0))
RES["eps_l_chosen"] = EL
print(f"\n  -> chosen A_ell = {EL}  (mean A closest to the previously reported ~10)\n",
      flush=True)

print(f"STAGE 2  uptake sweep at A_ell = {EL:.2f}   [the mechanism result]", flush=True)
print(f"{'beta':>7} | {'Rhat_c':>15} | {'A':>15} | {'mean rho':>9}", flush=True)
stage2 = {}
for b in BETA_GRID:
    a = agg([one(sd, EL, b) for sd in SEEDS2])
    stage2[b] = a
    print(f"{b:>7.2f} | {a['Rc'][0]:>7.3f}+-{a['Rc'][1]:<6.3f} | "
          f"{a['A'][0]:>7.2f}+-{a['A'][1]:<6.2f} | {a['rho'][0]:>9.4f}", flush=True)
RES["stage2_beta"] = {str(k): v for k, v in stage2.items()}
json.dump(RES, open(os.path.join(RESULTS, "production.json"), "w"), indent=1)

print(f"\nSTAGE 3  ablation at A_ell = {EL:.2f}", flush=True)
print(f"{'scenario':>8} | {'Rhat_c':>15} | {'A':>15} | {'Nc':>5}", flush=True)
CFG = {"full":  dict(lam1=4.5, eps_s=12.0, eps_l=EL),
       "nomem": dict(lam1=0.0, eps_s=12.0, eps_l=EL),
       "norep": dict(lam1=4.5, eps_s=0.0,  eps_l=EL),
       "noatt": dict(lam1=4.5, eps_s=12.0, eps_l=0.0)}
stage3 = {}
for name in ("full", "nomem", "norep", "noatt"):
    c = CFG[name]
    a = agg([one(sd, c["eps_l"], 1.0, eps_s=c["eps_s"], lam1=c["lam1"])
             for sd in SEEDS2])
    stage3[name] = a
    print(f"{name:>8} | {a['Rc'][0]:>7.3f}+-{a['Rc'][1]:<6.3f} | "
          f"{a['A'][0]:>7.2f}+-{a['A'][1]:<6.2f} | {a['Nc'][0]:>5.1f}", flush=True)
RES["stage3_ablation"] = stage3
json.dump(RES, open(os.path.join(RESULTS, "production.json"), "w"), indent=1)
print(f"\nDONE in {time.time()-t0:.0f}s -> results/production.json", flush=True)
