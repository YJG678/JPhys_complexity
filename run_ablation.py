"""Ablation study in the (strong-chemotaxis) regime of Experiment 1."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import json
import numpy as np
from hybrid_model import HybridSim, R_c, aggregation_index

BASE = dict(N=500, L=1.0, dt=0.05, v0=0.12, lam0=5.0, tau_m=0.25, kappa=20.0,
            mu_p=1.0, r_c=0.06, reg=0.03,
            short_backend="celllist", long_backend="direct_vec")
variants = {
    "full":           dict(lam1=4.5, eps_s=12.0, eps_l=0.02),
    "no_memory":      dict(lam1=0.0, eps_s=12.0, eps_l=0.02),
    "no_long_range":  dict(lam1=4.5, eps_s=12.0, eps_l=0.0),
    "no_short_range": dict(lam1=4.5, eps_s=0.0,  eps_l=0.02),
}
NSTEPS, M = 450, 6
out = {}
for name, ov in variants.items():
    rc, ag = [], []
    for s in range(M):
        sim = HybridSim(seed=300 + s, **{**BASE, **ov})
        sim.run(NSTEPS)
        rc.append(R_c(sim)); ag.append(aggregation_index(sim))
    out[name] = {"R_c": float(np.mean(rc)), "R_c_sd": float(np.std(rc)),
                 "agg": float(np.mean(ag)), "agg_sd": float(np.std(ag))}
    print(f"{name:16s}: R_c={out[name]['R_c']:.3f}+-{out[name]['R_c_sd']:.3f}  "
          f"agg={out[name]['agg']:.2f}+-{out[name]['agg_sd']:.2f}", flush=True)
json.dump({"variants": out, "NSTEPS": NSTEPS, "M": M, "N": BASE["N"]},
          open(os.path.join(RESULTS, "ablation.json"), "w"), indent=0)
print("saved ablation.json")
