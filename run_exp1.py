"""Experiment 1: validation against the coarse chemotaxis model.
Four scenarios on prescribed nutrient rho(x)=1/(1+|x-C|). Saves metrics+snapshot."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import sys, time
import numpy as np
from hybrid_model import HybridSim, R_c, aggregation_index, kl_to_nutrient

SCEN_ARG = sys.argv[1] if len(sys.argv) > 1 else "all"
N = 800
NSTEPS = 1000
REC = 25
base = dict(N=N, v0=0.12, lam0=5.0, tau_m=0.25, kappa=20, dt=0.05, seed=7)

cfg = {
    "rw":     dict(lam1=0.0, eps_s=0.0, eps_l=0.0),                    # pure random walk
    "chemo":  dict(lam1=4.5, eps_s=0.0, eps_l=0.0),                    # memory chemotaxis
    "rep":    dict(lam1=4.5, eps_s=12.0, r_c=0.06, eps_l=0.0),         # + short-range repulsion
    "full":   dict(lam1=4.5, eps_s=12.0, r_c=0.06, eps_l=0.02, reg=0.03),  # + long-range attraction
}

SCENARIOS = list(cfg) if SCEN_ARG == "all" else [SCEN_ARG]

metrics = {"Rc": R_c, "A": aggregation_index, "KL": kl_to_nutrient}

for SCEN in SCENARIOS:
    sim = HybridSim(**base, **cfg[SCEN],
                    short_backend="direct_vec", long_backend="direct_vec")
    t0 = time.time()
    hist = sim.run(NSTEPS, record_every=REC, metrics=metrics)
    np.savez(os.path.join(RESULTS, f"exp1_{SCEN}.npz"),
             t=np.array(hist["t"]), Rc=np.array(hist["Rc"]),
             A=np.array(hist["A"]), KL=np.array(hist["KL"]),
             X=sim.X, C=sim.C)
    print(f"[{SCEN}] done in {time.time()-t0:.1f}s | "
          f"Rc {hist['Rc'][0]:.3f}->{hist['Rc'][-1]:.3f} | "
          f"A {hist['A'][0]:.2f}->{hist['A'][-1]:.2f} | "
          f"KL {hist['KL'][0]:.3f}->{hist['KL'][-1]:.3f}", flush=True)
