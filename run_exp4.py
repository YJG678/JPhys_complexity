"""Experiment 4: mean-field convergence.
For each N, run R independent realisations of the (interaction-free) chemotactic
process, form the smoothed empirical density n^N on a grid, and measure the L1
fluctuation of individual realisations about the ensemble mean. This fluctuation
is the statistical distance of the empirical measure from the deterministic
mean-field density, and is predicted to scale as N^{-1/2}."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import time
import numpy as np
from hybrid_model import HybridSim

NBINS = 20
NSTEPS = 400
R = 16                       # realisations per N
Ns = [250, 500, 1000, 2000, 4000, 8000]
base = dict(v0=0.12, lam0=5.0, lam1=4.5, kappa=20, tau_m=0.25, dt=0.05)

def density(sim):
    H, _, _ = np.histogram2d(sim.X[:, 0], sim.X[:, 1], bins=NBINS,
                             range=[[0, 1], [0, 1]])
    return H / H.sum()

t0 = time.time()
errs = []
for N in Ns:
    dens = []
    for r in range(R):
        s = HybridSim(N=N, seed=1000 + r, **base)   # no interactions -> cheap
        s.run(NSTEPS)
        dens.append(density(s))
    dens = np.array(dens)
    mean = dens.mean(axis=0)
    # L1 distance of each realisation to the ensemble mean, averaged
    l1 = np.mean([np.sum(np.abs(d - mean)) for d in dens])
    errs.append(l1)
    print(f"N={N:>6}: mean L1 fluctuation = {l1:.4e}", flush=True)

Ns = np.array(Ns); errs = np.array(errs)
b, a = np.polyfit(np.log(Ns), np.log(errs), 1)
print(f"fitted error ~ N^{b:.3f}   (predicted -0.5)")
np.savez(os.path.join(RESULTS, "exp4.npz"), Ns=Ns, errs=errs, slope=b, const=np.exp(a))
print(f"done in {time.time()-t0:.1f}s")
