"""Time-step convergence of the splitting scheme, measured on the DETERMINISTIC
flow. The stochastic tumble probability 1-exp(-lambda*dt) is exact in law, so the
time-integration error resides entirely in the deterministic sub-steps
(self-propulsion transport, regularized forces, exact memory relaxation). We
therefore disable tumbling (lam0=lam1=0), integrate the same initial condition
with dt, dt/2, dt/4, dt/8, and measure the terminal periodic-distance difference
||X_dt(T)-X_{dt/2}(T)||_2. Explicit-Euler transport gives first-order (O(dt))."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import json
import numpy as np
from hybrid_model import HybridSim, min_image

L = 1.0
def terminal_positions(dt, N, T, seed=0):
    nsteps = int(round(T / dt))
    sim = HybridSim(N, L=L, seed=seed, dt=dt,
                    v0=0.08, tau_m=0.3, lam0=0.0, lam1=0.0, kappa=8.0,  # no tumbling
                    mu_p=1.0, eps_s=8.0, r_c=0.06, eps_l=0.02, reg=0.03,
                    short_backend="celllist", long_backend="direct_vec")
    sim.run(nsteps)
    return sim.X.copy()

def pdist(A, B):
    d = min_image(A - B, L)
    return np.sqrt(np.sum(d**2) / A.shape[0])   # RMS periodic displacement

N, T, seed = 800, 2.0, 5
dts = [0.08, 0.04, 0.02, 0.01, 0.005]
X = {dt: terminal_positions(dt, N, T, seed) for dt in dts}
for dt in dts:
    print(f"computed X(T) for dt={dt}", flush=True)

rows = []
for i in range(len(dts) - 1):
    e = pdist(X[dts[i]], X[dts[i + 1]])
    rows.append((dts[i], e))
    print(f"||X_dt - X_dt/2|| (dt={dts[i]}) = {e:.4e}")
d = np.array([r[0] for r in rows]); e = np.array([r[1] for r in rows])
rate = np.polyfit(np.log(d), np.log(e), 1)[0]
print(f"time-step convergence rate = {rate:.3f} (explicit-Euler transport -> ~1)")
json.dump({"dts": [r[0] for r in rows], "E": [r[1] for r in rows],
           "rate": float(rate), "N": N, "T": T}, open(os.path.join(RESULTS, "dt_refine.json"), "w"))
print("saved dt_refine.json")
