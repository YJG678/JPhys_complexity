"""Generate representative particle-configuration snapshots for Figures 1 and 2,
using exactly the parameters of Sections 7.4 (ablation) and 7.5 (regimes) so the
snapshots are consistent with the reported R_c / aggregation values."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import numpy as np
from hybrid_model import HybridSim, R_c, aggregation_index

# ---------- Ablation snapshots (Section 7.4 parameters) ----------
ABASE = dict(N=500, L=1.0, dt=0.05, v0=0.12, lam0=5.0, tau_m=0.25, kappa=20.0,
             mu_p=1.0, r_c=0.06, reg=0.03,
             short_backend="celllist", long_backend="direct_vec")
avariants = {
    "full":           dict(lam1=4.5, eps_s=12.0, eps_l=0.02),
    "no_memory":      dict(lam1=0.0, eps_s=12.0, eps_l=0.02),
    "no_short_range": dict(lam1=4.5, eps_s=0.0,  eps_l=0.02),
    "no_long_range":  dict(lam1=4.5, eps_s=12.0, eps_l=0.0),
}
ab = {}
for name, ov in avariants.items():
    s = HybridSim(seed=300, **{**ABASE, **ov})
    s.run(450)
    ab[name + "_X"] = s.X.copy()
    ab[name + "_C"] = np.asarray(s.C)
    ab[name + "_Rc"] = R_c(s)
    ab[name + "_agg"] = aggregation_index(s)
    print(f"ablation {name:15s}: Rc={R_c(s):.3f} agg={aggregation_index(s):.2f}", flush=True)
np.savez(os.path.join(RESULTS, "ablation_snaps.npz"), **ab)

# ---------- Regime snapshots (Section 7.5 sweep parameters) ----------
RBASE = dict(N=500, L=1.0, v0=0.12, lam0=5.0, kappa=20, tau_m=0.25, dt=0.05,
             eps_s=12.0, r_c=0.06, reg=0.03,
             short_backend="direct_vec", long_backend="direct_vec")
# (lambda1, eps_l) chosen from the sweep grid to land in each regime
points = {
    "dispersed":  dict(lam1=0.0, eps_l=0.0),
    "localized":  dict(lam1=4.5, eps_l=0.02),
    "clustered":  dict(lam1=1.5, eps_l=0.02),
    "collapsed":  dict(lam1=4.5, eps_l=0.1),
}
def classify(rc, c):
    if c > 12.0: return "collapsed"
    if rc < 0.20: return "localized"
    if c > 2.0: return "clustered"
    return "dispersed"
rg = {}
for name, ov in points.items():
    s = HybridSim(**RBASE, seed=21, **ov)
    s.run(400)
    rc, ag = R_c(s), aggregation_index(s)
    rg[name + "_X"] = s.X.copy(); rg[name + "_C"] = np.asarray(s.C)
    rg[name + "_Rc"] = rc; rg[name + "_agg"] = ag
    rg[name + "_par"] = np.array([ov["lam1"], ov["eps_l"]])
    print(f"regime {name:10s} (lam1={ov['lam1']}, eps_l={ov['eps_l']}): "
          f"Rc={rc:.3f} agg={ag:.2f} -> classified {classify(rc, ag)}", flush=True)
np.savez(os.path.join(RESULTS, "regime_snaps.npz"), **rg)
print("saved ablation_snaps.npz, regime_snaps.npz")
