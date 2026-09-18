"""Figure 7: force-evaluation accuracy.

Reads results/forceerr.json written by run_forceerr.py and writes
figures/fig_forceerr.pdf and .png.

NOTE.  No script in the first submission produced this figure; run_forceerr.py
computed and stored the data and nothing plotted it.  This script closes that gap.

Panel (a): the per-cell monopole error for an isolated cluster, against the
separation ratio s/d, with the predicted second-order reference law.
Panel (b): the global Barnes-Hut traversal error against the opening parameter
theta, with the cell-list error shown on the same axis -- it sits at machine
precision, which is the point of the comparison.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = os.path.join(RESULTS, "forceerr.json")
if not os.path.exists(src):
    sys.exit("results/forceerr.json not found -- run:  python run_forceerr.py")
D = json.load(open(src))
if "monopole" not in D:
    sys.exit("forceerr.json has no 'monopole' block -- re-run run_forceerr.py in full")

# palette and rcParams matched to the paper's other figures
BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c8c8c"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 300})

fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))

# ---------------------------------------------------- (a) per-cell monopole law
sd = np.array(D["monopole"]["sd"], float)
rel = np.array(D["monopole"]["rel"], float)
slope = D["monopole"]["slope"]
o = np.argsort(sd); sd, rel = sd[o], rel[o]
ax[0].loglog(sd, rel, "o", color=BLUE, ms=8, mec="white", mew=0.8, zorder=3,
             label=f"measured (slope {slope:.2f})")
ref = rel[0] * (sd / sd[0]) ** 2
ax[0].loglog(sd, ref, "--", color="k", alpha=0.65, lw=1.5, zorder=2,
             label=r"$\propto (s/d)^2$")
ax[0].set_xlabel(r"separation ratio $s/d$")
ax[0].set_ylabel("relative monopole error")
ax[0].set_title("(a) Per-cell monopole error law")
ax[0].legend(loc="upper left", fontsize=9.5, framealpha=0.9)

# ------------------------------------------------- (b) global Barnes-Hut error
th = np.array(sorted(float(t) for t in D["bh_theta"]))
key = {float(t): t for t in D["bh_theta"]}
e2 = np.array([D["bh_theta"][key[t]]["E2"] for t in th])
ei = np.array([D["bh_theta"][key[t]]["Einf"] for t in th])
cell_floor = max(v["E2"] for v in D["cell"].values())

ax[1].semilogy(th, e2, "s-", color=BLUE, ms=7, lw=1.8, label=r"$E_2$ (global)")
ax[1].semilogy(th, ei, "^-", color=ORANGE, ms=7, lw=1.8, label=r"$E_\infty$ (global)")
ax[1].axhline(cell_floor, ls=":", color=GREEN, lw=2.0,
              label=r"cell list ($\sim$ machine $\varepsilon$)")
ax[1].annotate(f"cell list: {cell_floor:.1e}", xy=(th[0], cell_floor),
               xytext=(0, 7), textcoords="offset points",
               fontsize=9, color=GREEN)
ax[1].set_xlabel(r"opening parameter $\theta$")
ax[1].set_ylabel("relative force error")
ax[1].set_title("(b) Global Barnes--Hut error")
ax[1].set_ylim(cell_floor / 30, max(ei.max(), e2.max()) * 60)
ax[1].legend(loc="center right", fontsize=9.5, framealpha=0.9)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGURES, f"fig_forceerr.{ext}"), bbox_inches="tight")
print("saved figures/fig_forceerr.pdf and .png")
print(f"  per-cell monopole slope        {slope:.3f}   (expected 2)")
print(f"  global E_2   over theta range  {e2.min():.2e} - {e2.max():.2e}")
print(f"  global E_inf over theta range  {ei.min():.2e} - {ei.max():.2e}")
print(f"  global E_2 fitted slope        {D['bh_slope']:.3f}")
print(f"  cell-list E_2 (max over N)     {cell_floor:.2e}")
