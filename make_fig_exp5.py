"""Figure 13: two- versus three-dimensional dynamics and 3D cost scaling.

Reads results/exp5.npz (and results/exp5_results.json) written by run_exp5.py
and writes figures/fig13_dim.pdf and .png.

NOTE.  No script in the first submission produced this figure: run_exp5.py
computed and stored the data and nothing plotted it.  This script closes that
gap; the numbers quoted in the panels are read from the stored results, so the
figure and the text cannot drift apart.

  (a) R_c(t) in d = 2 and d = 3 with the fitted early-time migration rates
  (b) final 2D configuration
  (c) final 3D configuration
  (d) wall-clock cost of one long-range evaluation in 3D, octree vs direct,
      with the fitted power laws
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401  (registers '3d')

src = os.path.join(RESULTS, "exp5.npz")
if not os.path.exists(src):
    sys.exit("results/exp5.npz not found -- run:  python run_exp5.py")
d = np.load(src)

BLUE, RED, GREY = "#1f77b4", "#d62728", "#8c8c8c"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 300})

fig = plt.figure(figsize=(13.0, 8.0))
gs_top = fig.add_gridspec(1, 3, left=0.065, right=0.975, top=0.885,
                          bottom=0.575, wspace=0.30)
gs_bot = fig.add_gridspec(1, 1, left=0.065, right=0.975, top=0.455,
                          bottom=0.075)

# ------------------------------------------------ (a) localisation transients
a = fig.add_subplot(gs_top[0, 0])
a.plot(d["t2"], d["Rc2"], "-", color=BLUE, lw=1.8,
       label=rf"2D, rate $={float(d['rate2']):.4f}$")
a.plot(d["t3"], d["Rc3"], "-", color=RED, lw=1.8,
       label=rf"3D, rate $={float(d['rate3']):.4f}$")
# early-time fits actually used for the quoted rates (first quarter of the run)
for tt, rr, rate, col in ((d["t2"], d["Rc2"], float(d["rate2"]), BLUE),
                          (d["t3"], d["Rc3"], float(d["rate3"]), RED)):
    k = max(2, len(tt) // 4)
    a.plot(tt[:k], rr[0] - rate * tt[:k], "--", color=col, lw=1.0, alpha=0.9)
a.set(xlabel="time $t$", ylabel="$R_c(t)$", title="(a) chemotactic localisation")
a.legend(fontsize=9, loc="upper right")

# ------------------------------------------------ (b) final 2D configuration
b = fig.add_subplot(gs_top[0, 1])
X2 = d["X2"]
b.scatter(X2[:, 0], X2[:, 1], s=5, color=BLUE, alpha=0.8, edgecolors="none")
b.scatter([0.5], [0.5], s=230, marker="*", c="gold", edgecolors="k",
          linewidths=0.8, zorder=4)
b.set(xlim=(0, 1), ylim=(0, 1), xlabel="$x$", ylabel="$y$",
      title=rf"(b) 2D final, $R_c={float(d['Rcf2']):.3f}$, "
            rf"$\mathcal{{A}}={float(d['clust2']):.2f}$")
b.set_aspect("equal"); b.grid(False)

# ------------------------------------------------ (c) final 3D configuration
c = fig.add_subplot(gs_top[0, 2], projection="3d")
X3 = d["X3"]
c.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=4, color=RED, alpha=0.55,
          edgecolors="none", depthshade=True)
c.scatter([0.5], [0.5], [0.5], s=160, marker="*", c="gold", edgecolors="k",
          linewidths=0.8, depthshade=False)
c.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
      xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], zticks=[0, 0.5, 1])
c.set_xlabel("$x$", labelpad=1); c.set_ylabel("$y$", labelpad=1)
c.set_zlabel("$z$", labelpad=1)
c.set_xticklabels(["0", "0.5", ""])   # blank: collides with the y=0 label
c.tick_params(labelsize=8, pad=1)
c.set_title(rf"(c) 3D final, $R_c={float(d['Rcf3']):.3f}$, "
            rf"$\mathcal{{A}}={float(d['clust3']):.2f}$", pad=2)
c.view_init(elev=20, azim=-60)
try:
    c.set_box_aspect((1, 1, 0.95), zoom=0.86)
except TypeError:
    pass

# ------------------------------------------------ (d) 3D cost scaling
e = fig.add_subplot(gs_bot[0, 0])
Nd_, Td_ = np.asarray(d["Nd"], float), np.asarray(d["Td"], float) * 1e3
No_, To_ = np.asarray(d["Noct"], float), np.asarray(d["Toct"], float) * 1e3
bd, bo = float(d["exp_direct"]), float(d["exp_oct"])
e.loglog(Nd_, Td_, "o", color=RED, ms=8, label=rf"direct $O(N^2)$: fit $N^{{{bd:.2f}}}$")
e.loglog(No_, To_, "^", color=BLUE, ms=8, label=rf"octree ($\theta=0.5$): fit $N^{{{bo:.2f}}}$")
for N_, T_, b_, col in ((Nd_, Td_, bd, RED), (No_, To_, bo, BLUE)):
    A = np.exp(np.log(T_).mean() - b_ * np.log(N_).mean())
    xs = np.array([N_.min() * 0.85, N_.max() * 1.2])
    e.loglog(xs, A * xs ** b_, "-", color=col, lw=1.2, alpha=0.85)
e.set(xlabel="number of agents $N$",
      ylabel="time per long-range evaluation (ms)",
      title="(d) 3D long-range force: octree versus direct summation")
e.legend(fontsize=10, loc="upper left")
e.set_xticks(Nd_)
e.set_xticklabels([f"{int(n)}" for n in Nd_])
e.minorticks_off()
e.grid(True, which="both", alpha=0.3)
# where the two fitted laws would meet
Ad = np.exp(np.log(Td_).mean() - bd * np.log(Nd_).mean())
Ao = np.exp(np.log(To_).mean() - bo * np.log(No_).mean())
if bd > bo:
    Nx = np.exp((np.log(Ao) - np.log(Ad)) / (bd - bo))
    e.annotate(rf"fitted laws cross at $N\approx{Nx/1e3:.0f}\times10^3$",
               xy=(0.985, 0.06), xycoords="axes fraction", ha="right",
               fontsize=9.5, color=GREY)

fig.suptitle("Experiment 5: two- versus three-dimensional dynamics "
             "and 3D cost scaling", fontsize=13)
fig.savefig(os.path.join(FIGURES, "fig13_dim.pdf"))
fig.savefig(os.path.join(FIGURES, "fig13_dim.png"), dpi=160)
print("saved fig13_dim.pdf / .png")
if os.path.exists(os.path.join(RESULTS, "exp5_results.json")):
    J = json.load(open(os.path.join(RESULTS, "exp5_results.json")))
    print("  quoted in the figure: "
          f"rate2D={J['migration_rate_2d']:.4f}, rate3D={J['migration_rate_3d']:.4f}, "
          f"octree exponent={J['oct_exp']:.2f}, direct exponent={J['direct_exp']:.2f}")
