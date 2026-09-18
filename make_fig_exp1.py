"""Figure 10: representative chemotaxis simulation (Experiment 1).

Reads results/exp1_{rw,chemo,rep,full}.npz written by run_exp1.py and writes
figures/fig_exp1.pdf and .png.

NOTE.  No script in the first submission produced this figure; it is added here so
that Figure 10 can be regenerated from the deposited data.

The KL-divergence panel of the original figure has been REMOVED: the quantity was
never defined in the text, and the particle-nutrient co-localisation it conveyed is
carried by R_c(t) in the first panel.  Pass --with-kl to restore it for comparison.
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ap = argparse.ArgumentParser()
ap.add_argument("--with-kl", action="store_true",
                help="restore the KL panel (the original, pre-revision layout)")
a = ap.parse_args()

# palette and rcParams matched to the paper's other figures
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#8c8c8c"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 300})

# scenario key -> (label, colour, linestyle).  The line style is a second encoding
# so the four series remain separable in greyscale and for colour-vision deficiency.
SCEN = [("rw",    "random walk",                 GREY,  (0, (1, 1.6))),
        ("chemo", "memory chemotaxis",           BLUE,  "-"),
        ("rep",   "chemotaxis + repulsion",      GREEN, (0, (5, 2))),
        ("full",  "chemotaxis + short + long",   RED,   "-")]
SNAPS = [("chemo", "final: memory chemotaxis",        BLUE),
         ("rep",   "final: chemotaxis + repulsion",   GREEN),
         ("full",  "final: chemotaxis + short + long", RED)]

D = {}
for k, *_ in SCEN:
    p = os.path.join(RESULTS, f"exp1_{k}.npz")
    if not os.path.exists(p):
        sys.exit(f"missing {p}\n  run:  python run_exp1.py        (all four scenarios)")
    D[k] = np.load(p)

ncol = 6
series = [("Rc",  r"$R_c(t)$", "Mean distance to nutrient centre"),
          ("A",   r"$A(t)$",   "Aggregation index")]
if a.with_kl:
    series.append(("KL", r"$D_{\mathrm{KL}}(n\,\|\,\rho)$", "KL divergence to nutrient profile"))

fig = plt.figure(figsize=(11.5, 6.6))
gs = gridspec.GridSpec(2, ncol, figure=fig, hspace=0.62, wspace=0.55)
span = ncol // len(series)

for j, (key, ylab, title) in enumerate(series):
    ax = fig.add_subplot(gs[0, j*span:(j+1)*span])
    for k, lab, col, ls in SCEN:
        ax.plot(D[k]["t"], D[k][key], color=col, ls=ls, lw=1.9, label=lab)
    ax.set_xlabel("time $t$"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.margins(x=0.02)
    if j == 0:
        handles, labels = ax.get_legend_handles_labels()

# one shared legend on its own row, so it occludes no data
fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.45),
           ncol=4, frameon=False, fontsize=10)

for j, (k, title, col) in enumerate(SNAPS):
    ax = fig.add_subplot(gs[1, j*2:(j+1)*2])
    X, C = D[k]["X"], D[k]["C"]
    ax.scatter(X[:, 0], X[:, 1], s=4, color=col, alpha=0.55, linewidths=0)
    ax.plot(C[0], C[1], "*", color="k", ms=15, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=0.2)

N = D["full"]["X"].shape[0]
fig.suptitle(f"Representative chemotaxis simulation ($N={N}$)", fontsize=13, y=0.97)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07)
stem = "fig_exp1_withKL" if a.with_kl else "fig_exp1"
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"), bbox_inches="tight")
print(f"saved figures/{stem}.pdf and .png")
for k, lab, *_ in SCEN:
    print(f"  {lab:<28} R_c {D[k]['Rc'][0]:.3f} -> {D[k]['Rc'][-1]:.3f}"
          f"   A {D[k]['A'][0]:.2f} -> {D[k]['A'][-1]:.2f}")
