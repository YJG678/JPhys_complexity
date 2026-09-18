"""Figure: uptake sweep -- the chemo-repulsive mechanism in isolation.
Reads results/production.json (produced by run_production.py).
Writes figures/fig_uptake.pdf and .png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = os.path.join(RESULTS, "production.json")
if not os.path.exists(src):
    sys.exit("results/production.json not found -- run  python run_production.py  first")
d = json.load(open(src))["stage2_beta"]
keys = {float(k): k for k in d}
beta = np.array(sorted(keys))
get = lambda q, i: np.array([d[keys[b]][q][i] for b in beta])
A, Ae = get('A', 0), get('A', 1)
R, Re = get('Rc', 0), get('Rc', 1)
P, Pe = get('rho', 0), get('rho', 1)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 8.5, "figure.dpi": 200, "savefig.dpi": 300})

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
panels = [
    (axes[0], A, Ae, BLUE,   r"aggregation index $\mathcal{A}$", True,  1.0, "Poisson"),
    (axes[1], R, Re, ORANGE, r"$\hat R_c$",                      False, 1.0, "dispersed"),
    (axes[2], P, Pe, AQUA,   r"mean nutrient $\bar\rho$",        False, None, None)]
for k, (ax, y, ye, col, lab, logy, ref, reflab) in enumerate(panels):
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.annotate(reflab, xy=(0.03 if k == 0 else 0.98, ref),
                    xycoords=("axes fraction", "data"),
                    ha="left" if k == 0 else "right", va="bottom",
                    fontsize=6.8, color=MUTED)
    ax.errorbar(beta, y, yerr=ye, color=col, lw=1.6, marker="o", ms=5.0,
                mfc=col, mec="white", mew=1.0, capsize=2.5, elinewidth=1.0,
                ecolor=col, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(beta.min()*0.75, beta.max()*1.35)
    ax.set_xticks(beta); ax.set_xticklabels([f"{b:g}" for b in beta])
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log"); ax.set_ylim(max(0.5, A.min()*0.5), A.max()*2.1)
        tk = [t for t in (1, 2, 5, 10, 20, 50) if A.min()*0.5 <= t <= A.max()*2.1]
        ax.set_yticks(tk); ax.set_yticklabels([str(t) for t in tk])
    ax.set_xlabel(r"uptake coefficient $\beta$")
    ax.set_title(lab, color=INK, pad=5)
    ax.text(-0.17, 1.06, "(" + "abc"[k] + ")", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=INK, va="top")
ax = axes[0]
ax.annotate(f"{A[0]:.1f}", xy=(beta[0], A[0]), xytext=(9, 6),
            textcoords="offset points", fontsize=7.2, color=INK, ha="left")
ax.annotate(f"{A[-1]:.2f}", xy=(beta[-1], A[-1]), xytext=(-9, -2),
            textcoords="offset points", fontsize=7.2, color=INK,
            ha="right", va="center")
ax.annotate(rf"$\times\,{A[0]/A[-1]:.0f}$ fall at fixed $A_\ell$", xy=(0.46, 0.42),
            xycoords="axes fraction", fontsize=7.2, color=MUTED, ha="center")
fig.tight_layout(w_pad=1.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGURES, f"fig_uptake.{ext}"), bbox_inches="tight")
print("A  :", np.round(A, 2), "+/-", np.round(Ae, 2))
print("Rc :", np.round(R, 3))
print("rho:", np.round(P, 4))
print("saved figures/fig_uptake.pdf and .png")
