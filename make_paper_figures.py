"""Four figures for the Computational and Applied Mathematics version, consistent
with the reported numbers in Sections 7.4, 7.5, 7.2-7.3, and 8.

  Figure 1  ablation_physics.pdf     - ablation snapshots + R_c / aggregation bars
  Figure 2  macroscopic_regimes.pdf  - regime heatmaps + configuration snapshots
  Figure 3  verification_meanfield.pdf - field-solver & particle convergence, mean field
  Figure 4  scalability.pdf          - wall-clock O(N^2) vs O(N log N); interaction counts
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "figure.dpi": 140})

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c8c8c"
REGCOL = {"dispersed": "#64748b", "localized": "#1f77b4",
          "clustered": "#2ca02c", "collapsed": "#d62728"}


def snapshot(ax, X, C, title, subtitle, edgecolor):
    """Beautiful particle-configuration scatter coloured by distance to source."""
    L = 1.0
    d = np.sqrt(np.minimum(np.abs(X[:, 0] - C[0]), L - np.abs(X[:, 0] - C[0]))**2 +
                np.minimum(np.abs(X[:, 1] - C[1]), L - np.abs(X[:, 1] - C[1]))**2)
    ax.scatter(X[:, 0], X[:, 1], c=d, cmap="viridis", s=10, alpha=0.85,
               edgecolors="none", vmin=0, vmax=0.5)
    ax.plot(C[0], C[1], marker="*", ms=16, color="white",
            markeredgecolor="black", markeredgewidth=1.2, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_edgecolor(edgecolor); s.set_linewidth(2.4)
    ax.set_title(title, color=edgecolor, fontsize=11, fontweight="bold", pad=4)
    ax.text(0.5, -0.07, subtitle, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.6)


# ============================ FIGURE 1 ============================
def figure1():
    ab = np.load(os.path.join(RESULTS, "ablation_snaps.npz"))
    means = json.load(open(os.path.join(RESULTS, "ablation.json")))["variants"]
    order = ["full", "no_memory", "no_short_range", "no_long_range"]
    letter = {"full": "A  full model", "no_memory": "B  no memory",
              "no_short_range": "C  no short-range", "no_long_range": "D  no long-range"}
    fig = plt.figure(figsize=(13.5, 7.4))
    gs = GridSpec(2, 4, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.18)
    for j, k in enumerate(order):
        ax = fig.add_subplot(gs[0, j])
        sub = f"$R_c={ab[k+'_Rc']:.3f}$,  agg $={ab[k+'_agg']:.1f}$"
        snapshot(ax, ab[k + "_X"], ab[k + "_C"], letter[k], sub, REGCOL[
            "localized" if k == "full" else
            "dispersed" if k in ("no_memory", "no_long_range") else "collapsed"])
    labels = ["full", "no\nmemory", "no short-\nrange", "no long-\nrange"]
    rc = [means[k]["R_c"] for k in order]; rce = [means[k]["R_c_sd"] for k in order]
    ag = [means[k]["agg"] for k in order]; age = [means[k]["agg_sd"] for k in order]
    x = np.arange(4)
    axb = fig.add_subplot(gs[1, :2])
    axb.bar(x, rc, yerr=rce, color=BLUE, alpha=0.9, capsize=4)
    axb.axhline(0.20, ls="--", color=GREY, lw=1); axb.text(3.4, 0.21, "localized $<0.2$",
                 fontsize=8, color=GREY, ha="right")
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set(ylabel="$R_c$ (mean distance to source)", title="(E) Localization diagnostic")
    axc = fig.add_subplot(gs[1, 2:])
    axc.bar(x, ag, yerr=age, color=ORANGE, alpha=0.9, capsize=4)
    axc.set_yscale("log"); axc.set_xticks(x); axc.set_xticklabels(labels)
    axc.set(ylabel="aggregation index", title="(F) Aggregation diagnostic")
    fig.suptitle("Ablation of model mechanisms: each term is physically necessary",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(FIGURES, "fig_ablation_physics.pdf")); fig.savefig(os.path.join(FIGURES, "fig_ablation_physics.png"), dpi=150)
    plt.close(fig); print("saved Figure 1")


# ============================ FIGURE 2 ============================
def figure2():
    """Regime heat maps + one representative snapshot per regime.

    The sweep values of eps_l are not equally spaced while the heat-map columns
    are, so the axes use column/row INDICES with the parameter values as tick
    labels (placing ticks at the raw values made the first three overlap and
    shifted every column by one).  The snapshots and their (lambda1, eps_l)
    labels are read from results/exp2.npz, i.e. from the same simulations that
    produced the classification, so they cannot disagree with panel (C)."""
    d = np.load(os.path.join(RESULTS, "exp2.npz"))
    lam1s, eps_ls = d["lam1s"], d["eps_ls"]
    Rc, Cl, Reg, Xall, rep = d["Rc"], d["Cl"], d["Reg"], d["X"], d["rep_idx"]
    C0 = d["C"] if "C" in d.files else np.array([0.5, 0.5])
    ny, nx = Rc.shape
    ext = [-0.5, nx - 0.5, -0.5, ny - 0.5]
    kw = dict(origin="lower", aspect="auto", extent=ext, interpolation="nearest")
    names = ["dispersed", "localized", "clustered", "collapsed"]

    fig = plt.figure(figsize=(13.5, 8.0))
    gst = GridSpec(1, 3, left=0.055, right=0.925, top=0.885, bottom=0.575,
                   wspace=0.45)
    gsb = GridSpec(1, 4, left=0.055, right=0.975, top=0.44, bottom=0.06,
                   wspace=0.22)
    a0, a1, a2 = (fig.add_subplot(gst[0, k]) for k in range(3))
    im0 = a0.imshow(Rc, cmap="viridis_r", **kw)
    a0.set(xlabel=r"long-range attraction $\epsilon_\ell$",
           ylabel=r"chemotactic strength $\lambda_1$",
           title="(A) mean distance $R_c$")
    fig.colorbar(im0, ax=a0, fraction=0.046, pad=0.03)
    im1 = a1.imshow(np.log10(np.maximum(Cl, 1e-12)), cmap="magma", **kw)
    a1.set(xlabel=r"long-range attraction $\epsilon_\ell$",
           title=r"(B) $\log_{10}$ aggregation index")
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.03)
    cmap = ListedColormap([REGCOL[k] for k in names])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
    im2 = a2.imshow(Reg, cmap=cmap, norm=norm, **kw)
    a2.set(xlabel=r"long-range attraction $\epsilon_\ell$",
           title="(C) regime map")
    cb = fig.colorbar(im2, ax=a2, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.03)
    cb.ax.set_yticklabels(names, fontsize=9)
    for a in (a0, a1, a2):
        a.grid(False)
        a.set_xticks(np.arange(nx))
        a.set_xticklabels([("0" if v == 0 else f"{v:g}") for v in eps_ls])
        a.set_yticks(np.arange(ny))
        a.set_yticklabels([f"{v:g}" for v in lam1s])
    for r in range(4):
        i, j = int(rep[r][0]), int(rep[r][1])
        if i < 0:
            continue
        a2.plot(j, i, marker="o", ms=9, mfc="none", mec="k", mew=1.6)
        a2.annotate("(" + "DEFG"[r] + ")", (j, i), textcoords="offset points",
                    xytext=(-30 if j >= nx - 1 else 9, 6),
                    fontsize=9, fontweight="bold")
    for r in range(4):
        ax = fig.add_subplot(gsb[0, r])
        i, j = int(rep[r][0]), int(rep[r][1])
        if i < 0:
            ax.axis("off")
            ax.text(0.5, 0.5, f"({chr(68+r)}) {names[r]}\nnot realised",
                    ha="center", va="center", fontsize=10)
            continue
        sub = (f"$\\lambda_1={lam1s[i]:g},\\ \\epsilon_\\ell={eps_ls[j]:g}$\n"
               f"$R_c={Rc[i,j]:.3f}$, agg $={Cl[i,j]:.1f}$")
        snapshot(ax, Xall[i, j], C0, f"{chr(68+r)}  {names[r]}", sub,
                 REGCOL[names[r]])
    fig.suptitle("Emergent macroscopic regimes over the "
                 r"$(\lambda_1,\epsilon_\ell)$ parameter space",
                 fontsize=13, fontweight="bold")
    plt.close(fig); print("saved Figure 2")


# ============================ FIGURE 3 ============================
def figure3():
    gr = json.load(open(os.path.join(RESULTS, "grid_refine.json")))
    dt = json.load(open(os.path.join(RESULTS, "dt_refine.json")))
    e4 = np.load(os.path.join(RESULTS, "exp4.npz"))
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))
    # (a) field solver spatial
    sp = gr["spatial"]; Ns = np.array(sorted(int(k) for k in sp)); es = np.array([sp[str(n)] for n in Ns])
    hs = 1.0 / Ns
    ax[0, 0].loglog(hs, es, "s-", color=RED, ms=8, label=f"measured (slope {gr['p_space']:.2f})")
    ax[0, 0].loglog(hs, es[0] * (hs / hs[0])**2, "--", color="k", alpha=0.6, label=r"$\propto h^2$")
    ax[0, 0].set(xlabel="grid spacing $h$", ylabel="$L^2$ error",
                 title="(a) Field solver: spatial (2nd order)"); ax[0, 0].legend()
    # (b) field solver temporal
    tp = gr["temporal"]; dts = np.array(sorted(float(k) for k in tp)); et = np.array([tp[k] for k in sorted(tp, key=lambda z: float(z))])
    ax[0, 1].loglog(dts, et, "^-", color=BLUE, ms=8, label=f"measured (slope {gr['p_time']:.2f})")
    ax[0, 1].loglog(dts, et[0] * (dts / dts[0]), "--", color="k", alpha=0.6, label=r"$\propto \Delta t$")
    ax[0, 1].set(xlabel="time step $\\Delta t$", ylabel="$L^2$ error",
                 title="(b) Field solver: temporal (1st order)"); ax[0, 1].legend()
    # (c) particle integrator time step
    dd = np.array(dt["dts"]); ee = np.array(dt["E"])
    ax[1, 0].loglog(dd, ee, "o-", color=GREEN, ms=8, label=f"measured (slope {dt['rate']:.2f})")
    ax[1, 0].loglog(dd, ee[0] * (dd / dd[0]), "--", color="k", alpha=0.6, label=r"$\propto \Delta t$")
    ax[1, 0].set(xlabel="time step $\\Delta t$", ylabel=r"$\|X_{\Delta t}-X_{\Delta t/2}\|_2$",
                 title="(c) Particle integrator (1st order)"); ax[1, 0].legend()
    # (d) mean-field N^-1/2
    Nm = np.asarray(e4["Ns"], float); em = np.asarray(e4["errs"], float); sl = float(e4["slope"])
    ax[1, 1].loglog(Nm, em, "D-", color=ORANGE, ms=8, label=f"measured (slope {sl:.3f})")
    ax[1, 1].loglog(Nm, em[0] * (Nm / Nm[0])**-0.5, "--", color="k", alpha=0.6, label=r"$\propto N^{-1/2}$")
    ax[1, 1].set(xlabel="number of particles $N$", ylabel="density fluctuation $E_N$",
                 title="(d) Mean-field convergence"); ax[1, 1].legend()
    fig.suptitle("Mathematical verification: solver convergence and the "
                 "mean-field limit", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES, "fig_verification_meanfield.pdf")); fig.savefig(os.path.join(FIGURES, "fig_verification_meanfield.png"), dpi=150)
    plt.close(fig); print("saved Figure 3")


# ============================ FIGURE 4 ============================
def figure4():
    """Wall-clock scaling and interaction counts."""
    nb = json.load(open(os.path.join(RESULTS, "exp3_numba.json")))
    cnts = np.load(os.path.join(RESULTS, "exp3_counts.npz"))
    Ns = np.array(nb["Ns"], float)
    cell = np.array(nb["cell"]) * 1e3
    bh = np.array(nb["bh_total"]) * 1e3

    legacy = False
    if nb.get("direct"):
        Nd = np.array(sorted(int(n) for n in nb["direct"]), float)
        Td = np.array([nb["direct"][str(int(n))] for n in Nd]) * 1e3
        bd = nb.get("slope_direct")
    else:                      # fall back to the uncompiled campaign
        legacy = True
        old = json.load(open(os.path.join(RESULTS, "exp3_results.json")))
        Nd = np.array(sorted(int(n) for n in old["direct"]), float)
        Td = np.array([old["direct"][str(int(n))] for n in Nd]) * 1e3
        bd = old["fits"]["direct"]
    bc, bb = nb.get("slope_cell"), nb.get("slope_bh")
    win = nb.get("fit_window", {})

    def lab(name, b, key):
        w = win.get(key, nb.get("fit_nmin", 0))
        tail = rf", fit $N\geq{w:g}$" if w else ""
        return rf"{name}: $N^{{{b:.2f}}}${tail}" if b else name

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].loglog(Nd, Td, "o-", color=RED, ms=7,
                 label=lab(r"direct $O(N^2)$" + (" [legacy run]" if legacy else ""),
                           bd, "direct"))
    Next = np.array([Nd[-1], 1e6])
    ax[0].loglog(Next, Td[-1] * (Next / Nd[-1]) ** 2, ":", color=RED, alpha=0.55,
                 label="direct (extrapolated)")
    ax[0].loglog(Ns, cell, "s-", color=GREEN, ms=7,
                 label=lab(r"cell list $O(N)$", bc, "cell"))
    ax[0].loglog(Ns, bh, "^-", color=BLUE, ms=7,
                 label=lab(r"Barnes--Hut", bb, "bh_total"))
    ax[0].axvline(1e6, color=GREY, ls=":", alpha=0.5)
    ax[0].set(xlabel="number of particles $N$",
              ylabel="wall-clock time per step (ms)",
              title="(a) Wall-clock scaling to $N=10^6$")
    ax[0].legend(fontsize=9)

    # (b) interaction counts
    def fexp(N, T, nmin=0):
        N = np.asarray(N, float); T = np.asarray(T, float)
        m = N >= nmin
        return np.polyfit(np.log(N[m]), np.log(T[m]), 1)[0]
    Nc = cnts["Ns"].astype(float); tot = cnts["tot"].astype(float)
    be = fexp(Nc, tot, 1e4)
    ax[1].loglog(Nc, tot, "^-", color=BLUE, ms=7,
                 label=rf"BH interactions: $N^{{{be:.2f}}}$")
    ax[1].loglog(Nc, tot[2] * (Nc / Nc[2]) * np.log(Nc) / np.log(Nc[2]), "--",
                 color="k", alpha=0.65, label=r"$\propto N\log N$")
    ax[1].loglog(Nc, tot[0] * (Nc / Nc[0]) ** 2, ":", color=GREY, alpha=0.7,
                 label=r"$\propto N^2$")
    ax[1].set(xlabel="number of particles $N$",
              ylabel="total accepted interactions",
              title=r"(b) Algorithmic complexity: $O(N\log N)$")
    ax[1].legend(fontsize=9)
    fig.suptitle("Algorithmic scalability to one million particles",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES, "fig_scalability.pdf"))
    fig.savefig(os.path.join(FIGURES, "fig_scalability.png"), dpi=150)
    plt.close(fig)
    print("saved Figure 4")
    print(f"  exponents used in the legend (from exp3_numba.json): "
          f"direct N^{bd:.3f}, cell N^{bc:.3f}, BH N^{bb:.3f}"
          + ("   [direct from the LEGACY uncompiled run]" if legacy else ""))


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
