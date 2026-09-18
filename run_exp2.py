"""Experiment 2: aggregation regimes (Figure 6).

Two-parameter sweep over chemotactic strength lambda1 and long-range
attraction eps_l, with short-range repulsion always active (so clusters have
finite size).  Each cell is classified by (R_c, aggregation index) into one of
four regimes.  The figure has two rows:

  top    (a) R_c, (b) log10 A, (c) regime classification   -- heat maps
  bottom one representative final configuration per regime

Notes on the plotting (these fix defects in the earlier version):
  * the sweep values of eps_l are NOT equally spaced, but the heat-map columns
    ARE.  Ticks are therefore placed on COLUMN INDICES and labelled with the
    eps_l values; putting ticks at the raw values on a linear extent made the
    first three labels overlap.
  * the snapshot for each regime is chosen deterministically from the
    classified grid (closest to that regime's median in (R_c, log10 A)) and the
    (lambda1, eps_l) printed in the panel title is read back out of the grid,
    so a snapshot can never disagree with panel (c).
  * the nutrient source C is drawn with a real star marker, and the periodic
    centre of mass of the particle cloud is drawn as well: under strong eps_l
    the collapse is driven by the isotropic attraction and is NOT pinned to the
    source, so the two need not coincide.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import time
import numpy as np
from hybrid_model import HybridSim, R_c, aggregation_index, min_image

N, NSTEPS = 500, 400
lam1s = np.array([0.0, 1.5, 3.0, 4.5])            # chemotactic strength
eps_ls = np.array([0.0, 0.005, 0.02, 0.05, 0.1])  # long-range attraction
base = dict(N=N, v0=0.12, lam0=5.0, kappa=20, tau_m=0.25, dt=0.05,
            eps_s=12.0, r_c=0.06, reg=0.03,
            short_backend="direct_vec", long_backend="direct_vec")

REGIME_NAMES = ["dispersed", "localisation", "clustering", "collapse"]
REGIME_COLORS = ["#cbd5e1", "#1f77b4", "#2ca02c", "#d62728"]


def classify(rc, c):
    """0 dispersed, 1 chemotactic localisation, 2 clustering, 3 collapse."""
    if c > 12.0:
        return 3
    if rc < 0.20:
        return 1
    if c > 2.0:
        return 2
    return 0


# ----------------------------------------------------------------------
# periodic centre of mass (circular mean, then one refinement sweep)
# ----------------------------------------------------------------------
def periodic_com(X, L):
    ang = 2 * np.pi * X / L
    c = np.arctan2(np.sin(ang).mean(axis=0), np.cos(ang).mean(axis=0))
    com = (c / (2 * np.pi)) * L % L
    for _ in range(3):                       # refine: mean of minimum images
        com = (com + min_image(X - com[None, :], L).mean(axis=0)) % L
    return com


def gyration_radius(X, L, com):
    d = min_image(X - com[None, :], L)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


# ----------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------
NPZ = os.path.join(RESULTS, "exp2.npz")
REPLOT = "--replot" in sys.argv          # re-draw from results/exp2.npz

Rc = np.zeros((len(lam1s), len(eps_ls)))
Cl = np.zeros_like(Rc)
Reg = np.zeros_like(Rc, dtype=int)
Xall = np.zeros((len(lam1s), len(eps_ls), N, 2))

L = 1.0
C0 = np.array([0.5, 0.5])
t0 = time.time()
if REPLOT:
    d = np.load(NPZ)
    lam1s, eps_ls = d["lam1s"], d["eps_ls"]
    Rc, Cl, Reg, Xall = d["Rc"], d["Cl"], d["Reg"], d["X"]
    print(f"re-plotting from {NPZ}")
else:
    for i, l1 in enumerate(lam1s):
        for j, el in enumerate(eps_ls):
            s = HybridSim(**base, lam1=float(l1), eps_l=float(el), seed=21)
            s.run(NSTEPS)
            Rc[i, j] = R_c(s)
            Cl[i, j] = aggregation_index(s)
            Reg[i, j] = classify(Rc[i, j], Cl[i, j])
            Xall[i, j] = s.X
        print(f"lam1={l1}: Rc={np.round(Rc[i],3)} A={np.round(Cl[i],1)} "
              f"regime={Reg[i]}", flush=True)
    print(f"done in {time.time()-t0:.1f}s")


# ----------------------------------------------------------------------
# deterministic choice of one representative cell per regime
# ----------------------------------------------------------------------
# representative (lambda1, eps_l) for each regime, quoted in the caption.
# They are verified against the classified grid below; if the grid ever moves,
# the code falls back to the median cell of that regime and says so.
PREFERRED = {0: (0.0, 0.0), 1: (4.5, 0.02), 2: (1.5, 0.02), 3: (4.5, 0.1)}


def _median_cell(Reg, Rc, lg, r):
    """Cell of regime r closest to that regime's median in standardised
    (R_c, log10 A); ties broken by larger lambda1 then larger eps_l."""
    cells = np.argwhere(Reg == r)
    mr, mc = np.median(Rc[Reg == r]), np.median(lg[Reg == r])
    sd_r, sd_c = max(Rc.std(), 1e-12), max(lg.std(), 1e-12)
    best, bestkey = None, None
    for (i, j) in cells:
        d = ((Rc[i, j] - mr) / sd_r) ** 2 + ((lg[i, j] - mc) / sd_c) ** 2
        key = (round(float(d), 12), -i, -j)
        if bestkey is None or key < bestkey:
            bestkey, best = key, (int(i), int(j))
    return best


def pick_representatives(Reg, Rc, Cl, lam1s, eps_ls):
    lg = np.log10(np.maximum(Cl, 1e-12))
    reps = {}
    for r in range(4):
        if not np.any(Reg == r):
            continue
        l1, el = PREFERRED[r]
        i = int(np.argmin(np.abs(lam1s - l1)))
        j = int(np.argmin(np.abs(eps_ls - el)))
        if Reg[i, j] == r:
            reps[r] = (i, j)
        else:
            reps[r] = _median_cell(Reg, Rc, lg, r)
            print(f"  ! preferred cell for '{REGIME_NAMES[r]}' "
                  f"(lam1={l1}, eps_l={el}) is now classified "
                  f"'{REGIME_NAMES[Reg[i, j]]}'; using "
                  f"(lam1={lam1s[reps[r][0]]:g}, eps_l={eps_ls[reps[r][1]]:g}) "
                  f"instead -- update PREFERRED and the caption.")
    return reps


reps = pick_representatives(Reg, Rc, Cl, lam1s, eps_ls)
for r, (i, j) in sorted(reps.items()):
    assert Reg[i, j] == r, "representative disagrees with classification"
    com = periodic_com(Xall[i, j], L)
    off = np.sqrt(np.sum(min_image(com - C0, L) ** 2))
    print(f"  {REGIME_NAMES[r]:13s} lam1={lam1s[i]:.1f} eps_l={eps_ls[j]:.3f} "
          f"R_c={Rc[i,j]:.3f} A={Cl[i,j]:.1f} "
          f"|com-C|={off:.3f} l_agg={gyration_radius(Xall[i,j],L,com):.3f}")

np.savez(NPZ,         # in --replot mode this only refreshes rep_idx
         lam1s=lam1s, eps_ls=eps_ls, Rc=Rc, Cl=Cl, Reg=Reg, X=Xall,
         rep_idx=np.array([reps.get(r, (-1, -1)) for r in range(4)]),
         N=N, nsteps=NSTEPS, C=C0, L=L)


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(13.5, 7.8))
gs_top = fig.add_gridspec(1, 3, left=0.055, right=0.925, top=0.885,
                          bottom=0.565, wspace=0.45)
gs_bot = fig.add_gridspec(1, 4, left=0.055, right=0.975, top=0.435,
                          bottom=0.065, wspace=0.30)
ax = [fig.add_subplot(gs_top[0, k]) for k in range(3)]

# index coordinates: columns/rows are equally spaced, labels carry the values
ny, nx = Rc.shape
ext = [-0.5, nx - 0.5, -0.5, ny - 0.5]
kw = dict(origin="lower", aspect="auto", extent=ext, interpolation="nearest")

im0 = ax[0].imshow(Rc, cmap="viridis_r", **kw)
ax[0].set(xlabel=r"long-range attraction $\epsilon_\ell$",
          ylabel=r"chemotactic strength $\lambda_1$",
          title=r"(a) mean distance $R_c$")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)

im1 = ax[1].imshow(np.log10(np.maximum(Cl, 1e-12)), cmap="magma", **kw)
ax[1].set(xlabel=r"long-range attraction $\epsilon_\ell$",
          title=r"(b) aggregation index $\log_{10}\mathcal{A}$")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.03)

cmap = ListedColormap(REGIME_COLORS)
norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
im2 = ax[2].imshow(Reg, cmap=cmap, norm=norm, **kw)
ax[2].set(xlabel=r"long-range attraction $\epsilon_\ell$",
          title="(c) regime classification")
cb = fig.colorbar(im2, ax=ax[2], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.03)
cb.ax.set_yticklabels(REGIME_NAMES, fontsize=9)

for a in ax:
    a.set_xticks(np.arange(nx))
    a.set_xticklabels([("0" if v == 0 else f"{v:g}") for v in eps_ls])
    a.set_yticks(np.arange(ny))
    a.set_yticklabels([f"{v:g}" for v in lam1s])

# mark the four representative cells on panel (c)
for r, (i, j) in sorted(reps.items()):
    ax[2].plot(j, i, marker="o", ms=9, mfc="none", mec="k", mew=1.6)
    dx = -30 if j >= nx - 1 else 9
    ax[2].annotate("(" + "defg"[r] + ")", (j, i), textcoords="offset points",
                   xytext=(dx, 6), fontsize=9, fontweight="bold")

# ---- bottom row: one snapshot per regime -----------------------------
for r in range(4):
    a = fig.add_subplot(gs_bot[0, r])
    tag = "(" + "defg"[r] + ")"
    if r not in reps:
        a.set_axis_off()
        a.text(0.5, 0.5, f"{tag} {REGIME_NAMES[r]}\nnot realised on this grid",
               ha="center", va="center", fontsize=10)
        continue
    i, j = reps[r]
    X = Xall[i, j]
    com = periodic_com(X, L)
    lagg = gyration_radius(X, L, com)
    a.scatter(X[:, 0], X[:, 1], s=4, c=REGIME_COLORS[r], edgecolors="none",
              alpha=0.85, zorder=2)
    a.scatter([C0[0]], [C0[1]], s=260, marker="*", c="gold", edgecolors="k",
              linewidths=0.8, zorder=4, label="nutrient source $C$")
    if Cl[i, j] > 2.0:     # a centre of mass is only meaningful once a
        # cluster exists; for a uniform cloud it is an arbitrary point.
        a.scatter([com[0]], [com[1]], s=70, marker="o", facecolors="none",
                  edgecolors="k", linewidths=1.4, zorder=5,
                  label=r"cluster centre $\pm\,\ell_{\rm agg}$")
        a.add_patch(plt.Circle(tuple(com), lagg, fill=False, ls="--", lw=0.9,
                               ec="k", zorder=3, clip_on=True))
    a.set(xlim=(0, L), ylim=(0, L), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    a.set_aspect("equal")
    a.set_title(f"{tag} {REGIME_NAMES[r]}\n"
                rf"$\lambda_1={lam1s[i]:g}$, $\epsilon_\ell={eps_ls[j]:g}$, "
                rf"$R_c={Rc[i,j]:.3f}$, $\mathcal{{A}}={Cl[i,j]:.1f}$",
                fontsize=9.5)
    if r == 1:
        a.legend(loc="upper left", fontsize=7, framealpha=0.9,
                 handletextpad=0.2, borderpad=0.25)

fig.suptitle(
    "Experiment 2: aggregation regimes "
    r"($N=500$, short-range repulsion active)",
    fontsize=13,
)

pdf_path = os.path.join(FIGURES, "fig6_regimes.pdf")
png_path = os.path.join(FIGURES, "fig6_regimes_final.png")

fig.savefig(pdf_path, format="pdf")
fig.savefig(png_path, format="png", dpi=160)

plt.close(fig)

print(f"saved: {pdf_path}")
print(f"saved: {png_path}")
