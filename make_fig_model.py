"""Figure 1: the model, quantitatively.

Replaces the purely schematic opening figure.  Every panel is computed from the
code in this package at the parameters actually used in the paper, so the figure
carries model-specific information rather than a cartoon.

  (a) turning-rate response lambda(rho - M) at the values used
  (b) a real single-particle trace of rho(t,X_i) against M_i(t), with the
      instantaneous turning rate below
  (c) both interaction kernels on one axis, with r_c, eps and xi marked
  (d) the timescale ladder that separates the regimes

Writes figures/fig1_model.pdf and .png.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hybrid_model import HybridSim, nutrient, tumble_rate

# ---- parameters of the reported runs ---------------------------------
v0, lam0, lam1, kap = 0.12, 5.0, 4.5, 20.0
tau_m, dt, L = 0.25, 0.05, 1.0
A_s, r_c, eps_s = 12.0, 0.06, 5e-3 * 0.06      # short range
A_l, reg = 0.10, 0.03                          # long range
kappaL = 8.66                                  # screening, kappa = 1/xi
xi = L / kappaL
D_rho, alpha, beta = 0.1, 0.5, 1.0

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c8c8c"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150, "savefig.dpi": 300})

fig = plt.figure(figsize=(12.4, 7.4))
gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, top=0.93, bottom=0.085,
                      hspace=0.42, wspace=0.34, height_ratios=[1.0, 1.0])

# ---------------------------------------------------- (a) turning response
a = fig.add_subplot(gs[0, 0])
q = np.linspace(-0.25, 0.25, 800)
lam = tumble_rate(q, lam0, lam1, kap)
a.plot(q, lam, color=BLUE, lw=2)
a.axhline(lam0, color=GREY, ls=":", lw=1)
a.axhline(lam0 - lam1, color=GREEN, ls="--", lw=1)
a.axhline(lam0 + lam1, color=RED, ls="--", lw=1)
a.axvline(0, color="k", lw=0.8, alpha=0.5)
a.annotate(r"$\lambda_0+\lambda_1=9.5$", (-0.24, lam0 + lam1), va="bottom",
           fontsize=8.5, color=RED)
a.annotate(r"$\lambda_0-\lambda_1=0.5$", (-0.24, lam0 - lam1 + 0.15), va="bottom",
           fontsize=8.5, color=GREEN)
a.annotate(r"$\lambda_0=5$", (-0.24, lam0 + 0.2), fontsize=8.5, color=GREY)
a.annotate("", xy=(1 / kap, 1.55), xytext=(-1 / kap, 1.55),
           arrowprops=dict(arrowstyle="<->", color="k", lw=1))
a.text(0, 2.05, r"sensing width $\kappa^{-1}=0.05$", ha="center", fontsize=8.5)
a.fill_between(q, 0, lam, where=q > 0, color=GREEN, alpha=0.10)
a.fill_between(q, 0, lam, where=q < 0, color=RED, alpha=0.10)
a.text(-0.145, 6.9, "worsening\n$\\rho<M_i$\nfrequent tumbles", ha="center",
       fontsize=8.5, color=RED)
a.text(0.145, 6.9, "improving\n$\\rho>M_i$\nlong runs", ha="center",
       fontsize=8.5, color=GREEN)
a.set(xlabel=r"memory-corrected signal $q=\rho(t,X_i)-M_i$",
      ylabel=r"turning rate $\lambda(q)$", ylim=(0, 10.4),
      title=r"(a) behavioural response $\lambda(q)=\lambda_0-\lambda_1\tanh(\kappa q)$")
ax2 = a.twinx()
ax2.set_ylim(a.get_ylim()); ax2.grid(False)
ticks = np.array([0.5, 2.0, 5.0, 9.5])
ax2.set_yticks(ticks)
ax2.set_yticklabels([f"{v0/t:.3f}" for t in ticks], fontsize=8.5)
ax2.set_ylabel(r"mean run length $v_0/\lambda$", fontsize=9, labelpad=1)

# ---------------------------------------------------- (b) memory trace
b = fig.add_subplot(gs[0, 1])
sim = HybridSim(N=200, v0=v0, lam0=lam0, lam1=lam1, kappa=kap, tau_m=tau_m,
                dt=dt, eps_s=0.0, r_c=r_c, eps_l=0.0, seed=7,
                short_backend="direct_vec", long_backend="direct_vec")
nst = 200
tr_t, tr_rho, tr_M = [], [], []
for n in range(nst):
    tr_t.append(n * dt)
    tr_rho.append(float(nutrient(sim.X, sim.C, L)[0]))
    tr_M.append(float(sim.M[0]))
    sim.step()
tr_t = np.array(tr_t); tr_rho = np.array(tr_rho); tr_M = np.array(tr_M)
b.plot(tr_t, tr_rho, color=BLUE, lw=1.6, label=r"$\rho(t,X_i(t))$ perceived")
b.plot(tr_t, tr_M, color=RED, lw=1.6, ls="--", label=r"$M_i(t)$ memory")
b.fill_between(tr_t, tr_M, tr_rho, where=tr_rho >= tr_M, color=GREEN, alpha=0.22,
               interpolate=True, lw=0)
b.fill_between(tr_t, tr_M, tr_rho, where=tr_rho < tr_M, color=RED, alpha=0.15,
               interpolate=True, lw=0)
k0 = int(np.argmax(tr_rho - tr_M))
b.annotate(r"lag $\tau_m=0.25$", xy=(tr_t[k0], 0.5 * (tr_rho[k0] + tr_M[k0])),
           xytext=(0.42, 0.30), textcoords="axes fraction", fontsize=8.5,
           arrowprops=dict(arrowstyle="->", lw=0.9))
b.set(xlabel="time $t$", ylabel="concentration",
      title=r"(b) temporal comparison along one trajectory ($N=200$, seed 7)")
b.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
bi = b.inset_axes([0.10, 0.58, 0.40, 0.36])
bi.plot(tr_t, tumble_rate(tr_rho - tr_M, lam0, lam1, kap), color="k", lw=1.0)
bi.axhline(lam0, color=GREY, ls=":", lw=0.8)
bi.set_ylim(0, 10); bi.tick_params(labelsize=7)
bi.set_title(r"$\lambda(t)$", fontsize=8, pad=2); bi.grid(alpha=0.2)

# ---------------------------------------------------- (c) interaction kernels
c = fig.add_subplot(gs[1, 0])
z = np.logspace(-3.9, np.log10(0.5), 1200)
Ks = A_s * np.clip(1 - z / r_c, 0, None) * z / np.sqrt(z ** 2 + eps_s ** 2)
Kl = A_l * np.exp(-z / xi) * z / (z ** 2 + reg ** 2)
c.loglog(z, Ks, color=GREEN, lw=2, label=r"repulsion $|K_s^\varepsilon|$, $A_s=12$")
c.loglog(z, Kl, color=ORANGE, lw=2, label=r"attraction $|K_\ell^\varepsilon|$, $A_\ell=0.10$")
c.loglog(z, A_l * z / (z ** 2 + reg ** 2), color=ORANGE, lw=1, ls=":",
         alpha=0.8, label="unscreened (not used)")
for xv, lbl, col in ((eps_s, r"$\varepsilon_s$", GREEN), (reg, r"$\varepsilon$", ORANGE),
                     (r_c, r"$r_c$", GREEN), (xi, r"$\xi=\kappa^{-1}$", ORANGE)):
    c.axvline(xv, color=col, ls="--", lw=0.9, alpha=0.65)
    c.annotate(lbl, (xv, 3.2e-3), color=col, fontsize=9, ha="center",
               backgroundcolor="white")
c.set(xlabel=r"separation $|z|$", ylabel="kernel magnitude",
      ylim=(2e-3, 3e1), xlim=(1.5e-4, 0.5),
      title="(c) the two interaction kernels on one axis")
c.legend(fontsize=8.5, loc="upper right")
c.grid(True, which="both", alpha=0.22)

# ---------------------------------------------------- (d) timescale ladder
d = fig.add_subplot(gs[1, 1])
rows = [
    (dt,                      r"time step $\Delta t$",                    GREY),
    (1 / (lam0 + lam1),       r"run, worsening $1/(\lambda_0+\lambda_1)$", RED),
    (xi ** 2 / D_rho,         r"diffusion over $\xi$: $\xi^2/D_\rho$",     BLUE),
    (1 / lam0,                r"run, adapted $1/\lambda_0$",               "k"),
    (tau_m,                   r"memory $\tau_m$",                          RED),
    (r_c / v0,                r"cross $r_c$: $r_c/v_0$",                  GREEN),
    (1 / (beta * 1.0),        r"uptake $1/\beta n_0$",                     BLUE),
    (1 / (lam0 - lam1),       r"run, improving $1/(\lambda_0-\lambda_1)$", GREEN),
    (1 / alpha,               r"decay $1/\alpha$",                         BLUE),
    (L / v0,                  r"cross domain $L/v_0$",                     "k"),
    (24.0,                    r"run horizon $T$",                          GREY),
]
ys = np.arange(len(rows))[::-1]
for y, (t, lbl, col) in zip(ys, rows):
    d.plot([t], [y], "o", color=col, ms=7)
    d.hlines(y, 1e-2, t, color=col, lw=1.2, alpha=0.35)
    d.text(t * 1.25, y, lbl, va="center", fontsize=8.6, color=col)
d.axvspan(1 / (lam0 + lam1), 1 / (lam0 - lam1), color=GREY, alpha=0.12, lw=0)
d.text(np.sqrt(1 / (lam0 ** 2 - lam1 ** 2)), -1.05,
       "run duration modulated by memory (factor 19)", ha="center",
       fontsize=8.5, color="k")
d.set_xscale("log")
d.set(xlim=(2.5e-2, 2.2e3), ylim=(-1.7, len(rows) - 0.3), yticks=[],
      xlabel="time (model units)",
      title="(d) separation of timescales at the reported parameters")
d.grid(True, axis="x", which="both", alpha=0.25)

fig.savefig(os.path.join(FIGURES, "fig1_model.pdf"))
fig.savefig(os.path.join(FIGURES, "fig1_model.png"), dpi=170)
print("saved fig1_model.pdf / .png")
print(f"  xi={xi:.4f}, run lengths v0/lambda = {v0/(lam0+lam1):.4f} .. {v0/(lam0-lam1):.4f}")
