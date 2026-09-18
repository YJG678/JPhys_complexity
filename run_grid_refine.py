"""Convergence of the semi-implicit nutrient field solver (Section: field solver).
Scheme:  (I - dt*D*Lap_h + dt*kappa*I) c^{n+1} = c^n + dt*S(t^{n+1}),
solved spectrally (the periodic 5-point Laplacian is diagonal in the DFT basis).
Manufactured exact solution  c*(x,y,t)=e^{-t}(1 + 0.5 sin(2pi x) cos(2pi y)).
Source S is taken from the CONTINUOUS PDE, so the spatial error reflects the
O(h^2) truncation of the discrete Laplacian; the time error is O(dt) (backward
Euler on diffusion+decay+source)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
FIGURES = os.path.join(HERE, "figures"); os.makedirs(FIGURES, exist_ok=True)
import json
import numpy as np

D, kappa, TWO_PI = 0.1, 0.5, 2 * np.pi

def cstar(x, y, t):
    return np.exp(-t) * (1.0 + 0.5 * np.sin(TWO_PI * x) * np.cos(TWO_PI * y))

def source(x, y, t):
    g = np.sin(TWO_PI * x) * np.cos(TWO_PI * y)
    return np.exp(-t) * ((kappa - 1.0) * (1.0 + 0.5 * g) + D * TWO_PI**2 * g)

def lap_eigenvalues(N, h):
    k = np.arange(N)
    lam1d = (2 * np.cos(TWO_PI * k / N) - 2) / h**2      # periodic 5-point
    return lam1d[:, None] + lam1d[None, :]

def solve(N, dt, T):
    h = 1.0 / N
    xs = (np.arange(N) + 0.0) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    c = cstar(X, Y, 0.0)
    lam = lap_eigenvalues(N, h)
    denom = 1.0 - dt * D * lam + dt * kappa           # multiplier in Fourier space
    nsteps = int(round(T / dt))
    for n in range(nsteps):
        t1 = (n + 1) * dt
        rhs = c + dt * source(X, Y, t1)
        c = np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))
    err = np.sqrt(np.mean((c - cstar(X, Y, nsteps * dt))**2))   # grid L2 (RMS)
    return err

# ---- spatial convergence: dt tiny, refine grid ----
T, dt_fine = 0.02, 2e-4
spatial = {}
for N in [16, 32, 64, 128]:
    spatial[N] = float(solve(N, dt_fine, T))
    print(f"space N={N:>4}: L2 err = {spatial[N]:.3e}", flush=True)
Ns = np.array(sorted(spatial)); es = np.array([spatial[n] for n in Ns])
p_space = np.polyfit(np.log(1.0 / Ns), np.log(es), 1)[0]
print(f"spatial order = {p_space:.3f} (expected ~2)")

# ---- temporal convergence: fine grid, refine dt ----
Nfix, T2 = 128, 0.1
temporal = {}
for dt in [0.02, 0.01, 0.005, 0.0025]:
    temporal[dt] = float(solve(Nfix, dt, T2))
    print(f"time dt={dt}: L2 err = {temporal[dt]:.3e}", flush=True)
dts = np.array(sorted(temporal)); et = np.array([temporal[d] for d in dts])
p_time = np.polyfit(np.log(dts), np.log(et), 1)[0]
print(f"temporal order = {p_time:.3f} (expected ~1)")

json.dump({"spatial": spatial, "p_space": float(p_space),
           "temporal": temporal, "p_time": float(p_time)},
          open(os.path.join(RESULTS, "grid_refine.json"), "w"), indent=0)
print("saved grid_refine.json")
