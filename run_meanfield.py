"""Mean-field convergence: E_N = || n_N - nbar ||_L2  versus N.

Implements the metric the manuscript states, Eq. (metric-meanfield):

    n_N(x) = (1/N) sum_i phi_eps(x - X_i)          smoothed empirical density
    nbar   = the same, from a large-N reference run
    E_N    = || n_N - nbar ||_{L2}                 grid L2 norm

and records the four quantities Referee 2 asked for -- reference particle number,
number of independent realisations, smoothing width, evaluation time -- in the
output file, so they can be read off rather than inferred.

NOTE ON THE FIRST SUBMISSION.  The script that produced the originally reported
exponent (run_exp4.py, retained in this package) measured a *different* quantity:
the L1 distance of each realisation from the ENSEMBLE MEAN AT THE SAME N, on a
20x20 histogram with no smoothing kernel.  That is a defensible finite-N
fluctuation measure, but it is not what Eq. (metric-meanfield) defines, and it has
no smoothing width and no large-N reference -- which is why those two numbers
could not be quoted.  This script computes the stated metric.  Both are reported
in the output so the two can be compared directly.

Usage:  python run_meanfield.py            (full)
        python run_meanfield.py --quick    (reduced)
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
import numpy as np
from hybrid_model import HybridSim

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()

# ---------------------------------------------------------------- parameters
L, M_GRID = 1.0, 64
EPS = 0.03                      # smoothing width of phi_eps (Gaussian)
if a.quick:
    NS        = [250, 500, 1000, 2000]
    N_REF     = 20000
    R         = 6
    NSTEPS    = 200
else:
    NS        = [250, 500, 1000, 2000, 4000, 8000]
    N_REF     = 100000
    R         = 16
    NSTEPS    = 400
DT   = 0.05
TEVAL = NSTEPS * DT
BASE = dict(v0=0.12, lam0=5.0, lam1=4.5, kappa=20, tau_m=0.25, dt=DT)

# ------------------------------------------------- smoothed empirical density
_k1 = 2*np.pi*np.fft.fftfreq(M_GRID, d=L/M_GRID)
_KX, _KY = np.meshgrid(_k1, _k1, indexing="ij")
_GAUSS = np.exp(-0.5*EPS**2*(_KX**2+_KY**2))      # Fourier transform of phi_eps

def smoothed_density(X):
    """n_N = (1/N) sum_i phi_eps(x - X_i), phi_eps a Gaussian of width EPS,
    on an M_GRID^2 periodic grid.  Normalised so that the grid integral is 1."""
    h = L/M_GRID
    g = X/h; i0 = np.floor(g).astype(np.int64); f = g-i0
    i0 %= M_GRID; i1 = (i0+1) % M_GRID
    H = np.zeros((M_GRID, M_GRID))
    for (ix, iy), w in (((i0[:,0],i0[:,1]),(1-f[:,0])*(1-f[:,1])),
                        ((i1[:,0],i0[:,1]),   f[:,0] *(1-f[:,1])),
                        ((i0[:,0],i1[:,1]),(1-f[:,0])*   f[:,1] ),
                        ((i1[:,0],i1[:,1]),   f[:,0] *   f[:,1] )):
        np.add.at(H, (ix, iy), w)
    H /= X.shape[0]*h*h                                    # int H dx = 1
    return np.real(np.fft.ifft2(np.fft.fft2(H)*_GAUSS))    # convolve with phi_eps

def l2(a_, b_):
    return float(np.sqrt(np.sum((a_-b_)**2)*(L/M_GRID)**2))

def hist20(X):
    Hh, _, _ = np.histogram2d(X[:,0], X[:,1], bins=20, range=[[0,L],[0,L]])
    return Hh/Hh.sum()

# -------------------------------------------------------- large-N reference
t0 = time.time()
print(f"reference run: N_ref = {N_REF}, t = {TEVAL}", flush=True)
ref = HybridSim(N=N_REF, seed=99991, **BASE); ref.run(NSTEPS)
nbar = smoothed_density(ref.X)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

# ------------------------------------------------------------------- sweep
print(f"\n{'N':>7} | {'E_N (stated metric)':>20} | {'L1 vs ensemble mean':>20}")
print("-"*56)
E_stated, E_legacy = [], []
for N in NS:
    dens_s, dens_h = [], []
    for r in range(R):
        s = HybridSim(N=N, seed=1000+r, **BASE); s.run(NSTEPS)
        dens_s.append(smoothed_density(s.X)); dens_h.append(hist20(s.X))
    e_s = float(np.mean([l2(d, nbar) for d in dens_s]))
    hm = np.array(dens_h).mean(axis=0)
    e_h = float(np.mean([np.sum(np.abs(d-hm)) for d in dens_h]))
    E_stated.append(e_s); E_legacy.append(e_h)
    print(f"{N:>7} | {e_s:>20.4e} | {e_h:>20.4e}", flush=True)

NSa = np.array(NS, float)
slope_stated = float(np.polyfit(np.log(NSa), np.log(E_stated), 1)[0])
slope_legacy = float(np.polyfit(np.log(NSa), np.log(E_legacy), 1)[0])
print(f"\nfitted exponent, stated metric  E_N ~ N^{slope_stated:.3f}   (expected -0.5)")
print(f"fitted exponent, legacy metric  E_N ~ N^{slope_legacy:.3f}   (expected -0.5)")

out = dict(
    metric="E_N = ||n_N - nbar||_L2, n_N = (1/N) sum_i phi_eps(x-X_i), phi_eps Gaussian",
    N=NS, E_stated=E_stated, E_legacy=E_legacy,
    slope_stated=slope_stated, slope_legacy=slope_legacy,
    # --- the four quantities Referee 2 asked for, recorded explicitly ---
    N_ref=N_REF, n_realisations=R, smoothing_width_eps=EPS, t_eval=TEVAL,
    grid=M_GRID, dt=DT, nsteps=NSTEPS, quick=bool(a.quick),
    seeds=dict(reference=99991, realisations=[1000+r for r in range(R)]))
json.dump(out, open(os.path.join(RESULTS, "meanfield.json"), "w"), indent=1)
print(f"\nsaved results/meanfield.json   (total {time.time()-t0:.1f}s)")
