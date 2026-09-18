"""Wall-clock scaling benchmark -> results/exp3_numba.json

This file was MISSING from the first submission: make_paper_figures.py reads
exp3_numba.json, and the wall-clock exponents quoted in the scalability table
(direct N^2.15, cell list N^0.99, Barnes-Hut N^1.38) and the "24 s / 26 s at
N = 10^6" timings are derived from it, but no script in the deposit produced it.
This script does.

Timing is at fixed number density (L propto sqrt(N)) and excludes one-time numba
compilation.  Every backend, INCLUDING direct summation, is timed with the same
number of repeats and reported as the MINIMUM over repeats: the minimum is the
standard estimator for a benchmark whose noise is one-sided (scheduler, cache,
page faults).  All repeats are stored so the scatter is visible.

The fitted exponents are computed here, with an explicit fit window and a
standard error, and stored in the JSON.  Everything downstream -- the figure
legend and verify_paper_numbers.py -- reads those stored values, so the figure,
the table and the text cannot quote three different exponents.

Usage:  python run_scaling_timing.py                  (up to N = 1e6; slow)
        python run_scaling_timing.py --max-n 100000   (faster)
        python run_scaling_timing.py --quick          (up to N = 20000)
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
import numpy as np
from hybrid_model import short_range_direct_vec, long_range_direct_vec

ap = argparse.ArgumentParser()
ap.add_argument("--max-n", type=float, default=1e6)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()
MAXN = 2e4 if a.quick else a.max_n

try:
    from bh_numba import long_range_bh_numba, short_range_celllist_numba
    HAVE_NUMBA = True
except Exception as e:                                    # pragma: no cover
    HAVE_NUMBA = False
    print(f"numba unavailable ({e}); compiled backends will be skipped")

DENSITY = 400.0
r_c, eps_s, eps_l, reg, theta = 0.06, 1.0, 1.0, 0.02, 0.5

def make(N, seed=0):
    L = np.sqrt(N / DENSITY)
    return np.random.default_rng(seed).random((N, 2)) * L, L

def timeit(fn, reps):
    """Minimum over `reps` timings; returns (best, [all timings])."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.min(ts)), [float(t) for t in ts]

GRID = [n for n in (1000, 2000, 4000, 8000, 10000, 20000, 50000,
                    100000, 200000, 500000, 1000000) if n <= MAXN]
DIRECT_GRID = [n for n in (1000, 2000, 4000, 8000) if n <= MAXN]

res = {"Ns": [], "cell": [], "bh_total": [], "direct": {},
       "raw": {"cell": {}, "bh_total": {}, "direct": {}},
       "density": DENSITY, "theta": theta, "r_c": r_c, "repeats": a.repeats,
       "numba": HAVE_NUMBA, "units": "seconds per force evaluation",
       "estimator": "min over repeats"}

if HAVE_NUMBA:                                            # warm up the JIT
    Xw, Lw = make(2000)
    long_range_bh_numba(Xw, Lw, eps_l, reg, theta)
    short_range_celllist_numba(Xw, Lw, eps_s, r_c)

print(f"{'N':>9} | {'cell list (s)':>14} | {'Barnes-Hut (s)':>15} | {'direct (s)':>12}")
print("-" * 60)
for N in GRID:
    X, L = make(N)
    if HAVE_NUMBA:
        tc, rc_ = timeit(lambda: short_range_celllist_numba(X, L, eps_s, r_c), a.repeats)
        tb, rb_ = timeit(lambda: long_range_bh_numba(X, L, eps_l, reg, theta), a.repeats)
        res["raw"]["cell"][str(N)] = rc_
        res["raw"]["bh_total"][str(N)] = rb_
    else:
        tc = tb = float("nan")
    td = float("nan")
    if N in DIRECT_GRID:
        # same number of repeats as the compiled backends -- timing direct
        # summation once made its fitted exponent the noisiest number in the
        # whole benchmark.
        td, rd_ = timeit(lambda: (short_range_direct_vec(X, L, eps_s, r_c),
                                  long_range_direct_vec(X, L, eps_l, reg)),
                         a.repeats)
        res["direct"][str(N)] = td
        res["raw"]["direct"][str(N)] = rd_
    res["Ns"].append(N); res["cell"].append(tc); res["bh_total"].append(tb)
    print(f"{N:>9} | {tc:>14.4f} | {tb:>15.4f} | "
          f"{'--' if np.isnan(td) else format(td,'.4f'):>12}", flush=True)

def fit(N, T, nmin=0):
    """Log-log least-squares slope with its standard error."""
    N = np.asarray(N, float); T = np.asarray(T, float)
    m = np.isfinite(T) & (N >= nmin) & (T > 0)
    if m.sum() < 2:
        return None, None
    x, y = np.log(N[m]), np.log(T[m])
    b, c = np.polyfit(x, y, 1)
    if m.sum() > 2:
        r = y - (b * x + c)
        se = float(np.sqrt((r @ r) / (m.sum() - 2) / np.sum((x - x.mean()) ** 2)))
    else:
        se = float("nan")
    return float(b), se

# Fit windows.  The compiled backends are fitted on the asymptotic part of the
# grid; direct summation never reaches it (it is limited by the O(N^2) distance
# matrix), so it is fitted on all of its points and its window is recorded
# separately rather than being silently different from the others.
NMIN = 1e4 if MAXN > 5e4 else 0
DMIN = 0.0
res["fit_nmin"] = NMIN                       # kept for backward compatibility
res["fit_window"] = {"cell": NMIN, "bh_total": NMIN, "direct": DMIN}
res["slope_cell"], res["se_cell"] = fit(res["Ns"], res["cell"], NMIN)
res["slope_bh"], res["se_bh"] = fit(res["Ns"], res["bh_total"], NMIN)
res["slope_direct"], res["se_direct"] = fit([int(k) for k in res["direct"]],
                                            [res["direct"][k] for k in res["direct"]],
                                            DMIN)
for k in ("cell", "bh_total"):
    if 1000000 in res["Ns"]:
        res[f"t_1e6_{k}"] = res[k][res["Ns"].index(1000000)]

print("\nfitted exponents:")
for k, se, win, lbl in (("slope_direct", "se_direct", DMIN, "direct       "),
                        ("slope_cell", "se_cell", NMIN, "cell list    "),
                        ("slope_bh", "se_bh", NMIN, "Barnes-Hut   ")):
    if res[k] is None:
        print(f"  {lbl} (insufficient data)"); continue
    e = res[se]
    pm = "" if e is None or not np.isfinite(e) else f" +/- {e:.3f}"
    print(f"  {lbl} N^{res[k]:.3f}{pm}   (fit over N >= {win:g})")
print("\nQuote these in Section 8.1 to 2 significant figures; the third digit "
      "is not reproducible across machines.")
json.dump(res, open(os.path.join(RESULTS, "exp3_numba.json"), "w"), indent=1)
print("\nsaved results/exp3_numba.json")
