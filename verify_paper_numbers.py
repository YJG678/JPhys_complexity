"""Check every numerical claim in Sections 7-8 against the stored results.

Reads only the files in results/ -- it runs no simulation -- and for each number
quoted in the paper prints the stored/recomputed value and a verdict:

    PASS     stored value agrees with the paper within tolerance
    FAIL     stored value disagrees  -> the paper must be corrected
    MISSING  no stored data          -> run the script named in the last column

Usage:  python verify_paper_numbers.py
Exit code 0 if nothing FAILs (MISSING is not a failure), 1 otherwise.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")
import numpy as np

def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    if name.endswith(".json"):
        return json.load(open(p))
    return dict(np.load(p))

FE, GR, DT = load("forceerr.json"), load("grid_refine.json"), load("dt_refine.json")
MF, TM, CN = load("meanfield.json"), load("exp3_numba.json"), load("exp3_counts.npz")
PR, E5 = load("production.json"), load("exp5.npz")

rows = []
def claim(section, what, paper, got, ok, script):
    rows.append((section, what, paper, got, ok, script))

def near(a, b, rel=0.05):
    return a is not None and b is not None and abs(a - b) <= rel * max(abs(b), 1e-30)

# ---------------------------------------------------------------- Section 7.1
if FE:
    e2 = max(v["E2"] for v in FE["cell"].values())
    ei = max(v["Einf"] for v in FE["cell"].values())
    claim("7.1", "cell-list E_2 <= 7.3e-16", "7.3e-16", f"{e2:.1e}", e2 <= 1e-15, "run_forceerr.py")
    claim("7.1", "cell-list E_inf <= 1.1e-15", "1.1e-15", f"{ei:.1e}", ei <= 2e-15, "run_forceerr.py")
    ms = FE.get("monopole", {}).get("slope")
    claim("7.1", "per-cell monopole slope", "2.03", f"{ms:.3f}" if ms else "-", near(ms, 2.03, 0.03), "run_forceerr.py")
    th = sorted(FE["bh_theta"], key=float)
    lo, hi = FE["bh_theta"][th[0]]["E2"], FE["bh_theta"][th[-1]]["E2"]
    claim("7.1", "BH global E_2 at theta=0.2", "2.7e-2", f"{lo:.2e}", near(lo, 2.7e-2, 0.1), "run_forceerr.py")
    claim("7.1", "BH global E_2 at theta=0.7", "4.5e-2", f"{hi:.2e}", near(hi, 4.5e-2, 0.1), "run_forceerr.py")
    claim("7.1", "BH global fitted slope", "0.38", f"{FE['bh_slope']:.3f}", near(FE["bh_slope"], 0.38, 0.1), "run_forceerr.py")
    eil = [FE["bh_theta"][t]["Einf"] for t in th]
    claim("7.1", "BH E_inf range", "0.08-0.18", f"{min(eil):.2f}-{max(eil):.2f}",
          min(eil) >= 0.06 and max(eil) <= 0.22, "run_forceerr.py")
else:
    claim("7.1", "force-accuracy numbers", "several", "-", None, "run_forceerr.py")

# ---------------------------------------------------------------- Section 7.2
if GR:
    claim("7.2", "field solver, spatial order", "1.96", f"{GR['p_space']:.3f}", near(GR["p_space"], 1.96, 0.03), "run_grid_refine.py")
    claim("7.2", "field solver, temporal order", "0.97", f"{GR['p_time']:.3f}", near(GR["p_time"], 0.97, 0.05), "run_grid_refine.py")
else:
    claim("7.2", "field-solver orders", "1.96 / 0.97", "-", None, "run_grid_refine.py")
if DT:
    r = DT.get("rate", DT.get("slope"))
    claim("7.2", "particle time-step order", "1.10", f"{r:.3f}" if r else "-", near(r, 1.10, 0.05), "run_dt_refine.py")
else:
    claim("7.2", "particle time-step order", "1.10", "-", None, "run_dt_refine.py")

# ---------------------------------------------------------------- Section 7.3
if MF:
    ss, sl = MF["slope_stated"], MF["slope_legacy"]
    claim("7.3", "mean-field exponent (as published)", "-0.505", f"{sl:.3f}", near(sl, -0.505, 0.06), "run_meanfield.py")
    claim("7.3", "mean-field exponent (stated metric)", "-0.5 expected", f"{ss:.3f}", near(ss, -0.5, 0.12), "run_meanfield.py")
    for k, lbl in (("N_ref", "reference N"), ("n_realisations", "realisations"),
                   ("smoothing_width_eps", "smoothing width eps"), ("t_eval", "evaluation time")):
        claim("7.3", f"R2.1: {lbl}", "was unstated", str(MF[k]), True, "run_meanfield.py")
else:
    claim("7.3", "mean-field exponent + the 4 parameters", "-0.505", "-", None, "run_meanfield.py")

# ------------------------------------------------------------ Section 7.4-7.5
if PR:
    ab = PR.get("stage3_ablation", {})
    for k, (rc, A) in {"full": (0.579, 5.93), "nomem": (1.040, 1.18),
                       "norep": (0.498, 56.45), "noatt": (0.870, 1.00)}.items():
        if k in ab:
            claim("7.4", f"ablation {k}: Rhat_c", f"{rc}", f"{ab[k]['Rc'][0]:.3f}", near(ab[k]["Rc"][0], rc, 0.08), "run_production.py")
            claim("7.4", f"ablation {k}: A", f"{A}", f"{ab[k]['A'][0]:.2f}", near(ab[k]["A"][0], A, 0.15), "run_production.py")
    bs = PR.get("stage2_beta", {})
    for b, A in (("0.1", 23.93), ("0.25", 21.74), ("0.5", 15.72), ("1.0", 5.93), ("2.0", 1.69)):
        if b in bs:
            claim("7.4", f"uptake sweep beta={b}: A", f"{A}", f"{bs[b]['A'][0]:.2f}", near(bs[b]["A"][0], A, 0.15), "run_production.py")
    claim("7.4", "retuned A_ell", "0.10", str(PR.get("eps_l_chosen")), PR.get("eps_l_chosen") == 0.1, "run_production.py")
else:
    claim("7.4", "ablation + uptake sweep tables", "several", "-", None, "run_production.py")

# ------------------------------------------------------------------ Section 8
# Wall-clock exponents.  These are MEASUREMENTS, not mathematical constants:
# they move by O(0.1) between machines, between runs, and with the fit window,
# because the benchmark is dominated by memory traffic at large N.  The paper
# should therefore quote them to two significant figures, and this check uses
# an absolute tolerance of 0.15 on the exponent rather than a relative one.
# Edit PAPER_8_1 to whatever Section 8.1 actually says.
PAPER_8_1 = {"slope_direct": 2.00, "slope_cell": 1.00, "slope_bh": 1.10}
EXP_TOL = 0.15

def near_abs(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol

if TM:
    for key, lbl in (("slope_direct", "direct wall-clock"),
                     ("slope_cell", "cell-list wall-clock"),
                     ("slope_bh", "Barnes-Hut wall-clock")):
        paper = PAPER_8_1[key]
        v = TM.get(key)
        se = TM.get("se_" + key.split("_")[1], None)
        got = "-" if v is None else (f"N^{v:.3f}" +
              (f"+/-{se:.3f}" if isinstance(se, float) and se == se else ""))
        claim("8.1", lbl, f"N^{paper}", got,
              near_abs(v, paper, EXP_TOL) if v else None, "run_scaling_timing.py")
    if TM.get("estimator") != "min over repeats":
        claim("8.1", "timing estimator", "min over repeats",
              str(TM.get("estimator", "median, direct timed once")), False,
              "re-run run_scaling_timing.py (current version)")
    for key, paper, lbl in (("t_1e6_cell", 0.4, "cell list at N=1e6 (s)"),
                            ("t_1e6_bh_total", 2.8, "Barnes-Hut at N=1e6 (s)")):
        v = TM.get(key)
        claim("8.1", lbl, f"{paper:.0f}", f"{v:.1f}" if v else "not reached", near(v, paper, 0.30) if v else None, "run_scaling_timing.py --max-n 1e6")
else:
    claim("8.1", "wall-clock exponents and N=1e6 timings", "N^2.00/1.00/1.10, 0.4s, 2.8s", "-", None, "run_scaling_timing.py")
if CN is not None:
    Ns, tot = np.asarray(CN["Ns"], float), np.asarray(CN["tot"], float)
    m = Ns >= 1e4
    if m.sum() >= 2:
        s = float(np.polyfit(np.log(Ns[m]), np.log(tot[m]), 1)[0])
        claim("8.1", "accepted interactions", "N^1.12", f"N^{s:.3f}", near(s, 1.12, 0.08), "run_interaction_counts.py")
        ref = float(np.polyfit(np.log(Ns[m]), np.log(Ns[m]*np.log(Ns[m])), 1)[0])
        claim("8.1", "N log N reference exponent", "N^1.09", f"N^{ref:.3f}", near(ref, 1.09, 0.05), "run_interaction_counts.py")
else:
    claim("8.1", "accepted-interaction exponent", "N^1.12", "-", None, "run_interaction_counts.py")
if E5 is not None:
    # run_exp5.py stores these as exp_oct / exp_direct; the older key names
    # b_oct / b_dir are accepted too so an old results file still verifies.
    for keys, paper, lbl in ((("exp_oct", "b_oct"), 1.34, "octree wall-clock"),
                             (("exp_direct", "b_dir"), 1.89, "3D direct wall-clock")):
        v = next((float(E5[k]) for k in keys if k in E5), None)
        claim("8.2", lbl, f"N^{paper}", f"N^{v:.3f}" if v is not None else "-",
              near_abs(v, paper, EXP_TOL) if v is not None else None, "run_exp5.py")
else:
    claim("8.2", "2D/3D scaling exponents", "N^1.34 / N^1.89", "-", None, "run_exp5.py")

# ------------------------------------------------------------------- report
quick_files = [n for n, d in (("production.json", PR), ("meanfield.json", MF)) if d and d.get("quick")]
if TM and TM.get("Ns") and max(TM["Ns"]) < 1e6:
    quick_files.append("exp3_numba.json (max N = %g)" % max(TM["Ns"]))
if quick_files:
    print()
    print("!" * 104)
    print("  WARNING: some stored results were produced at REDUCED settings:")
    for f in quick_files:
        print(f"      {f}")
    print("  Reduced runs do NOT reproduce the published numbers and will show as FAIL below.")
    print("  Re-run the named scripts WITHOUT --quick before judging any failure.")
    print("!" * 104)


W = (6, 40, 16, 16, 8)
print("=" * 104)
print("  VERIFICATION OF THE NUMERICAL CLAIMS IN SECTIONS 7-8")
print("  (reads results/ only; runs no simulation)")
print("=" * 104)
print(f"{'sec':<{W[0]}} {'quantity':<{W[1]}} {'paper':>{W[2]}} {'stored':>{W[3]}}  {'verdict':<{W[4]}} script")
print("-" * 104)
nf = nm = 0
for sec, what, paper, got, ok, script in rows:
    if ok is None:
        v = "MISSING"; nm += 1
    elif ok:
        v = "PASS"
    else:
        v = "**FAIL**"; nf += 1
    tail = f" {script}" if ok is None or not ok else ""
    print(f"{sec:<{W[0]}} {what:<{W[1]}} {str(paper):>{W[2]}} {str(got):>{W[3]}}  {v:<{W[4]}}{tail}")
print("-" * 104)
print(f"  {len(rows)-nf-nm} pass, {nf} fail, {nm} missing")
if nm:
    print("  MISSING entries need the named script run; they are not failures.")
print("=" * 104)
sys.exit(1 if nf else 0)

# ---------------------------------------------------------------- Section 8.1
# paste-ready sentence, built from the stored fit
if TM and TM.get("slope_cell") is not None:
    w = TM.get("fit_window", {})
    def two(x):
        return "-" if x is None else f"{x:.2f}"
    print("\nSection 8.1, ready to paste (2 s.f., from results/exp3_numba.json):")
    print(f"  direct summation scales as N^{two(TM.get('slope_direct'))}, the cell list "
          f"as N^{two(TM.get('slope_cell'))} and the Barnes-Hut traversal as "
          f"N^{two(TM.get('slope_bh'))}")
    print(f"  (least-squares fits over N >= {w.get('direct', 0):g}, "
          f"{w.get('cell', TM.get('fit_nmin', 0)):g} and "
          f"{w.get('bh_total', TM.get('fit_nmin', 0)):g} respectively; "
          f"minimum of {TM.get('repeats')} repeats)")
