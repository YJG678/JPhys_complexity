"""Master driver: reproduces every computational result in the paper.

    python run_all.py --quick     smoke test (~6 min, REDUCED settings)
    python run_all.py             full run   (~1-2 h at production settings)
    python run_all.py --list      list the stages and exit
    python run_all.py --only 5 12 run selected stages only

Outputs land in results/ (JSON, NPZ) and figures/ (PDF, PNG).
Stage 12 checks every number quoted in Sections 7-8 against results/ and needs
no simulation of its own; run it any time.

--quick does NOT reproduce the published numbers and is not meant to. Use it to
check the install, then drop the flag.
"""
import os, sys, time, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("results", "figures"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)

#      n, script,                      quick-args,  description,                              required
STAGES = [
    (1,  "test_coupled.py",            [],          "Verification suite (19 checks)",          True),
    (2,  "run_grid_refine.py",         [],          "Field solver convergence (Tab.2, Fig.8ab)", True),
    (3,  "run_dt_refine.py",           [],          "Particle time-step order (Tab.2, Fig.8c)", True),
    (4,  "run_forceerr.py",            [],          "Force accuracy (Tab.2, Fig.7)",           True),
    (5,  "run_meanfield.py",           ["--quick"], "Mean-field convergence (Tab.2, Fig.8d)",  True),
    (6,  "run_interaction_counts.py",  [],          "Interaction counts (Tab.6) [numba]",      False),
    (7,  "run_scaling_timing.py",      ["--quick"], "Wall-clock scaling (Tab.6, Fig.12) [numba]", False),
    (8,  "run_exp5.py",                [],          "2D vs 3D (Sec.8.2, Fig.13)",              False),
    (9,  "run_exp2.py",                [],          "Regime sweep (Fig.11)",                   False),
    (10, "run_production.py",          ["--quick"], "Coupled ablation + uptake sweep (Tab.4-5)", True),
    (11, "make_fig_uptake.py",         [],          "Uptake-sweep figure (Fig.10)",            True),
    (12, "verify_paper_numbers.py",    [],          "Check paper numbers against results/",    True),
]

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", nargs="*", type=int, default=None)
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

if a.list:
    for n, f, _, d, req in STAGES:
        print(f"  {n:>2}. {f:<28} {d}{'' if req else '   [optional]'}")
    sys.exit(0)

def have_numba():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False

NEEDS_NUMBA = {"run_interaction_counts.py", "run_scaling_timing.py"}
t0, failed = time.time(), []
for n, script, qargs, desc, required in STAGES:
    if a.only and n not in a.only:
        continue
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print(f"\n--- stage {n}: {desc}\n    SKIPPED (missing {script})")
        continue
    if script in NEEDS_NUMBA and not have_numba():
        print(f"\n--- stage {n}: {desc}\n    SKIPPED (numba not installed; pip install numba)")
        continue
    args = list(qargs) if a.quick else []
    print(f"\n{'='*74}\n  STAGE {n}: {desc}\n{'='*74}", flush=True)
    r = subprocess.run([sys.executable, path] + args, cwd=HERE)
    if r.returncode != 0 and script != "verify_paper_numbers.py":
        failed.append((n, script))
        print(f"  ** stage {n} exited with code {r.returncode} (continuing)")

print(f"\n{'='*74}")
print(f"  finished in {time.time()-t0:.0f}s")
print(f"  results -> {os.path.join(HERE,'results')}")
print(f"  figures -> {os.path.join(HERE,'figures')}")
if failed:
    print("  stages that failed: " + ", ".join(f"{n} ({s})" for n, s in failed))
print("="*74)
sys.exit(1 if failed else 0)
