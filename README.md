# Reproduction package
## A scalable hybrid particle–field method for memory-driven chemotactic active matter

Everything needed to reproduce the computational results in the paper. Pure Python;
no compilation step; one optional dependency.

---

## Quick start (Windows PowerShell)

```powershell
cd path\to\hybrid_chemotaxis_repro
python -m pip install -r requirements.txt
python run_all.py --quick
```

macOS / Linux is identical with `python3` in place of `python`.

`--quick` runs everything at reduced size in about four minutes and is the right first
step: it confirms the install works.

> **`--quick` will not reproduce the numbers in the paper, and is not meant to.** It uses
> `N = 150`, 400 steps and 2 seeds instead of `N = 300`, 1200 steps and 3–4 seeds, which
> is far too noisy to resolve the ablation or fix the retuned `A_ell`. Expect it to select
> a different `A_ell` and to show the mechanism trend only roughly. Use it to check that
> the code runs, then **drop the flag** for the production settings the paper reports
> (30–60 min on a laptop).

**Run the scripts from inside this folder.** Every script resolves its own location, so
`python run_production.py` works from anywhere, but the relative paths in this README
assume you are in the package directory.

### If you saw `ModuleNotFoundError: No module named 'coupled_model'`

That error means Python could not find `coupled_model.py` on its import path — almost
always because the script being run lives in a different folder from the modules. This
package is deliberately **flat**: every module and every script sits in this one
directory, and each script adds its own directory to `sys.path` on startup. So either

- keep all the files together in one folder (recommended), or
- if you move a driver script elsewhere, add at the top of it:
  ```python
  import sys; sys.path.insert(0, r"C:\path\to\hybrid_chemotaxis_repro")
  ```

There is no `simulation.py` in this package. If you have one of your own, put it in this
folder and its `from coupled_model import ...` will resolve.

---

## What is here

### Model modules

| file | what it is |
|---|---|
| **`coupled_model.py`** | **The model of the paper.** Particles coupled to a dynamic reaction–diffusion nutrient field: CIC deposition, semi-implicit spectral field solve with exact local uptake, CIC interpolation, screened long-range kernel, regularised short-range kernel. This is what the physics results use. |
| `hybrid_model.py` | The earlier static-field model (prescribed `rho(x) = 1/(1+|x-C|)`). Retained because the force-accuracy benchmarks are cleanest against it, and so the two can be compared directly. Not used for the physics results. |
| `bh_numba.py` | Compiled (numba) Barnes–Hut tree and cell list, for the large-`N` scaling benchmark only. |
| `model3d.py` | Three-dimensional variant (octree). |

### Scripts, and the paper element each produces

| script | produces | paper element |
|---|---|---|
| `test_coupled.py` | 19 verification checks on `coupled_model` | — |
| `run_grid_refine.py` | field-solver orders in space and time | Tab. 2, Fig. 8a,b |
| `run_dt_refine.py` | particle time-step order | Tab. 2, Fig. 8c |
| `run_forceerr.py` | cell-list exactness, Barnes–Hut error | Tab. 2, Fig. 7 |
| `run_meanfield.py` | mean-field convergence exponent **and the four parameters Referee 2 asked for** | Tab. 2, Fig. 8d, §7.3 |
| `run_interaction_counts.py` | accepted-interaction counts (needs numba) | Tab. 6, Fig. 12b,c |
| `run_scaling_timing.py` | wall-clock scaling exponents and the N=10⁶ timings (needs numba) | Tab. 6, Fig. 12a |
| `run_exp5.py` | 2D vs 3D comparison | §8.2, Fig. 13 |
| `run_exp2.py` | regime sweep | Fig. 11 |
| `run_production.py` | coupled ablation + uptake sweep | Tab. 4–5, Figs. 9–10 |
| `make_fig_uptake.py` | the uptake figure | Fig. 10 |
| **`verify_paper_numbers.py`** | **checks every number in §7–8 against `results/`** | — |
| `run_exp1.py`, `run_exp4.py`, `run_ablation.py`, `gen_snapshots.py`, `make_paper_figures.py` | earlier static-field drivers, retained for comparison | Figs. 9–11 (static field) |

---

## Verifying the paper's numbers

```powershell
python verify_paper_numbers.py
```

Runs no simulation. It reads `results/` and prints, for every number quoted in
Sections 7 and 8, the value the paper states, the value actually stored, and a verdict:

- **PASS** — agrees within tolerance
- **FAIL** — disagrees; the paper needs correcting
- **MISSING** — no stored data yet; the last column names the script to run

On a fresh checkout every row is MISSING. After `python run_all.py` they should all be
PASS. If any result was produced with `--quick`, the script prints a warning banner
before the table, because reduced runs will show as FAIL.

### Two gaps this replaces

Two numbers in the first submission could not be traced to code, and both are now fixed:

1. **The wall-clock scaling exponents** (direct `N^2.15`, cell list `N^0.99`,
   Barnes–Hut `N^1.38`) and the "24 s / 26 s at N = 10⁶" timings were derived from a file
   `exp3_numba.json` that no deposited script produced, and the exponents themselves were
   fitted at plot time and never stored. `run_scaling_timing.py` now produces that file
   and stores the fitted exponents in it.
2. **The mean-field exponent.** `run_exp4.py` (retained) measures the *L¹* distance of
   each realisation from the **ensemble mean at the same N**, on a 20×20 histogram with
   no smoothing kernel. That is not the metric Eq. (metric-meanfield) defines — which is
   an *L²* norm of the **φ_ε-smoothed** density against a **large-N reference** — and it
   is why the smoothing width and reference N could not be quoted: the script has
   neither. `run_meanfield.py` computes the stated metric, records all four parameters,
   and reports the legacy quantity alongside so the two can be compared.

---

Outputs go to `results/` (JSON, NPZ) and `figures/` (PDF, PNG). Both are created on first
run.

---

## Reproducing specific results

```powershell
python run_all.py --list              # show the stages
python run_all.py --only 6 7          # just the production runs and their figure
python run_production.py              # full production settings
python test_coupled.py                # verification suite alone
```

`run_production.py` runs three stages and writes `results/production.json`:

1. **Retune `A_ell`.** Sweeps the long-range strength and selects the value whose mean
   aggregation index is closest to 10. The selected value is recorded in the JSON as
   `eps_l_chosen`; with the shipped parameters it is **0.10**.
2. **Uptake sweep.** Holds `A_ell` fixed and sweeps `beta`. This is the mechanism result:
   the aggregation index falls monotonically as uptake strengthens.
3. **Ablation.** Full model versus no memory, no short-range repulsion, no long-range
   attraction.

---

## Expected output

The verification suite should end with `19/19 checks passed`. Representative values from
the full (non-`--quick`) run, on which the paper's tables are based:

```
field solver          spatial order 1.956      temporal order 0.972
particle integrator   time-step order 1.104
cell list vs direct   E2 <= 7.3e-16            (exact to machine precision)
Barnes-Hut, theta=0.2 E2 = 2.66e-02            (clustered configuration)

ablation (N=300, t=24, 4 seeds, A_ell=0.10)
  full model      Rhat_c 0.579+-0.032    A  5.93+-0.85
  no memory       Rhat_c 1.040+-0.039    A  1.18+-0.33
  no repulsion    Rhat_c 0.498+-0.087    A 56.45+-10.63
  no attraction   Rhat_c 0.870+-0.011    A  1.00+-0.04

uptake sweep (A_ell = 0.10 fixed)
  beta 0.10   Rhat_c 0.316    A 24.57    mean rho 0.1308
  beta 0.25   Rhat_c 0.327    A 21.45    mean rho 0.1038
  beta 0.50   Rhat_c 0.404    A 15.54    mean rho 0.0814
  beta 1.00   Rhat_c 0.579    A  5.93    mean rho 0.0499
  beta 2.00   Rhat_c 0.795    A  1.69    mean rho 0.0296
```

Random seeds are fixed (`numpy.random.default_rng(seed)`), so these are reproducible
exactly on the same NumPy version. Small differences across NumPy releases are expected
in the last digits and do not affect any conclusion.

---

## Notes on the model, for a reader comparing against the paper

- **The field is dynamic, not slaved.** At the default parameters the nutrient relaxes on
  `L^2/D_rho = 50` against a transport time `L/v0 = 8.3`, so it cannot be adiabatically
  eliminated. `hybrid_model.py`, by contrast, prescribes the field; running the two side
  by side (as `run_production.py` allows) shows what the coupling changes.
- **The long-range kernel is screened**, at `kappa = 1/xi` with
  `xi = sqrt(D_rho/(alpha + beta n0))` — the screening length the uptake generates. This
  is not a numerical convenience: for the unscreened `1/r` kernel the periodic sum is only
  conditionally convergent and the minimum-image convention is uncontrolled. Set
  `kappa_screen=0.0` in the `CoupledSim` constructor to recover the unscreened kernel for
  comparison. `test_coupled.py` T3 quantifies the difference.
- **The uptake term is integrated exactly**, `rho* = rho^n exp(-dt beta U^n)`, rather than
  explicitly. The explicit form is stable only for `dt*beta*max(U) < 1`, and `max(U)`
  grows without bound as particles aggregate, so it fails in the collapsed regime. The
  exponential update is unconditionally positive and stable; `test_coupled.py` T7 tests it
  at `dt*beta*max(U)` up to 500.
- **The short-range kernel is regularised at the origin** (`reg_s`), without which it has
  a jump there and is not in `W^{1,inf}`, contrary to the assumption used in the
  well-posedness theorem. At `reg_s = 3e-4` the force changes by 0.1%.

---

## Troubleshooting

**`ModuleNotFoundError`** — see the note above; keep the files in one folder.

**`ModuleNotFoundError: No module named 'numba'`** — only `run_interaction_counts.py`
needs it. `run_all.py` detects its absence and skips that stage. Install with
`pip install numba` if you want the scaling benchmark.

**Matplotlib backend errors on a headless machine** — the figure script already sets the
non-interactive `Agg` backend, so this should not arise; if it does, set
`MPLBACKEND=Agg` in the environment.

**A run seems slow** — try `--quick` first. The dominant cost is the O(N²) long-range
force in `coupled_model.long_range_screened_vec`; it is vectorised but quadratic, which is
fine at the `N = 300` used for the physics and is why the large-`N` scaling benchmark uses
the compiled tree in `bh_numba.py` instead.

---

## Citing

If you use this code, please cite the paper. Random seeds, parameter values and the
figure-generating scripts are all included so that every number in the paper can be traced
to the code that produced it.
