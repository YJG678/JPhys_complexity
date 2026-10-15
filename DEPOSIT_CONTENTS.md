# What is in this deposit

Everything needed to regenerate every number and figure in the paper.

## Solver
- `hybrid_model.py`   2D particle model, cell list, Barnes-Hut quadtree, metrics
- `coupled_model.py`  coupled particle-field model: screened long-range kernel,
                      exponential-uptake / semi-implicit field split, lattice-sum reference
- `model3d.py`        d-dimensional model and octree Barnes-Hut (3D)
- `bh_numba.py`       compiled array-based quadtree and cell list (optional, needs numba)

## Verification
- `test_coupled.py`          19 checks (positivity, kernel limits, screening,
                             minimum-image vs lattice sum, cell-list exactness)
- `verify_paper_numbers.py`  checks every value quoted in Sections 7-8 against
                             the stored results and prints PASS / FAIL / MISSING

## Drivers (regenerate the data from scratch)
- `run_all.py`  16-stage driver; `--quick` for a reduced grid, `--list` to see stages
- individual drivers: `run_forceerr.py`, `run_grid_refine.py`, `run_dt_refine.py`,
  `run_meanfield.py`, `run_production.py`, `run_ablation.py`, `run_exp1.py`,
  `run_exp2.py`, `run_exp3.py`, `run_exp4.py`, `run_exp5.py`,
  `run_scaling_timing.py`, `run_interaction_counts.py`, `gen_snapshots.py`

## Figure scripts (one per figure, reading only files in results/)
- `make_fig_model.py`     Figure 1   (model response, memory trace, kernels, timescales)
- `make_fig_forceerr.py`  Figure 7   (force accuracy)
- `make_fig_exp1.py`      Figure 9   (representative run)
- `make_fig_uptake.py`    Figure 10  (uptake sweep)
- `run_exp2.py`           Figure 11  (regime map; `--replot` redraws from stored data)
- `make_paper_figures.py` Figure 12  (scalability) and alternative composites
- `make_fig_exp5.py`      Figure 13  (2D vs 3D)

## Data
- `results/*.json`, `results/*.npz`  the stored output of every driver above
- `figures/*.pdf`, `figures/*.png`   the figures as published

## Parameters and seeds
Every driver sets its parameters and its random seed at the top of the file;
no parameter is read from the environment and no seed is drawn from the clock,
so a re-run reproduces the deposited results exactly on the same platform.

## Reproducing
    pip install -r requirements.txt     # numpy, scipy, matplotlib; numba optional
    python run_all.py                   # full run
    python run_all.py --quick           # reduced grids, minutes rather than hours
    python verify_paper_numbers.py      # check the paper's numbers against results/
