# Taylor Approximation Validity Study

Tools for studying the validity range of 1st-order (Jacobian) and 2nd-order (Hessian) Taylor approximations to the larnd-sim-jax differentiable simulation.

## Idea

Given nominal simulation parameters, compute the Jacobian and Hessian of the expected first-hit ADC per pixel w.r.t. each physics parameter. Then sweep perturbations and compare:

- **True:** full resimulation at perturbed parameter value
- **1st order:** `f(theta_0) + J * delta`
- **2nd order:** `f(theta_0) + J * delta + 0.5 * H * delta^2`

The "validity range" is the perturbation range where the approximation error stays below a configurable ADC threshold.

## Pipeline

The full differentiable pipeline per event:

1. `simulate_wfs` (LUT-based waveform simulation)
2. `simulate_probabilistic` (probabilistic FEE with average noise)
3. `get_average_hit_values` (collapse to expected first-hit ADC per pixel)

The Jacobian is computed via `jax.jacfwd` through this pipeline. The Hessian uses `jax.jacfwd(jax.jacfwd(...))` — this is possible because the FEE functions in `src/larndsim/fee_jax.py` use smooth (C-infinity) approximations (`_soft_max`, `_soft_where`) instead of hard `jnp.maximum`/`jnp.where`, and `lax.stop_gradient` on discrete `top_k` indices.

## Files

| File | Description |
|------|-------------|
| `scan_utils.py` | Core library: `setup_params_and_tracks`, `compute_event_scan`, `save_results` |
| `run_taylor_scan.py` | Standalone script: loops over parameters x events for one input file. Accepts `--input_id`. |
| `submit_taylor_scan.sh` | SLURM submission: `bash submit_taylor_scan.sh <id>` for one input, `bash submit_taylor_scan.sh all` for all 22 inputs. |
| `plot_taylor_scan.ipynb` | Loads all `results/taylor_scan_*.pkl`, merges events, plots per-parameter grids and validity ranges. No GPU needed. |
| `results/` | Output directory for pickle files (one per input file, gitignored) |
| `logs/` | SLURM log files (gitignored) |

## Running

```bash
# From repo root:

# Single input file
bash tests/taylor/submit_taylor_scan.sh 0

# All input files (submits 22 jobs)
bash tests/taylor/submit_taylor_scan.sh all

# Monitor
tail -f tests/taylor/logs/taylor_scan_0-<jobid>.out

# Plot (no GPU needed, loads from results/)
jupyter notebook tests/taylor/plot_taylor_scan.ipynb
```

## Configuration

In `run_taylor_scan.py`:
- `--input_id`: which `prepared_data/input_<id>.h5` to process
- `--params`: list of parameters (default: Ab kb eField long_diff tran_diff lifetime)
- `REL_DELTAS`: perturbation grid (edit in script)

In `plot_taylor_scan.ipynb`:
- `RESULTS_DIR`: path to results folder
- `THRESHOLDS`: ADC error thresholds for validity range computation

## Key codebase dependency

The smooth FEE functions (`_soft_max`, `_soft_where`, smooth `log_diff_ndtr`, `_find_one_hit_step` with `stop_gradient` on `top_k` indices) live in `src/larndsim/fee_jax.py`. These replace the original hard-switching operations to enable higher-order differentiation. Forward-pass values are identical to the originals (validated: mean abs difference < 1e-4 ADC).
