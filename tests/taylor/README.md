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
| `scan_utils.py` | Core library: `setup_params_and_tracks`, `compute_event_scan`, `compute_validity_range`, `save_results`/`load_results` |
| `run_taylor_scan.py` | Standalone script: loops over parameters x events, computes J/H and perturbation scan, saves to pickle. For SLURM. |
| `submit_taylor_scan.sh` | SLURM submission wrapper. Submit from repo root: `sbatch tests/taylor/submit_taylor_scan.sh` |
| `plot_taylor_scan.ipynb` | Loads pickle results, plots per-parameter event grids and validity range summaries. No GPU needed. |

## Configuration

Edit the top of `run_taylor_scan.py` (or the equivalent notebook):

- `PARAMS`: list of parameter names to scan (e.g. `['Ab', 'kb', 'eField', 'long_diff', 'tran_diff', 'lifetime']`)
- `REL_DELTAS`: perturbation grid (e.g. `np.linspace(-0.5, 0.5, 21)`)
- `INPUT_FILE`: track data file
- `OUTPUT_FILE`: where to save results pickle

In `plot_taylor_scan.ipynb`:

- `THRESHOLDS`: list of ADC error thresholds for validity range (e.g. `[0.1, 0.5]`)

## Running

```bash
# From repo root
sbatch tests/taylor/submit_taylor_scan.sh

# Monitor progress
tail -f tests/taylor/logs/taylor_scan-<jobid>.out

# Results saved incrementally to tests/taylor/taylor_scan_results.pkl
# Plot results (no GPU needed)
jupyter notebook tests/taylor/plot_taylor_scan.ipynb
```

## Key codebase dependency

The smooth FEE functions (`_soft_max`, `_soft_where`, smooth `log_diff_ndtr`, `_find_one_hit_step` with `stop_gradient` on `top_k` indices) live in `src/larndsim/fee_jax.py`. These replace the original hard-switching operations to enable higher-order differentiation. Forward-pass values are identical to the originals (validated: mean abs difference < 1e-4 ADC).
