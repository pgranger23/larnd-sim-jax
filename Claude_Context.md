# Claude Context File

## Purpose

This file serves as a persistent memory for Claude Code sessions working on this project. It should be read at the start of each fresh session to quickly understand:
- What this project is and its structure
- How to run the code
- Key decisions and findings from previous sessions
- Current state of work

**Edit this file organically as we work** to keep it up-to-date with new discoveries and decisions.

---

## Project Overview

**larnd-sim-jax** is a differentiable physics simulation framework for the DUNE Near Detector (Liquid Argon TPC). It's a JAX-based rewrite of the official DUNE larnd-sim, enabling gradient-based optimization of detector parameters.

- **Origin**: Fork of https://github.com/ynashed/larnd-sim
- **Language**: Python (100%)
- **Core Framework**: JAX (differentiable computing)

---

## Documentation

Technical documentation is stored in:
```
/sdf/home/c/cjesus/LArTPC/documentation/
```

Key documents:
- `LUT_simulation_pipeline.md` - Detailed explanation of LUT mode simulation (9-step pipeline, gradient flow, etc.)

---

## Development Figures

All development/debugging figures are stored in:
```
/sdf/home/c/cjesus/LArTPC/dev_figures/
```

**How to generate fit plots:**
```bash
singularity exec -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif python3 \
    optimize/analysis_scripts/plot_fit.py \
    --input_file fit_result/<test_name>/history_iter<N>_<label>.pkl \
    --output_dir /sdf/home/c/cjesus/LArTPC/dev_figures/<subfolder>
```

---

## Directory Structure

```
larnd-sim-jax/
├── src/larndsim/           # Main simulation package
│   ├── sim_jax.py          # Core simulation orchestration
│   ├── detsim_jax.py       # Pixel-level current generation
│   ├── consts_jax.py       # Parameters & configuration
│   ├── losses_jax.py       # Loss functions for fitting
│   ├── fee_jax.py          # ADC digitization
│   ├── quenching_jax.py    # Recombination models
│   └── drifting_jax.py     # Electron drift simulation
├── src/siren/              # SIREN surrogate for LUT (NEW)
│   ├── core.py             # SIREN model definition
│   ├── train_surrogate.py  # Training CLI entry point
│   ├── inference.py        # SurrogatePredictor wrapper
│   ├── training/           # Training utilities
│   └── analysis/           # Visualization & comparison tools
├── optimize/               # Optimization and fitting scripts
│   ├── example_run.py      # Main entry point for fitting
│   ├── fit_params.py       # Fitter classes (GradientDescent, Minuit, etc.)
│   ├── dataio.py           # Data loading
│   └── scripts/            # Parameter configs and job scripts
├── prepared_data/          # Test input files (input_*.h5)
├── tests/                  # Unit tests
└── docs/                   # Sphinx documentation
```

---

## How to Run

### Container Setup

The project uses a Singularity container with all dependencies pre-installed:
```
/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
```

**Container provides**: JAX 0.5.3 + CUDA 12, numpy, numba, optax, iminuit, h5py, etc.
**Container does NOT include**: larndsim itself (installed from source each run)

### Basic Execution Pattern

```bash
cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax

singularity exec -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
    pip3 install --quiet . && python3 -m optimize.example_run [args...]
"
```

### Minimal Test Run (CPU) - WORKING

```bash
singularity exec -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
pip3 install --quiet . && python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 2 \
    --params optimize/scripts/params_test.yaml \
    --input_file_sim prepared_data/input_1.h5 \
    --input_file_tgt prepared_data/input_1.h5 \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed 1 \
    --out_label test_run \
    --test_name test \
    --seed 3 \
    --optimizer_fn Adam \
    --iterations 10 \
    --max_batch_len 200 \
    --lr_scheduler exponential_decay \
    --lr_kw '{\"decay_rate\": 0.97}' \
    --track_z_bound 28 \
    --max_clip_norm_val 1 \
    --clip_from_range \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 0 \
    --signal_length 150 \
    --mode parametrized \
    --loss_fn mse_adc \
    --fit_type chain \
    --cpu_only \
    --mc_diff \
    --no-noise-guess \
    --no-noise-target \
    --sim_seed_strategy same"
```

**Note**: `--sim_seed_strategy same` is crucial for convergence when using `--mc_diff`.

### GPU Execution

For GPU runs, use `--nv` flag and remove `--cpu_only`:
```bash
singularity exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif ...
```

### Test Scripts

Pre-made test scripts in `optimize/`:
- `fit_test.sh` - Basic fitting test
- `scan_test.sh` - Parameter scan
- `minuit_test.sh` - Minuit minimizer test

---

## Key Parameters

Parameters to fit are defined in YAML files (e.g., `optimize/scripts/params_test.yaml`).

Common parameters:
- `eField` - Electric field
- Recombination: `Ab`, `kb`, `alpha`, `beta`
- Drift: `vdrift`, `lifetime`
- See `src/larndsim/consts_jax.py` for full list

---

## Current Branch

**Branch**: `cjesus/getting-started`

---

## Session Log

### 2026-01-13: Initial Setup
- Explored codebase structure
- Created branch `cjesus/getting-started`
- Verified container execution works
- Successfully ran minimal test (5 iterations fitting eField on CPU)
- First iteration ~10s (JIT compilation), subsequent ~2-3s each

### 2026-01-13: Debugging Fit Convergence
**Problem**: Initial runs with default settings showed eField diverging away from target.

**Root cause**: `--sim_seed_strategy different` (default) combined with `--mc_diff` causes stochastic mismatch between target and simulation, leading to incorrect gradients.

**Solution**: Use `--sim_seed_strategy same` when using `--mc_diff` to ensure deterministic comparison.

**Results comparison**:
| Setting | Target | Init | Final (10 iter) | Loss |
|---------|--------|------|-----------------|------|
| `different` seed | 0.505 | 0.500 | 0.456 (diverged) | ~0.00189 (increasing) |
| `same` seed | 0.505 | 0.500 | 0.507 (converged) | ~0.00008 (decreasing) |

**Visualization**: Plots saved in `plots/getting_started/` and `plots/same_seed/`

**Important flags for deterministic fitting**:
- `--sim_seed_strategy same` - Match seeds between target and simulation
- `--no-noise-guess --no-noise-target` - Disable electronics noise
- `--mc_diff` - Monte Carlo diffusion (requires same seed strategy)

### 2026-01-14: SIREN Surrogate for LUT

**Goal**: Create a neural network surrogate for the LUT (Look-Up Table) used in detector simulation.

**What was built** (`src/siren/`):
- SIREN model (Sinusoidal Representation Network) with 4D input: (diff, x, y, t) → response
- Training pipeline with checkpointing, resumption, and visualization
- LUT vs SIREN comparison tools

**LUT Structure**:
- Raw: `response_44_v2a_full_tick.npz` shape (45, 45, 1950)
- Response template: shape (100, 45, 45, 1950) after diffusion convolution
- Total: 394,875,000 data points

**How to train the surrogate**:
```bash
singularity exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
    pip3 install --quiet . && python3 -m src.siren.train_surrogate \
        --output_dir siren_training \
        --batch_size 65536 \
        --num_steps 50000
"
```

**Resume training**:
```bash
python3 -m src.siren.train_surrogate --resume siren_training/checkpoint_latest.npz
```

**Compare LUT vs SIREN**:
```bash
python3 -m src.siren.analysis.compare \
    --model siren_training/final_model.npz \
    --output_dir siren_training/comparison
```

**Key files**:
- `src/siren/core.py` - SIREN model (256×4 layers, ω₀=30)
- `src/siren/train_surrogate.py` - CLI training entry point
- `src/siren/inference.py` - `SurrogatePredictor` for inference
- `src/siren/training/dataset.py` - Loads and samples from response_template
- `src/siren/analysis/compare.py` - LUT vs SIREN comparison plots

**Training completed** (50k steps, ~16 min on A100):
- Final loss: ~0 (MSE)
- Model saved: `siren_training/final_model.npz`
- Checkpoints: `siren_training/checkpoint_step_*.npz`

**GPU Job Submission** (IMPORTANT: use PYTHONNOUSERSITE=1):
```bash
sbatch scripts/submit_siren_training.sh
# Monitor: ./scripts/monitor_siren.sh
```

### LUT Indexing - CRITICAL Understanding

**LUT indices = distance from pixel center** (not pixel IDs!)

From `detsim_jax.py:516-521`:
```python
i = (x_dist/params.response_bin_size).astype(int)  # Distance → index
```

Key parameters:
- `response_bin_size = 0.04434 cm`
- `pixel_pitch ≈ 0.4 cm`
- **1 pixel spacing ≈ 9 LUT bins**

**5×5 Pixel Grid → LUT Indices** (charge at main pixel center):
```
         col -2    col -1    col 0     col +1    col +2
row -2:  (18,18)   (18,9)    (18,0)    (18,9)    (18,18)
row -1:  (9,18)    (9,9)     (9,0)     (9,9)     (9,18)
row  0:  (0,18)    (0,9)     (0,0)     (0,9)     (0,18)    ← Main pixel
row +1:  (9,18)    (9,9)     (9,0)     (9,9)     (9,18)
row +2:  (18,18)   (18,9)    (18,0)    (18,9)    (18,18)
```

- **Main pixel center**: LUT[0, 0] → UNIPOLAR (collects charge)
- **Edge neighbors**: LUT[9, 0] or LUT[0, 9] → BIPOLAR
- **Corner neighbors**: LUT[9, 9] → BIPOLAR
- **Distant neighbors**: LUT[18, *] → weak BIPOLAR

Verified empirically:
- `LUT[0,0]`: unipolar, max=0.709
- `LUT[4,3]`: peak amplitude 3.36
- `LUT[22,22]`: bipolar, tiny (~0.003)

### SIREN Validation Results

**Good performance** at main pixel (0,0) and nearby:
- SIREN captures unipolar waveform shape well
- MAE ~0.004 vs amplitude ~3.4 → ~0.1% relative error

**Poor performance** at distant neighbors (e.g., 22,22):
- LUT is nearly flat (~0)
- SIREN shows spurious oscillations (~0.002 amplitude)
- This is a known SIREN limitation: sinusoidal activations can't represent flat regions

**Comparison plots**: `siren_training/comparison/`

### Validation Results: All 25 Pixels (Completed 2026-01-14)

**Script**: `src/siren/analysis/validate_all_pixels.py`

**Output**: `siren_training/comparison/validation_all_pixels/` (also copied to `/sdf/home/c/cjesus/LArTPC/dev_figures/siren_validation/`)

**Run command**:
```bash
srun --partition=ampere --account=mli:cider-ml --gpus=1 --time=00:15:00 \
  singularity exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet .
    python3 -m src.siren.analysis.validate_all_pixels \
        --model siren_training/final_model.npz \
        --output_dir siren_training/comparison/validation_all_pixels
"
```

**Results Summary**:

| Pixel Type | LUT Index | Max Amplitude | MAE | Relative Error |
|------------|-----------|---------------|-----|----------------|
| Main (0,0) | [0,0] | 0.709 | 0.0043 | 0.6% |
| Edge (±1,0) | [9,0] | 0.102 | 0.0021 | 2.1% |
| Corner (±1,±1) | [9,9] | 0.035 | 0.0023 | 6.5% |
| Distant (±2,±2) | [18,18] | 0.0046 | 0.0021 | 46% |

**Key Findings**:
- SIREN achieves consistent absolute MAE (~0.002) across all pixels
- Main pixel: excellent fit (0.6% relative error) - captures unipolar shape well
- Edge/corner neighbors: good fit (2-6% relative error)
- Distant pixels: high relative error (46%) but signal is tiny (~0.005 amplitude)
- The high relative error for distant pixels is acceptable since the signal is essentially noise-level

**Generated Plots**:
- `diff_0/all_pixels_diff0.png` - 5×5 grid, no diffusion
- `diff_49/all_pixels_diff49.png` - 5×5 grid, medium diffusion
- `diff_99/all_pixels_diff99.png` - 5×5 grid, high diffusion
- `summary.png` - Key pixels across all diffusion values

### 2026-01-14: CDF Training Mode (NEW)

**Goal**: Train SIREN on cumulative distribution (CDF) instead of raw response.

**Why CDF?**:
- Main pixel CDF: smooth monotonic curve [0→10], easy to learn
- Neighbor pixel CDF: stays near 0 (bipolar signals cancel out)
- Avoids oscillating bipolar signals that SIREN struggles with

**New CLI flags**:
```bash
--use_cdf              # Train on CDF/10 instead of raw response
--lambda_deriv 1.0     # Weight for derivative loss (0 = CDF only)
```

**CDF Training Commands**:
```bash
# CDF-only mode (recommended to start)
srun --partition=ampere --account=mli:cider-ml --gpus=1 --mem=32G --time=01:00:00 \
  singularity exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet .
    python3 -m src.siren.train_surrogate \
        --output_dir siren_training_cdf \
        --batch_size 8192 \
        --num_steps 50000 \
        --use_cdf \
        --lambda_deriv 0.0
"

# CDF + derivative loss (enforces derivative matches waveform)
python3 -m src.siren.train_surrogate \
    --output_dir siren_training_cdf_deriv \
    --batch_size 4096 \
    --num_steps 50000 \
    --use_cdf \
    --lambda_deriv 1.0
```

**CDF Verification** (from training output):
- CDF range: [0, 1.0] after /10 normalization
- Main pixel (0,0,0) final CDF value: 1.0000 ✓
- Neighbor (0,9,9) final CDF value: -0.0000 ✓

**Files modified**:
- `src/siren/training/config.py` - Added `use_cdf`, `lambda_deriv`
- `src/siren/training/dataset.py` - CDF computation, `sample_batch_cdf()`
- `src/siren/training/trainer.py` - `_compile_cdf_functions()`, derivative loss
- `src/siren/train_surrogate.py` - CLI flags

**Memory note**: CDF mode stores both templates (~6GB), use `--mem=32G` in srun.

### Run Name Parameter (NEW)

Use `--run_name` to organize multiple experiments:
```bash
python3 -m src.siren.train_surrogate --use_cdf --lambda_deriv 0.0 --run_name cdf_only
python3 -m src.siren.train_surrogate --use_cdf --lambda_deriv 1.0 --run_name cdf_deriv_1.0

# Results in:
# siren_training/cdf_only/
# siren_training/cdf_deriv_1.0/
```

### Training Visualization (CDF Mode)

During CDF training, `prediction_comparison.png` is generated at each validation step showing:
- **Top row**: LUT CDF vs SIREN CDF for main (0,0) and edge (0,9) pixels
- **Bottom row**: LUT response vs SIREN derivative (d(CDF)/dt)

Uses diff=49 as representative diffusion value. This helps monitor whether the trained CDF correctly recovers the original waveform when differentiated.

**Output files**:
- `training_progress.png` - Loss curves (2 panels: loss + LR)
- `prediction_cdf.png` - 3x3 grid of LUT vs SIREN CDF predictions
- `prediction_deriv.png` - 3x3 grid of LUT vs SIREN derivative (response)

### 2026-01-15: Training Improvements

#### Removed Train/Val Split

**Rationale**: The surrogate is trying to perfectly memorize the LUT, not generalize. Therefore, all data should be used for training (no validation holdout).

**Changes**:
- Removed `val_fraction` parameter
- Removed `_create_train_val_split()` method
- All 395M data points used for training
- Progress plot reduced from 3 subplots to 2 (train loss + LR)
- Renamed `val_every` → `plot_every` (for generating prediction plots)

#### Square Output Feature (`--square_output`)

**Problem**: SIREN predictions could go negative, but CDF values should be in [0, ∞).

**Solution**: When `--square_output` is enabled:
1. Keep `outermost_linear=True` (linear output layer) - matches LUCiD implementation
2. Square the linear output: output² ∈ [0, ∞)
3. Network learns to output values that when squared match the targets

**Usage**:
```bash
python3 -m src.siren.train_surrogate --use_cdf --square_output ...
```

**Technical notes**:
- Derivative via autodiff applies chain rule: d(f²)/dt = 2*f*df/dt
- The trainer's `_apply_model()` helper handles squaring consistently
- This approach works much better than squaring sin output (which caused learning issues)

#### Visualization Scripts

Created comprehensive visualization scripts for trained models:

**1. `src/siren/analysis/plot_cdf_pixels.py`** - 5×5 pixel grid (25 pixels)
```bash
python3 -m src.siren.analysis.plot_cdf_pixels \
    --model siren_training/seed42/final_model.npz \
    --output_dir siren_training/seed42/validation \
    --square_output
```

**2. `src/siren/analysis/plot_cdf_subgrid.py`** - Main pixel subgrid (LUT indices 0-4)
```bash
python3 -m src.siren.analysis.plot_cdf_subgrid \
    --model siren_training/seed42/final_model.npz \
    --output_dir siren_training/seed42/validation_subgrid \
    --square_output
```

**Plot settings**: figsize=(10,10), linewidth=2, 10 diffusion values

#### Optimizer State Saving (for proper resume)

**Problem**: Resuming training caused loss to jump from ~1e-8 to ~1e-5 because Adam optimizer state (momentum) was reset.

**Solution**: Save and restore full optimizer state in checkpoints.
- `save_checkpoint()` now saves `opt_state`
- `load_checkpoint()` restores `opt_state`
- Proper resume maintains training continuity

### Current Training Command

```bash
srun --partition=ampere --account=mli:cider-ml --gpus=1 --mem=32G --time=12:00:00 \
  singularity exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet .
    python3 -m src.siren.train_surrogate \
        --output_dir siren_training \
        --run_name seed42 \
        --use_cdf \
        --square_output \
        --learning_rate 1e-4 \
        --lr_decay_rate 0.99998 \
        --lr_min 0 \
        --num_steps 1000000 \
        --batch_size 65536 \
        --seed 42
"
```

**Multi-seed training** (for reproducibility studies):
- Seeds: 42, 43, 44, 45, 46
- 1M steps each, no min LR

### 2026-01-15: SIREN Surrogate Integration into Simulation

**Goal**: Add a third simulation mode ('surrogate') that uses the trained SIREN CDF model instead of the LUT.

**Key Insight**: The FEE code immediately computes `q_cumsum = (wfs * t_sampling).cumsum(axis=-1)` and only uses the cumsum. The SIREN CDF model outputs exactly this, so we can skip waveform generation entirely.

**Architecture**:
```
LUT mode:       electrons → simulate_drift_new() → simulate_signals_new() → waveforms → FEE:cumsum → q_cumsum → ADC
Surrogate mode: electrons → simulate_drift_surrogate() → simulate_signals_surrogate() → q_cumsum → ADC (direct!)
```

**Benefits**:
- Exact continuous diffusion values (no interpolation across 3 indices)
- Fractional bin positions (no integer discretization)
- Direct CDF output (bypasses waveform generation)

**Files Created/Modified**:

1. **`src/larndsim/surrogate_utils.py`** (NEW) - Coordinate transforms:
   - `long_diff_to_siren_diff()`: Map diffusion ticks → SIREN range [0, 99]
   - `position_to_siren_xy()`: Map distances → fractional bin indices [0, 44]
   - `normalize_siren_inputs()`: Normalize to [-1, 1] for SIREN
   - `siren_cdf_to_q_cumsum()`: Convert CDF output to charge

2. **`src/larndsim/consts_jax.py`**:
   - Added surrogate normalization parameters to `Params_template`
   - Added `load_surrogate()` function (handles both checkpoints and final_model.npz)

3. **`src/larndsim/sim_jax.py`**:
   - `simulate_drift_surrogate()`: Returns fractional coordinates
   - `simulate_signals_surrogate()`: Evaluates SIREN at coordinates, outputs q_cumsum
   - `simulate_surrogate()`: Main entry point

4. **`src/larndsim/fee_jax.py`**:
   - `get_adc_values_from_cumsum()`: FEE simulation from pre-computed cumsum

5. **`optimize/simulate.py`**:
   - Added 'surrogate' mode
   - Added `--surrogate_model` argument

6. **`documentation/SIREN_surrogate_integration.md`** - Full documentation

**Test Script**: `src/siren/test_surrogate.py`
```bash
python3 -m src.siren.test_surrogate
```

**Usage (Surrogate Simulation)**:
```bash
python -m optimize.simulate \
    --mode surrogate \
    --surrogate_model siren_training/seed42/final_model.npz \
    --input_file data/tracks.h5 \
    --output_file results/output \
    --electron_sampling_resolution 0.001 \
    --number_pix_neighbors 1 \
    --signal_length 150
```

**Coordinate Mappings**:
- Diffusion: `diff_siren = (long_diff_ticks - 0.001) / (10.0 - 0.001) * 99.0` → [0, 99]
- Position: `x_frac = x_dist / response_bin_size` → continuous [0, 44]
- CDF to charge: `q_cumsum = siren_output * 10 * n_electrons * t_sampling`

**Test Results** (seed46, 150k steps):
- SIREN evaluation: 0.044841 at main pixel center
- Full simulation produces ADC values: [84.8, 83.9, 82.8]
- q_cumsum max: 17068.83

**Checkpoint vs Final Model**:
- `checkpoint_*.npz`: Full training state (wrapped params, optimizer state) - for resume
- `final_model.npz`: Inference only (unwrapped params, model_config) - created at end
- `load_surrogate()` handles both formats automatically
