#!/usr/bin/env python3
"""
Comparison utilities for LUT vs SIREN surrogate.

Provides functions for comparing predictions, computing metrics, and visualizing differences.
"""

import numpy as np

# Use non-interactive backend (no X11 needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comparison metrics between predictions and targets.

    Args:
        predictions: Model predictions.
        targets: Ground truth values.

    Returns:
        Dictionary of metrics.
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)

    # Avoid division by zero for relative errors
    nonzero_mask = np.abs(targets) > 1e-8
    rel_errors = np.zeros_like(errors)
    rel_errors[nonzero_mask] = abs_errors[nonzero_mask] / np.abs(targets[nonzero_mask])

    metrics = {
        'mse': float(np.mean(errors ** 2)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(abs_errors)),
        'max_error': float(np.max(abs_errors)),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'median_abs_error': float(np.median(abs_errors)),
        'p95_abs_error': float(np.percentile(abs_errors, 95)),
        'p99_abs_error': float(np.percentile(abs_errors, 99)),
        'mean_rel_error': float(np.mean(rel_errors[nonzero_mask])) if nonzero_mask.sum() > 0 else 0.0,
        'median_rel_error': float(np.median(rel_errors[nonzero_mask])) if nonzero_mask.sum() > 0 else 0.0,
    }

    # Correlation
    if len(predictions) > 1:
        corr, _ = pearsonr(predictions.ravel(), targets.ravel())
        metrics['pearson_r'] = float(corr)
        metrics['r_squared'] = float(corr ** 2)

    return metrics


def compare_lut_siren(
    model,
    params: Dict,
    dataset,
    n_samples: int = 100000,
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare SIREN predictions with LUT values.

    Args:
        model: SIREN model instance.
        params: Model parameters.
        dataset: ResponseTemplateDataset instance.
        n_samples: Number of samples to compare.
        seed: Random seed for sampling.

    Returns:
        Tuple of (metrics, predictions, targets, coords).
    """
    rng = np.random.default_rng(seed)

    # Sample random indices
    indices = rng.choice(dataset.total_points, size=n_samples, replace=False)

    # Get coordinates and values using lazy computation
    coords = dataset._indices_to_coords(indices)
    targets = dataset.values[indices]

    # Normalize coordinates
    coords_norm = dataset.normalize_inputs(jnp.array(coords))

    # Get predictions
    preds_norm = model.apply(params, coords_norm)
    preds_norm = np.array(preds_norm).ravel()

    # Denormalize predictions
    predictions = np.array(dataset.denormalize_outputs(jnp.array(preds_norm)))

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    return metrics, predictions, targets, coords


def plot_comparison_slices(
    model,
    params: Dict,
    dataset,
    output_dir: str,
    n_slices: int = 5,
) -> None:
    """
    Plot comparison slices at different fixed values.

    Creates plots comparing LUT and SIREN predictions for:
    - Time slices (fixed t)
    - Spatial slices (fixed x, y)
    - Diffusion slices (fixed diff)

    Args:
        model: SIREN model instance.
        params: Model parameters.
        dataset: ResponseTemplateDataset instance.
        output_dir: Directory to save plots.
        n_slices: Number of slices per dimension.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define slice indices
    diff_indices = np.linspace(0, dataset.n_diff - 1, n_slices, dtype=int)
    x_indices = np.linspace(0, dataset.n_x - 1, n_slices, dtype=int)
    t_indices = np.linspace(0, dataset.n_t - 1, n_slices, dtype=int)

    # Plot 1: Time evolution at different diffusion values (fixed x=22, y=22)
    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))

    x_fixed, y_fixed = dataset.n_x // 2, dataset.n_y // 2

    for i, diff_idx in enumerate(diff_indices):
        # Get LUT values
        lut_slice = dataset.response_template[diff_idx, x_fixed, y_fixed, :]
        t_vals = np.arange(dataset.n_t)

        # Get SIREN predictions
        coords = np.stack([
            np.full(dataset.n_t, diff_idx, dtype=np.float32),
            np.full(dataset.n_t, x_fixed, dtype=np.float32),
            np.full(dataset.n_t, y_fixed, dtype=np.float32),
            t_vals.astype(np.float32),
        ], axis=1)
        coords_norm = dataset.normalize_inputs(jnp.array(coords))
        preds_norm = model.apply(params, coords_norm)
        siren_slice = np.array(dataset.denormalize_outputs(jnp.array(preds_norm.ravel())))

        # Plot comparison
        ax = axes[0, i]
        ax.plot(t_vals, lut_slice, 'b-', alpha=0.7, label='LUT')
        ax.plot(t_vals, siren_slice, 'r--', alpha=0.7, label='SIREN')
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Response')
        ax.set_title(f'diff={diff_idx}')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot difference
        ax = axes[1, i]
        diff = siren_slice - lut_slice
        ax.plot(t_vals, diff, 'g-', alpha=0.7)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('SIREN - LUT')
        ax.set_title(f'Error (MAE={np.abs(diff).mean():.4f})')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Time Evolution at x={x_fixed}, y={y_fixed}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'time_slices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Spatial heatmaps at fixed time and diffusion
    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))

    t_fixed = dataset.n_t // 2

    for i, diff_idx in enumerate(diff_indices):
        # Get LUT 2D slice
        lut_2d = dataset.response_template[diff_idx, :, :, t_fixed]

        # Get SIREN predictions for 2D grid
        x_grid, y_grid = np.meshgrid(np.arange(dataset.n_x), np.arange(dataset.n_y), indexing='ij')
        coords = np.stack([
            np.full(dataset.n_x * dataset.n_y, diff_idx, dtype=np.float32),
            x_grid.ravel().astype(np.float32),
            y_grid.ravel().astype(np.float32),
            np.full(dataset.n_x * dataset.n_y, t_fixed, dtype=np.float32),
        ], axis=1)
        coords_norm = dataset.normalize_inputs(jnp.array(coords))
        preds_norm = model.apply(params, coords_norm)
        siren_2d = np.array(dataset.denormalize_outputs(jnp.array(preds_norm.ravel()))).reshape(dataset.n_x, dataset.n_y)

        # Plot LUT
        ax = axes[0, i]
        im = ax.imshow(lut_2d, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Y bin')
        ax.set_ylabel('X bin')
        ax.set_title(f'LUT (diff={diff_idx})')

        # Plot SIREN
        ax = axes[1, i]
        im = ax.imshow(siren_2d, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Y bin')
        ax.set_ylabel('X bin')
        ax.set_title(f'SIREN (diff={diff_idx})')

        # Plot difference
        ax = axes[2, i]
        diff_2d = siren_2d - lut_2d
        vmax = max(abs(diff_2d.min()), abs(diff_2d.max()))
        im = ax.imshow(diff_2d, aspect='auto', origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Y bin')
        ax.set_ylabel('X bin')
        ax.set_title(f'SIREN - LUT (MAE={np.abs(diff_2d).mean():.4f})')

    plt.suptitle(f'Spatial Distribution at t={t_fixed}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_slices.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plots to {output_dir}")


def create_comparison_report(
    model_path: str,
    lut_path: str,
    output_dir: str,
    n_samples: int = 100000,
) -> Dict[str, float]:
    """
    Create a comprehensive comparison report.

    Args:
        model_path: Path to trained SIREN model.
        lut_path: Path to LUT file.
        output_dir: Output directory for report.
        n_samples: Number of samples for metrics.

    Returns:
        Dictionary of comparison metrics.
    """
    from ..inference import SurrogatePredictor
    from ..training.dataset import ResponseTemplateDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")

    # Load model
    predictor = SurrogatePredictor(model_path)

    # Load dataset
    dataset = ResponseTemplateDataset(
        lut_path=lut_path,
        val_fraction=0.0,  # Use all data
        normalize_inputs=True,
        normalize_outputs=True,
    )

    print("Computing metrics...")

    # Compare
    metrics, predictions, targets, coords = compare_lut_siren(
        predictor.model,
        predictor.params,
        dataset,
        n_samples=n_samples,
    )

    # Print metrics
    print("\nComparison Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create plots
    print("\nCreating comparison plots...")

    # Scatter plot
    from .visualize import plot_predictions
    plot_predictions(predictions, targets, str(output_dir / 'predictions.png'))
    plt.close()

    # Slice comparisons
    plot_comparison_slices(
        predictor.model,
        predictor.params,
        dataset,
        str(output_dir),
    )

    print(f"\nReport saved to {output_dir}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare LUT vs SIREN')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained SIREN model')
    parser.add_argument('--lut_path', type=str,
                        default='src/larndsim/detector_properties/response_44_v2a_full_tick.npz',
                        help='Path to LUT file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for comparison report')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='Number of samples for metrics')

    args = parser.parse_args()

    create_comparison_report(
        args.model,
        args.lut_path,
        args.output_dir,
        args.n_samples,
    )
