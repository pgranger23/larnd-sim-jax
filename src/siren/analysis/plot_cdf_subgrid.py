#!/usr/bin/env python3
"""
Visualization script for SIREN CDF model - main pixel subgrid (5x5 positions).

Generates CDF and derivative comparison plots for all 25 subgrid positions
within the main pixel (LUT indices 0-4 in x and y).

Usage:
    python -m src.siren.analysis.plot_cdf_subgrid \
        --model siren_training/cdf_square_v2/final_model.npz \
        --output_dir siren_training/cdf_square_v2/validation_subgrid
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from ..core import create_siren
from ..training.checkpointing import load_checkpoint
from ..training.dataset import ResponseTemplateDataset


def load_model_and_dataset(model_path: str, lut_path: str, square_output: bool):
    """Load SIREN model and CDF dataset."""
    # Load checkpoint (ignore opt_state for visualization)
    params, step, config, history, norm_params, dataset_stats, _ = load_checkpoint(model_path)

    model = create_siren(
        hidden_features=config['hidden_features'],
        hidden_layers=config['hidden_layers'],
        out_features=1,
        w0=config['w0'],
        outermost_linear=config.get('outermost_linear', True),
    )

    dataset = ResponseTemplateDataset(
        lut_path=lut_path,
        normalize_inputs=True,
        normalize_outputs=True,
        use_cdf=True,
    )

    return model, params, dataset, config


def get_cdf_and_deriv(
    model,
    params,
    dataset,
    diff_idx: int,
    lut_x: int,
    lut_y: int,
    square_output: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get LUT and SIREN CDF/derivative at specified position."""
    n_t = dataset.n_t

    lut_cdf = dataset.cdf_template[diff_idx, lut_x, lut_y, :]
    lut_response = dataset.response_template[diff_idx, lut_x, lut_y, :]

    t_vals = jnp.arange(n_t, dtype=jnp.float32)
    coords = jnp.stack([
        jnp.full(n_t, diff_idx, dtype=jnp.float32),
        jnp.full(n_t, lut_x, dtype=jnp.float32),
        jnp.full(n_t, lut_y, dtype=jnp.float32),
        t_vals,
    ], axis=1)
    coords_norm = dataset.normalize_inputs(coords)

    siren_raw = model.apply(params['params'], coords_norm).ravel()
    if square_output:
        siren_cdf = np.array(siren_raw ** 2)
    else:
        siren_cdf = np.array(siren_raw)

    t_range = dataset.norm_params.t_range
    scale_factor = (2.0 / (t_range[1] - t_range[0])) * 10.0

    def siren_scalar(coord):
        out = model.apply(params['params'], coord[None, :])[0, 0]
        if square_output:
            out = out ** 2
        return out

    grad_fn = jax.vmap(jax.grad(siren_scalar))
    grads = grad_fn(coords_norm)
    siren_deriv = np.array(grads[:, 3]) * scale_factor

    return lut_cdf, siren_cdf, lut_response, siren_deriv


def plot_subgrid_cdf(
    model,
    params,
    dataset,
    diff_idx: int,
    output_path: str,
    square_output: bool,
) -> Dict[str, float]:
    """Create 5x5 grid plot for CDF comparison of main pixel subgrid."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    metrics = {}

    # Subgrid indices: (0,0) through (4,4)
    for row_idx in range(5):
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            lut_x = row_idx
            lut_y = col_idx

            lut_cdf, siren_cdf, _, _ = get_cdf_and_deriv(
                model, params, dataset, diff_idx, lut_x, lut_y, square_output
            )

            mae = np.abs(lut_cdf - siren_cdf).mean()
            metrics[f"subgrid_{lut_x}_{lut_y}"] = {'cdf_mae': float(mae)}

            t_vals = np.arange(len(lut_cdf))
            ax.plot(t_vals, lut_cdf, 'b-', linewidth=2, label='LUT')
            ax.plot(t_vals, siren_cdf, 'r--', linewidth=2, label='SIREN')

            is_center = (lut_x == 0 and lut_y == 0)
            title_weight = 'bold' if is_center else 'normal'
            ax.set_title(f"({lut_x},{lut_y})\nMAE={mae:.4f}",
                        fontsize=9, fontweight=title_weight)

            if row_idx == 0 and col_idx == 4:
                ax.legend(loc='lower right', fontsize=7)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if is_center:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)

    fig.suptitle(f'CDF: Main Pixel Subgrid (diff={diff_idx})', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")
    return metrics


def plot_subgrid_deriv(
    model,
    params,
    dataset,
    diff_idx: int,
    output_path: str,
    square_output: bool,
) -> Dict[str, float]:
    """Create 5x5 grid plot for derivative comparison of main pixel subgrid."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    metrics = {}

    for row_idx in range(5):
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            lut_x = row_idx
            lut_y = col_idx

            _, _, lut_response, siren_deriv = get_cdf_and_deriv(
                model, params, dataset, diff_idx, lut_x, lut_y, square_output
            )

            mae = np.abs(lut_response - siren_deriv).mean()
            metrics[f"subgrid_{lut_x}_{lut_y}"] = {'deriv_mae': float(mae)}

            t_vals = np.arange(len(lut_response))
            ax.plot(t_vals, lut_response, 'b-', linewidth=2, label='LUT')
            ax.plot(t_vals, siren_deriv, 'r--', linewidth=2, label='SIREN')

            is_center = (lut_x == 0 and lut_y == 0)
            title_weight = 'bold' if is_center else 'normal'
            ax.set_title(f"({lut_x},{lut_y})\nMAE={mae:.4f}",
                        fontsize=9, fontweight=title_weight)

            if row_idx == 0 and col_idx == 4:
                ax.legend(loc='upper right', fontsize=7)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if is_center:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)

    fig.suptitle(f'Derivative: Main Pixel Subgrid (diff={diff_idx})', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Visualize SIREN CDF model - main pixel subgrid')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained SIREN model')
    parser.add_argument('--lut_path', type=str,
                        default='src/larndsim/detector_properties/response_44_v2a_full_tick.npz',
                        help='Path to LUT file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for plots')
    parser.add_argument('--diff_values', type=int, nargs='+',
                        default=[0, 11, 22, 33, 44, 55, 66, 77, 88, 99],
                        help='Diffusion indices to plot')
    parser.add_argument('--square_output', action='store_true',
                        help='Model uses squared output')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and dataset...")
    model, params, dataset, config = load_model_and_dataset(
        args.model, args.lut_path, args.square_output
    )

    square_output = args.square_output or config.get('square_output', False)
    if square_output:
        print("Using squared output mode")

    print(f"Dataset shape: {dataset.response_template.shape}")
    print(f"Diffusion values: {args.diff_values}")

    all_metrics = {}

    for diff_idx in args.diff_values:
        print(f"\nProcessing diff={diff_idx}...")

        diff_dir = output_dir / f"diff{diff_idx}"
        diff_dir.mkdir(parents=True, exist_ok=True)

        cdf_metrics = plot_subgrid_cdf(
            model, params, dataset, diff_idx,
            str(diff_dir / "subgrid_cdf.png"),
            square_output,
        )

        deriv_metrics = plot_subgrid_deriv(
            model, params, dataset, diff_idx,
            str(diff_dir / "subgrid_deriv.png"),
            square_output,
        )

        metrics = {}
        for key in cdf_metrics:
            metrics[key] = {**cdf_metrics[key], **deriv_metrics.get(key, {})}

        with open(diff_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        all_metrics[f"diff_{diff_idx}"] = metrics

    with open(output_dir / "all_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
