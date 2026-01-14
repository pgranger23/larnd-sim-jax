#!/usr/bin/env python3
"""
Validation script for SIREN surrogate at all 25 pixel centers.

Generates comparison plots of LUT vs SIREN predictions for:
- 5×5 pixel grid (1 main + 24 neighbors)
- 3 diffusion values (low, medium, high)

LUT Indexing:
- LUT indices = distance from pixel center in response_bin_size units
- 1 pixel spacing ≈ 9 LUT bins (pixel_pitch / response_bin_size)
- Main pixel center at (0,0), neighbors at (9,0), (0,9), (9,9), (18,*), etc.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


# 5×5 pixel grid to LUT index mapping
# Each entry is (lut_x_idx, lut_y_idx) for pixel at (row, col) offset from main pixel
# row/col offsets: -2, -1, 0, +1, +2
PIXEL_TO_LUT = {
    (-2, -2): (18, 18), (-2, -1): (18, 9), (-2, 0): (18, 0), (-2, 1): (18, 9), (-2, 2): (18, 18),
    (-1, -2): (9, 18),  (-1, -1): (9, 9),  (-1, 0): (9, 0),  (-1, 1): (9, 9),  (-1, 2): (9, 18),
    (0, -2): (0, 18),   (0, -1): (0, 9),   (0, 0): (0, 0),   (0, 1): (0, 9),   (0, 2): (0, 18),
    (1, -2): (9, 18),   (1, -1): (9, 9),   (1, 0): (9, 0),   (1, 1): (9, 9),   (1, 2): (9, 18),
    (2, -2): (18, 18),  (2, -1): (18, 9),  (2, 0): (18, 0),  (2, 1): (18, 9),  (2, 2): (18, 18),
}


def load_model_and_dataset(model_path: str, lut_path: str):
    """Load SIREN model and dataset."""
    from ..inference import SurrogatePredictor
    from ..training.dataset import ResponseTemplateDataset

    predictor = SurrogatePredictor(model_path)
    dataset = ResponseTemplateDataset(
        lut_path=lut_path,
        val_fraction=0.0,
        normalize_inputs=True,
        normalize_outputs=True,
    )

    return predictor, dataset


def get_waveforms(
    predictor,
    dataset,
    diff_idx: int,
    lut_x: int,
    lut_y: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get LUT and SIREN waveforms at specified position.

    Args:
        predictor: SurrogatePredictor instance.
        dataset: ResponseTemplateDataset instance.
        diff_idx: Diffusion index (0-99).
        lut_x: LUT x index (0-44).
        lut_y: LUT y index (0-44).

    Returns:
        Tuple of (lut_waveform, siren_waveform).
    """
    n_t = dataset.n_t

    # Get LUT waveform
    lut_waveform = dataset.response_template[diff_idx, lut_x, lut_y, :]

    # Get SIREN prediction
    t_vals = np.arange(n_t, dtype=np.float32)
    coords = np.stack([
        np.full(n_t, diff_idx, dtype=np.float32),
        np.full(n_t, lut_x, dtype=np.float32),
        np.full(n_t, lut_y, dtype=np.float32),
        t_vals,
    ], axis=1)

    coords_norm = dataset.normalize_inputs(jnp.array(coords))
    preds_norm = predictor.model.apply(predictor.params, coords_norm)
    siren_waveform = np.array(dataset.denormalize_outputs(jnp.array(preds_norm.ravel())))

    return lut_waveform, siren_waveform


def plot_pixel_grid(
    predictor,
    dataset,
    diff_idx: int,
    output_path: str,
) -> Dict[str, float]:
    """
    Create 5×5 grid plot comparing LUT vs SIREN for all pixels.

    Args:
        predictor: SurrogatePredictor instance.
        dataset: ResponseTemplateDataset instance.
        diff_idx: Diffusion index.
        output_path: Path to save figure.

    Returns:
        Dictionary of per-pixel MAE values.
    """
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    metrics = {}

    # Iterate through 5×5 grid
    for row_idx, row_offset in enumerate([-2, -1, 0, 1, 2]):
        for col_idx, col_offset in enumerate([-2, -1, 0, 1, 2]):
            ax = axes[row_idx, col_idx]

            # Get LUT indices
            lut_x, lut_y = PIXEL_TO_LUT[(row_offset, col_offset)]

            # Get waveforms
            lut_wave, siren_wave = get_waveforms(
                predictor, dataset, diff_idx, lut_x, lut_y
            )

            # Calculate MAE
            mae = np.abs(lut_wave - siren_wave).mean()
            max_amp = np.abs(lut_wave).max()
            rel_err = mae / max_amp if max_amp > 1e-6 else 0.0

            pixel_key = f"pixel_{row_offset}_{col_offset}"
            metrics[pixel_key] = {
                'mae': float(mae),
                'max_amplitude': float(max_amp),
                'relative_error': float(rel_err),
                'lut_indices': (int(lut_x), int(lut_y)),
            }

            # Plot
            t_vals = np.arange(len(lut_wave))
            ax.plot(t_vals, lut_wave, 'b-', alpha=0.7, linewidth=1.5, label='LUT')
            ax.plot(t_vals, siren_wave, 'r--', alpha=0.7, linewidth=1.5, label='SIREN')

            # Title
            is_main = (row_offset == 0 and col_offset == 0)
            title_color = 'green' if is_main else 'black'
            title_weight = 'bold' if is_main else 'normal'
            ax.set_title(
                f"Pixel ({row_offset:+d},{col_offset:+d})\nLUT[{lut_x},{lut_y}]\nMAE={mae:.4f}",
                fontsize=10,
                color=title_color,
                fontweight=title_weight,
            )

            # Only add legend to first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (ticks)', fontsize=8)
            ax.set_ylabel('Response', fontsize=8)
            ax.tick_params(labelsize=7)

            # Highlight main pixel subplot
            if is_main:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)

    plt.suptitle(f'LUT vs SIREN - All 25 Pixels (diff={diff_idx})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")
    return metrics


def create_summary_plot(
    predictor,
    dataset,
    diff_values: List[int],
    output_path: str,
) -> None:
    """
    Create summary plot showing key pixels across all diffusion values.

    Args:
        predictor: SurrogatePredictor instance.
        dataset: ResponseTemplateDataset instance.
        diff_values: List of diffusion indices.
        output_path: Path to save figure.
    """
    # Key pixels to show: main (0,0), edge (0,1), corner (1,1), distant (2,2)
    key_pixels = [
        ((0, 0), "Main (0,0)"),
        ((0, 1), "Edge (0,+1)"),
        ((1, 1), "Corner (+1,+1)"),
        ((2, 2), "Distant (+2,+2)"),
    ]

    fig, axes = plt.subplots(len(key_pixels), len(diff_values), figsize=(5*len(diff_values), 4*len(key_pixels)))

    for row_idx, ((row_off, col_off), label) in enumerate(key_pixels):
        lut_x, lut_y = PIXEL_TO_LUT[(row_off, col_off)]

        for col_idx, diff_idx in enumerate(diff_values):
            ax = axes[row_idx, col_idx]

            lut_wave, siren_wave = get_waveforms(
                predictor, dataset, diff_idx, lut_x, lut_y
            )

            mae = np.abs(lut_wave - siren_wave).mean()

            t_vals = np.arange(len(lut_wave))
            ax.plot(t_vals, lut_wave, 'b-', alpha=0.7, linewidth=1.5, label='LUT')
            ax.plot(t_vals, siren_wave, 'r--', alpha=0.7, linewidth=1.5, label='SIREN')

            ax.set_title(f"{label} | diff={diff_idx}\nMAE={mae:.4f}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (ticks)', fontsize=8)
            ax.set_ylabel('Response', fontsize=8)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('SIREN Validation Summary - Key Pixels Across Diffusion Values', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate SIREN at all 25 pixel centers')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained SIREN model')
    parser.add_argument('--lut_path', type=str,
                        default='src/larndsim/detector_properties/response_44_v2a_full_tick.npz',
                        help='Path to LUT file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for validation plots')
    parser.add_argument('--diff_values', type=int, nargs='+', default=[0, 49, 99],
                        help='Diffusion indices to validate')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and dataset...")
    predictor, dataset = load_model_and_dataset(args.model, args.lut_path)

    print(f"Dataset shape: {dataset.response_template.shape}")
    print(f"Diffusion values to validate: {args.diff_values}")

    all_metrics = {}

    # Generate plots for each diffusion value
    for diff_idx in args.diff_values:
        print(f"\nProcessing diff={diff_idx}...")

        # Create subdirectory
        diff_dir = output_dir / f"diff_{diff_idx}"
        diff_dir.mkdir(parents=True, exist_ok=True)

        # Generate 5×5 grid plot
        metrics = plot_pixel_grid(
            predictor, dataset, diff_idx,
            str(diff_dir / f"all_pixels_diff{diff_idx}.png")
        )

        # Save metrics
        with open(diff_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        all_metrics[f"diff_{diff_idx}"] = metrics

    # Create summary plot
    print("\nCreating summary plot...")
    create_summary_plot(
        predictor, dataset, args.diff_values,
        str(output_dir / "summary.png")
    )

    # Save combined metrics
    with open(output_dir / "all_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary statistics
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for diff_idx in args.diff_values:
        metrics = all_metrics[f"diff_{diff_idx}"]
        maes = [v['mae'] for v in metrics.values()]
        print(f"\nDiff={diff_idx}:")
        print(f"  Mean MAE across pixels: {np.mean(maes):.6f}")
        print(f"  Max MAE:  {np.max(maes):.6f}")
        print(f"  Min MAE:  {np.min(maes):.6f}")

        # Main pixel
        main_mae = metrics['pixel_0_0']['mae']
        print(f"  Main pixel MAE: {main_mae:.6f}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
