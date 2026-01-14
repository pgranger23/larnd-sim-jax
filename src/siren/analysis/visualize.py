"""
Visualization utilities for SIREN surrogate training.

Provides functions for plotting training curves, predictions, and diagnostics.
"""

import numpy as np

# Use non-interactive backend (no X11 needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json


def plot_training_history(
    history_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot training history from a history.json file.

    Args:
        history_path: Path to history.json file.
        output_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Training loss
    ax = axes[0, 0]
    steps = history.get('step', list(range(len(history['train_loss']))))
    ax.plot(steps, history['train_loss'], 'b-', alpha=0.7, label='Train')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Validation loss
    ax = axes[0, 1]
    if history.get('val_loss'):
        # Validation is less frequent, need to align steps
        val_steps = np.linspace(0, steps[-1], len(history['val_loss'])) if steps else list(range(len(history['val_loss'])))
        ax.plot(val_steps, history['val_loss'], 'r-', alpha=0.7, label='Validation')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Validation Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Validation Loss')

    # Plot 3: Learning rate
    ax = axes[1, 0]
    if history.get('learning_rate'):
        ax.plot(steps, history['learning_rate'], 'g-', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Learning Rate Schedule')

    # Plot 4: Combined train/val
    ax = axes[1, 1]
    ax.plot(steps, history['train_loss'], 'b-', alpha=0.7, label='Train')
    if history.get('val_loss'):
        val_steps = np.linspace(0, steps[-1], len(history['val_loss'])) if steps else list(range(len(history['val_loss'])))
        ax.plot(val_steps, history['val_loss'], 'r-', alpha=0.7, label='Validation')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Train vs Validation Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {output_path}")

    return fig


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    max_points: int = 10000,
) -> plt.Figure:
    """
    Plot predictions vs targets.

    Args:
        predictions: Model predictions.
        targets: Ground truth values.
        output_path: Optional path to save figure.
        figsize: Figure size.
        max_points: Maximum points to plot (for performance).

    Returns:
        Matplotlib figure.
    """
    # Subsample if needed
    if len(predictions) > max_points:
        idx = np.random.choice(len(predictions), max_points, replace=False)
        predictions = predictions[idx]
        targets = targets[idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Scatter plot
    ax = axes[0]
    ax.scatter(targets, predictions, alpha=0.1, s=1)
    lims = [min(targets.min(), predictions.min()), max(targets.max(), predictions.max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Predictions vs Targets')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Error distribution
    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='Zero error')
    ax.axvline(errors.mean(), color='g', linestyle='--', label=f'Mean: {errors.mean():.4f}')
    ax.set_xlabel('Error (Pred - Target)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Relative error vs target magnitude
    ax = axes[2]
    abs_targets = np.abs(targets)
    rel_errors = np.abs(errors) / (abs_targets + 1e-6)
    # Bin by target magnitude
    bins = np.percentile(abs_targets, np.linspace(0, 100, 21))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_errors = []
    for i in range(len(bins) - 1):
        mask = (abs_targets >= bins[i]) & (abs_targets < bins[i + 1])
        if mask.sum() > 0:
            bin_errors.append(np.median(rel_errors[mask]))
        else:
            bin_errors.append(np.nan)
    ax.plot(bin_centers, bin_errors, 'b-o')
    ax.set_xlabel('Target Magnitude')
    ax.set_ylabel('Median Relative Error')
    ax.set_title('Relative Error vs Target Magnitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions plot to {output_path}")

    return fig


def plot_loss_by_region(
    coords: np.ndarray,
    errors: np.ndarray,
    dim_names: List[str] = ['diff', 'x', 'y', 't'],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot error distribution by input dimension.

    Args:
        coords: Input coordinates of shape (N, 4).
        errors: Prediction errors of shape (N,).
        dim_names: Names for each dimension.
        output_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    abs_errors = np.abs(errors)

    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        # Bin by this dimension
        dim_vals = coords[:, i]
        bins = np.percentile(dim_vals, np.linspace(0, 100, 21))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_means = []
        bin_stds = []
        for j in range(len(bins) - 1):
            mask = (dim_vals >= bins[j]) & (dim_vals < bins[j + 1])
            if mask.sum() > 0:
                bin_means.append(np.mean(abs_errors[mask]))
                bin_stds.append(np.std(abs_errors[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        ax.plot(bin_centers, bin_means, 'b-o', label='Mean |error|')
        ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                        alpha=0.3, label='±1 std')
        ax.set_xlabel(name)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'Error vs {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved error by region plot to {output_path}")

    return fig


def create_training_report(
    output_dir: str,
    report_path: Optional[str] = None,
) -> None:
    """
    Create a comprehensive training report with all plots.

    Args:
        output_dir: Directory containing training outputs.
        report_path: Optional path for report directory (defaults to output_dir/report).
    """
    output_dir = Path(output_dir)
    report_dir = Path(report_path) if report_path else output_dir / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating training report in {report_dir}")

    # Plot training history
    history_path = output_dir / 'history.json'
    if history_path.exists():
        plot_training_history(
            str(history_path),
            str(report_dir / 'training_history.png'),
        )
        plt.close()

    print(f"Report saved to {report_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize SIREN training')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Training output directory')
    parser.add_argument('--report_dir', type=str, default=None,
                        help='Report output directory')

    args = parser.parse_args()
    create_training_report(args.output_dir, args.report_dir)
