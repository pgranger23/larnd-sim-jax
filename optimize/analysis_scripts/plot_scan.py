#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import argparse
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
plt.rcParams['font.size'] = 15

def parse_args():
    parser = argparse.ArgumentParser(description="Plot gradient scan results")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="scan_results",
        help="Path to the input dir with scan result files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scan_plots",
        help="Path to the output dir",
    )
    return parser.parse_args()


def load_scan_file(fname):
    """Load a scan pickle file and validate it."""
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    
    if 'config' not in results:
        raise ValueError(f"No config found in {fname}")
    
    config = results['config']
    if not hasattr(config, 'fit_type') or config.fit_type != 'scan':
        raise ValueError(f"Expected fit_type='scan', found {getattr(config, 'fit_type', 'unknown')} in {fname}")
    
    # Extract parameter names
    params = [key.replace('_grad', '') for key in results.keys() if '_grad' in key]
    
    return results, params, config


def extract_param_from_filename(fname, params):
    """
    Extract parameter name and batch number from filename.
    Expected format: history_{param}_batch{N}_{label}.pkl
    """
    basename = os.path.basename(fname)
    if not basename.startswith('history_'):
        return None, None
    
    parts = basename.split('_')
    
    # Try to match parameter name (can be multi-word like "long_diff")
    for i in range(1, len(parts)):
        potential_param = '_'.join(parts[1:i+1])
        if potential_param in params:
            # Found the parameter, now look for batch number
            batch_num = None
            for j in range(i+1, len(parts)):
                if parts[j].startswith('batch'):
                    try:
                        batch_num = int(parts[j].replace('batch', ''))
                    except ValueError:
                        pass
            return potential_param, batch_num
    
    return None, None


def extract_scan_data(results, params, config, fname):
    """
    Extract scan data for the parameter specified in filename.
    
    LikelihoodProfiler layout:
    - Data accumulates: [batch0_param0, batch0_param1, ..., batch1_param0, batch1_param1, ...]
    - Each scan is nb_iter points
    - Filename format: history_{param}_batch{N}_{label}.pkl
    
    Returns:
        param_values: (nbatches, nb_iter) array
        gradients: (nbatches, nb_iter) array  
        losses: (nbatches, nb_iter) array
        aux_data: dict with sub-loss terms (nbatches, nb_iter) arrays
        param_name: name of the parameter
        target: target value for this parameter
    """
    nparams = len(params)
    nb_iter = config.iterations
    total_points = len(results["losses_iter"])
    
    # Extract parameter name and batch number from filename
    param_name, batch_num = extract_param_from_filename(fname, params)
    
    if param_name is None:
        raise ValueError(f"Could not extract parameter name from filename: {fname}")
    
    if param_name not in params:
        raise ValueError(f"Parameter {param_name} not found in file {fname}")
    
    param_idx = params.index(param_name)
    
    # Validate data layout
    if total_points % nb_iter != 0:
        raise ValueError(f"Total points {total_points} not divisible by iterations {nb_iter}")
    
    n_scans = total_points // nb_iter
    
    # Extract data for all batches up to batch_num
    if batch_num is None:
        logger.warning(f"No batch number in filename, extracting all batches")
        num_batches = (n_scans + nparams - 1) // nparams  # Estimate total batches
    else:
        num_batches = batch_num + 1  # batch_num is 0-indexed
    
    param_values_list = []
    grad_list = []
    loss_list = []
    aux_data = {}  # Will store sub-losses per batch
    
    for b in range(num_batches):
        # For batch b, this parameter's scan is at: b * nparams + param_idx
        scan_idx = b * nparams + param_idx
        
        if scan_idx >= n_scans:
            logger.warning(f"Batch {b} exceeds available scans, stopping")
            break
        
        # Extract data for this scan
        # Note: _iter arrays have +1 offset (initial value at index 0)
        start_iter = 1 + scan_idx * nb_iter
        end_iter = start_iter + nb_iter
        start_data = scan_idx * nb_iter
        end_data = start_data + nb_iter
        
        param_values_list.append(results[f"{param_name}_iter"][start_iter:end_iter])
        grad_list.append(results[f"{param_name}_grad"][start_data:end_data])
        loss_list.append(results["losses_iter"][start_data:end_data])
        
        # Extract auxiliary data if available
        if 'aux_iter' in results and len(results['aux_iter']) > 0:
            aux_slice = results['aux_iter'][start_data:end_data]
            
            # Initialize aux_data dict on first batch
            if b == 0 and len(aux_slice) > 0:
                # Get available keys from first entry
                first_entry = aux_slice[0] if isinstance(aux_slice[0], dict) else {}
                for key in first_entry.keys():
                    aux_data[key] = []
            
            # Extract sub-loss values for this batch
            for key in aux_data.keys():
                batch_values = []
                for entry in aux_slice:
                    if isinstance(entry, dict) and key in entry:
                        val = entry[key]
                        # Convert to float if needed
                        if isinstance(val, (float, int)):
                            val = float(val)
                        else:
                            val = np.nan  # Non-numeric value, set to NaN
                        batch_values.append(val)
                    else:
                        batch_values.append(np.nan)
                aux_data[key].append(batch_values)
    
    param_values = np.array(param_values_list)
    gradients = np.array(grad_list)
    losses = np.array(loss_list)
    
    # Convert aux_data to arrays
    for key in aux_data.keys():
        aux_data[key] = np.array(aux_data[key])
    
    # Get target value
    target = results.get(f"{param_name}_target", [None])[0] if f"{param_name}_target" in results else None
    
    logger.info(f"Extracted {len(param_values_list)} batches for {param_name}")
    logger.info(f"Data shape: {param_values.shape}, range [{param_values.min():.6e}, {param_values.max():.6e}]")
    if aux_data:
        logger.info(f"Auxiliary data keys: {list(aux_data.keys())}")
    
    return param_values, gradients, losses, aux_data, param_name, target


def make_title(config):
    """Generate title string from config."""
    batch_size = config.max_batch_len
    noise = not config.no_noise
    seed_strategy = config.sim_seed_strategy
    sampling = config.electron_sampling_resolution * 1e4
    
    return (f"{batch_size:.0f}cm batches ; Noise: {'on' if noise else 'off'} ; "
            f"Random strategy: {seed_strategy} ; Sampling resolution: {sampling:.0f}um")


def plot_time(fname, ax=None):
    """Plot computation time per iteration."""
    results, params, config = load_scan_file(fname)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Get parameter name from filename
    param_name, _ = extract_param_from_filename(fname, params)
    if param_name is None:
        param_name = params[0]
    
    time = np.array(results["step_time"])
    
    ax.plot(time, label='Time')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Iteration')
    ax.set_title(f"Time per iteration for {param_name}")
    ax.get_figure().suptitle(make_title(config))
    ax.grid(True, alpha=0.3)


def plot_gradient_scan(fname, ax=None, plot_all=False):
    """
    Plot gradient and loss vs parameter value from scan.
    
    Args:
        fname: Path to pickle file
        ax: Matplotlib axis (creates new figure if None)
        plot_all: If True, plot all batches; if False, plot mean across batches
    """
    results, params, config = load_scan_file(fname)
    
    # Extract scan data
    param_values, gradients, losses, aux_data, param_name, target = extract_scan_data(
        results, params, config, fname
    )
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Average across batches if requested
    if not plot_all:
        param_values_plot = np.nanmean(param_values, axis=0)
        gradients_plot = np.nanmean(gradients, axis=0)
        losses_plot = np.nanmean(losses, axis=0)
    else:
        param_values_plot = param_values.T
        gradients_plot = gradients.T
        losses_plot = losses.T
    
    # Plot gradient on primary axis
    l1 = ax.plot(param_values_plot, gradients_plot, color='blue', label='gradient')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Gradient', color='blue')
    ax.tick_params(axis='y', colors='blue')
    
    # Plot loss on secondary axis
    ax2 = ax.twinx()
    l2 = ax2.plot(param_values_plot, losses_plot, color='green', label='loss')
    ax2.set_ylabel('Loss', color='green')
    ax2.tick_params(axis='y', colors='green')
    
    # Add reference lines
    l3 = ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='zero gradient')
    
    lines = l1[:1] + l2[:1] + [l3]
    if target is not None:
        l4 = ax.axvline(target, color='red', linestyle='--', linewidth=1, alpha=0.7, label='target')
        lines.append(l4)
    
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')
    
    ax.set_title(f"Gradient scan for {param_name}")
    ax.get_figure().suptitle(make_title(config))
    ax.grid(True, alpha=0.3)


def plot_subloss_scan(fname, ax=None, plot_all=False):
    """
    Plot sub-loss terms vs parameter value from scan.
    
    Args:
        fname: Path to pickle file
        ax: Matplotlib axis (creates new figure if None)
        plot_all: If True, plot all batches; if False, plot mean across batches
    """
    results, params, config = load_scan_file(fname)
    
    # Extract scan data
    param_values, gradients, losses, aux_data, param_name, target = extract_scan_data(
        results, params, config, fname
    )
    
    # Check if we have sub-loss data
    if not aux_data:
        if ax is not None:
            ax.text(0.5, 0.5, 'No auxiliary data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        return
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Average across batches if requested
    if not plot_all:
        param_values_plot = np.nanmean(param_values, axis=0)
    else:
        param_values_plot = param_values.T
    
    # Plot each sub-loss term
    colors = ['purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    lines = []
    
    for idx, (key, values) in enumerate(aux_data.items()):
        if not plot_all:
            values_plot = np.nanmean(values, axis=0)
        else:
            values_plot = values.T
        
        color = colors[idx % len(colors)]
        line = ax.plot(param_values_plot, values_plot, color=color, label=key, alpha=0.8)
        lines.extend(line)
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Sub-loss values')
    
    # Add target reference line
    if target is not None:
        ax.axvline(target, color='red', linestyle='--', linewidth=1, alpha=0.7, label='target')
    
    ax.legend(loc='best')
    ax.set_title(f"Sub-loss terms for {param_name}")
    ax.get_figure().suptitle(make_title(config))
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of scan files
    input_dir = Path(args.input_dir)
    list_of_files = sorted(input_dir.glob('*.pkl'))
    
    if not list_of_files:
        raise ValueError(f"No .pkl files found in {args.input_dir}")
    
    nparams = len(list_of_files)
    logger.info(f"Found {nparams} scan files in {args.input_dir}")
    
    # Calculate grid dimensions
    ncols = min(3, nparams)
    nrows = (nparams + ncols - 1) // ncols
    
    # Create figures for all batches
    fig_all, axs_all = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    fig_time, axs_time = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    fig_subloss_all, axs_subloss_all = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    
    for i, fname in enumerate(list_of_files):
        row, col = i // ncols, i % ncols
        
        try:
            plot_gradient_scan(str(fname), axs_all[row, col], plot_all=True)
            plot_time(str(fname), axs_time[row, col])
            plot_subloss_scan(str(fname), axs_subloss_all[row, col], plot_all=True)
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}", exc_info=True)
            axs_all[row, col].text(0.5, 0.5, f'Error: {e}', 
                                   ha='center', va='center', transform=axs_all[row, col].transAxes)
            axs_time[row, col].text(0.5, 0.5, f'Error: {e}',
                                    ha='center', va='center', transform=axs_time[row, col].transAxes)
            axs_subloss_all[row, col].text(0.5, 0.5, f'Error: {e}',
                                           ha='center', va='center', transform=axs_subloss_all[row, col].transAxes)
    
    # Hide unused subplots
    for i in range(nparams, nrows * ncols):
        row, col = i // ncols, i % ncols
        axs_all[row, col].axis('off')
        axs_time[row, col].axis('off')
        axs_subloss_all[row, col].axis('off')
    
    fig_all.tight_layout()
    fig_all.savefig(output_dir / 'gradient_scan.pdf')
    fig_all.savefig(output_dir / 'gradient_scan.png', dpi=300)
    
    fig_time.tight_layout()
    fig_time.savefig(output_dir / 'gradient_scan_time.pdf')
    fig_time.savefig(output_dir / 'gradient_scan_time.png', dpi=300)
    
    fig_subloss_all.tight_layout()
    fig_subloss_all.savefig(output_dir / 'subloss_scan.pdf')
    fig_subloss_all.savefig(output_dir / 'subloss_scan.png', dpi=300)
    
    logger.info(f"Saved all-batch plots to {output_dir}")
    
    # Create figure for averaged plots
    fig_avg, axs_avg = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    fig_subloss_avg, axs_subloss_avg = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    
    for i, fname in enumerate(list_of_files):
        row, col = i // ncols, i % ncols
        
        try:
            plot_gradient_scan(str(fname), axs_avg[row, col], plot_all=False)
            plot_subloss_scan(str(fname), axs_subloss_avg[row, col], plot_all=False)
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}", exc_info=True)
            axs_avg[row, col].text(0.5, 0.5, f'Error: {e}',
                                   ha='center', va='center', transform=axs_avg[row, col].transAxes)
            axs_subloss_avg[row, col].text(0.5, 0.5, f'Error: {e}',
                                           ha='center', va='center', transform=axs_subloss_avg[row, col].transAxes)
    
    # Hide unused subplots
    for i in range(nparams, nrows * ncols):
        row, col = i // ncols, i % ncols
        axs_avg[row, col].axis('off')
        axs_subloss_avg[row, col].axis('off')
    
    fig_avg.tight_layout()
    fig_avg.savefig(output_dir / 'gradient_scan_avg.pdf')
    fig_avg.savefig(output_dir / 'gradient_scan_avg.png', dpi=300)
    
    fig_subloss_avg.tight_layout()
    fig_subloss_avg.savefig(output_dir / 'subloss_scan_avg.pdf')
    fig_subloss_avg.savefig(output_dir / 'subloss_scan_avg.png', dpi=300)
    
    logger.info(f"Saved averaged plots to {output_dir}")


if __name__ == "__main__":
    main()