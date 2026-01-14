#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os
from pprint import pprint
import argparse
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
plt.rcParams['font.size'] = 15

def print_config(config):
    pprint(vars(config), indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot gradient scan results")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="scan_results",
        help="Path to the input dir. Expect single param per file",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the input file. Expect all params in the same file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="scan_plots",
        help="Path to the output dir",
    )
    return parser.parse_args()


def plot_time(fname, ax=None, ipar=0):
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    # print_config(results['config'])
    
    if 'fit_type' not in results['config']:
        raise ValueError(f"Expected fit_type in {fname}")
    
    if results['config'].fit_type != 'scan':
        raise ValueError(f"Expected fit_type scan, found {results['config']['fit_type']} in {fname}")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    params = [key.replace('_grad', '') for key in results.keys() if '_grad' in key]
    
    if ipar >= len(params):
        return
    
    nparams_in_file = len(params)
    nb_iter = results['config'].iterations
    total_data_points = len(results["losses_iter"])
    
    # Detect mode and select appropriate parameter
    if total_data_points % (nb_iter * nparams_in_file) == 0:
        # Multi-param file
        param = params[ipar]
    elif total_data_points % nb_iter == 0:
        # Single-param file: find which parameter was scanned
        scanned_param = None
        for p in params:
            param_values = np.array(results[f"{p}_iter"][1:])
            if len(np.unique(param_values)) > 1:
                scanned_param = p
                break
        param = scanned_param if scanned_param else params[0]
    else:
        param = params[0]

    title = f"{results['config'].max_batch_len:.0f}cm batches ; Noise: {'on' if not results['config'].no_noise else 'off'} ; Random strategy: {results['config'].sim_seed_strategy} ; Sampling resolution: {results['config'].electron_sampling_resolution*1e4:.0f}um"
    
    time = np.array(results["step_time"])

    ax.plot(time, label='Loss')
    ax.set_ylabel('Time (s)')
    ax.set_title(f"Time per iteration for {param}")
    ax.get_figure().suptitle(title)

def plot_gradient_scan(fname, ax=None, plot_all=False, ipar=0):
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    # print_config(results['config'])
    params = [key.replace('_grad', '') for key in results.keys() if '_grad' in key]

    if 'fit_type' not in results['config']:
        raise ValueError(f"Expected fit_type in {fname}")
    
    if results['config'].fit_type != 'scan':
        raise ValueError(f"Expected fit_type scan, found {results['config']['fit_type']} in {fname}")
    
    if ipar >= len(params):
        return
    
    batch_size = results['config'].max_batch_len
    noise = (not results['config'].no_noise)
    title = f"{batch_size:.0f}cm batches ; Noise: {'on' if noise else 'off'} ; Random strategy: {results['config'].sim_seed_strategy} ; Sampling resolution: {results['config'].electron_sampling_resolution*1e4:.0f}um"

    nparams_in_file = len(params)
    nb_iter = results['config'].iterations
    total_data_points = len(results["losses_iter"])
    
    # Detect if this is a single-param file or multi-param file
    # Single-param files: data for one parameter only (per-file scan mode)
    # Multi-param files: data for all parameters together (single-file scan mode)
    
    # Try multi-param layout first
    if total_data_points % (nb_iter * nparams_in_file) == 0:
        # Multi-param file: all parameters scanned together
        # Use the provided ipar to select the parameter
        param = params[ipar]
        nbatches = total_data_points // (nb_iter * nparams_in_file)
        param_value = np.array(results[f"{param}_iter"][1:]).reshape(nbatches, nparams_in_file, -1)[:, ipar, :]
        grad = np.array(results[f"{param}_grad"]).reshape(nbatches, nparams_in_file, -1)[:, ipar, :]
        loss = np.array(results["losses_iter"]).reshape(nbatches, nparams_in_file, -1)[:, ipar, :]
    elif total_data_points % nb_iter == 0:
        # Single-param file: only one parameter was scanned
        # Find which parameter actually has varying values
        scanned_param = None
        for p in params:
            param_values = np.array(results[f"{p}_iter"][1:])
            if len(np.unique(param_values)) > 1:  # This parameter varies
                scanned_param = p
                break
        
        if scanned_param is None:
            # Fallback: use first parameter
            logger.warning(f"Could not determine which parameter was scanned in {fname}, using {params[0]}")
            scanned_param = params[0]
        
        param = scanned_param
        nbatches = total_data_points // nb_iter
        param_value = np.array(results[f"{param}_iter"][1:]).reshape(nbatches, -1)
        grad = np.array(results[f"{param}_grad"]).reshape(nbatches, -1)
        loss = np.array(results["losses_iter"]).reshape(nbatches, -1)
    else:
        raise ValueError(f"Cannot determine data layout: losses_iter length {total_data_points} is not divisible by nb_iter ({nb_iter}) or nb_iter*nparams ({nb_iter}*{nparams_in_file}={nb_iter*nparams_in_file})")

    target = results[f"{param}_target"]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if not plot_all:
        grad = np.nanmean(grad, axis=0)
        loss = np.nanmean(loss, axis=0)

    # grad = np.nanmedian(np.array(results[f"{param}_grad"][:]).reshape(-1, nbatches), axis=0)
    # loss = np.nanmedian(np.array(results["losses_iter"][:]).reshape(-1, nbatches), axis=0)
    # pprint(results[f"{param}_grad"][:41])
    l1 = ax.plot(param_value.T, grad.T, color='blue', label="gradient")
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', colors='green')
    ax.tick_params(axis='y', colors='blue')

    l2 = ax2.plot(param_value.T, loss.T, color='green', label="loss")
    l3 = ax.axvline(target, color='red', label='target')
    l4 = ax.axhline(0, color='red', label='target')
    
    lns = l1[:1] + l2[:1] + [l3]
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)
    ax.set_xlabel(param)
    ax.get_figure().suptitle(title)
    # print(len(results[f"{param}_grad"]))

if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine the number of parameters and create file list
    if args.input_file is not None:
        if not os.path.exists(args.input_file):
            raise ValueError(f"Input file {args.input_file} does not exist")
        
        # Load file to determine number of parameters
        with open(args.input_file, 'rb') as f:
            results = pickle.load(f)
        params = [key.replace('_grad', '') for key in results.keys() if '_grad' in key]
        nparams = len(params)
        
        # Create list with same file repeated for each parameter
        list_of_files = [args.input_file] * nparams
        param_indices = list(range(nparams))
        logger.info(f"Using input file {args.input_file} with {nparams} parameters: {params}")
    else:
        list_of_files = glob.glob(f'{input_dir}/*.pkl')
        if not list_of_files:
            raise ValueError(f"No .pkl files found in {input_dir}")
        
        nparams = len(list_of_files)
        # For directory mode, each file contains a single parameter (ipar=0)
        param_indices = [0] * nparams
        logger.info(f"Found {nparams} files in {input_dir}")

    # Calculate grid dimensions
    ncols = min(3, nparams)
    nrows = (nparams + ncols - 1) // ncols  # Ceiling division
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)
    fig_time, axs_time = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)

    for i, (f, ipar) in enumerate(zip(list_of_files, param_indices)):
        row, col = i // ncols, i % ncols
        ax = axs[row, col]
        ax_time = axs_time[row, col]
        
        plot_gradient_scan(f, ax, True, ipar)
        plot_time(f, ax_time, ipar)
    
    # Hide unused subplots
    for i in range(nparams, nrows * ncols):
        row, col = i // ncols, i % ncols
        axs[row, col].axis('off')
        axs_time[row, col].axis('off')
    
    fig.tight_layout()
    fig.savefig(f'{output_dir}/gradient_scan.pdf')
    fig.savefig(f'{output_dir}/gradient_scan.png', dpi=300)

    fig_time.tight_layout()
    fig_time.savefig(f'{output_dir}/gradient_scan_time.pdf')
    fig_time.savefig(f'{output_dir}/gradient_scan_time.png', dpi=300)

    # Second figure for averaged plots
    fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), squeeze=False)

    for i, (f, ipar) in enumerate(zip(list_of_files, param_indices)):
        row, col = i // ncols, i % ncols
        ax = axs[row, col]
        plot_gradient_scan(f, ax, False, ipar)
    
    # Hide unused subplots
    for i in range(nparams, nrows * ncols):
        row, col = i // ncols, i % ncols
        axs[row, col].axis('off')
    
    fig.tight_layout()
    fig.savefig(f'{output_dir}/gradient_scan_avg.pdf')
    fig.savefig(f'{output_dir}/gradient_scan_avg.png', dpi=300)