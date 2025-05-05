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

    param = params[ipar]
    nparams = len(params)

    nb_iter = results['config'].iterations

    target = results[f"{param}_target"]

    # nbatches = results['config'].max_nbatch
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # max_index = (len(results["losses_iter"])//nbatches)*nbatches

    # param_value = np.average(np.array(results[f"{param}_iter"][1:max_index + 1]).reshape(-1, nbatches), axis=1)
    param_value = np.sort(np.unique(results[f"{param}_iter"][1:]))
    nbatches = len(results["losses_iter"])//(nb_iter*nparams)
    if len(results["losses_iter"]) % (nb_iter*nparams) != 0:
        raise ValueError(f"Expected losses_iter to be divisible by param_value, found {len(results['losses_iter'])} and {param_value}")


    param_value = np.array(results[f"{param}_iter"][1:]).reshape(nbatches, nparams, -1)[:, ipar, :]
    grad = np.array(results[f"{param}_grad"]).reshape(nbatches, nparams, -1)[:, ipar, :]
    loss = np.array(results["losses_iter"]).reshape(nbatches, nparams, -1)[:, ipar, :]

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

    if args.input_file is not None:
        if not os.path.exists(args.input_file):
            raise ValueError(f"Input file {args.input_file} does not exist")
        list_of_files = [args.input_file] * 9
        logger.info(f"Using input file {args.input_file}")
    else:
        list_of_files = glob.glob(f'{input_dir}/*.pkl')
        logger.info(f"Found {len(list_of_files)} files in {input_dir}")

    fig, axs = plt.subplots(3, 3, figsize=(20, 15))

    for i, f in enumerate(list_of_files):
        ax = axs[i//3, i%3]
        if args.input_file is not None:
            plot_gradient_scan(f, ax, True, i)
        else:
            plot_gradient_scan(f, ax, True, 0)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/gradient_scan.pdf')
    fig.savefig(f'{output_dir}/gradient_scan.png', dpi=300)

    fig, axs = plt.subplots(3, 3, figsize=(20, 15))

    for i, f in enumerate(list_of_files):
        ax = axs[i//3, i%3]
        if args.input_file is not None:
            plot_gradient_scan(f, ax, False, i)
        else:
            plot_gradient_scan(f, ax, False, 0)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/gradient_scan_avg.pdf')
    fig.savefig(f'{output_dir}/gradient_scan_avg.png', dpi=300)