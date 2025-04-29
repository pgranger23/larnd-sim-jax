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
from pprint import pprint


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
plt.rcParams['font.size'] = 15

def print_config(config):
    pprint(vars(config), indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot minuit fit results")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file. Expect all params in the same file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="scan_plots",
        help="Path to the output dir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.input_file):
        raise ValueError(f"Input file {args.input_file} does not exist")
    logger.info(f"Using input file {args.input_file}")

    with open(args.input_file, 'rb') as f:
        results = pickle.load(f)
    minuit_res = results['minuit_result'][0]
    pprint(vars(results['config']))
    
    nfits = len(results['minuit_result'])

    list_params = [key.replace('_target', '') for key in results.keys() if '_target' in key]
    if nfits == 1:
        logger.info("Only one fit found, just printing results")
        logger.info(f"Is the fit valid? {minuit_res['valid']}")
        logger.info(f"Loss at minimum: {minuit_res['fval']}")
        logger.info(f"Estimated distance to minimum: {minuit_res['edm']}")
        for par in list_params:
            logger.info(f"{par}: Target: {results[f'{par}_target'][0]} Minuit: {minuit_res['params'][par]} Error: {minuit_res['params'][par]/results[f'{par}_target'][0] - 1:.2%}")
    else:
        logger.info("Multiple fits found, making the plots")
        fig, axs = plt.subplots(3, 3, figsize=(20, 15))
        fig2, axs2 = plt.subplots(3, 3, figsize=(20, 15))

        batch_size = results['config'].max_batch_len
        noise = (not results['config'].no_noise)
        title = f"{batch_size:.0f}cm batches ; Noise: {'on' if noise else 'off'} ; Random strategy: {results['config'].sim_seed_strategy} ; Sampling resolution: {results['config'].electron_sampling_resolution*1e4:.0f}um"

        pprint(results['step_time'])

        for i, par in enumerate(list_params):
            ax = axs[i//3, i%3]
            ax2 = axs2[i//3, i%3]
            par_results = [results['minuit_result'][j]['params'][par] for j in range(nfits)]
            par_loss = [results['minuit_result'][j]['fval'] for j in range(nfits)]
            par_target = results[f'{par}_target'][0]
            par_range = (par_target/2, par_target*2)
            flags = [False, False]
            for j in range(nfits):
                if results['minuit_result'][j]['valid']:
                    color = 'b'
                    idx = 0
                else:
                    color = 'r'
                    idx = 1
                if flags[idx]:
                    label = None
                else:
                    flags[idx] = True
                    label = 'valid' if idx==0 else 'invalid'
                ax.errorbar(par_results[j], par_loss[j], xerr=results['minuit_result'][j]['errors'][par], fmt='o', color=color, label=label)
            ax2.hist(par_results, range=par_range, bins=20, histtype='step')

            ax.axvline(results[f'{par}_target'][0], color='r', linestyle='--', label='target')
            ax2.axvline(results[f'{par}_target'][0], color='r', linestyle='--', label='target')
            ax.set_yscale('log')
            ax.set_xlabel(par)
            ax.set_ylabel('Loss')
            ax.legend()
            ax2.legend()

            ax2.set_xlabel(par)
        fig.suptitle(title)
        fig2.suptitle(title)
        fig.tight_layout()
        fig2.tight_layout()

        input_basename = os.path.basename(args.input_file)
        input_basename = os.path.splitext(input_basename)[0]
        fig.savefig(os.path.join(output_dir, f"{input_basename}_minuit_fit.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"{input_basename}_minuit_fit.pdf"))
        fig2.savefig(os.path.join(output_dir, f"{input_basename}_minuit_fit_hist.png"), dpi=300)
        fig2.savefig(os.path.join(output_dir, f"{input_basename}_minuit_fit_hist.pdf"))
    
