#!/usr/bin/env python3
"""Compute Taylor scan results for all parameters and events in one input file.

All parameters are made differentiable simultaneously (single JAX compilation).
Jacobian and Hessian are computed once per event for all parameters.

Usage:
    python3 tests/taylor/run_taylor_scan.py --input_id 0
    python3 tests/taylor/run_taylor_scan.py --input_id 15
"""

import os, sys, argparse
os.environ['JAX_PLATFORMS'] = 'cuda'

# Add tests/taylor to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch  # must be imported before jax
import numpy as np
from scan_utils import setup_params_and_tracks, compute_event_all_params, save_results

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=int, required=True, help='Input file ID (e.g. 0, 10, 21)')
args = parser.parse_args()

# --- Configuration ---
INPUT_FILE = f'prepared_data/input_{args.input_id}.h5'
PARAMS = ['Ab', 'kb', 'eField', 'long_diff', 'tran_diff', 'lifetime']
N_POINTS = 21

# Per-parameter perturbation half-ranges (fraction of nominal)
PARAM_RANGES = {
    'Ab':        0.20,
    'kb':        0.50,
    'eField':    0.005,
    'long_diff': 0.50,
    'tran_diff': 0.50,
    'lifetime':  0.50,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RESULTS_DIR, f'taylor_scan_{args.input_id}.pkl')

print(f'Input: {INPUT_FILE}')
print(f'Output: {OUTPUT_FILE}')
print(f'Parameters: {PARAMS}')
print(f'Ranges: {PARAM_RANGES}')

# --- Setup (single compilation for all params) ---
ref_params, response, events, fields = setup_params_and_tracks(INPUT_FILE, PARAMS)
print(f'Loaded {len(events)} events: {sorted(events.keys())}')

# --- Compute: one event at a time, all params per event ---
results = {p: {} for p in PARAMS}

for eid in sorted(events.keys()):
    print(f'\n===== Event {eid} =====', flush=True)
    try:
        event_results = compute_event_all_params(
            events[eid], ref_params, response, fields,
            PARAMS, PARAM_RANGES, N_POINTS,
        )
        for param_name in PARAMS:
            results[param_name][eid] = event_results[param_name]
            H = event_results[param_name]['H_scalar']
            n = event_results[param_name]['n_active']
        print(f'  {n} pix, H=[' + ', '.join(
            f'{p}:{event_results[p]["H_scalar"]:.1f}' for p in PARAMS
        ) + ']', flush=True)
        save_results(results, OUTPUT_FILE)
    except Exception as e:
        print(f'  FAILED: {e}', flush=True)

print(f'\nDone. Results saved to {OUTPUT_FILE}')
