#!/usr/bin/env python3
"""Compute Taylor scan results for all parameters and events in one input file.

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
from scan_utils import setup_params_and_tracks, compute_event_scan, save_results

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=int, required=True, help='Input file ID (e.g. 0, 10, 21)')
parser.add_argument('--params', nargs='+', default=['Ab', 'kb', 'eField', 'long_diff', 'tran_diff', 'lifetime'],
                    help='Parameters to scan')
args = parser.parse_args()

# --- Configuration ---
INPUT_FILE = f'prepared_data/input_{args.input_id}.h5'
PARAMS = args.params
REL_DELTAS = np.linspace(-0.50, 0.50, 21)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RESULTS_DIR, f'taylor_scan_{args.input_id}.pkl')

print(f'Input: {INPUT_FILE}')
print(f'Output: {OUTPUT_FILE}')
print(f'Parameters: {PARAMS}')

# --- Setup ---
ref_params_base, response, events, fields = setup_params_and_tracks(INPUT_FILE)
print(f'Loaded {len(events)} events: {sorted(events.keys())}')

# --- Compute ---
results = {}

for param_name in PARAMS:
    print(f'\n===== Parameter: {param_name} =====', flush=True)
    results[param_name] = {}

    for eid in sorted(events.keys()):
        print(f'  Event {eid}...', end=' ', flush=True)
        try:
            res = compute_event_scan(
                events[eid], ref_params_base, response, fields,
                param_name, REL_DELTAS,
            )
            results[param_name][eid] = res
            save_results(results, OUTPUT_FILE)
            print(f'{res["n_active"]} pix, H={res["H_scalar"]:.2f}', flush=True)
        except Exception as e:
            print(f'FAILED: {e}', flush=True)

print(f'\nDone. Results saved to {OUTPUT_FILE}')
