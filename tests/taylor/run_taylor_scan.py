#!/usr/bin/env python3
"""Compute validity results for all parameters and events. Run via SLURM."""

import os, sys
os.environ['JAX_PLATFORMS'] = 'cuda'

# Add tests/taylor to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch  # must be imported before jax
import numpy as np
from scan_utils import setup_params_and_tracks, compute_event_scan, save_results

# --- Configuration (same as notebook) ---
INPUT_FILE = 'prepared_data/input_0.h5'
PARAMS = ['Ab', 'kb', 'eField', 'long_diff', 'tran_diff', 'lifetime']
REL_DELTAS = np.linspace(-0.50, 0.50, 21)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'taylor_scan_results.pkl')

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
