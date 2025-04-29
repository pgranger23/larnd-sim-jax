#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import argparse
import sys
import traceback
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.sim_jax import prepare_tracks, simulate, simulate_parametrized
from pprint import pprint
import numpy as np
import h5py
import jax

from .fit_params import ParamFitter
from .dataio import TracksDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# jax.config.update('jax_log_compiles', True)
jax.config.update('jax_platform_name', 'cpu')

def load_lut(config):
    response = np.load(config.lut_file)
    extended_response = np.zeros((50, 50, 1891))
    extended_response[:45, :45, :] = response
    response = extended_response
    baseline = np.sum(response[:, :, :-config.signal_length+1], axis=-1)
    response = np.concatenate([baseline[..., None], response[..., -config.signal_length+1:]], axis=-1)
    return response

def main(config):
    if config.lut_file == "" and config.mode == 'lut':
        return 1, 'Error: LUT file is required for mode "lut"'

    Params = build_params_class([])
    ref_params = load_detector_properties(Params, config.detector_props, config.pixel_layouts)
    ref_params = ref_params.replace(
        electron_sampling_resolution=config.electron_sampling_resolution,
        number_pix_neighbors=config.number_pix_neighbors,
        signal_length=config.signal_length,
        time_window=config.signal_length,
        )
    
    if not config.noise:
        ref_params = ref_params.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)

    tracks, fields, original_tracks = prepare_tracks(ref_params, config.input_file)
    logger.info(f"Loaded {len(tracks)} segments from {config.input_file}")

    if args.mode == 'lut':
        response = load_lut(config)
        ref, pixels_ref, ticks_ref, pix_matching, electrons, ticks_electrons = simulate(ref_params, response, tracks, fields, rngseed=config.seed)
    else:
        ref, pixels_ref, ticks_ref, pix_matching, electrons, ticks_electrons = simulate_parametrized(ref_params, tracks, fields, rngseed=config.seed, diffusion_in_current_sim=config.diffusion_in_current_sim)

    with h5py.File(config.output_file, 'w') as f:
        f.create_dataset('adc', data=ref)
        f.create_dataset('pixels', data=pixels_ref)
        f.create_dataset('ticks', data=ticks_ref)
        # f.create_dataset('pixel_signals', data=signals_ref)
    
    return 0, 'Success'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", dest="input_file",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input data file")
    parser.add_argument("--output_file", dest="output_file",
                        help="Output data file", required=True)
    parser.add_argument("--detector_props", dest="detector_props",
                        default="src/larndsim/detector_properties/module0.yaml",
                        help="Path to detector properties YAML file")
    parser.add_argument("--pixel_layouts", dest="pixel_layouts",
                        default="src/larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument('--mode', type=str, help='Mode used to simulate the induced current on the pixels', choices=['lut', 'parametrized'], default='lut')
    parser.add_argument('--electron_sampling_resolution', type=float, required=True, help='Electron sampling resolution')
    parser.add_argument('--number_pix_neighbors', type=int, required=True, help='Number of pixel neighbors')
    parser.add_argument('--signal_length', type=int, required=True, help='Signal length')
    parser.add_argument('--lut_file', type=str, required=False, default="", help='Path to the LUT file')
    parser.add_argument('--noise', action='store_true', help='Add noise to the simulation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--diffusion_in_current_sim', action='store_true', help='Use diffusion in current simulation')

    try:
        args = parser.parse_args()
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    logger.info(status_message)
    exit(retval)
