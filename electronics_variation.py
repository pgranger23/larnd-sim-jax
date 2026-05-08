#!/usr/bin/env python3

import argparse
import os
import sys
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import logging
from numpy.lib import recfunctions as rfn

# Enforce CPU if --gpu is not specified
if '--gpu' not in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import simulate_wfs, pad_size
from larndsim.fee_jax import get_adc_values, digitize
from larndsim.losses_jax import adc2charge
from optimize.dataio import chop_tracks

logger = logging.getLogger("electronics_variation")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_events(filename, sampling_resolution, swap_xz=True, n_events=-1):
    """
    Loads events from an edepsim HDF5 file and returns a list of chopped tracks,
    where each element in the list corresponds to one event.
    """
    with h5py.File(filename, 'r') as f:
        tracks = np.array(f['segments'])

    if not 't0' in tracks.dtype.names:
        tracks = rfn.append_fields(tracks, 't0', np.zeros(tracks.shape[0]), usemask=False)

    track_fields = tracks.dtype.names

    replace_map = {
            'event_id': 'eventID',
            'traj_id': 'trackID',
        }
    track_fields = tuple([replace_map.get(field, field) if field in replace_map else field for field in track_fields])
    tracks.dtype.names = track_fields

    if n_events > 0:
        evID = np.unique(tracks['eventID'])[:n_events]
        ev_msk = np.isin(tracks['eventID'], evID)
        tracks = tracks[ev_msk]

    if swap_xz:
        x_start = np.copy(tracks['x_start'] )
        x_end = np.copy(tracks['x_end'])
        x = np.copy(tracks['x'])

        tracks['x_start'] = np.copy(tracks['z_start'])
        tracks['x_end'] = np.copy(tracks['z_end'])
        tracks['x'] = np.copy(tracks['z'])

        tracks['z_start'] = x_start
        tracks['z_end'] = x_end
        tracks['z'] = x

    unique_events, first_indices = np.unique(tracks['eventID'], return_index=True)

    first_indices = np.sort(first_indices)
    last_indices = np.r_[first_indices[1:] - 1, len(tracks) - 1]
    
    tracks_unstructured = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)

    logger.info(f"Loaded {len(first_indices)} events from {filename}")
    
    events = []
    for i in range(len(first_indices)):
        event_tracks = tracks_unstructured[first_indices[i]:last_indices[i] + 1, :]
        chopped_event = chop_tracks(event_tracks, track_fields, sampling_resolution)
        events.append(chopped_event)
        
    return events, track_fields

def group_events_to_batches(events, batch_size):
    """
    Groups events into batches such that each event is fully contained in a batch.
    """
    batches = []
    current_batch = []
    current_size = 0
    for event_chopped in events:
        event_size = event_chopped.shape[0]
        if current_size + event_size > batch_size and current_batch:
            batches.append(np.vstack(current_batch))
            current_batch = []
            current_size = 0
        current_batch.append(event_chopped)
        current_size += event_size
    if current_batch:
        batches.append(np.vstack(current_batch))
    return batches

def main(args):
    if args.gpu:
        jax.config.update('jax_platform_name', 'cuda')
    else:
        jax.config.update('jax_platform_name', 'cpu')

    # Load parameters
    Params = build_params_class([]) # No gradients needed
    params = load_detector_properties(Params, args.detector_props, args.pixel_layouts)
    
    # Load LUT
    if args.mode == 'lut':
        if not args.lut_file:
            raise ValueError("LUT file is required for mode 'lut'")
        response_template, params = load_lut(args.lut_file, params)
    else:
        # For parametrized mode, we don't need the LUT
        response_template = None
    
    # Update params with command line args
    params = params.replace(
        electron_sampling_resolution=args.electron_sampling_resolution,
        number_pix_neighbors=args.number_pix_neighbors,
        signal_length=args.signal_length,
        time_window=args.signal_length,
        mc_diff=args.mc_diff,
        diffusion_in_current_sim=args.diffusion_in_current_sim
    )
    
    if not args.noise:
        params = params.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)

    # Load events
    events, track_fields = load_events(args.input_file, args.electron_sampling_resolution, swap_xz=args.swap_xz, n_events=args.n_events)
    
    # Batch events
    batches = group_events_to_batches(events, args.batch_size)
    logger.info(f"Grouped events into {len(batches)} batches")

    # Output file
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    with h5py.File(args.output_file, 'w') as out_f:
        # Save metadata
        out_f.attrs['n_variations'] = args.n_variations
        out_f.attrs['batch_size'] = args.batch_size
        out_f.attrs['seed'] = args.seed if args.seed is not None else -1

        for i_batch, batch_tracks in tqdm(enumerate(batches), total=len(batches), desc="Processing batches"):
            # Pad batch to avoid frequent recompilations
            size = batch_tracks.shape[0]
            padded_size = pad_size(size, "batch_size", 0.5)
            batch_tracks_padded = np.pad(batch_tracks, ((0, padded_size - size), (0, 0)), mode='constant')
            
            tracks_jax = jnp.array(batch_tracks_padded)
            
            # 1. Run simulation up to waveforms (common for all electronics variations)
            if args.mode == 'lut':
                wfs, unique_pixels = simulate_wfs(params, response_template, tracks_jax, track_fields)
            else:
                # Parametrized mode logic for WFs is not as direct in the current repo as simulate_wfs,
                # but we can follow the logic from simulate_parametrized in sim_jax.py
                # For this script, we'll assume LUT mode is preferred as in simulate.py
                raise NotImplementedError("Parametrized mode for waveforms is not yet supported in this script. Use --mode lut.")

            # Mask out invalid pixels (ID = -1)
            valid_mask = unique_pixels != -1
            wfs_valid = wfs[valid_mask]
            unique_pixels_valid = unique_pixels[valid_mask]
            
            batch_group = out_f.create_group(f'batch_{i_batch}')
            batch_group.create_dataset('waveform', data=wfs_valid)
            batch_group.create_dataset('unique_pixels', data=unique_pixels_valid)
            
            # 2. Run N different electronics simulations
            for i_var in range(args.n_variations):
                # Generate a different seed for each variation
                current_seed = (args.seed if args.seed is not None else 0) + i_var
                key = jax.random.PRNGKey(current_seed)
                
                # get_adc_values returns (Npixels, MAX_ADC_VALUES)
                full_adc, full_ticks = get_adc_values(params, wfs, key)
                
                # Filter valid pixels and convert to ADC counts
                full_adc_valid = full_adc[valid_mask]
                full_ticks_valid = full_ticks[valid_mask]
                
                # Convert integrated charge to ADC and then back to physical charge (ke-)
                # digitize converts charge (mV) to ADC. get_adc_values returns integrated charge in electrons.
                # params.GAIN is in mV/ke-.
                
                # Actually, get_adc_values returns integrated charge in electrons.
                # Let's digitize it first to get ADCs as requested
                adcs_valid = digitize(params, full_adc_valid)
                
                # And convert back to charge in ke- for hit_charge
                charges_valid = adc2charge(adcs_valid, params)
                
                # Save variation results
                var_group = batch_group.create_group(f'variation_{i_var}')
                var_group.create_dataset('hit_adc', data=adcs_valid)
                var_group.create_dataset('hit_time', data=full_ticks_valid)
                var_group.create_dataset('hit_charge', data=charges_valid)
                var_group.attrs['seed'] = current_seed

    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LArND-Sim-JAX electronics variation study")
    parser.add_argument("--input_file", type=str, required=True, help="Input edepsim HDF5 file")
    parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 file")
    parser.add_argument("--detector_props", type=str, default="src/larndsim/detector_properties/module0.yaml", help="Path to detector properties YAML")
    parser.add_argument("--pixel_layouts", type=str, default="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml", help="Path to pixel layouts YAML")
    parser.add_argument("--lut_file", type=str, default="", help="Path to the LUT file (required for mode 'lut')")
    parser.add_argument("--mode", type=str, choices=['lut', 'parametrized'], default='lut', help="Simulation mode")
    parser.add_argument("--n_variations", type=int, default=5, help="Number of electronics simulations per batch")
    parser.add_argument("--batch_size", type=int, default=500, help="Target batch size (in segments)")
    parser.add_argument("--n_events", type=int, default=-1, help="Number of events to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    parser.add_argument("--noise", action='store_true', default=True, help="Enable noise (default: True)")
    parser.add_argument("--no-noise", dest='noise', action='store_false', help="Disable noise")
    parser.add_argument("--swap_xz", action='store_true', default=True, help="Swap X and Z coordinates (default: True)")
    parser.add_argument("--electron_sampling_resolution", type=float, default=0.001, help="Electron sampling resolution")
    parser.add_argument("--number_pix_neighbors", type=int, default=1, help="Number of pixel neighbors")
    parser.add_argument("--signal_length", type=int, default=200, help="Signal length (ticks)")
    parser.add_argument("--mc_diff", action='store_true', help="Use Monte Carlo diffusion")
    parser.add_argument("--diffusion_in_current_sim", action='store_true', help="Use diffusion in current simulation")

    args = parser.parse_args()
    main(args)
