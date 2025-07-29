#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import argparse
import sys
import traceback
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import prepare_tracks, simulate_new, simulate_parametrized, id2pixel, get_pixel_coordinates
from larndsim.losses_jax import get_hits_space_coords
from pprint import pprint
import numpy as np
import h5py
import jax
from tqdm import tqdm
from numpy.lib import recfunctions as rfn
from larndsim.sim_jax import pad_size
from .dataio import chop_tracks, jax_from_structured
import jax.numpy as jnp
from larndsim.fee_jax import digitize
from larndsim.losses_jax import adc2charge

# from ctypes import cdll
# libcudart = cdll.LoadLibrary('libcudart.so')


def load_events_as_batch(filename, sampling_resolution, swap_xz=True, n_events=-1):
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
    
    tracks = rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32)

    logger.info(f"Loaded {len(first_indices)} events from {filename} with {tracks.shape[0]} tracks")
    
    return [chop_tracks(tracks[first_indices[i]:last_indices[i] + 1, :], track_fields, sampling_resolution) for i in range(len(first_indices))], track_fields


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# jax.config.update('jax_log_compiles', True)


def main(config):
    if os.path.isfile(config.output_file):
        os.remove(config.output_file)
    if config.lut_file == "" and config.mode == 'lut':
        return 1, 'Error: LUT file is required for mode "lut"'
    
    if not config.gpu:
        jax.config.update('jax_platform_name', 'cpu')

    pars = []
    if config.jac:
        def sim_wrapper(params, tracks):
            adcs, pixel_x, pixel_y, pixel_z, ticks, event, unique_pixels, pix_renumbering, electrons, _ = simulate_parametrized(params, tracks, fields, rngseed=config.seed)
            return jnp.stack([adcs, ticks], axis=-1)
        pars = ['Ab', 'kb', 'eField', 'long_diff', 'tran_diff', 'lifetime', 'shift_z']
    Params = build_params_class(pars)
    ref_params = load_detector_properties(Params, config.detector_props, config.pixel_layouts)

    if args.mode == 'lut':
        response, ref_params = load_lut(config.lut_file, ref_params)
    
    
    params_to_apply = [
        'diffusion_in_current_sim',
        'mc_diff',
        'electron_sampling_resolution',
        'number_pix_neighbors',
        'signal_length',
    ]


    ref_params = ref_params.replace(**{k: getattr(config, k) for k in params_to_apply}, time_window=config.signal_length)
    
    if not config.noise:
        ref_params = ref_params.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)


    dataset, fields = load_events_as_batch(config.input_file, config.electron_sampling_resolution, swap_xz=True, n_events=config.n_events)

    if config.out_np:
        l_adc, l_Q, l_ticks, l_eventID, l_pix_x, l_pix_y, l_pix_z = [], [], [], [], [], [], []

    # libcudart.cudaProfilerStart()
    for ibatch, batch in tqdm(enumerate(dataset), desc="Loading tracks", total=len(dataset)):
        size = batch.shape[0]
        size = pad_size(size, "batch_size", 0.5)
        batch = np.pad(batch, ((0, size - batch.shape[0]), (0, 0)), mode='constant', constant_values=0)
        tracks = jax.device_put(batch)

        if args.mode == 'lut':
            adcs, pixel_x, pixel_y, pixel_z, ticks, event, unique_pixels = simulate_new(ref_params, response, tracks, fields, rngseed=config.seed)
        else:
            adcs, pixel_x, pixel_y, pixel_z, ticks, event, unique_pixels, pix_renumbering, electrons, wfs = simulate_parametrized(ref_params, tracks, fields, rngseed=config.seed)
        if config.jac:
            jac_res = jax.jacfwd(sim_wrapper)(ref_params, tracks)

        adc_lowest = digitize(ref_params, ref_params.DISCRIMINATION_THRESHOLD)
        adcs_clean = adcs - adc_lowest
        mask = (adcs_clean.flatten() != 0) & (jnp.repeat(event, 10) != -1)
        Q = adc2charge(adcs.flatten()[mask], ref_params)

        if not config.out_np:
            with h5py.File(config.output_file, 'a') as f:
                group = f.create_group(f"batch_{ibatch}")
                group.create_dataset('adc_clean', data=adcs_clean.flatten()[mask])
                group.create_dataset('adc', data=adcs.flatten()[mask])
                group.create_dataset('Q', data=Q)
                group.create_dataset('pixels', data=jnp.repeat(unique_pixels, 10)[mask])
                group.create_dataset('ticks', data=ticks.flatten()[mask])
                group.create_dataset('eventID', data=jnp.repeat(event, 10)[mask])
                group.create_dataset('pix_x', data=jnp.repeat(pixel_x, 10)[mask])
                group.create_dataset('pix_y', data=jnp.repeat(pixel_y, 10)[mask])
                group.create_dataset('pix_z', data=pixel_z.flatten()[mask])

                if config.save_wfs:
                    group.create_dataset('wfs', data=jnp.repeat(wfs, 10, axis=0)[mask, :])
                if config.jac:
                    for par in pars:
                        group.create_dataset(f'jac_{par}_adc', data=getattr(jac_res, par)[:, :, 0].flatten()[mask])
                        group.create_dataset(f'jac_{par}_ticks', data=getattr(jac_res, par)[:, :, 1].flatten()[mask])

        else:
            l_adc.append(adcs.flatten()[mask])
            l_Q.append(Q)
            l_ticks.append(ticks.flatten()[mask])
            l_eventID.append(jnp.repeat(event, 10)[mask])
            l_pix_x.append(jnp.repeat(pixel_x, 10)[mask])
            l_pix_y.append(jnp.repeat(pixel_y, 10)[mask])
            l_pix_z.append(pixel_z.flatten()[mask])

    if config.out_np:
        jnp.savez(config.output_file, adcs=np.concatenate(l_adc), Q=np.concatenate(l_Q), x=np.concatenate(l_pix_x), y=np.concatenate(l_pix_y), z=np.concatenate(l_pix_z), ticks=np.concatenate(l_ticks), event_id=np.concatenate(l_eventID))

    # libcudart.cudaProfilerStop()
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
                        default="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument('--mode', type=str, help='Mode used to simulate the induced current on the pixels', choices=['lut', 'parametrized'], default='lut')
    parser.add_argument('--electron_sampling_resolution', type=float, required=True, help='Electron sampling resolution')
    parser.add_argument('--number_pix_neighbors', type=int, required=True, help='Number of pixel neighbors')
    parser.add_argument('--signal_length', type=int, required=True, help='Signal length')
    parser.add_argument('--lut_file', type=str, required=False, default="", help='Path to the LUT file')
    parser.add_argument('--noise', action='store_true', help='Add noise to the simulation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--diffusion_in_current_sim', action='store_true', help='Use diffusion in current simulation')
    parser.add_argument('--batch_size', type=float, default=500, help='Batch size for simulation')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for simulation')
    parser.add_argument('--jac', action='store_true', help='Compute jacobian')
    parser.add_argument('--mc_diff', action='store_true', help='Use Monte Carlo diffusion')
    parser.add_argument('--save_wfs', action='store_true', help='Save waveforms')
    parser.add_argument('--n_events', type=int, default=-1, help='Number of events to be simulated')
    parser.add_argument('--out_np', action='store_true', default=False, help='store target-like output in npz')

    try:
        args = parser.parse_args()
        if args.save_wfs and args.jac:
            raise ValueError("Cannot save waveforms and compute jacobian at the same time. Please choose one of the two options.")

        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    logger.info(status_message)
    exit(retval)
