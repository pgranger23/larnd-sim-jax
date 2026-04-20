#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import sys
import argparse
import traceback

# Enforce CPU if --gpu is not specified
if '--gpu' not in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import simulate_stochastic, simulate_parametrized, simulate_wfs
from larndsim.losses_jax import get_hits_space_coords
from larndsim.detsim_jax import validate_event_ids_for_packing, validate_local_event_ids
from pprint import pprint
import numpy as np
import h5py
import jax
from tqdm import tqdm
from numpy.lib import recfunctions as rfn
from larndsim.sim_jax import pad_size
from .dataio import TracksDataset
import jax.numpy as jnp
from larndsim.fee_jax import digitize
from larndsim.losses_jax import adc2charge

# from ctypes import cdll
# libcudart = cdll.LoadLibrary('libcudart.so')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# jax.config.update('jax_log_compiles', True)


def main(config):
    output_filename = config.output_file
    if not config.out_np:
        if not output_filename.endswith('.h5'):
            output_filename += '.h5'

    if os.path.isfile(output_filename):
        os.remove(output_filename)
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

    if config.mode == 'lut':
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

    dataset = TracksDataset(
        filename=config.input_file,
        nevents=config.n_events,
        max_nbatch=None,
        swap_xz=True,
        random_nevents=False,
        data_seed=config.seed if config.seed is not None else 42,
        max_batch_len=config.max_batch_len,
        print_input=False,
        chopped=config.chop,
        pad=False,
        electron_sampling_resolution=config.electron_sampling_resolution,
        live_selection=False,
    )
    fields = dataset.get_track_fields()
    evt_col = fields.index("eventID")

    if config.out_np:
        l_adc, l_Q, l_ticks, l_eventID, l_pix_x, l_pix_y, l_pix_z, l_hit_prob = [], [], [], [], [], [], [], []

    # libcudart.cudaProfilerStart()
    for ibatch in tqdm(range(len(dataset)), desc="Loading tracks", total=len(dataset)):
        batch = dataset[ibatch]
        size = pad_size(batch.shape[0], "batch_size", 0.5)
        batch = dataset.pad_batch(batch, size, ibatch)

        global_event_ids = dataset.get_batch_global_event_ids(ibatch)

        event_ids = batch[:, evt_col].astype(np.int64)

        # Validate local event ID namespace before overflow checks
        validate_local_event_ids(event_ids, context=f"simulate batch {ibatch}")
        validate_event_ids_for_packing(ref_params, event_ids, kind="pixel", context=f"simulate batch {ibatch}")
        validate_event_ids_for_packing(ref_params, event_ids, kind="bin", context=f"simulate batch {ibatch}")

        # Get mapping from local event IDs back to global IDs
        local_to_global = {i: int(gid) for i, gid in enumerate(global_event_ids)}

        tracks = jax.device_put(batch)
        rngseed = ibatch if config.seed is None else config.seed
        if config.mode == 'lut':
            wfs, unique_pixels = simulate_wfs(ref_params, response, tracks, fields)
            adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, hit_pixels = simulate_stochastic(ref_params, wfs, unique_pixels, rngseed=rngseed)
            print("Unique pixels:", unique_pixels)
            print("Hit pixels:", hit_pixels)
        else:
            
            adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, hit_pixels = simulate_parametrized(ref_params, tracks, fields, rngseed=rngseed)
            wfs = None
        if config.jac:
            jac_res = jax.jacfwd(sim_wrapper)(ref_params, tracks)

        adc_lowest = digitize(ref_params, ref_params.DISCRIMINATION_THRESHOLD)
        adcs_clean = adcs - adc_lowest
        mask = (adcs_clean.flatten() != 0) & (event.flatten() != -1)
        Q = adc2charge(adcs.flatten()[mask], ref_params)

        if not config.out_np:
            with h5py.File(output_filename, 'a') as f:
                batch_group = f.create_group(f"batch_{ibatch}")

                # Split output per-event using local→global ID mapping
                for local_event_id in np.unique(event[mask]).astype(int):
                    if local_event_id < 0:  # Skip padding
                        continue

                    global_event_id = local_to_global.get(local_event_id, local_event_id)
                    event_mask = (event.flatten()[mask] == local_event_id)

                    # Create per-event subgroup
                    event_group = batch_group.create_group(f"event_{global_event_id}")
                    event_group.create_dataset('adc_clean', data=adcs_clean.flatten()[mask][event_mask])
                    event_group.create_dataset('adc', data=adcs.flatten()[mask][event_mask])
                    event_group.create_dataset('Q', data=Q[event_mask])
                    event_group.create_dataset('pixels', data=hit_pixels[mask][event_mask])
                    event_group.create_dataset('ticks', data=ticks.flatten()[mask][event_mask])
                    event_group.create_dataset('eventID', data=np.full(event_mask.sum(), global_event_id, dtype=np.int64))
                    event_group.create_dataset('pix_x', data=pixel_x[mask][event_mask])
                    event_group.create_dataset('pix_y', data=pixel_y[mask][event_mask])
                    event_group.create_dataset('pix_z', data=pixel_z.flatten()[mask][event_mask])

                    if config.jac:
                        for par in pars:
                            event_group.create_dataset(f'jac_{par}_adc', data=getattr(jac_res, par)[:, :, 0].flatten()[mask][event_mask])
                            event_group.create_dataset(f'jac_{par}_ticks', data=getattr(jac_res, par)[:, :, 1].flatten()[mask][event_mask])

                    if config.save_wfs:
                        event_group.create_dataset('wfs', data=wfs)


        else:
            l_adc.append(adcs.flatten()[mask])
            l_Q.append(Q)
            l_ticks.append(ticks.flatten()[mask])
            l_eventID.append(event.flatten()[mask])
            l_pix_x.append(pixel_x.flatten()[mask])
            l_pix_y.append(pixel_y.flatten()[mask])
            l_pix_z.append(pixel_z.flatten()[mask])
            l_hit_prob.append(hit_prob.flatten()[mask])

    if config.out_np:
        jnp.savez(config.output_file, adcs=np.concatenate(l_adc), Q=np.concatenate(l_Q), x=np.concatenate(l_pix_x), y=np.concatenate(l_pix_y), z=np.concatenate(l_pix_z), ticks=np.concatenate(l_ticks), hit_prob=np.concatenate(l_hit_prob), event_id=np.concatenate(l_eventID))

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
    parser.add_argument('--electron_sampling_resolution', type=float, required=True, default=0.1, help='Electron sampling resolution')
    parser.add_argument('--number_pix_neighbors', type=int, required=True, help='Number of pixel neighbors')
    parser.add_argument('--signal_length', type=int, required=True, help='Signal length')
    parser.add_argument('--lut_file', type=str, required=False, default="", help='Path to the LUT file')
    parser.add_argument('--noise', action='store_true', help='Add noise to the simulation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--diffusion_in_current_sim', action='store_true', help='Use diffusion in current simulation')
    parser.add_argument('--batch_size', type=float, default=500, help='Batch size for simulation')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for simulation')
    parser.add_argument('--jac', action='store_true', help='Compute jacobian')
    parser.add_argument('--mc_diff', action='store_true', help='Use Monte Carlo diffusion')
    parser.add_argument('--save_wfs', action='store_true', help='Save waveforms')
    parser.add_argument('--n_events', type=int, default=-1, help='Number of events to be simulated')
    parser.add_argument('--out_np', action='store_true', default=False, help='store target-like output in npz')
    parser.add_argument('--max_batch_len', type=float, default=50., help='Maximum trajectory length budget used while preparing tracks')
    parser.add_argument('--chop', action='store_true', default=False, help='Enable segment chopping in data loading (default: disabled)')

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
