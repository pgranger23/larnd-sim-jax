#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import argparse
import yaml
import sys, os
import traceback
import json
import cProfile
import jax

from .fit_params import ParamFitter
from .dataio import TracksDataset, DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# jax.config.update('jax_log_compiles', True)

def make_param_list(config):
    if len(config.param_list) == 1 and os.path.splitext(config.param_list[0])[1] == ".yaml":
        with open(config.param_list[0], 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        for key in config_dict.keys():
            logger.info(f"Setting lr {config_dict[key]} for {key}")
        param_list = config_dict
    else:
        param_list = config.param_list
    return param_list


def main(config):
    if config.cpu_only:
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')

    if config.debug_nans:
        jax.config.update("jax_debug_nans", True)
    else:
        jax.config.update("jax_debug_nans", False)

    if config.non_deterministic:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=false'
    else:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

    logger.info(f"Jax devices: {jax.devices()}")

    if config.print_input:
        logger.info(f"fit label: {config.out_label}")

    if config.lut_file == "" and config.mode == 'lut':
        return 1, 'Error: LUT file is required for mode "lut"'

    iterations = config.iterations
    max_nbatch = config.max_nbatch

    if iterations is not None:
        if max_nbatch is None or iterations < max_nbatch or max_nbatch < 0:
            max_nbatch = iterations

    dataset_sim = TracksDataset(filename=config.input_file_sim, ntrack=config.data_sz, max_nbatch=max_nbatch, seed=config.data_seed, random_ntrack=config.random_ntrack, 
                            track_len_sel=config.track_len_sel, max_abs_costheta_sel=config.max_abs_costheta_sel, min_abs_segz_sel=config.min_abs_segz_sel, track_z_bound=config.track_z_bound, max_batch_len=config.max_batch_len, print_input=config.print_input, electron_sampling_resolution=config.electron_sampling_resolution)

    dataset_target = TracksDataset(filename=config.input_file_tgt, ntrack=config.data_sz, max_nbatch=max_nbatch, seed=config.data_seed, random_ntrack=config.random_ntrack,
                            track_len_sel=config.track_len_sel, max_abs_costheta_sel=config.max_abs_costheta_sel, min_abs_segz_sel=config.min_abs_segz_sel, track_z_bound=config.track_z_bound, max_batch_len=config.max_batch_len, print_input=config.print_input, electron_sampling_resolution=config.electron_sampling_resolution)

    # check if the track in sim and target are consistent
    if len(dataset_sim) != len(dataset_target):
        raise Exception("target and sim inpputs are different in size.")

    batch_sz = config.batch_sz
    if config.max_batch_len is not None and batch_sz != 1:
        logger.warning("Need batch size == 1 for splitting in dx chunks. Setting now...")
        batch_sz = 1

    tracks_dataloader_sim = DataLoader(dataset_sim,
                                  shuffle=config.data_shuffle, 
                                  batch_size=batch_sz)

    tracks_dataloader_target = DataLoader(dataset_target,
                                  shuffle=config.data_shuffle,
                                  batch_size=batch_sz)

    # check if tracks_dataloader_sim and tracks_dataloader_target have the same size
    if len(tracks_dataloader_sim) != len(tracks_dataloader_target):
        raise Exception("target and sim inpputs are different in size.")

    # For readout noise: no_noise overrides if explicitly set to True. Otherwise, turn on noise
    # individually for target and guess
    param_list = make_param_list(config)
    logger.info(f"Param list: {param_list}")
    param_fit = ParamFitter(param_list, dataset_sim.get_track_fields(),
                            detector_props=config.detector_props, pixel_layouts=config.pixel_layouts,
                            load_checkpoint=config.load_checkpoint, lr=config.lr, 
                            readout_noise_target=(not config.no_noise) and (not config.no_noise_target),
                            readout_noise_guess=(not config.no_noise) and (not config.no_noise_guess),
                            out_label=config.out_label, test_name=config.test_name, norm_scheme=config.norm_scheme,
                            max_clip_norm_val=config.max_clip_norm_val, clip_from_range=config.clip_from_range,
                            optimizer_fn=config.optimizer_fn,
                            lr_scheduler=config.lr_scheduler, lr_kw=config.lr_kw,
                            no_adc=config.no_adc, loss_fn=config.loss_fn, loss_fn_kw=config.loss_fn_kw, shift_no_fit=config.shift_no_fit,
                            link_vdrift_eField=config.link_vdrift_eField,
                            set_target_vals=config.set_target_vals, vary_init=config.vary_init, seed_init=config.seed_init, compute_target_hessian=config.compute_target_hessian,
                            config = config, profile_gradient=config.profile_gradient, scan_tgt_nom=config.scan_tgt_nom, epoch_size=len(tracks_dataloader_sim), keep_in_memory=config.keep_in_memory,
                            sim_seed_strategy=config.sim_seed_strategy)
    param_fit.make_target_sim(seed=config.seed, fixed_range=config.fixed_range)

    # jax.profiler.start_trace("/tmp/tensorboard")

    # with cProfile.Profile() as pr:
    param_fit.fit(tracks_dataloader_sim, tracks_dataloader_target, epochs=config.epochs, iterations=iterations, shuffle=config.data_shuffle, save_freq=config.save_freq)
    # pr.dump_stats('prof.prof')
    # jax.profiler.stop_trace()
    return 0, 'Fitting successful'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", dest="param_list", default=[], nargs="+", required=True,
                        help="List of parameters to optimize. See consts_ep.py")
    parser.add_argument("--input_file_sim", dest="input_file_sim",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input sim data file")
    parser.add_argument("--input_file_tgt", dest="input_file_tgt",
                        default="/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5",
                        help="Input target data file")
    parser.add_argument("--detector_props", dest="detector_props",
                        default="larndsim/detector_properties/module0.yaml",
                        help="Path to detector properties YAML file")
    parser.add_argument("--pixel_layouts", dest="pixel_layouts",
                        default="larndsim/pixel_layouts/multi_tile_layout-2.2.16.yaml",
                        help="Path to pixel layouts YAML file")
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", type=str, default=None,
                        help="Path to checkpoint Pickle (pkl) file")
    parser.add_argument("--lr", dest="lr", default=1, type=float,
                        help="Learning rate -- used for all params")
    parser.add_argument("--batch_sz", dest="batch_sz", default=1, type=int,
                        help="Batch size for fitting (tracks).")
    parser.add_argument("--epochs", dest="epochs", default=100, type=int,
                        help="Number of epochs")
    parser.add_argument("--seed", dest="seed", default=2, type=int,
                        help="Random seed for target construction")
    parser.add_argument("--data_seed", dest="data_seed", default=3, type=int,
                        help="Random seed for data picking if not using the whole set")
    parser.add_argument("--seed-init", dest="seed_init", default=30, type=int,
                        help="Random seed for initial condition. Only used if --vary-init is passed.")
    parser.add_argument("--vary-init", dest="vary_init", default=False, action="store_true",
                        help="Randomly sample initial guess (vs starting at nominal value)")
    parser.add_argument("--data_sz", dest="data_sz", default=None, type=int,
                        help="Data size for fitting (number of tracks); input negative values to run on the whole dataset")
    parser.add_argument("--no-noise", dest="no_noise", default=False, action="store_true",
                        help="Flag to turn off readout noise (both target and guess)")
    parser.add_argument("--no-noise-target", dest="no_noise_target", default=False, action="store_true",
                        help="Flag to turn off readout noise (just target, guess has noise)")
    parser.add_argument("--no-noise-guess", dest="no_noise_guess", default=False, action="store_true",
                        help="Flag to turn off readout noise (just guess, target has noise)")
    parser.add_argument("--data_shuffle", dest="data_shuffle", default=False, action="store_true",
                        help="Flag of data shuffling")
    parser.add_argument("--save_freq", dest="save_freq", default=10, type=int,
                        help="Save frequency of the result")
    parser.add_argument("--random_ntrack", dest="random_ntrack", default=False, action="store_true",
                        help="Flag of whether sampling the tracks randomly or sequentially")
    parser.add_argument("--track_len_sel", dest="track_len_sel", default=2., type=float,
                        help="Track selection requirement on track length.")
    parser.add_argument("--max_abs_costheta_sel", dest="max_abs_costheta_sel", default=0.966, type=float,
                        help="Theta is the angle of track wrt to the z axis. Remove tracks which are very colinear with z.")
    parser.add_argument("--min_abs_segz_sel", dest="min_abs_segz_sel", default=15., type=float,
                        help="Remove track segments that are close to the cathode.")
    parser.add_argument("--track_z_bound", dest="track_z_bound", default=28., type=float,
                        help="Set z bound to keep healthy set of tracks")
    parser.add_argument("--out_label", dest="out_label", default="",
                        help="Label for output pkl file")
    parser.add_argument("--test_name", dest="test_name", default="",
                        help="Name of the test")
    parser.add_argument("--fixed_range", dest="fixed_range", default=None, type=float,
                        help="Construct target by sampling in a certain range (fraction of nominal)")
    parser.add_argument("--norm_scheme", dest="norm_scheme", default="divide",
                        help="Normalization scheme to use for params. Right now, divide (by nom) and standard (subtract mean, div by variance)")
    parser.add_argument("--max_clip_norm_val", dest="max_clip_norm_val", default=None, type=float,
                        help="If passed, does gradient clipping (norm)")
    parser.add_argument("--clip_from_range", dest="clip_from_range", default=False, action="store_true",
                        help="Flag to clip the fitted parameters at the nominal ranges")
    parser.add_argument("--optimizer_fn", dest="optimizer_fn", default="Adam",
                        help="Choose optimizer function (here Adam vs SGD")
    parser.add_argument("--lr_scheduler", dest="lr_scheduler", default=None,
                        help="Schedule learning rate, e.g. ExponentialLR")
    parser.add_argument("--lr_kw", dest="lr_kw", default=None, type=json.loads,
                        help="kwargs for learning rate scheduler, as string dict")
    parser.add_argument("--no_adc", dest="no_adc", default=False, action="store_true",
                        help="Don't include ADC in loss (e.g. for vdrift)")
    parser.add_argument("--iterations", dest="iterations", default=None, type=int,
                        help="Number of iterations to run. Overrides epochs.")
    parser.add_argument("--loss_fn", dest="loss_fn", default=None,
                        help="Loss function to use. Named options are SDTW and space_match.")
    parser.add_argument("--loss_fn_kw", dest="loss_fn_kw", default=None, type=json.loads,
                        help="Loss function keyword arguments.")
    parser.add_argument("--max_batch_len", dest="max_batch_len", default=None, type=float,
                        help="Max dx [cm] per batch. If passed, will add tracks to batch until overflow, splitting where needed")
    parser.add_argument("--max_nbatch", dest="max_nbatch", default=None, type=int,
                        help="Upper number of different batches taken from the data, given the max_batch_len. Overrides data_sz.")
    parser.add_argument("--print_input", dest="print_input", default=False, action="store_true",
                        help="print the event and track id per batch.")
    parser.add_argument("--shift-no-fit", dest="shift_no_fit", default=[], nargs="+", 
                        help="Set of params to shift in target sim without fitting them (robustness/separability check).")
    parser.add_argument("--set-target-vals", dest="set_target_vals", default=[], nargs="+", 
                        help="Explicitly set values of target. Syntax is <param1> <val1> <param2> <val2>...")
    parser.add_argument("--link-vdrift-eField", dest="link_vdrift_eField", default=False, action="store_true",
                        help="Link vdrift and eField in fitting")
    parser.add_argument("--profile_gradient", dest="profile_gradient", default=False, action="store_true",
                        help="To profile the gradient and loss instead of making an actual fit.")
    parser.add_argument("--scan_tgt_nom", dest="scan_tgt_nom", default=False, action="store_true",
                        help="Set the gradient and loss scan target to the parameter nominal value, otherwise there will be a target throw.")
    parser.add_argument('--mode', type=str, help='Mode used to simulate the induced current on the pixels', choices=['lut', 'parametrized'], default='lut')
    parser.add_argument('--electron_sampling_resolution', type=float, required=True, help='Electron sampling resolution')
    parser.add_argument('--number_pix_neighbors', type=int, required=True, help='Number of pixel neighbors')
    parser.add_argument('--signal_length', type=int, required=True, help='Signal length')
    parser.add_argument('--lut_file', type=str, required=False, default="", help='Path to the LUT file')
    parser.add_argument('--keep_in_memory', default=False, action="store_true", help='Keep the expected output of each batch in memory')
    parser.add_argument('--compute_target_hessian', default=False, action="store_true", help='Computes the Hessian at the target for every batch')
    parser.add_argument('--non_deterministic', default=False, action="store_true", help='Make the computation slightly non-deterministic for faster computation')
    parser.add_argument('--debug_nans', default=False, action="store_true", help='Debug NaNs (much slower)')
    parser.add_argument('--cpu_only', default=False, action="store_true", help='Run on CPU only')
    parser.add_argument('--sim_seed_strategy', default="different", type=str, choices=['same', 'different', 'random', 'constant'],
                        help='Strategy to choose the seed for the simulation (the seed for target is the batch id). It can be "same" (same for target and sim), "different" (different for target and sim but constant across epochs), "random" (different between target and sim and random across epochs), "constant" (the seed is constant across batches).')

    try:
        args = parser.parse_args()
        retval, status_message = main(args)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Fitting failed.'

    logger.info(status_message)
    exit(retval)
