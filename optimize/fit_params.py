import os, sys
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .utils import get_id_map, all_sim, embed_adc_list
from .ranges import ranges
from larndsim.sim_jax import simulate, simulate_parametrized, get_size_history
from larndsim.losses_jax import params_loss, params_loss_parametrized, mse_adc, mse_time, mse_time_adc, chamfer_3d, sdtw_adc, sdtw_time, sdtw_time_adc
from larndsim.consts_jax import build_params_class, load_detector_properties
from larndsim.softdtw_jax import SoftDTW
from jax.flatten_util import ravel_pytree
import logging
import torch
import optax
import jax
import jax.numpy as jnp
from time import time
from jax import value_and_grad

from tqdm import tqdm

from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def normalize_param(param_val, param_name, scheme="divide", undo_norm=False):
    if scheme == "divide":
        if undo_norm:
            out_val = param_val * ranges[param_name]['nom']
        else:
            out_val = param_val / ranges[param_name]['nom']

        return out_val
    elif scheme == "standard":
        sigma = (ranges[param_name]['up'] - ranges[param_name]['down']) / 2.

        if undo_norm:
            out_val = param_val*sigma**2 + ranges[param_name]['nom']
        else:
            out_val = (param_val - ranges[param_name]['nom']) / sigma**2
            
        return out_val
    elif scheme == "none":
        return param_val
    else:
        raise ValueError(f"No normalization method called {scheme}")
    
def extract_relevant_params(params, relevant):
    return {par: getattr(params, par) for par in relevant}

def update_params(params, update):
    return params.replace(**{key: getattr(params, key) + val for key, val in update.items()})

def format_hessian(hess):
    flatten_hessian, _ = ravel_pytree(hess)
    return flatten_hessian.tolist()

def remove_noise_from_params(params):
    noise_params = ('RESET_NOISE_CHARGE', 'UNCORRELATED_NOISE_CHARGE')
    return params.replace(**{key: 0. for key in noise_params})

class ParamFitter:
    def __init__(self, relevant_params, track_fields,
                 detector_props, pixel_layouts, load_checkpoint = None,
                 lr=None, optimizer=None, lr_scheduler=None, lr_kw=None, 
                 loss_fn=None, loss_fn_kw=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", norm_scheme="divide", max_clip_norm_val=None, optimizer_fn="Adam",
                 no_adc=False, shift_no_fit=[], link_vdrift_eField=False,
                 set_target_vals=[], vary_init=False, seed_init=30, profile_gradient = False, epoch_size=1, keep_in_memory=False,
                 compute_target_hessian=False,
                 config = {}):
        if optimizer_fn == "Adam":
            self.optimizer_fn = optax.adam
        elif optimizer_fn == "SGD":
            self.optimizer_fn = optax.sgd
        else:
            raise NotImplementedError("Only SGD and Adam supported")
        self.optimizer_fn_name = optimizer_fn

        self.no_adc = no_adc
        self.shift_no_fit = shift_no_fit
        self.link_vdrift_eField = link_vdrift_eField

        self.out_label = out_label
        self.norm_scheme = norm_scheme
        self.max_clip_norm_val = max_clip_norm_val
        if self.max_clip_norm_val is not None:
            logger.info(f"Will clip gradient norm at {self.max_clip_norm_val}")

        self.profile_gradient = profile_gradient

        self.compute_target_hessian = compute_target_hessian

        self.current_mode = config.mode
        self.electron_sampling_resolution = config.electron_sampling_resolution
        self.number_pix_neighbors = config.number_pix_neighbors
        self.signal_length = config.signal_length

        if self.current_mode == 'lut':
            self.lut_file = config.lut_file
            self.load_lut()

        self.track_fields = track_fields
        if type(relevant_params) == dict:
            self.relevant_params_list = list(relevant_params.keys())
            self.relevant_params_dict = relevant_params
        elif type(relevant_params) == list:
            self.relevant_params_list = relevant_params
            self.relevant_params_dict = None
        else:
            raise TypeError("relevant_params must be list of param names or list of dicts with learning rates")

        is_continue = False
        if load_checkpoint is not None:
            history = pickle.load(open(load_checkpoint, "rb"))
            is_continue = True

        self.target_val_dict = None
        if len(set_target_vals) > 0:
            if len(set_target_vals) % 2 != 0:
                raise ValueError("Incorrect format for set_target_vals!")
            
            self.target_val_dict = {}
            for i_val in range(len(set_target_vals)//2):
                param_name = set_target_vals[2*i_val]
                param_val = set_target_vals[2*i_val+1]
                self.target_val_dict[param_name] = float(param_val)

        # Normalize parameters to init at 1, or random, or set to checkpointed values
        
        Params = build_params_class(self.relevant_params_list)
        ref_params = load_detector_properties(Params, detector_props, pixel_layouts)
        ref_params = ref_params.replace(
            electron_sampling_resolution=self.electron_sampling_resolution,
            number_pix_neighbors=self.number_pix_neighbors,
            signal_length=self.signal_length,
            time_window=self.signal_length)

        initial_params = {}

        for param in self.relevant_params_list:
            if is_continue:
                initial_params[param] = history[param][-1]
            else:
                if vary_init:
                    logger.info("Running with random initial guess")
                    init_val = np.random.uniform(low=ranges[param]['down'], 
                                                  high=ranges[param]['up'])
                    initial_params[param] = init_val

        self.current_params = ref_params.replace(**initial_params)

        #Only do it now to not inpact current_params
        if not readout_noise_target:
            logger.info("Not simulating electronics noise for target")
            ref_params = remove_noise_from_params(ref_params)
        if not readout_noise_guess:
            logger.info("Not simulating electronics noise for guesses")
            self.current_params = remove_noise_from_params(self.current_params)

        self.ref_params = ref_params

        self.params_normalization = ref_params.replace(**{key: getattr(self.current_params, key) if getattr(self.current_params, key) != 0. else 1. for key in self.relevant_params_list})
        self.norm_params = ref_params.replace(**{key: 1. if getattr(self.current_params, key) != 0. else 0. for key in self.relevant_params_list})

        self.learning_rates = {}

        if lr_scheduler is not None and lr_kw is not None:
            lr_scheduler_fn = getattr(optax, lr_scheduler)
            logger.info(f"Using learning rate scheduler {lr_scheduler}")
        else:
            lr_scheduler_fn = optax.constant_schedule
            lr_kw = {}

        if self.relevant_params_dict is None:
            if lr is None:
                raise ValueError("Need to specify lr for params")
            else:
                self.learning_rates = {par: lr_scheduler_fn(lr, transition_steps=epoch_size, **lr_kw) for par in self.relevant_params_list}
        else:
            self.learning_rates = {key: lr_scheduler_fn(float(value), transition_steps=epoch_size, **lr_kw) for key, value in self.relevant_params_dict.items()}
        
        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr

        if optimizer is None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.max_clip_norm_val),
                optax.multi_transform({key: self.optimizer_fn(value) for key, value in self.learning_rates.items()},
                            {key: key for key in self.relevant_params_list})
            )
        else:
            raise ValueError("Passing directly optimizer is not supported")
        
        self.opt_state = self.optimizer.init(extract_relevant_params(self.norm_params, self.relevant_params_list))

        loss_functions = {
            "mse_adc": (mse_adc, {}),
            "mse_time": (mse_time, {}),
            "mse_time_adc": (mse_time_adc, {'alpha': 0.5}),
            "chamfer_3d": (chamfer_3d, {}),
            "sdtw_adc": (sdtw_adc, {'gamma': 1.}),
            "sdtw_time": (sdtw_time, {'gamma': 1.}),
            "sdtw_time_adc": (sdtw_time_adc, {'gamma': 1., 'alpha': 0.5})
        }

        # Set up loss function -- can pass in directly, or choose a named one
        if loss_fn is None or loss_fn == "space_match":
            loss_fn = "mse_adc"
        elif loss_fn == "SDTW":
            loss_fn = "sdtw_adc"
    
        if isinstance(loss_fn, str):
            if loss_fn not in loss_functions:
                raise ValueError(f"Loss function {loss_fn} not supported")
            self.loss_fn, self.loss_fn_kw = loss_functions[loss_fn]
            if loss_fn_kw is not None:
                self.loss_fn_kw.update(loss_fn_kw) #Adding the user defined kwargs
            logger.info(f"Using loss function {loss_fn} with kwargs {self.loss_fn_kw}")

        else:
            self.loss_fn = loss_fn
            self.loss_fn_kw = {}
            logger.info("Using custom loss function")

        if loss_fn in ['sdtw_adc', 'sdtw_time', 'sdtw_time_adc']: #Need to setup the sdtw class for the loss function
            self.loss_fn_kw['dstw'] = SoftDTW(**self.loss_fn_kw)

        if is_continue:
            self.training_history = history
        else:
            self.training_history = {}
            for param in self.relevant_params_list:
                self.training_history[param] = []
                self.training_history[param+"_grad"] = []
                self.training_history[param+"_iter"] = []

                self.training_history[param + '_target'] = []
                self.training_history[param + '_init'] = [getattr(self.current_params, param)]
                # self.training_history[param + '_lr'] = [lr_dict[param]]
            self.training_history['step_time'] = []
            for param in self.shift_no_fit:
                self.training_history[param + '_target'] = []

            self.training_history['losses'] = []
            self.training_history['losses_iter'] = []
            self.training_history['norm_scheme'] = self.norm_scheme
        #     self.training_history['fit_diffs'] = self.fit_diffs
            self.training_history['optimizer_fn_name'] = self.optimizer_fn_name
            if self.compute_target_hessian:
                self.training_history['hessian'] = []
            self.training_history['size_history'] = []
            self.training_history['memory'] = []

        self.training_history['config'] = config

        self.keep_in_memory = keep_in_memory
        if keep_in_memory:
            
            self.targets = {}

    def load_lut(self):
        response = np.load(self.lut_file)
        extended_response = np.zeros((50, 50, 1891))
        extended_response[:45, :45, :] = response
        response = extended_response
        baseline = np.sum(response[:, :, :-self.signal_length+1], axis=-1)
        response = np.concatenate([baseline[..., None], response[..., -self.signal_length+1:]], axis=-1)
        self.response = response

    def clip_values(self, mini=0.01, maxi=100):
        cur_norm_values = extract_relevant_params(self.norm_params, self.relevant_params_list)
        cur_norm_values = {key: jnp.array(max(mini, min(maxi, val))) for key, val in cur_norm_values.items()}
        self.norm_params = self.norm_params.replace(**cur_norm_values)

    def clip_values_from_range(self):
        self.norm_params = self.norm_params.replace(
            **{key: jnp.array(max(ranges[key]['down']/getattr(self.params_normalization, key), min(ranges[key]['up']/getattr(self.params_normalization, key), getattr(self.norm_params, key)))) for key in self.relevant_params_list}
            )

    def update_params(self):
        self.current_params = self.norm_params.replace(**{key: getattr(self.norm_params, key)*getattr(self.params_normalization, key) for key in self.relevant_params_list})

    def make_target_sim(self, seed=2, fixed_range=None):
        np.random.seed(seed)
        logger.info("Constructing target param simulation")

        self.target_params = {}

        if self.profile_gradient:
            logger.info("Using the fitter in a gradient profile mode. Setting targets to nominal values")
            for param in self.relevant_params_list:
                self.target_params[param] = ranges[param]['nom']
        elif self.target_val_dict is not None:
            if set(self.relevant_params_list + self.shift_no_fit) != set(self.target_val_dict.keys()):
                logger.debug(set(self.relevant_params_list + self.shift_no_fit))
                logger.debug(set(self.target_val_dict.keys()))
                raise ValueError("Must specify all parameters if explicitly setting target")

            logger.info("Explicitly setting targets:")
            for param in self.target_val_dict.keys():
                param_val = self.target_val_dict[param]
                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        else:
            for param in self.relevant_params_list + self.shift_no_fit:
                if fixed_range is not None:
                    param_val = np.random.uniform(low=ranges[param]['nom']*(1.-fixed_range), 
                                                high=ranges[param]['nom']*(1.+fixed_range))
                else:
                    param_val = np.random.uniform(low=ranges[param]['down'], 
                                                high=ranges[param]['up'])

                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        self.target_params = self.ref_params.replace(**self.target_params)

    
    def fit(self, dataloader, epochs=300, iterations=None, shuffle=False, 
            save_freq=10, print_freq=1):
        # If explicit number of iterations, scale epochs accordingly
        if iterations is not None:
            epochs = iterations // len(dataloader) + 1

        # make a folder for the pixel target
        if os.path.exists('target_' + self.out_label):
            shutil.rmtree('target_' + self.out_label, ignore_errors=True)
        os.makedirs('target_' + self.out_label)

        # make a folder for the fit result
        if not os.path.exists('fit_result'):
            os.makedirs('fit_result')

        # Include initial value in training history (if haven't loaded a checkpoint)
        for param in self.relevant_params_list:
            if len(self.training_history[param]) == 0:
                self.training_history[param].append(getattr(self.current_params, param))
                self.training_history[param+'_target'].append(getattr(self.target_params, param))
            if len(self.training_history[param+"_iter"]) == 0:
                self.training_history[param+"_iter"].append(getattr(self.current_params, param))
        for param in self.shift_no_fit:
            if len(self.training_history[param+'_target']) == 0:
                self.training_history[param+'_target'].append(getattr(self.target_params, param))

        if iterations is not None:
            pbar_total = iterations
        else:
            pbar_total = len(dataloader) * epochs

        if self.profile_gradient:
            logger.info("Using the fitter in a gradient profile mode. The sampling will follow a regular grid.")
            nb_var_params = len(self.relevant_params_list)
            logger.info(f"{nb_var_params} parameters are to be scanned.")
            nb_steps = int(epochs**(1./nb_var_params))
            logger.info(f"Each parameter will be scanned with {nb_steps} steps")
            grids_1d = [list(range(nb_steps))]*nb_var_params
            steps_grids = list(np.meshgrid(*grids_1d))

            for i, param in enumerate(self.relevant_params_list):
                lower = ranges[param]['down']
                upper = ranges[param]['up']
                param_step = (upper - lower)/(nb_steps - 1)
                steps_grids[i] = steps_grids[i].astype(float)
                steps_grids[i] *= param_step
                steps_grids[i] += lower
                steps_grids[i] = steps_grids[i].ravel()

        # The training loop
        total_iter = 0
        with tqdm(total=pbar_total) as pbar:
            for epoch in range(epochs):
                if epoch == 2: libcudart.cudaProfilerStart()
                # Losses for each batch -- used to compute epoch loss
                losses_batch=[]

                if self.profile_gradient:
                    new_param_values = {}
                    for i, param in enumerate(self.relevant_params_list):
                        new_param_values[param] = steps_grids[i][epoch]
                    logger.info(f"Stepping parameter values: {new_param_values}")
                    self.current_params = self.current_params.replace(**new_param_values)
                for i, selected_tracks_bt_torch in enumerate(dataloader):
                    start_time = time()
                    
                    #Convert torch tracks to jax
                    selected_tracks_bt_torch = torch.flatten(selected_tracks_bt_torch, start_dim=0, end_dim=1)
                    selected_tracks = jax.device_put(selected_tracks_bt_torch.numpy())

                    #Simulate the output for the whole batch
                    loss_ev = []

                    #Simulating the reference during the first epoch
                    fname = 'target_' + self.out_label + '/batch' + str(i) + '_target.npz'
                    if epoch == 0:
                        if self.current_mode == 'lut':
                            ref_adcs, ref_unique_pixels, ref_ticks = simulate(self.target_params, self.response, selected_tracks, self.track_fields, i) #Setting a different random seed for each target
                        else:
                            ref_adcs, ref_unique_pixels, ref_ticks = simulate_parametrized(self.target_params, selected_tracks, self.track_fields, i) #Setting a different random seed for each target

                        if self.compute_target_hessian:
                            if self.current_mode == 'lut':
                                hess, aux = jax.jacfwd(jax.jacrev(params_loss, (0), has_aux=True), has_aux=True)(self.target_params, self.response, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0, loss_fn=self.loss_fn, **self.loss_fn_kw)
                            else:
                                ref_adcs, ref_unique_pixels, ref_ticks = simulate_parametrized(self.target_params, selected_tracks, self.track_fields)
                                hess, aux = jax.jacfwd(jax.jacrev(params_loss_parametrized, (0), has_aux=True), has_aux=True)(self.target_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=0, loss_fn=self.loss_fn, **self.loss_fn_kw)
                            self.training_history['hessian'].append(format_hessian(hess))

                        # embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                        #Saving the target for the batch
                        #TODO: See if we have to do this for each event
                        
                        with open(fname, 'wb') as f:
                            jnp.savez(f, adcs=ref_adcs, unique_pixels=ref_unique_pixels, ticks=ref_ticks)
                            if self.keep_in_memory:
                                self.targets[i] = (ref_adcs, ref_unique_pixels, ref_ticks)

                    else:
                        #Loading the target
                        if self.keep_in_memory:
                            ref_adcs, ref_unique_pixels, ref_ticks = self.targets[i]
                        else:
                            with open(fname, 'rb') as f:
                                loaded = jnp.load(f)
                                ref_adcs = loaded['adcs']
                                ref_unique_pixels = loaded['unique_pixels']
                                ref_ticks = loaded['ticks']

                    # Simulate and get output
                    if self.current_mode == 'lut':
                        (loss_val, aux), grads = value_and_grad(params_loss, (0), has_aux = True)(self.current_params, self.response, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=i, loss_fn=self.loss_fn, **self.loss_fn_kw)
                    else:
                        (loss_val, aux), grads = value_and_grad(params_loss_parametrized, (0), has_aux = True)(self.current_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks, self.track_fields, rngkey=i, loss_fn=self.loss_fn, **self.loss_fn_kw)
                    scaled_grads = {key: getattr(grads, key)*getattr(self.params_normalization, key) for key in self.relevant_params_list}
                    if not self.profile_gradient:
                        leaves = jax.tree_util.tree_leaves(grads)
                        hasNaN = any(jnp.isnan(leaf).any() for leaf in leaves)
                        if hasNaN:
                            logger.warning("Got NaN gradients! Skipping update for this batch")
                        else:
                            updates, self.opt_state = self.optimizer.update(scaled_grads, self.opt_state)
                            # self.current_params = update_params(self.norm_params, updates)
                            self.norm_params = update_params(self.norm_params, updates)
                            #Clipping param values
                            self.clip_values_from_range()
                            self.update_params()

                    stop_time = time()

                    for param in self.relevant_params_list:
                        self.training_history[param+"_grad"].append(scaled_grads[param].item())
                    self.training_history['step_time'].append(stop_time - start_time)

                    self.training_history['losses_iter'].append(loss_val.item())
                    for param in self.relevant_params_list:
                        self.training_history[param+"_iter"].append(getattr(self.current_params, param).item())

                    self.training_history['size_history'].append(get_size_history())
                    self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

                    if iterations is not None:
                        if total_iter % print_freq == 0:
                            for param in self.relevant_params_list:
                                logger.info(f"{param} {getattr(self.current_params,param)} {scaled_grads[param]}")
                            
                        if total_iter % save_freq == 0:
                            with open(f'fit_result/history_{param}_iter{total_iter}_{self.out_label}.pkl', "wb") as f_history:
                                pickle.dump(self.training_history, f_history)

                            if os.path.exists(f'fit_result/history_{param}_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                os.remove(f'fit_result/history_{param}_iter{total_iter-save_freq}_{self.out_label}.pkl') 

                    total_iter += 1
                    pbar.update(1)
                    
                    if iterations is not None:
                        if total_iter >= iterations:
                            break
            libcudart.cudaProfilerStop()