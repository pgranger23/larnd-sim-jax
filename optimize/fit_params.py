import os, sys
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .ranges import ranges
from larndsim.sim_jax import simulate, simulate_parametrized, get_size_history
from larndsim.losses_jax import params_loss, params_loss_parametrized, mse_adc, mse_time, mse_time_adc, chamfer_3d, sdtw_adc, sdtw_time, sdtw_time_adc
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.softdtw_jax import SoftDTW
from jax.flatten_util import ravel_pytree
import logging
import optax
import jax
import jax.numpy as jnp
from time import time
from jax import value_and_grad, grad
import iminuit

from tqdm import tqdm

from ctypes import cdll
# libcudart = cdll.LoadLibrary('libcudart.so')

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

def format_hessian(hess):
    flatten_hessian, _ = ravel_pytree(hess)
    return flatten_hessian.tolist()

def remove_noise_from_params(params):
    noise_params = ('RESET_NOISE_CHARGE', 'UNCORRELATED_NOISE_CHARGE')
    return params.replace(**{key: 0. for key in noise_params})

class ParamFitter:
    def __init__(self, relevant_params, track_fields,
                 detector_props, pixel_layouts,
                 loss_fn=None, loss_fn_kw=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", test_name="this_test",
                 shift_no_fit=[], set_target_vals=[], vary_init=False, keep_in_memory=False,
                 compute_target_hessian=False, sim_seed_strategy="different",
                 target_seed=0, target_fixed_range=None,
                 adc_norm=10, match_z=True,
                 diffusion_in_current_sim=False,
                 mc_diff = False,
                 read_target=False,
                 config = {}):

        self.read_target = read_target
        self.shift_no_fit = shift_no_fit
        self.detector_props = detector_props
        self.pixel_layouts = pixel_layouts
        self.readout_noise_target = readout_noise_target
        self.readout_noise_guess = readout_noise_guess
        self.vary_init = vary_init
        self.target_seed = target_seed
        self.target_fixed_range = target_fixed_range
        self.diffusion_in_current_sim = diffusion_in_current_sim
        self.mc_diff = mc_diff

        self.out_label = out_label
        self.test_name = test_name

        self.compute_target_hessian = compute_target_hessian

        self.current_mode = config.mode
        self.electron_sampling_resolution = config.electron_sampling_resolution
        self.number_pix_neighbors = config.number_pix_neighbors
        self.signal_length = config.signal_length

        self.track_fields = track_fields
        if 'eventID' in self.track_fields:
            self.evt_id = 'eventID'
            self.trj_id = 'trackID'
        else:
            self.evt_id = 'event_id'
            self.trj_id = 'traj_id'

        if type(relevant_params) == dict:
            self.relevant_params_list = list(relevant_params.keys())
            self.relevant_params_dict = relevant_params
        elif type(relevant_params) == list:
            self.relevant_params_list = relevant_params
            self.relevant_params_dict = None
        else:
            raise TypeError("relevant_params must be list of param names or list of dicts with learning rates")

        self.target_val_dict = None
        if len(set_target_vals) > 0:
            if len(set_target_vals) % 2 != 0:
                raise ValueError("Incorrect format for set_target_vals!")
            
            self.target_val_dict = {}
            for i_val in range(len(set_target_vals)//2):
                param_name = set_target_vals[2*i_val]
                param_val = set_target_vals[2*i_val+1]
                self.target_val_dict[param_name] = float(param_val)

        self.setup_params()
        if not self.read_target:
            self.make_target_sim()

        if self.current_mode == 'lut':
            self.lut_file = config.lut_file
            self.load_lut()

        loss_functions = {
            "mse_adc": (mse_adc, {}),
            "mse_time": (mse_time, {}),
            "mse_time_adc": (mse_time_adc, {'alpha': 0.5}),
            "chamfer_3d": (chamfer_3d, {'adc_norm': adc_norm, 'match_z': match_z}),
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

        self.training_history = {}
        for param in self.relevant_params_list:
            self.training_history[param] = []
            self.training_history[param + '_target'] = []
            self.training_history[param + '_init'] = [getattr(self.current_params, param)]

        for param in self.shift_no_fit:
            self.training_history[param + '_target'] = []

        if self.compute_target_hessian:
            self.training_history['hessian'] = []
        self.training_history['size_history'] = []
        self.training_history['memory'] = []

        self.training_history['config'] = config

        self.keep_in_memory = keep_in_memory
        self.sim_seed_strategy = sim_seed_strategy
        if keep_in_memory:
            self.targets = {}

    def setup_params(self):
        Params = build_params_class(self.relevant_params_list)
        ref_params = load_detector_properties(Params, self.detector_props, self.pixel_layouts)
        ref_params = ref_params.replace(
            electron_sampling_resolution=self.electron_sampling_resolution,
            number_pix_neighbors=self.number_pix_neighbors,
            signal_length=self.signal_length,
            time_window=self.signal_length)
        
        params_to_apply = [
            "diffusion_in_current_sim",
            "mc_diff"
        ]

        ref_params = ref_params.replace(**{key: getattr(self, key) for key in params_to_apply})

        initial_params = {}

        if self.vary_init:
            logger.info("Running with random initial guess")
            for param in self.relevant_params_list:
                init_val = np.random.uniform(low=ranges[param]['down'], 
                                            high=ranges[param]['up'])
                initial_params[param] = init_val

        self.current_params = ref_params.replace(**initial_params)

        #Only do it now to not inpact current_params
        if not self.readout_noise_guess:
            logger.info("Not simulating electronics noise for guesses")
            self.current_params = remove_noise_from_params(self.current_params)

        self.ref_params = ref_params

        self.params_normalization = ref_params.replace(**{key: getattr(self.current_params, key) if getattr(self.current_params, key) != 0. else 1. for key in self.relevant_params_list})
        self.norm_params = ref_params.replace(**{key: 1. if getattr(self.current_params, key) != 0. else 0. for key in self.relevant_params_list})

    def load_lut(self):
        self.response = load_lut(self.lut_file)

    def update_params(self):
        self.current_params = self.norm_params.replace(**{key: getattr(self.norm_params, key)*getattr(self.params_normalization, key) for key in self.relevant_params_list})

    def make_target_sim(self):
        np.random.seed(self.target_seed)
        logger.info("Constructing target param simulation")

        self.target_params = {}

        if self.target_val_dict is not None:
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
                if self.target_fixed_range is not None:
                    param_val = np.random.uniform(low=ranges[param]['nom']*(1.-self.target_fixed_range), 
                                                high=ranges[param]['nom']*(1.+self.target_fixed_range))
                else:
                    param_val = np.random.uniform(low=ranges[param]['down'], 
                                                high=ranges[param]['up'])

                logger.info(f'{param}, target: {param_val}, init {getattr(self.current_params, param)}')    
                self.target_params[param] = param_val
        self.target_params = self.ref_params.replace(**self.target_params)
        if not self.readout_noise_target:
            logger.info("Not simulating electronics noise for target")
            self.target_params = remove_noise_from_params(self.target_params)

    def get_simulated_target(self, target, i, evts_sim, regen=False):
        #Reading the reference
        if self.read_target:
            with open(target, 'rb') as f:
                loaded = jnp.load(f, allow_pickle=True)
                try:
                    mask = jnp.isin(loaded['event_id'], evts_sim)
                except:
                    mask = jnp.isin(loaded['event'], evts_sim)
                try:
                    ref_adcs = loaded['adcs'][mask]
                except:
                    print("no adcs in the input target")
                try:
                    ref_Q = loaded['Q'][mask]
                except:
                    ref_Q = adc2charge(ref_adcs, self.current_params)
                # switch x and z
                # as x is the drift in data, and z is the drift in sim
                ref_pixel_x = loaded['z'][mask]
                ref_pixel_y = loaded['y'][mask]
                ref_pixel_z = loaded['x'][mask]
                ref_ticks = loaded['ticks'][mask]
                try:
                    ref_event = loaded['event_id'][mask]
                except:
                    ref_event = loaded['event'][mask]
        else:
            #Simulating the reference during the first epoch
            fname = 'target_' + self.out_label + '/batch' + str(i) + '_target.npz'
            if regen or not os.path.exists(fname):
                if self.current_mode == 'lut':
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, _, _, _, _ = simulate(self.target_params, self.response, target, self.track_fields, i+1) #Setting a different random seed for each target
                else:
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, _, _, _, _ = simulate_parametrized(self.target_params, target, self.track_fields, i+1) #Setting a different random seed for each target

                if self.compute_target_hessian:
                    logger.error("Computing target hessian is not implemented yet")
                    raise NotImplementedError("Computing target hessian is not implemented yet")
                    # logger.info("Computing target hessian")
                    # if self.current_mode == 'lut':
                    #     hess, aux = jax.jacfwd(jax.jacrev(params_loss, (0), has_aux=True), has_aux=True)(self.target_params, self.response, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks_tgt, self.track_fields, rngkey=i, loss_fn=self.loss_fn, diffusion_in_current_sim=self.diffusion_in_current_sim, **self.loss_fn_kw)
                    # else:
                    #     hess, aux = jax.jacfwd(jax.jacrev(params_loss_parametrized, (0), has_aux=True), has_aux=True)(self.target_params, ref_adcs, ref_unique_pixels, ref_ticks, selected_tracks_tgt, self.track_fields, rngkey=i, loss_fn=self.loss_fn, diffusion_in_current_sim=self.diffusion_in_current_sim, **self.loss_fn_kw)
                    # self.training_history['hessian'].append(format_hessian(hess))

                # embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                #Saving the target for the batch
                #TODO: See if we have to do this for each event
                
                with open(fname, 'wb') as f:
                    jnp.savez(f, adcs=ref_adcs, pixel_x=ref_pixel_x, pixel_y=ref_pixel_y, pixel_z=ref_pixel_z, ticks=ref_ticks, event=ref_event)
                    if self.keep_in_memory:
                        self.targets[i] = (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event)

            else:
                #Loading the target
                if self.keep_in_memory:
                    ref_adcs, ref_unique_pixels, ref_ticks = self.targets[i]
                else:
                    with open(fname, 'rb') as f:
                        loaded = jnp.load(f)
                        ref_adcs = loaded['adcs']
                        ref_pixel_x = loaded['pixel_x']
                        ref_pixel_y = loaded['pixel_y']
                        ref_pixel_z = loaded['pixel_z']
                        ref_ticks = loaded['ticks']
                        ref_event = loaded['event']
        
        return ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event

    def compute_loss(self, tracks, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, with_loss=True, with_grad=True, epoch=0):
        if self.sim_seed_strategy == "same":
            rngkey = i + 1
        elif self.sim_seed_strategy == "different":
            rngkey = -i - 1 #Need some offset otherwise batch 0 has same seed
        elif self.sim_seed_strategy == "different_epoch":
            rngkey = -i - 1 - epoch * 10000
        elif self.sim_seed_strategy == "random":
            rngkey = np.random.randint(0, 1000000)
        elif self.sim_seed_strategy == "constant":
            rngkey = 0
        else:
            raise ValueError("Unknown sim_seed_strategy. Must be same, different or random")

        assert(with_loss or with_grad)
        loss_val, grads, aux = None, None, None

        # Simulate and get output
        if self.current_mode == 'lut':
            if with_loss and with_grad:
                (loss_val, aux), grads = value_and_grad(params_loss, (0), has_aux = True)(self.current_params, self.response, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, tracks, self.track_fields, rngkey=rngkey, loss_fn=self.loss_fn, **self.loss_fn_kw)
            elif with_loss:
                loss_val, aux = params_loss(self.current_params, self.response, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, tracks, self.track_fields, rngkey=rngkey, loss_fn=self.loss_fn, **self.loss_fn_kw)
            elif with_grad:
                grads, aux = grad(params_loss, (0), has_aux=True)(self.current_params, self.response, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, tracks, self.track_fields, rngkey=rngkey, loss_fn=self.loss_fn, **self.loss_fn_kw)
        else:
            if with_loss and with_grad:
                (loss_val, aux), grads = value_and_grad(params_loss_parametrized, (0), has_aux = True)(self.current_params, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, tracks, self.track_fields, rngkey=rngkey, loss_fn=self.loss_fn, **self.loss_fn_kw)
            elif with_loss:
                loss_val, aux = params_loss_parametrized(self.current_params, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, tracks, self.track_fields, rngkey=rngkey, loss_fn=self.loss_fn, **self.loss_fn_kw)
            elif with_grad:
                grads, aux = grad(params_loss_parametrized, (0), has_aux=True)(self.current_params, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, tracks, self.track_fields, rngkey=rngkey, loss_fn=self.loss_fn, **self.loss_fn_kw)
        return loss_val, grads, aux
    
    def prepare_fit(self):
        # make a folder for the pixel target
        if os.path.exists('target_' + self.out_label):
            shutil.rmtree('target_' + self.out_label, ignore_errors=True)
        os.makedirs('target_' + self.out_label)

        # make a folder for the fit result
        if not os.path.exists(f'fit_result/{self.test_name}'):
            os.makedirs(f'fit_result/{self.test_name}')

        for param in self.relevant_params_list:
            self.training_history[param] = []
            self.training_history[param+"_grad"] = []
            self.training_history[param+"_iter"] = []
        self.training_history['step_time'] = []
        self.training_history['losses_iter'] = []

        # Include initial value in training history (if haven't loaded a checkpoint)
        for param in self.relevant_params_list:
            if len(self.training_history[param]) == 0:
                self.training_history[param].append(getattr(self.current_params, param))
                if not self.read_target:
                    self.training_history[param+'_target'].append(getattr(self.target_params, param))
            if len(self.training_history[param+"_iter"]) == 0:
                self.training_history[param+"_iter"].append(getattr(self.current_params, param))
        if not self.read_target:
            for param in self.shift_no_fit:
                if len(self.training_history[param+'_target']) == 0:
                    self.training_history[param+'_target'].append(getattr(self.target_params, param))

    def fit(self):
        raise NotImplementedError("Fit method not implemented. Use a derived class")

        

class GradientDescentFitter(ParamFitter):
    def __init__(self, optimizer_fn="Adam", max_clip_norm_val=False, clip_from_range=False,
                 lr_scheduler=None, optimizer=None, lr=None, lr_kw=None, epoch_size=1, **kwargs):
        super().__init__(**kwargs)

        if optimizer_fn == "Adam":
            self.optimizer_fn = optax.adam
        elif optimizer_fn == "SGD":
            self.optimizer_fn = optax.sgd
        else:
            raise NotImplementedError("Only SGD and Adam supported")
        self.optimizer_fn_name = optimizer_fn

        self.max_clip_norm_val = max_clip_norm_val
        if self.max_clip_norm_val is not None:
            logger.info(f"Will clip gradient norm at {self.max_clip_norm_val}")

        self.clip_from_range = clip_from_range

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
                if lr_scheduler_fn == optax.constant_schedule:
                    self.learning_rates = {par: lr_scheduler_fn(lr) for par in self.relevant_params_list}
                else:
                    self.learning_rates = {par: lr_scheduler_fn(lr, transition_steps=epoch_size, staircase=True, **lr_kw) for par in self.relevant_params_list}
        else:
            if lr_scheduler_fn == optax.constant_schedule:
                self.learning_rates = {key: lr_scheduler_fn(float(value)) for key, value in self.relevant_params_dict.items()}
            else:
                self.learning_rates = {key: lr_scheduler_fn(float(value), transition_steps=epoch_size, staircase=True, **lr_kw) for key, value in self.relevant_params_dict.items()}

        
        # Set up optimizer -- can pass in directly, or construct as SGD from relevant params and/or lr

        if optimizer is None:
            if self.max_clip_norm_val is not None:
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(self.max_clip_norm_val),
                    optax.multi_transform({key: self.optimizer_fn(value) for key, value in self.learning_rates.items()},
                                {key: key for key in self.relevant_params_list})
                )
            else:
                self.optimizer = optax.chain(
                    optax.multi_transform({key: self.optimizer_fn(value) for key, value in self.learning_rates.items()},
                                {key: key for key in self.relevant_params_list})
                )
        else:
            raise ValueError("Passing directly optimizer is not supported")
    
        self.opt_state = self.optimizer.init(extract_relevant_params(self.norm_params, self.relevant_params_list))

        self.training_history['optimizer_fn_name'] = self.optimizer_fn_name

    def clip_values(self, mini=0.01, maxi=100):
        cur_norm_values = extract_relevant_params(self.norm_params, self.relevant_params_list)
        cur_norm_values = {key: jnp.array(max(mini, min(maxi, val))) for key, val in cur_norm_values.items()}
        self.norm_params = self.norm_params.replace(**cur_norm_values)

    def clip_values_from_range(self):
        self.norm_params = self.norm_params.replace(
            **{key: jnp.array(max(ranges[key]['down']/getattr(self.params_normalization, key), min(ranges[key]['up']/getattr(self.params_normalization, key), getattr(self.norm_params, key)))) for key in self.relevant_params_list}
            )

    def apply_updates(self, params, update):
        setattr(self, params, getattr(self, params).replace(**{key: getattr(getattr(self, params), key) + val for key, val in update.items()}))

    def process_grads(self, grads):
        scaled_grads = {key: getattr(grads, key)*getattr(self.params_normalization, key) for key in self.relevant_params_list}

        leaves = jax.tree_util.tree_leaves(grads)
        hasNaN = any(jnp.isnan(leaf).any() for leaf in leaves)
        if hasNaN:
            logger.warning("Got NaN gradients! Skipping update for this batch")
        else:
            updates, self.opt_state = self.optimizer.update(scaled_grads, self.opt_state)
            # self.current_params = update_params(self.norm_params, updates)
            self.apply_updates('norm_params', updates)
            if self.clip_from_range:
                #Clipping param values
                self.clip_values_from_range()
            self.update_params()
        return scaled_grads

    def fit(self, dataloader_sim, target, epochs=300, iterations=None, save_freq=10, print_freq=1):

        self.prepare_fit()

        if iterations is not None:
            pbar_total = iterations
        else:
            pbar_total = len(dataloader_sim) * epochs

        if not self.read_target:
            # If explicit number of iterations, scale epochs accordingly
            if len(dataloader_sim) != len(dataloader_target):
                raise Exception("Sim and target inputs do not match in size. Panic.")


        if iterations is not None:
            epochs = iterations // len(dataloader_sim) + 1

        # The training loop
        total_iter = 0
        terminate_fit = False
        with tqdm(total=pbar_total) as pbar:
            for epoch in range(epochs):
                if terminate_fit:
                    break
                logger.info(f"epoch {epoch}")
                # if epoch == 2: libcudart.cudaProfilerStart()

                for i in range(len(dataloader_sim)):
                    start_time = time()

                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.track_fields))
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = jnp.unique(selected_tracks_sim[:, self.track_fields.index(self.evt_id)])

                    # target
                    if not self.read_target:
                        selected_tracks_bt_tgt = dataloader_target[i].reshape(-1, len(self.track_fields))
                        target = jax.device_put(selected_tracks_bt_tgt)
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event = self.get_simulated_target(target, i, evts_sim, regen=False)

                    # loss
                    loss_val, grads, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, epoch=epoch)

                    modified_grads = self.process_grads(grads) #Grads are modified ans applied in this function

                    stop_time = time()

                    for param in self.relevant_params_list:
                        self.training_history[param+"_grad"].append(modified_grads[param].item())
                    self.training_history['step_time'].append(stop_time - start_time)

                    self.training_history['losses_iter'].append(loss_val.item())
                    for param in self.relevant_params_list:
                        #TODO: Need to check why this is not consistent
                        if type(getattr(self.current_params, param)) == float:
                            self.training_history[param + '_iter'].append(getattr(self.current_params, param))
                        else:
                            self.training_history[param + '_iter'].append(getattr(self.current_params, param).item())

                    self.training_history['size_history'].append(get_size_history())
                    if 'cuda' in jax.devices():
                        self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

                    if iterations is not None or total_iter == (iterations-1):
                        if total_iter % print_freq == 0:
                            for param in self.relevant_params_list:
                                logger.info(f"{param} {getattr(self.current_params,param)} {modified_grads[param]}")
                            
                        if total_iter % save_freq == 0:
                            with open(f'fit_result/{self.test_name}/history_iter{total_iter}_{self.out_label}.pkl', "wb") as f_history:
                                pickle.dump(self.training_history, f_history)

                            if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')

                    total_iter += 1
                    pbar.update(1)
                    
                    if iterations is not None:
                        if total_iter >= iterations:
                            with open(f'fit_result/{self.test_name}/history_iter{total_iter}_{self.out_label}.pkl', "wb") as f_history:
                                pickle.dump(self.training_history, f_history)

                            if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                                os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')

                            if os.path.exists('target_' + self.out_label):
                                shutil.rmtree('target_' + self.out_label, ignore_errors=True)

                            terminate_fit = True
                            break

            with open(f'fit_result/{self.test_name}/history_iter{total_iter}_{self.out_label}.pkl', "wb") as f_history:
                pickle.dump(self.training_history, f_history)

            if os.path.exists(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl'):
                os.remove(f'fit_result/{self.test_name}/history_iter{total_iter-save_freq}_{self.out_label}.pkl')

            if os.path.exists('target_' + self.out_label):
                shutil.rmtree('target_' + self.out_label, ignore_errors=True)

            # libcudart.cudaProfilerStop()


class LikelihoodProfiler(ParamFitter):
    def __init__(self, scan_tgt_nom=False, **kwargs):
        self.scan_tgt_nom = scan_tgt_nom
        super().__init__(**kwargs)
        

    def make_target_sim(self, seed=2, fixed_range=None):
        if self.scan_tgt_nom:
            target_params = {}
            logger.info("Using the fitter in a gradient profile mode. Setting targets to nominal values")
            for param in self.relevant_params_list:
                target_params[param] = ranges[param]['nom']
            self.target_params = self.ref_params.replace(**target_params)
            if not self.readout_noise_target:
                logger.info("Not simulating electronics noise for target")
                self.target_params = remove_noise_from_params(self.target_params)
        else:
            super().make_target_sim()

    def fit(self, dataloader_sim, dataloader_target, iterations=100, **kwargs):

        self.prepare_fit()

        logger.info("Using the fitter in a gradient profile mode. The sampling will follow a regular grid.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        nb_var_params = len(self.relevant_params_list)
        logger.info(f"{nb_var_params} parameters are to be scanned.")
        nb_steps = iterations
        logger.info(f"Each parameter will be scanned with {nb_steps} steps")

        self.ref_params = self.current_params

        for i in range(len(dataloader_sim)):
            logger.info(f"Batch {i}/{len(dataloader_target)}")

            # sim
            selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.track_fields))
            selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
            evts_sim = jnp.unique(selected_tracks_sim[:, self.track_fields.index(self.evt_id)])

            # target
            if not self.read_target:
                selected_tracks_bt_tgt = dataloader_target[i].reshape(-1, len(self.track_fields))
                target = jax.device_put(selected_tracks_bt_tgt)
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event = self.get_simulated_target(target, i, evts_sim, regen=False)

            for param in self.relevant_params_list:
                lower = ranges[param]['down']
                upper = ranges[param]['up']
                param_step = (upper - lower)/(nb_steps - 1)

                for iter in tqdm(range(nb_steps)):
                    start_time = time()
                    new_param_values = {param: lower + iter*param_step}
                    self.current_params = self.ref_params.replace(**new_param_values)
                    loss_val, grads, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event)

                    stop_time = time()

                    for par in self.relevant_params_list:
                        self.training_history[par+"_grad"].append(getattr(grads, par).item())
                    self.training_history['step_time'].append(stop_time - start_time)

                    self.training_history['losses_iter'].append(loss_val.item())
                    for par in self.relevant_params_list:
                        #TODO: Need to check why this is not consistent
                        if type(getattr(self.current_params, par)) == float:
                            self.training_history[par + '_iter'].append(getattr(self.current_params, par))
                        else:
                            self.training_history[par + '_iter'].append(getattr(self.current_params, par).item())

                    self.training_history['size_history'].append(get_size_history())
                    if 'cuda' in jax.devices():
                        self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

        with open(f'fit_result/{self.test_name}/history_iter{iterations}_{self.out_label}.pkl', "wb") as f_history:
            pickle.dump(self.training_history, f_history)

        if os.path.exists('target_' + self.out_label):
            shutil.rmtree('target_' + self.out_label, ignore_errors=True)

class MinuitFitter(ParamFitter):
    def __init__(self, separate_fits=True, minimizer_strategy=1, minimizer_tol=1e-5, **kwargs):
        super().__init__(**kwargs)

        self.separate_fits = separate_fits
        self.minimizer_strategy = minimizer_strategy
        self.minimizer_tol = minimizer_tol

    def configure_minimizer(self, loss_wrapper, grad_wrapper=None,):
        self.minimizer = iminuit.Minuit(
            loss_wrapper,
            [getattr(self.current_params, param) for param in self.relevant_params_list],
            grad=grad_wrapper,
            name=self.relevant_params_list
        )
        for param in self.relevant_params_list:
            lower = ranges[param]['down']
            upper = ranges[param]['up']

            self.minimizer.limits[param] = (lower, upper)
            self.minimizer.errors[param] = (upper - lower)/10.
            self.minimizer.fixed = False
        
        self.minimizer.strategy = self.minimizer_strategy  # 0, 1 or 2. Maybe, on 0, it doesn't use the grad func? Try out
        self.minimizer.errordef = 1  # definition of "1 sigma": 0.5 for NLL, 1 for chi2
        self.minimizer.tol = (  # stopping value, EDM < tol
            self.minimizer_tol / 0.002 / 1  # iminuit multiplies by default with 0.002
        )
        self.minimizer.print_level = 2  # 0, 1 or 2. Verbosity level

    def prepare_fit(self):
        self.training_history["minuit_result"] = []
        return super().prepare_fit()

    def push_minuit_result(self, result):
        result_dict = {
            "fval": result.fval,
            "edm": result.fmin.edm,
            "covariance": result.covariance,
            "params": result.values.to_dict(),
            "errors": result.errors.to_dict(),
            "valid": result.valid,
        }
        self.training_history["minuit_result"].append(result_dict)
        if 'cuda' in jax.devices():
            self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

    def fit(self, dataloader_sim, dataloader_target, **kwargs):
        self.prepare_fit()
        logger.info("Using the fitter in a Minuit mode.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        logger.info(f"Running in {'separate' if self.separate_fits else 'joint'} fit mode")
        

        if self.separate_fits:
            for i in range(len(dataloader_sim)):
                logger.info(f"Batch {i}/{len(dataloader_target)}")
                start_time = time()

                # sim
                selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.track_fields))
                selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                evts_sim = jnp.unique(selected_tracks_sim[:, self.track_fields.index(self.evt_id)])

                # target
                if not self.read_target:
                    selected_tracks_bt_tgt = dataloader_target[i].reshape(-1, len(self.track_fields))
                    target = jax.device_put(selected_tracks_bt_tgt)
                ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event = self.get_simulated_target(target, i, evts_sim, regen=False)

                def loss_wrapper(args):
                    # Update the current params with the new values
                    self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                    loss_val, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, with_grad=False)
                    return loss_val

                def grad_wrapper(args):
                    # Update the current params with the new values
                    self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                    _, grads, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, with_loss=False)
                    return [getattr(grads, key) for key in self.relevant_params_list]

                self.configure_minimizer(loss_wrapper, grad_wrapper)
                result = self.minimizer.migrad()

                stop_time = time()
                self.training_history['step_time'].append(stop_time - start_time)
                self.push_minuit_result(result)

        else:
            # Joint fit
            def loss_wrapper(args):
                # Update the current params with the new values
                self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                avg_loss = 0
                for i in range(len(dataloader_sim)):
                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.track_fields))
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = jnp.unique(selected_tracks_sim[:, self.track_fields.index(self.evt_id)])

                    # target
                    if not self.read_target:
                        selected_tracks_bt_tgt = dataloader_target[i].reshape(-1, len(self.track_fields))
                        target = jax.device_put(selected_tracks_bt_tgt)
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event = self.get_simulated_target(target, i, evts_sim, regen=False)

                    loss_val, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, with_grad=False)
                    avg_loss += loss_val
                return avg_loss/len(dataloader_target)
            
            def grad_wrapper(args):
                # Update the current params with the new values
                self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                avg_grad = [0 for _ in range(len(self.relevant_params_list))]
                for i in range(len(dataloader_sim)):
                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.track_fields))
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = jnp.unique(selected_tracks_sim[:, self.track_fields.index(self.evt_id)])

                    # target
                    if not self.read_target:
                        selected_tracks_bt_tgt = dataloader_target[i].reshape(-1, len(self.track_fields))
                        target = jax.device_put(selected_tracks_bt_tgt)
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event = self.get_simulated_target(target, i, evts_sim, regen=False)

                    _, grads, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_event, with_loss=False)
                    avg_grad = [getattr(grads, key) + avg_grad[i] for i, key in enumerate(self.relevant_params_list)]
                return [g/len(dataloader_target) for g in avg_grad]

            self.configure_minimizer(loss_wrapper, grad_wrapper)
            result = self.minimizer.migrad()
            self.push_minuit_result(result)
        
        with open(f'fit_result/{self.test_name}/history_minuit_{self.out_label}.pkl', "wb") as f_history:
            pickle.dump(self.training_history, f_history)



