import os, sys

from requests import options
larndsim_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(0, larndsim_dir)
import shutil
import pickle
import numpy as np
from .ranges import ranges
from larndsim.sim_jax import get_size_history
from larndsim.losses_jax import mse_adc, mse_time, mse_time_adc, chamfer_3d, sdtw_adc, sdtw_time, sdtw_time_adc, adc2charge, nll_loss, llhd_loss #, sinkhorn_loss
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

from .strategies import LUTSimulation, LUTProbabilisticSimulation, ParametrizedSimulation, GenericLossStrategy

from ctypes import cdll
# libcudart = cdll.LoadLibrary('libcudart.so')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def value_and_grad_fwd(f, argnums=0, has_aux=False):
    """
    Computes the primal value and the gradient using forward-mode autodiff,
    with full support for auxiliary data.
    """
    def wrapper(*args, **kwargs):
        out = f(*args, **kwargs)
        if has_aux:
            y, aux = out
            # Return 'y' for jacfwd to differentiate, and (y, aux) to pass through
            return y, (y, aux)
        else:
            y = out
            # Return 'y' for jacfwd, and a copy of 'y' to pass through
            return y, y

    # We use jacfwd with has_aux=True so it captures our injected auxiliary data
    fwd_fn = jax.jacfwd(wrapper, argnums=argnums, has_aux=True)

    def val_and_grad_fn(*args, **kwargs):
        grad, aux_out = fwd_fn(*args, **kwargs)
        if has_aux:
            y, aux = aux_out
            return (y, aux), grad
        else:
            y = aux_out
            return y, grad
            
    return val_and_grad_fn

def serialize_value(v):
    if hasattr(v, 'shape'):
        # Check if it's a scalar (0-dimensional) or multi-dimensional
        if len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1):
            # Scalar JAX array -> Python float
            return float(v)
        else:
            # Multi-dimensional JAX array -> numpy array
            return np.array(v)
    elif hasattr(v, 'item'):
        # Fallback for other array-like objects with item()
        return float(v)
    else:
        # Already Python type
        return v


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
    def __init__(self, relevant_params, set_init_params, sim_track_fields, tgt_track_fields,
                 detector_props, pixel_layouts,
                 loss_fn=None, loss_fn_kw=None, readout_noise_target=True, readout_noise_guess=False, 
                 out_label="", test_name="this_test",
                 shift_no_fit=[], set_target_vals=[], vary_init=False, keep_in_memory=False,
                 compute_target_hessian=False, sim_seed_strategy="different",
                 target_seed=0, target_fixed_range=None,
                 adc_norm=1, match_z=True,
                 diffusion_in_current_sim=False,
                 mc_diff = False,
                 read_target=False,
                 probabilistic_sim=False,
                 sz_mini_bt=1, shuffle_bt=False,
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
        self.probabilistic_sim = probabilistic_sim
        self.sz_mini_bt = sz_mini_bt
        self.shuffle_bt = shuffle_bt

        self.sim_track_fields = sim_track_fields
        self.tgt_track_fields = tgt_track_fields
        if 'eventID' in self.sim_track_fields:
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

        self.set_init_params = set_init_params

        if self.current_mode == 'lut':
            self.lut_file = config.lut_file
        else:
            self.lut_file = None

        self.setup_params()

        if not self.read_target:
            self.make_target_sim()

        loss_functions = {
            "mse_adc": (mse_adc, {}),
            "mse_time": (mse_time, {}),
            "mse_time_adc": (mse_time_adc, {'alpha': 0.5}),
            "chamfer_3d": (chamfer_3d, {'adc_norm': adc_norm, 'match_z': match_z}),
            "sdtw_adc": (sdtw_adc, {'gamma': 1.}),
            "sdtw_time": (sdtw_time, {'gamma': 1.}),
            "sdtw_time_adc": (sdtw_time_adc, {'gamma': 1., 'alpha': 0.5}),
            "nll": (nll_loss, {'adc_norm': adc_norm, 'sigma': 1.0}),
            "llhd": (llhd_loss, {}),
            #"sinkhorn_loss": (sinkhorn_loss, {})
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
        
        # Set up loss strategy based on loss function and simulation type
        if loss_fn == 'llhd':
            from .strategies import ProbabilisticLossStrategy
            self.loss_strategy = ProbabilisticLossStrategy(loss_fn=llhd_loss, **self.loss_fn_kw)
        elif self.probabilistic_sim:
            # Use CollapsedProbabilisticLossStrategy for probabilistic simulation with deterministic losses
            from .strategies import CollapsedProbabilisticLossStrategy
            self.loss_strategy = CollapsedProbabilisticLossStrategy(loss_fn=self.loss_fn, **self.loss_fn_kw)
            logger.info(f"Using CollapsedProbabilisticLossStrategy with {loss_fn}")
        else:
            self.loss_strategy = GenericLossStrategy(self.loss_fn, **self.loss_fn_kw)

        self.training_history = {}
        for param in self.relevant_params_list:
            self.training_history[param] = []
            self.training_history[param + '_target'] = []
            self.training_history[param + '_init'] = [getattr(self.current_params, param)]

        for param in self.shift_no_fit:
            self.training_history[param + '_target'] = []

        if self.compute_target_hessian:
            self.training_history['hessian'] = []
            self.training_history['gradient'] = []
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
        
        if self.lut_file is not None:
            self.response, ref_params = load_lut(self.lut_file, ref_params)
        
        params_to_apply = [
            "diffusion_in_current_sim",
            "mc_diff"
        ]

        ref_params = ref_params.replace(**{key: getattr(self, key) for key in params_to_apply})

        # Initialize Simulation Strategy
        if self.current_mode == 'lut':
            if self.probabilistic_sim:
                self.sim_strategy = LUTProbabilisticSimulation(self.response)
            else:
                self.sim_strategy = LUTSimulation(self.response)
        elif self.current_mode == 'parametrized':
            self.sim_strategy = ParametrizedSimulation()
        else:
             raise ValueError(f"Unknown mode {self.current_mode}")

        initial_params = {}

        if self.vary_init:
            logger.info("Running with random initial guess")
            for param in self.relevant_params_list:
                init_val = np.random.uniform(low=ranges[param]['down'], 
                                            high=ranges[param]['up'])
                initial_params[param] = init_val

        elif len(self.set_init_params) > 0:
            if len(self.set_init_params) % 2 != 0:
                raise ValueError("Incorrect format for set_init_params!")
            for i_val in range(len(self.set_init_params)//2):
                param_name = self.set_init_params[2*i_val]
                param_val = self.set_init_params[2*i_val+1]
                initial_params[param_name] = float(param_val)

        self.current_params = ref_params.replace(**initial_params)

        self.ref_params = ref_params

        self.params_normalization = ref_params.replace(**{key: getattr(self.current_params, key) if getattr(self.current_params, key) != 0. else 1. for key in self.relevant_params_list})
        self.norm_params = ref_params.replace(**{key: 1. if getattr(self.current_params, key) != 0. else 0. for key in self.relevant_params_list})

        #Only do it now to not inpact current_params (ref_params?)
        #FIXME It's a problem if the noise parameters are to be fitted
        if not self.readout_noise_guess:
            logger.info("Not simulating electronics noise for guesses")
            self.current_params = remove_noise_from_params(self.current_params)
            self.params_normalization = remove_noise_from_params(self.params_normalization)
            self.norm_params = remove_noise_from_params(self.norm_params)

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
                if 'event_id' in loaded:
                    mask = jnp.isin(loaded['event_id'], evts_sim)
                    ref_event = loaded['event_id'][mask]
                elif 'event' in loaded:
                    mask = jnp.isin(loaded['event'], evts_sim)
                    ref_event = loaded['event'][mask]
                else:
                    raise ValueError("No event_id or event in the target file")

                if not 'adcs' in loaded:
                    raise ValueError("No adcs in the target file")

                ref_adcs = loaded['adcs'][mask]
                if 'Q' in loaded:
                    ref_Q = loaded['Q'][mask]
                else:
                    ref_Q = adc2charge(ref_adcs, self.current_params)
    
                # switch x and z
                # as x is the drift in data, and z is the drift in sim
                ref_pixel_x = loaded['z'][mask]
                ref_pixel_y = loaded['y'][mask]
                ref_pixel_z = loaded['x'][mask]
                ref_ticks = loaded['ticks'][mask]
                ref_hit_prob = jnp.ones(ref_adcs.shape[0]) #Assuming all hits are valid
        else:
            #Simulating the reference during the first epoch
            fname = 'target_' + self.out_label + '/batch' + str(i) + '_target.npz'
            if regen or not os.path.exists(fname):
                
                # Target generation should always produce "hits" (stochastic),
                # even if the fitting strategy is probabilistic.
                target_strategy = self.sim_strategy
                if isinstance(self.sim_strategy, LUTProbabilisticSimulation):
                    target_strategy = LUTSimulation(self.response)
                
                prediction = target_strategy.predict(self.target_params, target, self.tgt_track_fields, rngkey=i+1)
                
                ref_adcs = prediction['adcs']
                ref_pixel_x = prediction['pixel_x']
                ref_pixel_y = prediction['pixel_y']
                ref_pixel_z = prediction['pixel_z']
                ref_ticks = prediction['ticks']
                ref_hit_prob = prediction['hit_prob']
                ref_event = prediction['event']
                ref_pixel_id = prediction.get('hit_pixels', prediction.get('unique_pixels'))

                # embed_target = embed_adc_list(self.sim_target, target, pix_target, ticks_list_targ)
                #Saving the target for the batch
                #TODO: See if we have to do this for each event
                
                with open(fname, 'wb') as f:
                    jnp.savez(f, adcs=ref_adcs, pixel_x=ref_pixel_x, pixel_y=ref_pixel_y, pixel_z=ref_pixel_z, ticks=ref_ticks, hit_prob=ref_hit_prob, event=ref_event, pixel_id=ref_pixel_id)
                    if self.keep_in_memory:
                        self.targets[i] = (ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id)

            else:
                #Loading the target
                if self.keep_in_memory:
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.targets[i]
                else:
                    with open(fname, 'rb') as f:
                        loaded = jnp.load(f)
                        ref_adcs = loaded['adcs']
                        ref_pixel_x = loaded['pixel_x']
                        ref_pixel_y = loaded['pixel_y']
                        ref_pixel_z = loaded['pixel_z']
                        ref_ticks = loaded['ticks']
                        ref_event = loaded['event']
                        ref_hit_prob = loaded['hit_prob']
                        ref_pixel_id = loaded['pixel_id'] if 'pixel_id' in loaded else jnp.zeros_like(ref_adcs, dtype=int) # Fallback

        return ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id

    def compute_loss(self, tracks, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True, with_hess=False, epoch=0):
        if self.probabilistic_sim:
            rngkey = None
        else:
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
        loss_val, grads, aux, hess, aux_hess = None, None, None, None, None

        target_data = {
            'adcs': ref_adcs,
            'pixel_x': ref_pixel_x,
            'pixel_y': ref_pixel_y,
            'pixel_z': ref_pixel_z,
            'ticks': ref_ticks,
            'hit_prob': ref_hit_prob,
            'event': ref_event,
            'pixel_id': ref_pixel_id
        }

        def loss_wrapper(params):
            prediction = self.sim_strategy.predict(params, tracks, self.sim_track_fields, rngkey)
            return self.loss_strategy.compute(params, prediction, target_data)

        # Simulate and get output
        if with_loss and with_grad:
            (loss_val, aux), grads = value_and_grad(loss_wrapper, has_aux = True)(self.current_params)
        elif with_loss:
            loss_val, aux = loss_wrapper(self.current_params)
        elif with_grad:
            grads, aux = grad(loss_wrapper, has_aux=True)(self.current_params)
 
        if with_hess:
            hess, aux_hess = jax.jacfwd(jax.jacrev(loss_wrapper, has_aux=True), has_aux=True)(self.current_params)

        return loss_val, grads, aux, hess, aux_hess
    
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
        self.training_history['aux_iter'] = []

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

    def fit(self, *args, **kwargs):
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
                elif lr_scheduler_fn == optax.exponential_decay:
                    self.learning_rates = {par: lr_scheduler_fn(lr, transition_steps=epoch_size, staircase=True, **lr_kw) for par in self.relevant_params_list}
                elif lr_scheduler_fn == optax.warmup_exponential_decay_schedule:
                    self.learning_rates = {par: lr_scheduler_fn(peak_value=lr, transition_steps=epoch_size, staircase=True, **lr_kw) for par in self.relevant_params_list}
                else:
                    raise ValueError("The specified optimizer schedules are not yet supported")
        else:
            if lr_scheduler_fn == optax.constant_schedule:
                self.learning_rates = {key: lr_scheduler_fn(float(value)) for key, value in self.relevant_params_dict.items()}
            elif lr_scheduler_fn == optax.exponential_decay:
                self.learning_rates = {key: lr_scheduler_fn(float(value), transition_steps=epoch_size, staircase=True, **lr_kw) for key, value in self.relevant_params_dict.items()}
            elif lr_scheduler_fn == optax.warmup_exponential_decay_schedule:
                self.learning_rates = {key: lr_scheduler_fn(peak_value=float(value), transition_steps=epoch_size, staircase=True, **lr_kw) for key, value in self.relevant_params_dict.items()}
            else:
                raise ValueError("The specified optimizer schedules are not yet supported")

        
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

    def add_grads(self, g1, g2):
        return jax.tree_util.tree_map(lambda a, b: a + b, g1, g2)

    def fit(self, dataloader_sim, target, epochs=300, iterations=None, save_freq=10, print_freq=1):

        self.prepare_fit()

        if iterations is not None:
            pbar_total = iterations
        else:
            pbar_total = len(dataloader_sim) * epochs

        if not self.read_target:
            # If explicit number of iterations, scale epochs accordingly
            if len(dataloader_sim) != len(target):
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

                # shuffle batches
                indices = np.arange(len(dataloader_sim))
                if self.shuffle_bt:
                    np.random.shuffle(indices)

                for i_bt, i in enumerate(indices):
                    start_time = time()

                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = jnp.unique(selected_tracks_sim[:, self.sim_track_fields.index(self.evt_id)])

                    # target
                    if not self.read_target:
                        selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                        this_target = jax.device_put(selected_tracks_bt_tgt)
                    else:
                        this_target = target
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)

                    # loss
                    loss_val, grads, aux, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, epoch=epoch, with_loss=True, with_grad=True)

                    # split the code, ugly, but no additional operation if there's no averaging
                    if self.sz_mini_bt > 1:
                        # initialize batch grads and mean
                        if i_bt == 0:
                            summed_grads = jax.tree_util.tree_map(jnp.zeros_like, grads)
                            bt_loss = []

                        summed_grads = self.add_grads(summed_grads, grads)
                        bt_loss.append(loss_val.item())

                        if i_bt != 0 and i_bt % self.sz_mini_bt == 0:
                            update_grads = jax.tree_util.tree_map(lambda x: x / self.sz_mini_bt, summed_grads)
                            avg_loss = np.mean(bt_loss)
 
                            modified_grads = self.process_grads(update_grads) #Grads are modified ans applied in this function
                            stop_time = time()

                            for param in self.relevant_params_list:
                                self.training_history[param+"_grad"].append(modified_grads[param].item())
                            self.training_history['step_time'].append(stop_time - start_time)

                            self.training_history['losses_iter'].append(avg_loss) # type: ignore
                            aux_serializable = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                            self.training_history['aux_iter'].append(aux_serializable) # type: ignore
                            for param in self.relevant_params_list:
                                #TODO: Need to check why this is not consistent
                                if type(getattr(self.current_params, param)) == float:
                                    self.training_history[param + '_iter'].append(getattr(self.current_params, param))
                                else:
                                    self.training_history[param + '_iter'].append(getattr(self.current_params, param).item())

                            self.training_history['size_history'].append(get_size_history())
                            if 'cuda' in jax.devices():
                                self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

                            if iterations is not None:
                                if total_iter % print_freq == 0:
                                    for param in self.relevant_params_list:
                                        logger.info(f"{param} {getattr(self.current_params,param)} {modified_grads[param]}")

                            # zero-ing out the grads and loss
                            summed_grads = jax.tree_util.tree_map(jnp.zeros_like, grads)
                            bt_loss = []
                    else:

                        modified_grads = self.process_grads(grads) #Grads are modified ans applied in this function

                        stop_time = time()

                        for param in self.relevant_params_list:
                            self.training_history[param+"_grad"].append(modified_grads[param].item())
                        self.training_history['step_time'].append(stop_time - start_time)

                        self.training_history['losses_iter'].append(loss_val.item()) # type: ignore
                        aux_serializable = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                        for param in self.relevant_params_list:
                            #TODO: Need to check why this is not consistent
                            if type(getattr(self.current_params, param)) == float:
                                self.training_history[param + '_iter'].append(getattr(self.current_params, param))
                            else:
                                self.training_history[param + '_iter'].append(getattr(self.current_params, param).item())

                        self.training_history['size_history'].append(get_size_history())
                        if 'cuda' in jax.devices():
                            self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

                        if iterations is not None:
                            if total_iter % print_freq == 0:
                                for param in self.relevant_params_list:
                                    logger.info(f"{param} {getattr(self.current_params,param)} {modified_grads[param]}")
                            
                    if iterations is not None:
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

    def fit(self, dataloader_sim, target, iterations=100, **kwargs):

        self.prepare_fit()

        logger.info("Using the fitter in a gradient profile mode. The sampling will follow a regular grid.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        nb_var_params = len(self.relevant_params_list)
        logger.info(f"{nb_var_params} parameters are to be scanned.")
        nb_steps = iterations
        logger.info(f"Each parameter will be scanned with {nb_steps} steps")

        self.ref_params = self.current_params

        for i in range(len(dataloader_sim)):
            logger.info(f"Batch {i}/{len(target)}")

            # sim
            selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
            selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
            evts_sim = jnp.unique(selected_tracks_sim[:, self.sim_track_fields.index(self.evt_id)])

            # target
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)

            for param in self.relevant_params_list:
                lower = ranges[param]['down']
                upper = ranges[param]['up']
                param_step = (upper - lower)/(nb_steps - 1)

                for iter in tqdm(range(nb_steps)):
                    start_time = time()
                    # if iter == 3:
                    #     options = jax.profiler.ProfileOptions()
                    #     options.host_tracer_level = 2
                    #     jax.profiler.start_trace("/sdf/home/p/pgranger/profile-data", profiler_options=options)
                    #     # libcudart.cudaProfilerStart()
                    # if iter == 9:
                    #     jax.profiler.stop_trace()
                    #     # libcudart.cudaProfilerStop()
                    new_param_values = {param: lower + iter*param_step}
                    self.current_params = self.ref_params.replace(**new_param_values)
                    loss_val, grads, aux, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True)

                    stop_time = time()

                    for par in self.relevant_params_list:
                        self.training_history[par+"_grad"].append(getattr(grads, par).item())
                    self.training_history['step_time'].append(stop_time - start_time)

                    self.training_history['losses_iter'].append(loss_val.item()) # type: ignore
                    aux_serializable = {k: serialize_value(v) for k, v in aux.items()} if aux is not None else {}
                    self.training_history['aux_iter'].append(aux_serializable) # type: ignore
                    for par in self.relevant_params_list:
                        #TODO: Need to check why this is not consistent
                        if type(getattr(self.current_params, par)) == float:
                            self.training_history[par + '_iter'].append(getattr(self.current_params, par))
                        else:
                            self.training_history[par + '_iter'].append(getattr(self.current_params, par).item())

                    self.training_history['size_history'].append(get_size_history())
                    if 'cuda' in jax.devices():
                        self.training_history['memory'].append(jax.devices('cuda')[0].memory_stats())

                with open(f'fit_result/{self.test_name}/history_{param}_batch{i}_{self.out_label}.pkl', "wb") as f_history:
                    pickle.dump(self.training_history, f_history)
                if os.path.exists(f'fit_result/{self.test_name}/history_{param}_batch{i-1}_{self.out_label}.pkl'):
                    os.remove(f'fit_result/{self.test_name}/history_{param}_batch{i-1}_{self.out_label}.pkl')

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
            self.minimizer.fixed[param] = False
        
        self.minimizer.strategy = self.minimizer_strategy  # 0, 1 or 2. Maybe, on 0, it doesn't use the grad func? Try out
        self.minimizer.errordef = 1  # definition of "1 sigma": 0.5 for NLL, 1 for chi2
        self.minimizer.tol = (  # stopping value, EDM < tol
            self.minimizer_tol / 0.002 / 1  # iminuit multiplies by default with 0.002
        )
        self.minimizer.print_level = 3  # 0, 1 or 2. Verbosity level

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

    def fit(self, dataloader_sim, target, **kwargs):
        self.prepare_fit()
        logger.info("Using the fitter in a Minuit mode.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        logger.info(f"Running in {'separate' if self.separate_fits else 'joint'} fit mode")

        def get_target(self, i, evts_sim, target):
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)
            return ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id

        if self.separate_fits:
            for i in range(len(dataloader_sim)):
                logger.info(f"Batch {i}/{len(dataloader_sim)}")
                start_time = time()

                # sim
                selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                evts_sim = jnp.unique(selected_tracks_sim[:, self.sim_track_fields.index(self.evt_id)])

                # target
                if not self.read_target:
                    selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                    this_target = jax.device_put(selected_tracks_bt_tgt)
                else:
                    this_target = target
                ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)

                def loss_wrapper(args): # type: ignore
                    # Update the current params with the new values
                    self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                    loss_val, _, _ , _, _= self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_grad=False)
                    return loss_val

                def grad_wrapper(args): # type: ignore
                    # Update the current params with the new values
                    self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                    _, grads, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=False)
                    return [getattr(grads, key) for key in self.relevant_params_list]

                self.configure_minimizer(loss_wrapper, grad_wrapper)
                result = self.minimizer.migrad()

                stop_time = time()
                self.training_history['step_time'].append(stop_time - start_time)
                self.push_minuit_result(result)

        else:
            # Joint fit
            def loss_wrapper(args):
                logger.debug(f"Loss wrapper called with args: {args}")
                # Update the current params with the new values

                self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                tot_loss = 0
                for i in range(len(dataloader_sim)):
                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = jnp.unique(selected_tracks_sim[:, self.sim_track_fields.index(self.evt_id)])

                    # target
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = get_target(self, i, evts_sim, target)

                    loss_val, _, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_grad=False, with_loss=True)
                    tot_loss += loss_val # type: ignore
                logger.debug(f"Total loss: {tot_loss}")
                return tot_loss
            
            def grad_wrapper(args):
                logger.debug(f"Grad wrapper called with args: {args}")
                # Update the current params with the new values
                self.current_params = self.current_params.replace(**{key: args[i] for i, key in enumerate(self.relevant_params_list)})
                tot_grad = [0 for _ in range(len(self.relevant_params_list))]
                for i in range(len(dataloader_sim)):
                    # sim
                    selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
                    selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
                    evts_sim = jnp.unique(selected_tracks_sim[:, self.sim_track_fields.index(self.evt_id)])

                    # target
                    ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = get_target(self, i, evts_sim, target)

                    _, grads, _, _, _ = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=False)
                    tot_grad = [getattr(grads, key) + tot_grad[i] for i, key in enumerate(self.relevant_params_list)]
                logger.debug(f"Average gradient: {[g/len(dataloader_sim) for g in tot_grad]}")
                return [g for g in tot_grad]

            self.configure_minimizer(loss_wrapper, grad_wrapper)
            result = self.minimizer.migrad()
            self.push_minuit_result(result)
        
        with open(f'fit_result/{self.test_name}/history_minuit_{self.out_label}.pkl', "wb") as f_history:
            pickle.dump(self.training_history, f_history)


class HessianCalculator(ParamFitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not len(self.set_init_params) > 0:
            raise ValueError("Remember to set the initial parameter values for Hessian calculation!")

    def fit(self, dataloader_sim, target, **kwargs):

        self.prepare_fit()

        logger.info("Using the fitter for Hessian matrix calculation.")
        logger.warning(f"Arguments {kwargs} are ignored in this mode.")

        self.ref_params = self.current_params

        self.training_history['n_hit'] = []
        for i in range(len(dataloader_sim)):
            logger.info(f"Batch {i}/{len(target)}")

            # sim
            selected_tracks_bt_sim = dataloader_sim[i].reshape(-1, len(self.sim_track_fields))
            selected_tracks_sim = jax.device_put(selected_tracks_bt_sim)
            evts_sim = jnp.unique(selected_tracks_sim[:, self.sim_track_fields.index(self.evt_id)])

            # target
            if not self.read_target:
                selected_tracks_bt_tgt = target[i].reshape(-1, len(self.tgt_track_fields))
                this_target = jax.device_put(selected_tracks_bt_tgt)
            else:
                this_target = target
            ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id = self.get_simulated_target(this_target, i, evts_sim, regen=False)
            n_hit = np.sum(ref_hit_prob)
            self.training_history['n_hit'].append(n_hit)

            if self.current_mode == 'lut':
                loss_val, grads, aux, hess, aux_hess = self.compute_loss(selected_tracks_sim, i, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, ref_pixel_id, with_loss=True, with_grad=True, with_hess=True)
                self.training_history['hessian'].append(format_hessian(hess))
                self.training_history['gradient'].append(format_hessian(grads))
                self.training_history['losses_iter'].append(loss_val.item())

            else:
                hess, aux = jax.jacfwd(jax.jacrev(params_loss_parametrized, (0), has_aux=True), has_aux=True)(self.fit_params, ref_adcs, ref_pixel_x, ref_pixel_y, ref_pixel_z, ref_ticks, ref_hit_prob, ref_event, selected_tracks_sim, self.sim_track_fields, rngkey=i, loss_fn=self.loss_fn, **self.loss_fn_kw)
                self.training_history['hessian'].append(format_hessian(hess))

            with open(f'fit_result/{self.test_name}/history_batch{i}_{self.out_label}.pkl', "wb") as f_history:
                pickle.dump(self.training_history, f_history)
            if os.path.exists(f'fit_result/{self.test_name}/history_batch{i-1}_{self.out_label}.pkl'):
                os.remove(f'fit_result/{self.test_name}/history_batch{i-1}_{self.out_label}.pkl')

        if os.path.exists('target_' + self.out_label):
            shutil.rmtree('target_' + self.out_label, ignore_errors=True)
