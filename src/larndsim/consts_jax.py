"""
Module containing constants needed by the simulation
"""

import numpy as np
import yaml
from flax import struct
import jax
import jax.numpy as jnp
import dataclasses
from types import MappingProxyType

@dataclasses.dataclass
class Params_template:
    """
    Template class for simulation parameters in LArND-Sim.

    Attributes:
        eField (float): Electric field strength.
        Ab (float): Recombination parameter A (Birks/Box model).
        kb (float): Recombination parameter k (Birks/Box model).
        lifetime (float): Electron lifetime in microseconds.
        vdrift (float): Electron drift velocity in mm/us.
        long_diff (float): Longitudinal diffusion coefficient.
        tran_diff (float): Transverse diffusion coefficient.
        tpc_borders (jax.Array): TPC border coordinates.
        box (int): Box model flag (1 for Box, 0 for Birks).
        birks (int): Birks model flag (1 for Birks, 0 for Box).
        lArDensity (float): Liquid argon density in g/cm^3.
        alpha (float): Alpha parameter for recombination.
        beta (float): Beta parameter for recombination.
        MeVToElectrons (float): Conversion factor from MeV to electrons.
        pixel_pitch (float): Pixel pitch in mm.
        n_pixels_x (tuple): Number of pixels in x-direction.
        n_pixels_y (tuple): Number of pixels in y-direction.
        max_radius (int): Maximum radius for pixel clustering.
        max_active_pixels (int): Maximum number of active pixels.
        drift_length (float): Drift length in mm.
        t_sampling (float): Sampling time in microseconds.
        time_interval (float): Time interval for simulation in microseconds.
        time_padding (float): Time padding in microseconds.
        min_step_size (float): Minimum step size in mm.
        time_max (float): Maximum simulation time in microseconds.
        time_window (float): Time window for signal integration in microseconds.
        e_charge (float): Elementary charge in Coulombs.
        temperature (float): Temperature in Kelvin.
        response_bin_size (float): Response bin size in microseconds.
        number_pix_neighbors (int): Number of neighboring pixels considered.
        electron_sampling_resolution (float): Electron sampling resolution.
        signal_length (float): Length of the signal window.
        MAX_ADC_VALUES (int): Maximum number of ADC values stored per pixel.
        DISCRIMINATION_THRESHOLD (float): Discrimination threshold for signal.
        ADC_HOLD_DELAY (int): ADC hold delay in clock cycles.
        CLOCK_CYCLE (float): Clock cycle time in microseconds.
        GAIN (float): Front-end gain in mV/ke-.
        V_CM (float): Common-mode voltage in mV.
        V_REF (float): Reference voltage in mV.
        V_PEDESTAL (float): Pedestal voltage in mV.
        ADC_COUNTS (int): Number of ADC counts.
        RESET_NOISE_CHARGE (float): Reset noise in electrons.
        UNCORRELATED_NOISE_CHARGE (float): Uncorrelated noise in electrons.
        ELECTRON_MOBILITY_PARAMS (tuple): Parameters for electron mobility calculation.
        shift_x (float): Shift of TPC in x-direction.
        shift_y (float): Shift of TPC in y-direction.
        shift_z (float): Shift of TPC in z-direction.
        size_margin (float): Margin added to TPC size.
    """
    eField: float = struct.field(pytree_node=False)
    Ab: float = struct.field(pytree_node=False)
    kb: float = struct.field(pytree_node=False)
    lifetime: float = struct.field(pytree_node=False)
    vdrift: float = struct.field(pytree_node=False)
    long_diff: float = struct.field(pytree_node=False)
    tran_diff: float = struct.field(pytree_node=False)
    tpc_borders: jax.Array = struct.field(pytree_node=False)
    box: int = struct.field(pytree_node=False)
    birks: int = struct.field(pytree_node=False)
    lArDensity: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    beta: float = struct.field(pytree_node=False)
    MeVToElectrons: float = struct.field(pytree_node=False)
    pixel_pitch: float = struct.field(pytree_node=False)
    # n_pixels: tuple = struct.field(pytree_node=False)
    n_pixels_x: tuple = struct.field(pytree_node=False)
    n_pixels_y: tuple = struct.field(pytree_node=False)
    max_radius: int = struct.field(pytree_node=False)
    max_active_pixels: int = struct.field(pytree_node=False)
    drift_length: float = struct.field(pytree_node=False)
    t_sampling: float = struct.field(pytree_node=False)
    time_interval: float = struct.field(pytree_node=False)
    time_padding: float = struct.field(pytree_node=False)
    min_step_size: float = struct.field(pytree_node=False)
    time_max: float = struct.field(pytree_node=False)
    time_window: float = struct.field(pytree_node=False)
    e_charge: float = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    response_bin_size: float = struct.field(pytree_node=False)
    number_pix_neighbors: int = struct.field(pytree_node=False)
    electron_sampling_resolution: float = struct.field(pytree_node=False)
    signal_length: float = struct.field(pytree_node=False)
    tran_diff_bin_edges: jax.Array = struct.field(pytree_node=False)
    #: Maximum number of ADC values stored per pixel
    MAX_ADC_VALUES: int = struct.field(pytree_node=False)
    #: Discrimination threshold
    DISCRIMINATION_THRESHOLD: float = struct.field(pytree_node=False)
    #: ADC hold delay in clock cycles
    ADC_HOLD_DELAY: int = struct.field(pytree_node=False)
    #: Clock cycle time in :math:`\mu s`
    CLOCK_CYCLE: float = struct.field(pytree_node=False)
    #: Front-end gain in :math:`mV/ke-`
    GAIN: float = struct.field(pytree_node=False)
    #: Common-mode voltage in :math:`mV`
    V_CM: float = struct.field(pytree_node=False)
    #: Reference voltage in :math:`mV`
    V_REF: float = struct.field(pytree_node=False)
    #: Pedestal voltage in :math:`mV`
    V_PEDESTAL: float = struct.field(pytree_node=False)
    #: Number of ADC counts
    ADC_COUNTS: int = struct.field(pytree_node=False)
    # if readout_noise:
        #: Reset noise in e-
        # self.RESET_NOISE_CHARGE = 900
        # #: Uncorrelated noise in e-
        # self.UNCORRELATED_NOISE_CHARGE = 500
    # else:
    RESET_NOISE_CHARGE: float = struct.field(pytree_node=False)
    UNCORRELATED_NOISE_CHARGE: float = struct.field(pytree_node=False)

    ELECTRON_MOBILITY_PARAMS: tuple = struct.field(pytree_node=False)

    #Shifts of the TPC with respect to the true positions
    shift_x: float = struct.field(pytree_node=False)
    shift_y: float = struct.field(pytree_node=False)
    shift_z: float = struct.field(pytree_node=False)
    size_margin: float = struct.field(pytree_node=False)
    diffusion_in_current_sim: bool = struct.field(pytree_node=False, default=True)
    mc_diff: bool = struct.field(pytree_node=False, default=False)

def build_params_class(params_with_grad):
    """
    Dynamically creates a dataclass based on the fields of `Params_template`, modifying the metadata of fields
    specified in `params_with_grad` to allow gradient calculation.

    Args:
        params_with_grad (list of str): List of parameter names for which gradient calculation is required.
            For these parameters, the `pytree_node=False` metadata is removed.

    Returns:
        type: A new dataclass type named "Params" with fields matching those of `Params_template`, where
            specified fields have updated metadata to support gradient computation.

    Notes:
        - This function assumes the existence of a `Params_template` dataclass and the `struct` module
          providing a `dataclass` decorator.
        - The returned class can be instantiated with the same fields as `Params_template`.
    """
    template_fields = dataclasses.fields(Params_template)
    # Removing the pytree_node=False for the variables requiring gradient calculation
    for param in params_with_grad:
        for field in template_fields:
            if field.name == param:
                field.metadata = MappingProxyType({})
                break
    #Dynamically creating the class from the fields and passing it to struct that will itself pass it to dataclass, ouf...
    base_class = type("Params", (object, ), {field.name: field for field in template_fields})
    base_class.__annotations__ = {field.name: field.type for field in template_fields}
    return struct.dataclass(base_class)

@jax.jit
def get_vdrift(params):
    """
    Calculation of the electron drift velocity w.r.t temperature and electric
    field.
    References:
        - https://lar.bnl.gov/properties/trans.html (summary)
        - https://doi.org/10.1016/j.nima.2016.01.073 (parameterization)
        
    Args:
        params (Params): Parameters object containing the electric field (in :math:`kV/cm`) 
            and temperature (in :math:`K`) values.
        
    Returns:
        float: electron drift velocity in :math:`cm/\mu s`
    """
    a0, a1, a2, a3, a4, a5 = params.ELECTRON_MOBILITY_PARAMS

    num = a0 + a1 * params.eField + a2 * pow(params.eField, 1.5) + a3 * pow(params.eField, 2.5)
    denom = 1 + (a1 / a0) * params.eField + a4 * pow(params.eField, 2) + a5 * pow(params.eField, 3)
    temp_corr = pow(params.temperature / 89, -1.5)

    mu = num / denom * temp_corr / 1000 #* V / kV

    return mu*params.eField


def load_detector_properties(params_cls, detprop_file, pixel_file):
    """
    Loads detector properties and pixel geometry from YAML files and initializes a parameter class.
    This function reads detector and pixel layout properties from the provided YAML files,
    processes and converts the relevant parameters, and returns an instance of the given
    parameter class (`params_cls`) with the loaded and computed values.

    Args:
        params_cls (type): The class to instantiate with the loaded parameters. Must support
            initialization with keyword arguments matching the keys in `params_dict`.
        detprop_file (str): Path to the YAML file containing detector properties.
        pixel_file (str): Path to the YAML file containing pixel layout and geometry.

    Returns:
        params_cls: An instance of `params_cls` initialized with the loaded and computed parameters.

    Notes:
        - The function expects specific keys to be present in the YAML files, such as
            'tpc_centers', 'time_interval', 'drift_length', 'vdrift_static', 'eField', etc.
        - Pixel positions and borders are converted from millimeters to centimeters.
        - The function computes TPC and tile borders, pixel mappings, and other derived parameters.
        - Only parameters matching `params_cls.__match_args__` are passed to the class initializer.
    """

    params_dict = {
        "eField": 0.50,
        "Ab": 0.8,
        "kb": 0.0486,
        "vdrift": 0.1648,
        "lifetime": 2.2e3,
        "long_diff": 4.0e-6,
        "tran_diff": 8.8e-6,
        "shift_x": 0.,
        "shift_y": 0.,
        "shift_z": 0.,
        "box": 1,
        "birks": 2,
        "lArDensity": 1.38,
        "alpha": 0.93,
        "beta": 0.207,
        "MeVToElectrons": 4.237e+04,
        "temperature": 87.17,
        "max_active_pixels": 0,
        "max_radius": 0,
        "min_step_size": 0.001, #cm
        "time_max": 0,
        "time_window": 189.1, #us,
        "e_charge": 1.602e-19,
        "t_sampling": 0.1,
        "time_padding": 190,
        "time_interval": [0, 200], #us
        "drift_length": 0,
        "response_bin_size": 0.04434,
        "number_pix_neighbors": 1,
        "electron_sampling_resolution": 0.001,
        "signal_length": 150,
        "MAX_ADC_VALUES": 10,
        "DISCRIMINATION_THRESHOLD": 7e3,
        "ADC_HOLD_DELAY": 15,
        "CLOCK_CYCLE": 0.1,
        "GAIN": 4e-3,
        "V_CM": 288,
        "V_REF": 1300,
        "V_PEDESTAL": 580,
        "ADC_COUNTS": 2**8,
        "RESET_NOISE_CHARGE": 900,
        "UNCORRELATED_NOISE_CHARGE": 500,
        "ELECTRON_MOBILITY_PARAMS": (551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053),
        "size_margin": 2e-2,
        "tran_diff_bin_edges": jnp.linspace(-0.22, 0.22, 6),
        "diffusion_in_current_sim": True,
        "mc_diff": False,
        "tpc_centers": np.array([[0, 0, 0], [0, 0, 0]]), # Placeholder for TPC centers
    }

    mm2cm = 0.1
    params_dict['tpc_borders'] = np.zeros((0, 3, 2))
    params_dict['tile_borders'] = np.zeros((2,2))

    with open(detprop_file) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)

    for key, value in detprop.items():
        if key in params_dict:
            if isinstance(value, (list, np.ndarray)):
                params_dict[key] = np.array(value)
            else:
                params_dict[key] = value
        else:
            raise ValueError(f"Key '{key}' in detector properties file is not recognized.")

    params_dict['tpc_centers'][:, [2, 0]] = params_dict['tpc_centers'][:, [0, 2]]

    with open(pixel_file, 'r') as pf:
        tile_layout = yaml.load(pf, Loader=yaml.FullLoader)

    params_dict['pixel_pitch'] = tile_layout['pixel_pitch'] * mm2cm
    chip_channel_to_position = tile_layout['chip_channel_to_position']
    params_dict['pixel_connection_dict'] = {tuple(pix): (chip_channel//1000,chip_channel%1000) for chip_channel, pix in chip_channel_to_position.items()}
    params_dict['tile_chip_to_io'] = tile_layout['tile_chip_to_io']

    params_dict['xs'] = np.array(list(chip_channel_to_position.values()))[:,0] * params_dict['pixel_pitch']
    params_dict['ys'] = np.array(list(chip_channel_to_position.values()))[:,1] * params_dict['pixel_pitch']
    params_dict['tile_borders'][0] = [-(max(params_dict['xs'])+params_dict['pixel_pitch'])/2, (max(params_dict['xs'])+params_dict['pixel_pitch'])/2]
    params_dict['tile_borders'][1] = [-(max(params_dict['ys'])+params_dict['pixel_pitch'])/2, (max(params_dict['ys'])+params_dict['pixel_pitch'])/2]

    params_dict['tile_positions'] = np.array(list(tile_layout['tile_positions'].values())) * mm2cm
    params_dict['tile_orientations'] = np.array(list(tile_layout['tile_orientations'].values()))
    tpcs = np.unique(params_dict['tile_positions'][:,0])
    params_dict['tpc_borders'] = np.zeros((len(tpcs), 3, 2))

    for itpc,tpc_id in enumerate(tpcs):
        this_tpc_tile = params_dict['tile_positions'][params_dict['tile_positions'][:,0] == tpc_id]
        this_orientation = params_dict['tile_orientations'][params_dict['tile_positions'][:,0] == tpc_id]
        x_border = min(this_tpc_tile[:,2])+params_dict['tile_borders'][0][0]+params_dict['tpc_centers'][itpc][0], \
                    max(this_tpc_tile[:,2])+params_dict['tile_borders'][0][1]+params_dict['tpc_centers'][itpc][0]
        y_border = min(this_tpc_tile[:,1])+params_dict['tile_borders'][1][0]+params_dict['tpc_centers'][itpc][1], \
                    max(this_tpc_tile[:,1])+params_dict['tile_borders'][1][1]+params_dict['tpc_centers'][itpc][1]
        z_border = min(this_tpc_tile[:,0])+params_dict['tpc_centers'][itpc][2], \
                    max(this_tpc_tile[:,0])+detprop['drift_length']*this_orientation[:,0][0]+params_dict['tpc_centers'][itpc][2]

        params_dict['tpc_borders'][itpc] = (x_border, y_border, z_border)
    
    #: Number of pixels per axis
    # params_dict['n_pixels'] = len(np.unique(params_dict['xs']))*2, len(np.unique(params_dict['ys']))*4
    params_dict['n_pixels_x'] = len(np.unique(params_dict['xs']))*2
    params_dict['n_pixels_y'] = len(np.unique(params_dict['ys']))*4

    params_dict['n_pixels_per_tile'] = len(np.unique(params_dict['xs'])), len(np.unique(params_dict['ys']))

    params_dict['tile_map'] = ((7,5,3,1),(8,6,4,2)),((16,14,12,10),(15,13,11,9))
    params_dict['tpc_borders'] = jnp.asarray(params_dict['tpc_borders'])
    filtered_dict = {key: value for key, value in params_dict.items() if key in params_cls.__match_args__}
    return params_cls(**filtered_dict)