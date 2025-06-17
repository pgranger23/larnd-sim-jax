import jax.numpy as jnp
from jax import jit
import jax
import numpy as np
from numpy.lib import recfunctions as rfn
from functools import partial
import logging
# from jax.experimental import checkify

# from larndsim.consts_jax import consts
from larndsim.detsim_jax import generate_electrons, get_pixels, id2pixel, accumulate_signals, accumulate_signals_parametrized, current_lut, get_pixel_coordinates, current_mc, apply_tran_diff
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.fee_jax import get_adc_values, digitize
from optimize.dataio import chop_tracks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

size_history_dict = {}

def load_data(fname, invert_xz=True):
    import h5py
    with h5py.File(fname, 'r') as f:
        tracks = np.array(f['segments'])
    if invert_xz:
        x_start = np.copy(tracks['x_start'] )
        x_end = np.copy(tracks['x_end'])
        x = np.copy(tracks['x'])

        tracks['x_start'] = np.copy(tracks['z_start'])
        tracks['x_end'] = np.copy(tracks['z_end'])
        tracks['x'] = np.copy(tracks['z'])

        tracks['z_start'] = x_start
        tracks['z_end'] = x_end
        tracks['z'] = x

    selected_tracks = tracks
    dtype = selected_tracks.dtype
    return rfn.structured_to_unstructured(tracks, copy=True, dtype=np.float32), dtype

def set_pixel_plane(params, tracks, fields):
    zMin = np.minimum(params.tpc_borders[:, 2, 1] - params.size_margin, params.tpc_borders[:, 2, 0] - params.size_margin)
    zMax = np.maximum(params.tpc_borders[:, 2, 1] + params.size_margin, params.tpc_borders[:, 2, 0] + params.size_margin)

    cond = tracks[:, fields.index("x")][..., None] >= params.tpc_borders[:, 0, 0][None, ...] - params.size_margin
    cond = np.logical_and(tracks[:, fields.index("x")][..., None] <= params.tpc_borders[:, 0, 1][None, ...] + params.size_margin, cond)
    cond = np.logical_and(tracks[:, fields.index("y")][..., None] >= params.tpc_borders[:, 1, 0][None, ...] - params.size_margin, cond)
    cond = np.logical_and(tracks[:, fields.index("y")][..., None] <= params.tpc_borders[:, 1, 1][None, ...] + params.size_margin, cond)
    cond = np.logical_and(tracks[:, fields.index("z")][..., None] >= zMin[None, ...], cond)
    cond = np.logical_and(tracks[:, fields.index("z")][..., None] <= zMax[None, ...], cond)

    mask = cond.sum(axis=-1) >= 1
    pixel_plane = cond.astype(int).argmax(axis=-1)
    tracks[:, fields.index('pixel_plane')] = pixel_plane
    return tracks

def pad_size(cur_size, tag, pad_threshold=0.05):
    global size_history_dict

    if tag not in size_history_dict:
        size_history_dict[tag] = []
    size_history = size_history_dict[tag]

    #If an input with this shape has already been used, we are fine
    if cur_size in size_history:
        logger.debug(f"Input size {cur_size} already existing.")
        return cur_size
    #Otherwise we want to see if there is something available not too far
    for size in size_history:
        if cur_size <= size <= cur_size*(1 + pad_threshold):
            logger.debug(f"Input size {cur_size} not existing. Using close size of {size}")
            return size
    #If nothing exists we will have to recompile. We still use some padding to try limiting further recompilations if the size is reduced
    new_size = int(cur_size*(1 + pad_threshold/2) + 0.5)
    size_history.append(new_size)
    size_history.sort()
    logger.debug(f"Input size {cur_size} not existing. Creating new size of {new_size}")
    return new_size

def get_size_history():
    return size_history_dict

@partial(jit, static_argnames=['fields'])
def shift_tracks(params, tracks, fields):
    shifted_tracks = tracks.at[:, fields.index("x_start")].subtract(params.shift_x)
    shifted_tracks = shifted_tracks.at[:, fields.index("x_end")].subtract(params.shift_x)
    shifted_tracks = shifted_tracks.at[:, fields.index("x")].subtract(params.shift_x)
    shifted_tracks = shifted_tracks.at[:, fields.index("y_start")].subtract(params.shift_y)
    shifted_tracks = shifted_tracks.at[:, fields.index("y_end")].subtract(params.shift_y)
    shifted_tracks = shifted_tracks.at[:, fields.index("y")].subtract(params.shift_y)
    shifted_tracks = shifted_tracks.at[:, fields.index("z_start")].subtract(params.shift_z)
    shifted_tracks = shifted_tracks.at[:, fields.index("z_end")].subtract(params.shift_z)
    shifted_tracks = shifted_tracks.at[:, fields.index("z")].subtract(params.shift_z)
    return shifted_tracks


@partial(jit, static_argnames=['fields'])
def simulate_drift(params, tracks, fields, rngkey):
    #Shifting tracks
    new_tracks = shift_tracks(params, tracks, fields)
    #Quenching and drifting
    new_tracks = quench(params, new_tracks, 2, fields)
    new_tracks = drift(params, new_tracks, fields)

    #Simulating the electron generation according to the diffusion coefficients

    if params.mc_diff:
        electrons = generate_electrons(new_tracks, fields, rngkey, not params.diffusion_in_current_sim)
    else:
        electrons = apply_tran_diff(params, new_tracks, fields)
    #Getting the pixels where the electrons are
    pIDs = get_pixels(params, electrons, fields)

    return electrons, pIDs

@jit
def get_renumbering(pIDs, unique_pixels):
    #Getting the renumbering of the pixels
    pIDs = pIDs.ravel()
    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs, method='sort')

    #Only getting the electrons for which the pixels are in the active region
    mask = (pix_renumbering < unique_pixels.size) & (unique_pixels[pix_renumbering] == pIDs)
    return mask, pix_renumbering

@partial(jit, static_argnames=['fields'])
def simulate_signals(params, electrons, mask_indices, pix_renumbering, unique_pixels, response, rngkey, fields):
    pix_renumbering = jnp.take(pix_renumbering, mask_indices, mode='fill', fill_value=0)
    npix = (2*params.number_pix_neighbors + 1)**2
    elec_ids = mask_indices//npix
    electrons_renumbered = jnp.take(electrons, elec_ids, mode='fill', fill_value=0, axis=0)
    #Getting the pixel coordinates
    xpitch, ypitch, plane, eid = id2pixel(params, unique_pixels)
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)
    #Getting the right indices for the currents
    t0, currents_idx = current_lut(params, response, electrons_renumbered, pixels_coord[pix_renumbering], fields)
    npixels = unique_pixels.shape[0]
    nticks_wf = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((npixels, nticks_wf))

    start_ticks = (t0/params.t_sampling + 0.5).astype(int) - params.signal_length
    
    wfs = accumulate_signals(wfs, currents_idx, electrons_renumbered[:, fields.index("n_electrons")], response, pix_renumbering, start_ticks, params.signal_length)
    integral, ticks = get_adc_values(params, wfs[:, 1:], rngkey)

    adcs = digitize(params, integral)
    return adcs, ticks, start_ticks, wfs[:, 1:]

@partial(jit, static_argnames=['fields'])
def simulate_signals_parametrized(params, electrons, pIDs, unique_pixels, rngkey, fields):
    xpitch, ypitch, plane, eid = id2pixel(params, pIDs)
    
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)
    t0, signals = current_mc(params, electrons, pixels_coord, fields)

    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)

    nticks_wf = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((unique_pixels.shape[0], nticks_wf))

    #TODO: Check what should be the correct implementation: when is the actual trigger reference time?
    start_ticks = t0 - signals.shape[1]

    wfs = accumulate_signals_parametrized(wfs, signals, pix_renumbering, start_ticks)
    integral, ticks = get_adc_values(params, wfs[:, 1:], rngkey)
    adcs = digitize(params, integral)
    return adcs, ticks, pix_renumbering, start_ticks, wfs[:, 1:]

from typing import Any, Tuple, List

def simulate_parametrized(params: Any, tracks: jnp.ndarray, fields: List[str], rngseed: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulates the signal from the drifted electrons and returns the ADC values, unique pixels, ticks, renumbering of the pixels, electrons and start ticks.
    Args:
        params (Any): Parameters of the simulation.
        tracks (jnp.ndarray): Tracks of the particles as a JAX array.
        fields (List[str]): List of field names corresponding to the tracks.
        rngseed (int): Random seed for the simulation.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
            - adcs: ADC values.
            - unique_pixels: Unique pixels.
            - ticks: Ticks of the signals.
            - pix_renumbering: Renumbering of the pixels.
            - electrons: Electrons generated.
            - start_ticks: Start ticks of the signals.
            - wfs: Waveforms of the signals.

    """

    master_key = jax.random.key(rngseed)
    rngkey1, rngkey2 = jax.random.split(master_key)
    electrons, pIDs = simulate_drift(params, tracks, fields, rngkey1)
    pIDs = pIDs.ravel()
    unique_pixels = jnp.unique(pIDs)
    padded_size = pad_size(unique_pixels.shape[0], "unique_pixels")

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_size - unique_pixels.shape[0]), mode='constant', constant_values=-1))

    adcs, ticks, pix_renumbering, start_ticks, wfs = simulate_signals_parametrized(params, electrons, pIDs, unique_pixels, rngkey2, fields)
    return adcs, unique_pixels, ticks, pix_renumbering, electrons, start_ticks, wfs

def simulate(params, response, tracks, fields, rngseed = 0):
    master_key = jax.random.key(rngseed)
    rngkey1, rngkey2 = jax.random.split(master_key)
    electrons, pIDs = simulate_drift(params, tracks, fields, rngkey1)

    main_pixels = pIDs[:, 2*params.number_pix_neighbors*(params.number_pix_neighbors+1)] #Getting the main pixel
    #Sorting the pixels and getting the unique ones
    unique_pixels = jnp.unique(main_pixels.ravel())
    padded_size = pad_size(unique_pixels.shape[0], "unique_pixels")

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_size - unique_pixels.shape[0]), mode='constant', constant_values=-1))

    mask, pix_renumbering = get_renumbering(pIDs, unique_pixels)
    mask_indices = jnp.nonzero(mask)[0]

    padded_size = pad_size(mask_indices.shape[0], "pix_renumbering")
    mask_indices = jnp.pad(mask_indices, (0, padded_size - mask_indices.shape[0]), mode='constant', constant_values=-1)

    # errors = checkify.user_checks | checkify.index_checks | checkify.float_checks
    # checked_f = checkify.checkify(accumulate_signals, errors=errors)
    # err, wfs = checked_f(wfs, currents_idx, electrons[:, fields.index("n_electrons")], response, pix_renumbering, start_ticks - earliest_tick, params.signal_length)
    # err.throw()

    adcs, ticks, start_ticks, wfs =  simulate_signals(params, electrons, mask_indices, pix_renumbering, unique_pixels, response, rngkey2, fields)
    return adcs, unique_pixels, ticks, pix_renumbering, electrons, start_ticks, wfs

def prepare_tracks(params, tracks_file, invert_xz=True):
    tracks, dtype = load_data(tracks_file, invert_xz)
    fields = dtype.names

    tracks = set_pixel_plane(params, tracks, fields)
    original_tracks = tracks.copy()
    tracks = chop_tracks(tracks, fields, params.electron_sampling_resolution)
    tracks = jnp.array(tracks)

    return tracks, fields, original_tracks

def backtrack_electrons(unique_pixels, pixId, hit_t0, pix_renumbering, electrons, start_ticks):
    #For given hit, returning the list of electrons that deposited charge in the pixel during this time
    pix_idx = jnp.searchsorted(unique_pixels, pixId, side='left')
    #If pixel not found, return empty array
    if pix_idx == unique_pixels.shape[0] or unique_pixels[pix_idx] != pixId:
        return jnp.array([]), jnp.array([])
    matching_electrons = electrons[pix_renumbering == pix_idx]
    #Getting the electrons that are in the time window
    t0 = start_ticks[pix_renumbering == pix_idx]
    matching_electrons = matching_electrons[jnp.logical_and(t0 >= hit_t0 - 50, t0 <= hit_t0 + 50)] #TODO: Put sensible values for time window
    return matching_electrons