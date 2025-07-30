import jax.numpy as jnp
from jax import jit, random, debug
import jax
import numpy as np
from numpy.lib import recfunctions as rfn
from functools import partial
import logging
# from jax.experimental import checkify

# from larndsim.consts_jax import consts
from larndsim.detsim_jax import generate_electrons, get_pixels, id2pixel, accumulate_signals, accumulate_signals_parametrized, current_lut, get_pixel_coordinates, current_mc, apply_tran_diff, get_hit_z, pixel2id, get_bin_shifts, density_2d
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.fee_jax import get_adc_values, digitize, get_adc_values_average_noise, select_split_roi
from optimize.dataio import chop_tracks
from larndsim.consts_jax import get_vdrift

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    zMin = jnp.minimum(params.tpc_borders[:, 2, 1] - params.size_margin, params.tpc_borders[:, 2, 0] - params.size_margin)
    zMax = jnp.maximum(params.tpc_borders[:, 2, 1] + params.size_margin, params.tpc_borders[:, 2, 0] + params.size_margin)

    cond = tracks[:, fields.index("x")][..., None] >= params.tpc_borders[:, 0, 0][None, ...] - params.size_margin
    cond = jnp.logical_and(tracks[:, fields.index("x")][..., None] <= params.tpc_borders[:, 0, 1][None, ...] + params.size_margin, cond)
    cond = jnp.logical_and(tracks[:, fields.index("y")][..., None] >= params.tpc_borders[:, 1, 0][None, ...] - params.size_margin, cond)
    cond = jnp.logical_and(tracks[:, fields.index("y")][..., None] <= params.tpc_borders[:, 1, 1][None, ...] + params.size_margin, cond)
    cond = jnp.logical_and(tracks[:, fields.index("z")][..., None] >= zMin[None, ...], cond)
    cond = jnp.logical_and(tracks[:, fields.index("z")][..., None] <= zMax[None, ...], cond)

    mask = cond.sum(axis=-1) >= 1
    pixel_plane = cond.astype(int).argmax(axis=-1)
    tracks[:, fields.index('pixel_plane')] = pixel_plane
    return tracks


def pad_size(cur_size, tag, pad_threshold=0.05):
    """
    Pads the input size(s) to avoid frequent recompilations. Works for N-dimensional input sizes.
    Args:
        cur_size (int or tuple/list of ints): Current size(s) of the input.
        tag (str): Tag to identify the input type.
        pad_threshold (float): Padding threshold.
    Returns:
        tuple: Padded size(s).
    """
    global size_history_dict

    # Ensure cur_size is a tuple for uniformity
    if isinstance(cur_size, (int, np.integer)):
        cur_size = (cur_size,)
        return_val = lambda x: x[0]  # If the input was a single integer, return an integer
    else:
        cur_size = tuple(cur_size)
        return_val = lambda x: x  # If the input was a tuple, return a tuple

    if tag not in size_history_dict:
        size_history_dict[tag] = []
    size_history = size_history_dict[tag]

    # If an input with this shape has already been used, we are fine
    if cur_size in size_history:
        logger.debug(f"Input size {cur_size} already existing for {tag}.")
        return return_val(cur_size)

    # Otherwise, see if there is something available not too far in all dimensions
    for size in size_history:
        if all(cs <= s <= cs * (1 + pad_threshold) for cs, s in zip(cur_size, size)):
            logger.debug(f"Input size {cur_size} not existing for {tag}. Using close size of {size}")
            return return_val(size)

    # If nothing exists, create a new padded size
    new_size = tuple(int(cs * (1 + pad_threshold / 2) + 0.5) for cs in cur_size)
    size_history.append(new_size)
    size_history.sort()
    logger.debug(f"Input size {cur_size} not existing for {tag}. Creating new size of {new_size}")

    return return_val(new_size)

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
    #pIDs have the shape of (n_seg/n_electrons, n_pixels)
    pIDs = pIDs.ravel()

    # The "jit" padding for unique_pixels is set to -1
    # With the sort, the padding will be at the beginning
    # Therefore pix_renumbering 0 is the padding
    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs, method='sort')

    #Only getting the electrons for which the pixels are in the active region
    mask = (pix_renumbering > 0) & (pix_renumbering < (unique_pixels.size+1)) & (unique_pixels[pix_renumbering] == pIDs)

    return mask, pix_renumbering

@partial(jit, static_argnames=['fields'])
def simulate_signals(params, electrons, mask_indices, pix_renumbering, unique_pixels, response, rngkey, fields):
    """
    Simulates the signals from the drifted electrons and returns the ADC values, unique pixels, ticks, renumbering of the pixels, electrons and start ticks.
    Args:
        params (Any): Parameters of the simulation.
        electrons (jnp.ndarray): Drifted electrons as a JAX array.
        mask_indices (jnp.ndarray): Mask indices for the pixels.
        pix_renumbering (jnp.ndarray): Renumbering of the pixels.
        unique_pixels (jnp.ndarray): Unique pixel identifiers.
        response (jnp.ndarray): Response function for the simulation.
        rngkey (jax.random.PRNGKey): Random key for the simulation.
        fields (List[str]): List of field names corresponding to the electrons.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
            - adcs: ADC values.
            - pixel_x: X coordinates of the pixels.
            - pixel_y: Y coordinates of the pixels.
            - pixel_z: Z coordinates of the pixels.
            - ticks: Ticks corresponding to the ADC values.
            - hit_prob: Hit probabilities.
            - event: Event numbers.
            - unique_pixels: Unique pixel identifiers.
    """


    pix_renumbering = jnp.take(pix_renumbering, mask_indices, mode='fill', fill_value=0) # should we fill with 0? it's a valid index
    npix = (2*params.number_pix_neighbors + 1)**2
    elec_ids = mask_indices//npix
    electrons_renumbered = jnp.take(electrons, elec_ids, mode='fill', fill_value=0, axis=0)

    #Getting the pixel coordinates
    xpitch, ypitch, plane, event = id2pixel(params, unique_pixels)
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)

    #Getting the right indices for the currents
    t0, currents_idx = current_lut(params, response, electrons_renumbered, pixels_coord[pix_renumbering], fields)
    npixels = unique_pixels.shape[0]
    nticks_wf = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((npixels, nticks_wf))

    # start_ticks = response.shape[-1] - (t0/params.t_sampling).astype(int) - params.signal_length #Start tick from distance to the end of the cathode
    cathode_ticks = (t0/params.t_sampling).astype(int) #Start tick from distance to the end of the cathode
    response_cum = jnp.cumsum(response, axis=-1)
    wfs = accumulate_signals(wfs, currents_idx, electrons_renumbered[:, fields.index("n_electrons")], response, response_cum, pix_renumbering, cathode_ticks, params.signal_length)
    # The first time tick of wfs has the signal which would be out of range, but still have the response. It is meant to be discarded.
    integral, ticks = get_adc_values(params, wfs[:, 1:], rngkey)

    pixel_x = pixels_coord[:, 0]
    pixel_y = pixels_coord[:, 1]
    pixel_z  = get_hit_z(params, ticks.flatten(), jnp.repeat(plane, 10))

    adcs = digitize(params, integral)
    hit_prob = ticks < wfs.shape[1] - 2  # Assuming hit probability is based on whether ticks are within the waveform length

    adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels, nb_valid = parse_output(params, adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels)

    return adcs[:nb_valid], pixel_x[:nb_valid], pixel_y[:nb_valid], pixel_z[:nb_valid], ticks[:nb_valid], hit_prob[:nb_valid], event[:nb_valid], unique_pixels[:nb_valid]


@partial(jit, static_argnames=['fields'])
def simulate_signals_parametrized(params, electrons, pIDs, unique_pixels, rngkey, fields):
    """
    Simulates the signals from the drifted electrons and returns the ADC values, unique pixels, ticks, renumbering of the pixels, electrons and start ticks.
    Args:
        params (Any): Parameters of the simulation.
        electrons (jnp.ndarray): Drifted electrons as a JAX array.
        pIDs (jnp.ndarray): Pixel IDs for the electrons.
        unique_pixels (jnp.ndarray): Unique pixel identifiers.
        rngkey (jax.random.PRNGKey): Random key for the simulation.
        fields (List[str]): List of field names corresponding to the electrons.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
            - adcs: ADC values.
            - pixel_x: X coordinates of the pixels.
            - pixel_y: Y coordinates of the pixels.
            - pixel_z: Z coordinates of the pixels.
            - ticks: Ticks corresponding to the ADC values.
            - event: Event numbers.
            - unique_pixels: Unique pixel identifiers.
    """

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

    pixel_x, pixel_y, pixel_plane, event = id2pixel(params, unique_pixels)
    pixel_coords = get_pixel_coordinates(params, pixel_x, pixel_y, pixel_plane)
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]
    pixel_z  = get_hit_z(params, ticks.flatten(), jnp.repeat(pixel_plane, 10))
    ref_hit_prob = ticks < wfs.shape[1] - 2  # Assuming hit probability is based on whether ticks are within the waveform length

    return adcs, pixel_x, pixel_y, pixel_z, ticks, ref_hit_prob, event

from typing import Any, Tuple, List

def simulate_parametrized(params: Any, tracks: jnp.ndarray, fields: List[str], rngseed: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulates the signal from the drifted electrons and returns the ADC values, pixel coordinates, ticks, hit probabilities, event numbers, and unique pixel identifiers.
    Args:
        params (Any): Parameters of the simulation.
        tracks (jnp.ndarray): Tracks of the particles as a JAX array.
        fields (List[str]): List of field names corresponding to the tracks.
        rngseed (int): Random seed for the simulation.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
            - adcs: ADC values.
            - pixel_x: X coordinates of the pixels.
            - pixel_y: Y coordinates of the pixels.
            - pixel_z: Z coordinates of the pixels.
            - ticks: Ticks corresponding to the ADC values.
            - hit_prob: Hit probabilities.
            - event: Event numbers.
            - unique_pixels: Unique pixel identifiers.
    """

    master_key = jax.random.key(rngseed)
    rngkey1, rngkey2 = jax.random.split(master_key)
    electrons, pIDs = simulate_drift(params, tracks, fields, rngkey1)
    pIDs = pIDs.ravel()
    unique_pixels = jnp.unique(pIDs)
    padded_size = pad_size(unique_pixels.shape[0], "unique_pixels")

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_size - unique_pixels.shape[0]), mode='constant', constant_values=-1))

    adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event = simulate_signals_parametrized(params, electrons, pIDs, unique_pixels, rngkey2, fields)

    adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels, nb_valid = parse_output(params, adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels)

    return adcs[:nb_valid], pixel_x[:nb_valid], pixel_y[:nb_valid], pixel_z[:nb_valid], ticks[:nb_valid], hit_prob[:nb_valid], event[:nb_valid], unique_pixels[:nb_valid]

@partial(jit, static_argnames=['fields'])
def simulate_drift_new(params, tracks, fields):
    #Shifting tracks
    new_tracks = shift_tracks(params, tracks, fields)
    #Quenching and drifting
    new_tracks = quench(params, new_tracks, 2, fields)
    new_tracks = drift(params, new_tracks, fields)

    #Getting the pixels where the electrons are
    main_electrons = new_tracks

    #Doing long_diff stuffs, MC only for now
    # if params.mc_diff:
    #     rnd_pos = random.normal(rngkey, tracks.shape[0])*main_electrons[:, fields.index("long_diff")]
    #     main_electrons = main_electrons.at[:, fields.index('z')].set(main_electrons[:, fields.index('z')] + rnd_pos)

    bins_pitches = get_bin_shifts(params, main_electrons, fields)

    #Doing some tran diff fake stuff

    nb_tran_diff_bins = params.nb_tran_diff_bins
    nb_tran_diff_bins_sym = (nb_tran_diff_bins - 1) // 2
    bins = jnp.linspace(-nb_tran_diff_bins*params.pixel_pitch/params.nb_sampling_bins_per_pixel,
                        nb_tran_diff_bins*params.pixel_pitch/params.nb_sampling_bins_per_pixel,
                        nb_tran_diff_bins + 1)
    x0 = main_electrons[:, fields.index("x")] % (params.pixel_pitch/params.nb_sampling_bins_per_pixel)
    y0 = main_electrons[:, fields.index("y")] % (params.pixel_pitch/params.nb_sampling_bins_per_pixel)
    sigma = main_electrons[:, fields.index("tran_diff")]
    tran_diff_weights = density_2d(bins, x0, y0, sigma)

    nelectrons = (tran_diff_weights*main_electrons[:, fields.index("n_electrons")][:, None, None]).reshape(-1) #Multiplying by the number of electron

    z_cathode = jnp.take(params.tpc_borders, main_electrons[:, fields.index("pixel_plane")].astype(int), axis=0)[..., 2, 1]
    t0 = (jnp.abs(main_electrons[:, fields.index('z')] - z_cathode)) / get_vdrift(params) #Getting t0 as the equivalent time to cathode
    t0_after_diff = (jnp.ones((main_electrons.shape[0], nb_tran_diff_bins, nb_tran_diff_bins))*t0[:, None, None]).reshape(-1) #Broadcasting t0 to the shape of the tran_diff_weights

    #Need to convert long_diff into a tick number
    long_diff = main_electrons[:, fields.index("long_diff")]/ get_vdrift(params)/ params.t_sampling

    long_diff = long_diff[:, None, None].repeat(nb_tran_diff_bins**2, axis=-1).reshape(-1) #Broadcasting long_diff to the shape of the tran_diff_weights


    bin_shifts = jnp.mgrid[-nb_tran_diff_bins_sym:nb_tran_diff_bins_sym+1, -nb_tran_diff_bins_sym:nb_tran_diff_bins_sym+1]

    bins_pitches_new = (bins_pitches[..., jnp.newaxis, jnp.newaxis] + bin_shifts).swapaxes(1, -1)
    pix_pitches = bins_pitches_new // params.nb_sampling_bins_per_pixel
    pixels = pixel2id(params, pix_pitches[..., 0], pix_pitches[..., 1], main_electrons[:, fields.index("pixel_plane")][:, None, None].astype(int), main_electrons[:, fields.index("eventID")][:, None, None].astype(int))
    main_pixels = pixels[:, nb_tran_diff_bins_sym, nb_tran_diff_bins_sym] #Getting the main pixel, not considering pixels that would only see some diffusion charge
    currents_idx = jnp.abs(bins_pitches_new % params.nb_sampling_bins_per_pixel - params.nb_sampling_bins_per_pixel//2 + 0.5).reshape(-1, 2).astype(int)


    #########################################################
    #################Adding neighbors########################
    #########################################################
    pix_grid = jnp.mgrid[-params.number_pix_neighbors:params.number_pix_neighbors+1, -params.number_pix_neighbors:params.number_pix_neighbors+1]
    bin_shifts_neighbors = pix_grid * params.nb_sampling_bins_per_pixel
    currents_idx_neigh = jnp.moveaxis(jnp.abs(bins_pitches[:, :, None, None] % params.nb_sampling_bins_per_pixel - params.nb_sampling_bins_per_pixel//2 + 0.5 - bin_shifts_neighbors).astype(int), 1, -1).reshape(-1, 2)

    principal_pitches = pix_pitches[:, nb_tran_diff_bins_sym, nb_tran_diff_bins_sym, :] #Getting the main pixel
    new_pitches = jnp.moveaxis((principal_pitches[:, :, None, None] + pix_grid), 1, -1)
    pIDs = pixel2id(params, new_pitches[..., 0], new_pitches[..., 1], main_electrons[:, fields.index("pixel_plane")][:, None, None].astype(int), main_electrons[:, fields.index("eventID")][:, None, None].astype(int))
    pIDs_neigh = pIDs.at[:, params.number_pix_neighbors, params.number_pix_neighbors].set(-999) # Getting rid of the main pixel, which is already in the main_pixels
    nelectrons_neigh = main_electrons[:, fields.index("n_electrons")]
    t0_neigh = t0
    return main_pixels, pixels, nelectrons, t0_after_diff, long_diff, currents_idx, pIDs_neigh ,currents_idx_neigh, nelectrons_neigh, t0_neigh

@jit
def simulate_signals_new(params, unique_pixels, pixels, t0_after_diff, response_template, nelectrons, long_diff, currents_idx, nelectrons_neigh, mask_indices, pix_renumbering_neigh, t0_neigh, currents_idx_neigh):
    pix_renumbering = jnp.searchsorted(unique_pixels, pixels.ravel(), method='sort')
    #Getting the right indices for the currents
   
    Npixels = unique_pixels.shape[0]
    Nticks = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((Npixels, Nticks))
    wfs = wfs.ravel()

    cathode_ticks = (t0_after_diff/params.t_sampling).astype(int) #Start tick from distance to the end of the cathode
    response_cum = jnp.cumsum(response_template, axis=-1)

    # Compute indices for updating wfs, taking into account start_ticks
    start_ticks = response_template.shape[-1] - params.signal_length - cathode_ticks
    time_ticks = start_ticks[..., None] + jnp.arange(params.signal_length)

    time_ticks = jnp.where((time_ticks <= 0 ) | (time_ticks >= Nticks - 1), 0, time_ticks+1) # it should be start_ticks +1 in theory but we cheat by putting the cumsum in the garbage too when strarting at 0 to mimic the expected behavior

    start_indices = pix_renumbering * Nticks

    end_indices = start_indices[..., None] + time_ticks

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    charge = (nelectrons[:, None]*jnp.ones((1, params.signal_length), dtype=jnp.float32)).reshape(-1) #Broadcasting the charge to the shape of the signal

    Ntemplates, Nx, Ny, Nt = response_template.shape

    template_values = params.long_diff_template

    idx = jnp.searchsorted(template_values, long_diff)
    # Should make sure in first place that template values go from 0 to > max possible diff value

    idx = jnp.clip(idx, 1, template_values.shape[0] - 2) #Ensuring idx is within bounds

    x0 = template_values[idx - 1]
    x1 = template_values[idx]
    x2 = template_values[idx + 1]

    # Quadratic interpolation coefficients
    a = (long_diff - x1) * (long_diff - x2) / ((x0 - x1) * (x0 - x2))
    b = (long_diff - x0) * (long_diff - x2) / ((x1 - x0) * (x1 - x2))
    c = (long_diff - x0) * (long_diff - x1) / ((x2 - x0) * (x2 - x1))

    a = (a[:, None]*jnp.ones((1, params.signal_length), dtype=jnp.float32)).reshape(-1) #Broadcasting the coefficients to the shape of the signal
    b = (b[:, None]*jnp.ones((1, params.signal_length), dtype=jnp.float32)).reshape(-1) #Broadcasting the coefficients to the shape of the signal
    c = (c[:, None]*jnp.ones((1, params.signal_length), dtype=jnp.float32)).reshape(-1) #Broadcasting the coefficients to the shape of the signal

    signal_indices = jnp.ravel((idx[..., None]*Nx*Ny + currents_idx[..., 0, None]*Ny + currents_idx[..., 1, None])*Nt + jnp.arange(response_template.shape[-1] - params.signal_length, response_template.shape[-1]))

    # Update wfs with accumulated signals
    wfs = wfs.at[(flat_indices,)].add((response_template.take(signal_indices))*charge*b)
    wfs = wfs.at[(flat_indices,)].add((response_template.take(signal_indices - Nx*Ny*Nt))*charge*a)
    wfs = wfs.at[(flat_indices,)].add((response_template.take(signal_indices + Nx*Ny*Nt))*charge*c)

    #Now correct for the missed ticks at the beginning
    integrated_start = response_cum.take(jnp.ravel((currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt + response_template.shape[-1] - params.signal_length))
    real_start = response_cum.take(jnp.ravel((currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt + cathode_ticks))
    difference = (integrated_start - real_start)*nelectrons

    start_ticks = jnp.where((start_ticks <= 0 ) | (start_ticks >= Nticks - 1), 0, start_ticks) + pix_renumbering * Nticks
    wfs = wfs.at[start_ticks].add(difference)

    wfs = wfs.reshape((Npixels, Nticks))


    pix_renumbering_neigh = jnp.take(pix_renumbering_neigh, mask_indices, mode='fill', fill_value=0)
    npix = (2*params.number_pix_neighbors + 1)**2
    elec_ids = mask_indices//npix
    nelectrons_neigh = jnp.take(nelectrons_neigh, elec_ids, mode='fill', fill_value=0)
    currents_idx_neigh = jnp.take(currents_idx_neigh, mask_indices, mode='fill', fill_value=0, axis=0)

    t0_neighbors = jnp.take(t0_neigh, elec_ids, mode='fill', fill_value=0)

    cathode_ticks_neigh = (t0_neighbors/params.t_sampling).astype(int) #Start tick from distance to the end of the cathode
    #WARNING: Assuming here that response_template[0] corresponds to no diff
    wfs = accumulate_signals(wfs, currents_idx_neigh, nelectrons_neigh, response_template[0], response_cum, pix_renumbering_neigh, cathode_ticks_neigh, params.signal_length)

    return wfs

@jit
def parse_output(params, adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels):
    mask = hit_prob > params.hit_prob_threshold

    max_length = mask.shape[0] * mask.shape[1]

    adcs_output = jnp.zeros((max_length,), dtype=adcs.dtype)
    pixel_x_output = jnp.zeros((max_length,), dtype=pixel_x.dtype)
    pixel_y_output = jnp.zeros((max_length,), dtype=pixel_y.dtype)
    pixel_z_output = jnp.zeros((max_length,), dtype=pixel_z.dtype)
    ticks_output = jnp.zeros((max_length,), dtype=ticks.dtype)
    hit_prob_output = jnp.zeros((max_length,), dtype=hit_prob.dtype)
    event_output = jnp.zeros((max_length,), dtype=event.dtype)
    unique_pixels_output = jnp.zeros((max_length,), dtype=unique_pixels.dtype)

    flat_mask = mask.flatten()

    new_index = jnp.where(flat_mask, jnp.cumsum(flat_mask), max_length - 1)
    adcs_output = adcs_output.at[new_index].set(adcs.flatten())
    pixel_x_output = pixel_x_output.at[new_index].set(jnp.repeat(pixel_x, mask.shape[1]))
    pixel_y_output = pixel_y_output.at[new_index].set(jnp.repeat(pixel_y, mask.shape[1]))
    pixel_z_output = pixel_z_output.at[new_index].set(pixel_z.flatten())
    ticks_output = ticks_output.at[new_index].set(ticks.flatten())
    hit_prob_output = hit_prob_output.at[new_index].set(hit_prob.flatten())
    event_output = event_output.at[new_index].set(jnp.repeat(event, mask.shape[1]))
    unique_pixels_output = unique_pixels_output.at[new_index].set(jnp.repeat(unique_pixels, mask.shape[1]))
    nb_valid = jnp.sum(flat_mask)

    return adcs_output, pixel_x_output, pixel_y_output, pixel_z_output, ticks_output, hit_prob_output, event_output, unique_pixels_output, nb_valid

@jit
def fee_sim_from_split(params, small_rois, large_rois, small_roi_idx, large_roi_idx, small_roi_start, large_roi_start, unique_pixels, max_tick_nb):
    integral_small, ticks_small, no_hit_prob_small = get_adc_values_average_noise(params, small_rois)
    integral_large, ticks_large, no_hit_prob_large = get_adc_values_average_noise(params, large_rois)

    integral = jnp.zeros((unique_pixels.shape[0], integral_small.shape[1]))

    integral = integral.at[small_roi_idx, :].set(integral_small[:small_roi_idx.shape[0], :])
    integral = integral.at[large_roi_idx, :].set(integral_large[:large_roi_idx.shape[0], :])

    ticks = jnp.zeros((unique_pixels.shape[0], ticks_small.shape[1]))
    ticks = ticks.at[small_roi_idx, :].set(ticks_small[:small_roi_idx.shape[0], :] + small_roi_start[:, None])
    ticks = ticks.at[large_roi_idx, :].set(ticks_large[:large_roi_idx.shape[0], :] + large_roi_start[:, None])
    ticks = jnp.minimum(ticks, max_tick_nb)  # Ensure ticks do not exceed waveform length

    no_prob = jnp.zeros((unique_pixels.shape[0], no_hit_prob_small.shape[1]))
    no_prob = no_prob.at[small_roi_idx, :].set(no_hit_prob_small[:small_roi_idx.shape[0], :])
    no_prob = no_prob.at[large_roi_idx, :].set(no_hit_prob_large[:large_roi_idx.shape[0], :])
    hit_prob = 1 - no_prob

    return integral, ticks, hit_prob

def simulate_new(params, response_template, tracks, fields, rngseed=None):
    """
    Simulates the signal from the drifted electrons and returns the ADC values, unique pixels, ticks, renumbering of the pixels, electrons and start ticks.
    Args:
        params (Any): Parameters of the simulation.
        response (jnp.ndarray): Response function.
        tracks (jnp.ndarray): Tracks of the particles as a JAX array.
        fields (List[str]): List of field names corresponding to the tracks.
        rngseed (int): Random seed for the simulation.
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]: 
            - adcs: ADC values.
            - pixel_x: X coordinates of the pixels.
            - pixel_y: Y coordinates of the pixels.
            - pixel_z: Z coordinates of the pixels.
            - ticks: Ticks of the signals.
            - hit_prob: Probability of a hit in the pixel.
            - event: Event IDs.
            - unique_pixels: Unique pixels.

    """

    main_pixels, pixels, nelectrons, t0_after_diff, long_diff, currents_idx, pIDs_neigh, currents_idx_neigh, nelectrons_neigh, t0_neigh = simulate_drift_new(params, tracks, fields)

    ################################################
    ################################################

    #Sorting the pixels and getting the unique ones
    unique_pixels = jnp.unique(main_pixels.ravel())
    padded_unique, padded_mask = pad_size((unique_pixels.shape[0], pIDs_neigh.size), "unique_pixels", 0.2)

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_unique - unique_pixels.shape[0]), mode='constant', constant_values=-1))

    mask, pix_renumbering_neigh = get_renumbering(pIDs_neigh, unique_pixels)
    mask_indices = jnp.nonzero(mask)[0]
    # padded_size = pad_size(mask_indices.shape[0], "pix_renumbering", 0.2)
    mask_indices = jnp.pad(mask_indices, (0, padded_mask - mask_indices.shape[0]), mode='constant', constant_values=-1)

    ###############################################
    ###############################################

    wfs = simulate_signals_new(params, unique_pixels, pixels, t0_after_diff, response_template, nelectrons, long_diff, currents_idx, nelectrons_neigh, mask_indices, pix_renumbering_neigh, t0_neigh, currents_idx_neigh)

    ###############################################
    ###############################################

    # integral, ticks = get_adc_values(params, wfs[:, 1:], rngkey2)

    if rngseed is not None:
        integral, ticks = get_adc_values(params, wfs[:, 1:], jax.random.key(rngseed))
        hit_prob = ticks < wfs.shape[1] - 2  # Assuming hit probability is based on whether ticks are within the waveform length
    else:
        small_rois, small_roi_start, small_roi_idx, large_rois, large_roi_start, large_roi_idx = select_split_roi(params, wfs[:, 1:])
        padded_small_nb = pad_size(small_rois.shape[0], "wfs_roi", 0.2) #We already enforced the length of the samllest ones, no need for padding
        padded_large_nb, padded_large_length = pad_size((large_rois.shape[0], large_rois.shape[1]), "wfs_roi", 0.2)
        small_rois = jnp.pad(small_rois, ((0, padded_small_nb - small_rois.shape[0]), (0, 0)), mode='constant', constant_values=0)
        large_rois = jnp.pad(large_rois, ((0, padded_large_nb - large_rois.shape[0]), (0, padded_large_length - large_rois.shape[1])), mode='constant', constant_values=0)
        small_roi_idx = jnp.pad(small_roi_idx, (0, padded_small_nb - small_roi_idx.shape[0]), mode='constant', constant_values=wfs.shape[0] - 1)  # Fill with the last index
        large_roi_idx = jnp.pad(large_roi_idx, (0, padded_large_nb - large_roi_idx.shape[0]), mode='constant', constant_values=wfs.shape[0] - 1)
        small_roi_start = jnp.pad(small_roi_start, (0, padded_small_nb - small_roi_start.shape[0]), mode='constant', constant_values=wfs.shape[1] - 2)
        large_roi_start = jnp.pad(large_roi_start, (0, padded_large_nb - large_roi_start.shape[0]), mode='constant', constant_values=wfs.shape[1] - 2)

        integral, ticks, hit_prob = fee_sim_from_split(params, small_rois, large_rois, small_roi_idx, large_roi_idx, small_roi_start, large_roi_start, unique_pixels, wfs.shape[1] - 2)
        
        # integral, ticks, no_hit_prob = get_adc_values_average_noise(params, wfs[:, 1:])
        # hit_prob = 1 - no_hit_prob

    adcs = digitize(params, integral)

    pixel_x, pixel_y, pixel_plane, event = id2pixel(params, unique_pixels)
    pixel_coords = get_pixel_coordinates(params, pixel_x, pixel_y, pixel_plane)
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]
    pixel_z  = get_hit_z(params, ticks.flatten(), jnp.repeat(pixel_plane, 10))

    adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels, nb_valid = parse_output(params, adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels)

    return adcs[:nb_valid], pixel_x[:nb_valid], pixel_y[:nb_valid], pixel_z[:nb_valid], ticks[:nb_valid], hit_prob[:nb_valid], event[:nb_valid], unique_pixels[:nb_valid]


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
