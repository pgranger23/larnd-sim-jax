import jax.numpy as jnp
from jax import jit, random, debug
import jax
import numpy as np
from numpy.lib import recfunctions as rfn
from functools import partial
import logging
# from jax.experimental import checkify

# from larndsim.consts_jax import consts
from larndsim.detsim_jax import generate_electrons, get_pixels, id2pixel, accumulate_signals, accumulate_signals_parametrized, current_lut, get_pixel_coordinates, current_mc, apply_tran_diff, get_hit_z, pixel2id, get_bin_shifts
from larndsim.quenching_jax import quench
from larndsim.drifting_jax import drift
from larndsim.fee_jax import get_adc_values, digitize, get_adc_values_average_noise_vmap
from optimize.dataio import chop_tracks
from larndsim.consts_jax import get_vdrift, RecombinationMode

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
    new_tracks = quench(params, new_tracks, fields)
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
def simulate_signals(params, unique_pixels, pixels, t0_after_diff, response_template,
                             nelectrons, long_diff, tran_diff, currents_idx, currents_frac,
                             nelectrons_neigh, pix_renumbering_neigh, t0_neigh,
                             currents_idx_neigh, currents_frac_neigh):

    Npixels = unique_pixels.shape[0]
    Nticks = int(params.time_interval[1] / params.t_sampling) + 1
    n_long, n_tran, Nx, Ny, Nt_sig = response_template.shape  # 5D: (n_long, n_tran, Nx, Ny, sig_len)
    sig_len = params.signal_length
    # response_template: 5D, trimmed to last sig_len ticks by load_lut.
    # response_cum: 3D (Nx, Ny, Nt_cum), zero-diffusion only (no σ_T, no σ_L).
    response_cum = params.response_cum   # (Nx, Ny, Nt_cum)
    Nt_cum = response_cum.shape[-1]      # = original_Nt - sig_len + 1
    long_stride = n_tran * Nx * Ny * Nt_sig  # flat stride for one longitudinal-template step

    # --- 1. PREPARE MAIN PIXEL DATA (DIFFERENTIABLE TIME SHIFT) ---
    pix_renum = jnp.searchsorted(unique_pixels, pixels.ravel(), method='sort')
    mask = (unique_pixels[pix_renum] == pixels.ravel())
    pix_renum = jnp.where(mask, pix_renum, -1)

    # Extract continuous tick and fractional remainder for the gradient
    float_ticks_main = t0_after_diff / params.t_sampling
    cathode_ticks_main = jnp.clip(jnp.floor(float_ticks_main).astype(int), 0, Nt_cum - 1)
    frac_main = float_ticks_main - cathode_ticks_main  # gradient flows through here

    # ── Longitudinal: quadratic interpolation (unchanged) ────────────────────
    template_vals = params.long_diff_template
    idx = jnp.clip(jnp.searchsorted(template_vals, long_diff), 1, template_vals.shape[0] - 2)
    x0_l, x1_l, x2_l = template_vals[idx - 1], template_vals[idx], template_vals[idx + 1]
    a = (long_diff - x1_l) * (long_diff - x2_l) / ((x0_l - x1_l) * (x0_l - x2_l))
    b = (long_diff - x0_l) * (long_diff - x2_l) / ((x1_l - x0_l) * (x1_l - x2_l))
    c = (long_diff - x0_l) * (long_diff - x1_l) / ((x2_l - x0_l) * (x2_l - x1_l))

    # ── Transverse σ_T: quadratic interpolation ───────────────────────────────
    tran_diff_vals = params.tran_diff_template   # (n_tran,)
    ti = jnp.clip(jnp.searchsorted(tran_diff_vals, tran_diff), 1, n_tran - 2)  # (N,)
    xt0, xt1, xt2 = tran_diff_vals[ti - 1], tran_diff_vals[ti], tran_diff_vals[ti + 1]
    at = (tran_diff - xt1) * (tran_diff - xt2) / ((xt0 - xt1) * (xt0 - xt2))
    bt = (tran_diff - xt0) * (tran_diff - xt2) / ((xt1 - xt0) * (xt1 - xt2))
    ct = (tran_diff - xt0) * (tran_diff - xt1) / ((xt2 - xt0) * (xt2 - xt1))

    # ── Spatial position: Catmull-Rom bicubic interpolation ───────────────────
    def _cr_wts(frac):
        """Catmull-Rom weights for frac ∈ [0, 1).  Returns (N, 4)."""
        t2, t3 = frac ** 2, frac ** 3
        w0 = -0.5 * t3 +       t2 - 0.5 * frac
        w1 =  1.5 * t3 - 2.5 * t2 + 1.0
        w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * frac
        w3 =  0.5 * t3 - 0.5 * t2
        return jnp.stack([w0, w1, w2, w3], axis=-1)   # (N, 4)

    wx = _cr_wts(currents_frac[:, 0])   # (N, 4)
    wy = _cr_wts(currents_frac[:, 1])   # (N, 4)
    # 4 clamped x/y indices for each electron: (N, 4)
    ixs = jnp.clip(currents_idx[:, 0:1] + jnp.array([-1, 0, 1, 2]), 0, Nx - 1)
    iys = jnp.clip(currents_idx[:, 1:2] + jnp.array([-1, 0, 1, 2]), 0, Ny - 1)

    # Signal Indices (Main) - Split across adjacent ticks
    start_ticks_0 = Nt_cum - 1 - cathode_ticks_main
    start_ticks_1 = start_ticks_0 - 1

    time_ticks_0 = start_ticks_0[..., None] + jnp.arange(sig_len)
    time_ticks_1 = start_ticks_1[..., None] + jnp.arange(sig_len)

    time_ticks_0 = jnp.where((time_ticks_0 <= 0) | (time_ticks_0 >= Nticks - 1), 0, time_ticks_0 + 1)
    time_ticks_1 = jnp.where((time_ticks_1 <= 0) | (time_ticks_1 >= Nticks - 1), 0, time_ticks_1 + 1)

    main_flat_indices_0 = (pix_renum[:, None] * Nticks + time_ticks_0).ravel()
    main_flat_indices_1 = (pix_renum[:, None] * Nticks + time_ticks_1).ravel()

    # ── Signal values: long quadratic × σ_T quadratic × spatial Catmull-Rom ──
    # 3 long × 3 tran × 4 x × 4 y = 144 template lookups, all unrolled by JAX JIT.
    local_t = jnp.arange(sig_len)
    main_vals_2d = jnp.zeros((idx.shape[0], sig_len))
    for l_off, lw in zip([idx - 1, idx, idx + 1], [a, b, c]):
        for t_off, tw in zip([ti - 1, ti, ti + 1], [at, bt, ct]):
            for kx in range(4):
                for ky in range(4):
                    flat_base = (
                        l_off * long_stride
                        + (t_off * Nx * Ny + ixs[:, kx] * Ny + iys[:, ky]) * Nt_sig
                    )  # (N,)
                    vals = response_template.take(flat_base[:, None] + local_t)  # (N, sig_len)
                    w = lw * tw * wx[:, kx] * wy[:, ky]                          # (N,)
                    main_vals_2d = main_vals_2d + vals * w[:, None]

    main_vals_2d = main_vals_2d * nelectrons[:, None]

    # Weight the values by the sub-tick fraction
    main_vals_0 = (main_vals_2d * (1.0 - frac_main[:, None])).ravel()
    main_vals_1 = (main_vals_2d * frac_main[:, None]).ravel()

    # --- 2. PREPARE NEIGHBOR DATA (DIFFERENTIABLE TIME SHIFT) ---
    npix_neigh = (2 * params.number_pix_neighbors + 1)**2
    elec_ids_neigh = jnp.repeat(jnp.arange(nelectrons_neigh.shape[0]), npix_neigh)

    neigh_charge = jnp.take(nelectrons_neigh, elec_ids_neigh)
    neigh_t0 = jnp.take(t0_neigh, elec_ids_neigh)

    float_ticks_neigh = neigh_t0 / params.t_sampling
    cathode_ticks_neigh = jnp.clip(jnp.floor(float_ticks_neigh).astype(int), 0, Nt_cum - 1)
    frac_neigh = float_ticks_neigh - cathode_ticks_neigh

    neigh_start_ticks_0 = Nt_cum - 1 - cathode_ticks_neigh
    neigh_start_ticks_1 = neigh_start_ticks_0 - 1

    neigh_time_ticks_0 = neigh_start_ticks_0[..., None] + jnp.arange(sig_len)
    neigh_time_ticks_1 = neigh_start_ticks_1[..., None] + jnp.arange(sig_len)

    neigh_time_ticks_0 = jnp.where((neigh_time_ticks_0 <= 0) | (neigh_time_ticks_0 >= Nticks - 1), 0, neigh_time_ticks_0 + 1)
    neigh_time_ticks_1 = jnp.where((neigh_time_ticks_1 <= 0) | (neigh_time_ticks_1 >= Nticks - 1), 0, neigh_time_ticks_1 + 1)

    neigh_flat_indices_0 = (pix_renumbering_neigh[:, None] * Nticks + neigh_time_ticks_0).ravel()
    neigh_flat_indices_1 = (pix_renumbering_neigh[:, None] * Nticks + neigh_time_ticks_1).ravel()

    # Neighbor signal: zero-diffusion template (long[0], tran[0]) + Catmull-Rom position
    neigh_no_diff = response_template[0, 0]   # (Nx, Ny, Nt_sig)
    neigh_wx = _cr_wts(currents_frac_neigh[:, 0])   # (N_neigh, 4)
    neigh_wy = _cr_wts(currents_frac_neigh[:, 1])   # (N_neigh, 4)
    neigh_ixs = jnp.clip(currents_idx_neigh[:, 0:1] + jnp.array([-1, 0, 1, 2]), 0, Nx - 1)
    neigh_iys = jnp.clip(currents_idx_neigh[:, 1:2] + jnp.array([-1, 0, 1, 2]), 0, Ny - 1)

    neigh_vals_2d = jnp.zeros((neigh_charge.shape[0], sig_len))
    for kx in range(4):
        for ky in range(4):
            n_flat = (neigh_ixs[:, kx] * Ny + neigh_iys[:, ky]) * Nt_sig   # (N_neigh,)
            vals_n = neigh_no_diff.take(n_flat[:, None] + local_t)           # (N_neigh, sig_len)
            w_n    = neigh_wx[:, kx] * neigh_wy[:, ky]                      # (N_neigh,)
            neigh_vals_2d = neigh_vals_2d + vals_n * w_n[:, None]

    neigh_vals_2d = neigh_vals_2d * neigh_charge[:, None]

    neigh_vals_0 = (neigh_vals_2d * (1.0 - frac_neigh[:, None])).ravel()
    neigh_vals_1 = (neigh_vals_2d * frac_neigh[:, None]).ravel()

    # --- 3. BOUNDARY CORRECTIONS (CUMSUMS) ---
    # response_cum is now 3D (Nx, Ny, Nt_cum) — zero diffusion, no n_long axis.
    # Use nearest-bin for the cumsum lookup (smooth quantity, doesn't need interpolation).

    # Main Corrections
    base_curr = (currents_idx[:, 0] * Ny + currents_idx[:, 1]) * Nt_cum
    val_cum_0 = response_cum.take(base_curr + cathode_ticks_main)
    val_cum_1 = response_cum.take(base_curr + jnp.clip(cathode_ticks_main + 1, 0, Nt_cum - 1))
    interp_cum = val_cum_0 * (1.0 - frac_main) + val_cum_1 * frac_main

    diff_main_interp = (response_cum.take(base_curr + Nt_cum - 1) - interp_cum) * nelectrons

    idx_corr_main_0 = jnp.where((start_ticks_0 <= 0) | (start_ticks_0 >= Nticks - 1), 0, start_ticks_0) + pix_renum * Nticks
    idx_corr_main_1 = jnp.where((start_ticks_1 <= 0) | (start_ticks_1 >= Nticks - 1), 0, start_ticks_1) + pix_renum * Nticks

    diff_main_0 = diff_main_interp * (1.0 - frac_main)
    diff_main_1 = diff_main_interp * frac_main

    # Neighbor Corrections
    base_curr_neigh = (currents_idx_neigh[:, 0] * Ny + currents_idx_neigh[:, 1]) * Nt_cum
    val_cum_neigh_0 = response_cum.take(base_curr_neigh + cathode_ticks_neigh)
    val_cum_neigh_1 = response_cum.take(base_curr_neigh + jnp.clip(cathode_ticks_neigh + 1, 0, Nt_cum - 1))
    interp_cum_neigh = val_cum_neigh_0 * (1.0 - frac_neigh) + val_cum_neigh_1 * frac_neigh

    diff_neigh_interp = (response_cum.take(base_curr_neigh + Nt_cum - 1) - interp_cum_neigh) * neigh_charge
    
    idx_corr_neigh_0 = jnp.where((neigh_start_ticks_0 <= 0) | (neigh_start_ticks_0 >= Nticks - 1), 0, neigh_start_ticks_0) + pix_renumbering_neigh * Nticks
    idx_corr_neigh_1 = jnp.where((neigh_start_ticks_1 <= 0) | (neigh_start_ticks_1 >= Nticks - 1), 0, neigh_start_ticks_1) + pix_renumbering_neigh * Nticks
    
    diff_neigh_0 = diff_neigh_interp * (1.0 - frac_neigh)
    diff_neigh_1 = diff_neigh_interp * frac_neigh

    # --- 4. UNIFIED SEGMENT SUM ---
    # We now concatenate the _0 and _1 variants to smoothly accumulate both
    all_indices = jnp.concatenate([
        main_flat_indices_0, main_flat_indices_1, 
        neigh_flat_indices_0, neigh_flat_indices_1, 
        idx_corr_main_0, idx_corr_main_1, 
        idx_corr_neigh_0, idx_corr_neigh_1
    ])
    
    all_values = jnp.concatenate([
        main_vals_0, main_vals_1, 
        neigh_vals_0, neigh_vals_1, 
        diff_main_0, diff_main_1, 
        diff_neigh_0, diff_neigh_1
    ])

    wfs_flat = jax.ops.segment_sum(
        all_values, 
        all_indices, 
        num_segments=Npixels * Nticks,
        indices_are_sorted=False
    )

    return wfs_flat.reshape(Npixels, Nticks)


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
    ref_hit_prob = jnp.where(ticks < wfs.shape[1] - 3, 1., 0.)  # Assuming hit probability is based on whether ticks are within the waveform length

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
    # Cache field indices for better performance
    x_idx = fields.index("x")
    y_idx = fields.index("y")
    z_idx = fields.index("z")
    n_electrons_idx = fields.index("n_electrons")
    pixel_plane_idx = fields.index("pixel_plane")
    eventID_idx = fields.index("eventID")
    long_diff_idx = fields.index("long_diff")
    tran_diff_idx = fields.index("tran_diff")
    
    #Shifting tracks
    new_tracks = shift_tracks(params, tracks, fields)
    new_tracks = quench(params, new_tracks, fields)
    new_tracks = drift(params, new_tracks, fields)

    main_electrons = new_tracks
    n_electrons_base = main_electrons[:, n_electrons_idx]   # (N,)

    # Sub-pixel bin position for each electron: (N, 2)
    bins_pitches = get_bin_shifts(params, main_electrons, fields)

    # Electron charge — no longer split across a tran-diff grid; the template handles spreading.
    nelectrons = n_electrons_base  # (N,)

    z_cathode = jnp.take(params.tpc_borders, main_electrons[:, pixel_plane_idx].astype(int), axis=0)[..., 2, 1]
    t0 = jnp.abs(main_electrons[:, z_idx] - z_cathode) / get_vdrift(params)  # (N,)
    t0_after_diff = t0  # (N,)

    # Longitudinal diffusion in ticks: (N,)
    long_diff = main_electrons[:, long_diff_idx] / get_vdrift(params) / params.t_sampling

    # Transverse diffusion sigma — used for template index lookup: (N,)
    tran_diff = main_electrons[:, tran_diff_idx]

    # Pixel assignment from sub-pixel bin position
    pix_pitches = bins_pitches // params.nb_sampling_bins_per_pixel  # (N, 2)
    main_pixels = pixel2id(params, pix_pitches[..., 0], pix_pitches[..., 1],
                           main_electrons[:, pixel_plane_idx].astype(int),
                           main_electrons[:, eventID_idx].astype(int))  # (N,)

    # In-pixel (x, y) bin index for response template lookup: (N, 2)
    # Keep the float value for sub-bin Catmull-Rom position interpolation.
    currents_xi = jnp.abs(
        bins_pitches % params.nb_sampling_bins_per_pixel
        - params.nb_sampling_bins_per_pixel // 2 + 0.5
    )  # (N, 2), continuous bin coord: 0.0 = centre of bin 0, 1.0 = centre of bin 1, ...
    currents_idx  = currents_xi.astype(int)       # (N, 2), integer bin index
    currents_frac = currents_xi - currents_idx     # (N, 2), fractional offset ∈ [0, 1)

    #########################################################
    #################Adding neighbors########################
    #########################################################
    pix_grid = jnp.mgrid[-params.number_pix_neighbors:params.number_pix_neighbors+1,
                         -params.number_pix_neighbors:params.number_pix_neighbors+1]
    bin_shifts_neighbors = pix_grid * params.nb_sampling_bins_per_pixel
    currents_xi_neigh = jnp.abs(
        bins_pitches[:, :, None, None] % params.nb_sampling_bins_per_pixel
        - params.nb_sampling_bins_per_pixel // 2 + 0.5
        - bin_shifts_neighbors
    )  # (N, 2, 2n+1, 2n+1), float
    currents_idx_neigh  = jnp.moveaxis(currents_xi_neigh.astype(int), 1, -1).reshape(-1, 2)
    currents_frac_neigh = jnp.moveaxis(
        currents_xi_neigh - currents_xi_neigh.astype(int), 1, -1
    ).reshape(-1, 2)

    new_pitches = jnp.moveaxis((pix_pitches[:, :, None, None] + pix_grid), 1, -1)  # (N, 2n+1, 2n+1, 2)
    pIDs = pixel2id(params, new_pitches[..., 0], new_pitches[..., 1],
                    main_electrons[:, pixel_plane_idx][:, None, None].astype(int),
                    main_electrons[:, eventID_idx][:, None, None].astype(int))
    pIDs_neigh = pIDs.at[:, params.number_pix_neighbors, params.number_pix_neighbors].set(-999)
    nelectrons_neigh = n_electrons_base
    t0_neigh = t0
    return main_pixels, nelectrons, t0_after_diff, long_diff, tran_diff, currents_idx, currents_frac, pIDs_neigh, currents_idx_neigh, currents_frac_neigh, nelectrons_neigh, t0_neigh

@jit
def simulate_signals_new(params, unique_pixels, pixels, t0_after_diff, response_template, nelectrons, long_diff, tran_diff, currents_idx, nelectrons_neigh, pix_renumbering_neigh, t0_neigh, currents_idx_neigh):
    """
    Simulates electronic signals on detector pixels using LUT-based current response templates.
    
    This function accumulates charge signals from electrons drifting to detector pixels, applying:
    - Longitudinal diffusion interpolation using quadratic interpolation between templates
    - Transverse diffusion via binned current response lookup
    - Neighbor pixel contributions from charge sharing
    - Time-dependent signal shape based on drift time
    
    The function uses a flattened pixel indexing scheme for efficient parallel updates and handles
    both main pixel contributions and neighbor pixel contributions separately.
    
    Args:
        params: Simulation parameters containing:
            - time_interval: Tuple of (t_start, t_end) in microseconds
            - t_sampling: Time sampling interval in microseconds
            - signal_length: Number of time samples in the signal template
            - number_pix_neighbors: Number of neighbor pixels in each direction
            - long_diff_template: Array of longitudinal diffusion values for template indexing
        unique_pixels (jnp.ndarray): 1D array of unique pixel IDs that received charge.
            Shape: (Npixels,). May contain padding with -1 values.
        pixels (jnp.ndarray): 2D array of pixel IDs for all electron bins after transverse diffusion.
            Shape: (Nelectrons, nb_tran_diff_bins, nb_tran_diff_bins).
        t0_after_diff (jnp.ndarray): 1D array of drift times from cathode to anode for each electron bin.
            Shape: (Nelectrons * nb_tran_diff_bins^2,) in microseconds.
        response_template (jnp.ndarray): 4D lookup table of current response templates.
            Shape: (Ntemplates, Nx, Ny, Nt) where:
                - Ntemplates: Number of longitudinal diffusion templates
                - Nx, Ny: Spatial bins for transverse position (typically nb_sampling_bins_per_pixel)
                - Nt: Time samples in template
        nelectrons (jnp.ndarray): 1D array of number of electrons in each transverse diffusion bin.
            Shape: (Nelectrons * nb_tran_diff_bins^2,).
        long_diff (jnp.ndarray): 1D array of longitudinal diffusion values (in time ticks) for each bin.
            Shape: (Nelectrons * nb_tran_diff_bins^2,).
        currents_idx (jnp.ndarray): 2D array of spatial bin indices (x, y) for current response lookup.
            Shape: (Nelectrons * nb_tran_diff_bins^2, 2).
        nelectrons_neigh (jnp.ndarray): 1D array of total electrons per original electron deposition
            that contribute to neighbor pixels. Shape: (Nelectrons,).
        pix_renumbering_neigh (jnp.ndarray): 1D array mapping neighbor pixel positions to indices 
            in unique_pixels. Shape: (Nelectrons * (2*number_pix_neighbors+1)^2,).
        t0_neigh (jnp.ndarray): 1D array of drift times for neighbor contributions.
            Shape: (Nelectrons,) in microseconds.
        currents_idx_neigh (jnp.ndarray): 2D array of spatial bin indices for neighbor pixel lookups.
            Shape: (Nelectrons * (2*number_pix_neighbors+1)^2, 2).
    
    Returns:
        jnp.ndarray: Simulated waveforms for each pixel.
            Shape: (Npixels, Nticks) where Nticks = time_interval[1]/t_sampling + 1.
            The first time tick (index 0) serves as a "garbage collector" for out-of-range signals
            and should typically be discarded in downstream processing.
    
    Algorithm:
        1. Flatten waveform array for efficient scatter operations
        2. For main pixel contributions:
           a. Compute time placement: start_ticks based on cathode distance
           b. Interpolate longitudinal diffusion using quadratic interpolation between 3 templates
           c. Extract response template values using currents_idx for spatial bins
           d. Accumulate weighted signals: template * charge * interpolation_weights
           e. Correct for signals that started before trigger (using cumulative sum)
        3. For neighbor pixel contributions:
           a. Expand electron counts to all neighbor positions
           b. Use first template (no diffusion) for neighbors
           c. Accumulate similar to main pixels but without diffusion interpolation
           d. Apply cumulative sum correction
        4. Reshape flattened waveform back to (Npixels, Nticks)
    
    Notes:
        - Time tick 0 is used as a "garbage bin" for out-of-range signals and should be excluded
        - Invalid pixel IDs (-1) are mapped to index 0 (the garbage pixel)
        - The function assumes response_template[0] has no longitudinal diffusion for neighbors
        - Signal accumulation uses JAX's .at[].add() for automatic gradient support
        - Quadratic interpolation provides smooth transitions between diffusion templates
    """
    pix_renumbering = jnp.searchsorted(unique_pixels, pixels.ravel(), method='sort')
    #Getting the right indices for the currents
   
    Npixels = unique_pixels.shape[0]
    Nticks = int(params.time_interval[1]/params.t_sampling) + 1 #Adding one first element to serve as a garbage collector
    wfs = jnp.zeros((Npixels, Nticks))
    wfs = wfs.ravel()

    # response_template has been trimmed to the last signal_length ticks by load_lut.
    # params.response_cum holds the zero-diffusion (no σ_T, no σ_L) cumsum prefix.
    # Shape: (Nx, Ny, Nt_cum) where Nt_cum = original_Nt - signal_length + 1.
    response_cum = params.response_cum   # (Nx, Ny, Nt_cum)
    Nt_cum = response_cum.shape[-1]      # = original_Nt - signal_length + 1

    cathode_ticks = jnp.clip((t0_after_diff/params.t_sampling).astype(int), 0, Nt_cum - 1)

    # With the trimmed template, the output start tick formula becomes:
    # start_ticks = Nt_cum - 1 - cathode_ticks   (was Nt - sig_len - cathode_ticks)
    start_ticks = Nt_cum - 1 - cathode_ticks
    time_ticks = start_ticks[..., None] + jnp.arange(params.signal_length)

    time_ticks = jnp.where((time_ticks <= 0 ) | (time_ticks >= Nticks - 1), 0, time_ticks+1) # it should be start_ticks +1 in theory but we cheat by putting the cumsum in the garbage too when strarting at 0 to mimic the expected behavior

    start_indices = pix_renumbering * Nticks

    end_indices = start_indices[..., None] + time_ticks

    # Flatten the indices
    flat_indices = jnp.ravel(end_indices)

    # More efficient broadcasting - use broadcast_to instead of ones multiplication
    charge = jnp.broadcast_to(nelectrons[:, None], (nelectrons.shape[0], params.signal_length)).reshape(-1)

    n_long, n_tran, Nx, Ny, Nt_sig = response_template.shape  # 5D: (n_long, n_tran, Nx, Ny, sig_len)
    long_stride = n_tran * Nx * Ny * Nt_sig  # stride per longitudinal-template step
    tran_diff_vals = params.tran_diff_template  # (n_tran,) sigma values
    tran_idx = jnp.clip(jnp.searchsorted(tran_diff_vals, tran_diff), 0, n_tran - 1)

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

    # More efficient broadcasting for interpolation coefficients
    signal_shape = (a.shape[0], params.signal_length)
    a_broadcast = jnp.broadcast_to(a[:, None], signal_shape).reshape(-1)
    b_broadcast = jnp.broadcast_to(b[:, None], signal_shape).reshape(-1)
    c_broadcast = jnp.broadcast_to(c[:, None], signal_shape).reshape(-1)

    # 5D template flat index: long*long_stride + tran*(Nx*Ny*Nt_sig) + x*(Ny*Nt_sig) + y*Nt_sig + t
    signal_indices = jnp.ravel(
        (idx[..., None] * n_tran * Nx * Ny + tran_idx[..., None] * Nx * Ny +
         currents_idx[..., 0, None] * Ny + currents_idx[..., 1, None]) * Nt_sig + jnp.arange(Nt_sig))

    # Cache template lookups to avoid repeated computation
    template_values_at_indices = response_template.take(signal_indices)
    template_values_minus = response_template.take(signal_indices - long_stride)
    template_values_plus = response_template.take(signal_indices + long_stride)

    # Update wfs with accumulated signals - combine multiplications efficiently
    wfs = wfs.at[(flat_indices,)].add(template_values_at_indices * charge * b_broadcast)
    wfs = wfs.at[(flat_indices,)].add(template_values_minus * charge * a_broadcast)
    wfs = wfs.at[(flat_indices,)].add(template_values_plus * charge * c_broadcast)

    #Now correct for the missed ticks at the beginning
    # response_cum: (Nx, Ny, Nt_cum), zero diffusion — no n_long axis.
    base_indices = (currents_idx[..., 0]*Ny + currents_idx[..., 1])*Nt_cum
    integrated_start = response_cum.take(jnp.ravel(base_indices + Nt_cum - 1))
    real_start = response_cum.take(jnp.ravel(base_indices + cathode_ticks))
    difference = (integrated_start - real_start)*nelectrons

    start_ticks_corr = jnp.where((start_ticks <= 0 ) | (start_ticks >= Nticks - 1), 0, start_ticks) + pix_renumbering * Nticks
    wfs = wfs.at[start_ticks_corr].add(difference)

    wfs = wfs.reshape((Npixels, Nticks))

    # Optimize neighbor pixel processing
    npix = (2*params.number_pix_neighbors + 1)**2
    elec_ids = jnp.arange(pix_renumbering_neigh.shape[0])//npix
    
    # Use take with mode='fill' for safe indexing
    nelectrons_neigh = jnp.take(nelectrons_neigh, elec_ids, mode='fill', fill_value=0)
    t0_neighbors = jnp.take(t0_neigh, elec_ids, mode='fill', fill_value=0)

    cathode_ticks_neigh = jnp.clip((t0_neighbors/params.t_sampling).astype(int), 0, Nt_cum - 1)
    # Neighbors use zero-diffusion template: long[0], tran[0] -> (Nx, Ny, Nt_sig)
    wfs = accumulate_signals(wfs, currents_idx_neigh, nelectrons_neigh, response_template[0, 0], response_cum, pix_renumbering_neigh, cathode_ticks_neigh, params.signal_length)

    return wfs

@jit
def parse_output(params, adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels):
    mask = (hit_prob > params.hit_prob_threshold) & (event[:, None] >= 0) & (unique_pixels[:, None] >= 0) #Getting rid of hits that are below the hit probability threshold and events that are not valid

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

    new_index = jnp.where(flat_mask, jnp.cumsum(flat_mask) - 1, max_length - 1)
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
def select_split_roi(params, wfs):
    roi_threshold = params.roi_threshold
    Npix, Nticks = wfs.shape
    roi_start = jnp.argmax(wfs > roi_threshold, axis=1)
    roi_end = Nticks - jnp.argmax(wfs[:, ::-1] > roi_threshold, axis=1) - 1

    mask_small_rois = ((roi_end - roi_start) < params.roi_split_length) & (roi_start > 0)
    nb_small_rois = jnp.sum(mask_small_rois)
    return nb_small_rois, mask_small_rois, roi_start

@partial(jit, static_argnames=["padded_small_nb", "padded_large_nb"])
def fee_sim_from_split(params, padded_small_nb, padded_large_nb, wfs, mask_small_rois, roi_start, max_tick_nb):
    Npix = wfs.shape[0]


    small_roi_idx = jnp.argwhere(mask_small_rois, size=padded_small_nb, fill_value=0)[:, 0]
    large_roi_idx = jnp.argwhere(~mask_small_rois, size=padded_large_nb, fill_value=0)[:, 0]

    small_roi_start = roi_start[small_roi_idx]

    large_rois = wfs.at[large_roi_idx, :].get()
    small_rois = wfs.at[small_roi_idx[:, None], (jnp.arange(params.roi_split_length) + small_roi_start[:, None])].get()

    ticks_distrib_small, charge_distrib_small = get_adc_values_average_noise_vmap(params, small_rois)
    ticks_distrib_large, charge_distrib_large = get_adc_values_average_noise_vmap(params, large_rois)

    charge_distrib = jnp.zeros((Npix, charge_distrib_small.shape[1], charge_distrib_small.shape[2]))

    charge_distrib = charge_distrib.at[small_roi_idx, :, :].set(charge_distrib_small[:small_roi_idx.shape[0], :, :])
    charge_distrib = charge_distrib.at[large_roi_idx, :, :].set(charge_distrib_large[:large_roi_idx.shape[0], :, :])


    ticks_distrib = jnp.zeros((Npix, ticks_distrib_small.shape[1], ticks_distrib_small.shape[2]))
    ticks_distrib = ticks_distrib.at[small_roi_idx, :, small_roi_start:small_roi_start + params.roi_split_length].set(ticks_distrib_small[:small_roi_idx.shape[0], :, :])
    ticks_distrib = ticks_distrib.at[large_roi_idx, :, :].set(ticks_distrib_large[:large_roi_idx.shape[0], :, :])
    ticks_distrib = jnp.minimum(ticks_distrib, max_tick_nb)  # Ensure ticks do not exceed waveform length

    return charge_distrib, ticks_distrib

def simulate_wfs(params, response_template, tracks, fields):
    """
    Simulates the signal from the drifted electrons and returns waveforms and unique pixel identifiers.
    
    This function performs the complete drift simulation pipeline: simulating electron drift,
    accumulating signals on pixels, and generating the corresponding waveforms.
    
    Args:
        params (Any): Parameters of the simulation.
        response_template (jnp.ndarray): Response function template for signal generation.
        tracks (jnp.ndarray): Tracks of the particles as a JAX array.
        fields (List[str]): List of field names corresponding to the tracks.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - wfs: Waveforms as a 2D JAX array with shape (Npixels, Nticks-1), where Npixels 
              is the number of unique active pixels and Nticks-1 is the number of time samples 
              (the first tick is excluded as it serves as a garbage collector).
            - unique_pixels: 1D JAX array of unique pixel identifiers that were active during 
              the simulation, with shape (Npixels,).
    """

    main_pixels, nelectrons, t0_after_diff, long_diff, tran_diff, currents_idx, currents_frac, pIDs_neigh, currents_idx_neigh, currents_frac_neigh, nelectrons_neigh, t0_neigh = simulate_drift_new(params, tracks, fields)

    ################################################
    ################################################

    #Sorting the pixels and getting the unique ones
    unique_pixels = jnp.unique(main_pixels.ravel())
    padded_unique = pad_size(unique_pixels.shape[0], "unique_pixels", 0.2)

    unique_pixels = jnp.sort(jnp.pad(unique_pixels, (0, padded_unique - unique_pixels.shape[0]), mode='constant', constant_values=-1))

    pix_renumbering_neigh= jnp.searchsorted(unique_pixels, pIDs_neigh.ravel(), method='sort')

    mask = (pix_renumbering_neigh < unique_pixels.size) & (unique_pixels[pix_renumbering_neigh] == pIDs_neigh.ravel())
    pix_renumbering_neigh = pix_renumbering_neigh.at[~mask].set(0) #ASSUMES THAT THERE IS ALWAYS A -1 PIXID
    # mask_indices = jnp.nonzero(mask)[0]
    # padded_size = pad_size(mask_indices.shape[0], "pix_renumbering", 0.2)
    # mask_indices = jnp.pad(mask_indices, (0, padded_mask - mask_indices.shape[0]), mode='constant', constant_values=-1)

    ###############################################
    ###############################################

    wfs = simulate_signals(params, unique_pixels, main_pixels, t0_after_diff, response_template, nelectrons, long_diff, tran_diff, currents_idx, currents_frac, nelectrons_neigh, pix_renumbering_neigh, t0_neigh, currents_idx_neigh, currents_frac_neigh)


    return wfs[:, 1:], unique_pixels

def simulate_stochastic(params, wfs, unique_pixels, rngseed):
    """
    Simulates the signal from the drifted electrons and returns the ADC values, pixel coordinates, ticks, hit probabilities, event numbers, and unique pixel identifiers.
    Args:
        params: Parameters of the simulation.
        wfs (jnp.ndarray): Waveforms as a JAX array.
        unique_pixels (jnp.ndarray): Unique pixel identifiers.
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
    integral, ticks = get_adc_values(params, wfs, jax.random.key(rngseed))
    hit_prob = jnp.where(ticks < wfs.shape[1] - 3, 1., 0.)  # Assuming hit probability is based on whether ticks are within the waveform length
    adcs = digitize(params, integral)

    pixel_x, pixel_y, pixel_plane, event = id2pixel(params, unique_pixels)
    pixel_coords = get_pixel_coordinates(params, pixel_x, pixel_y, pixel_plane)
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]
    pixel_z  = get_hit_z(params, ticks.flatten(), jnp.repeat(pixel_plane, 10))

    adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, hit_pixels, nb_valid = parse_output(params, adcs, pixel_x, pixel_y, pixel_z, ticks, hit_prob, event, unique_pixels)

    return adcs[:nb_valid], pixel_x[:nb_valid], pixel_y[:nb_valid], pixel_z[:nb_valid], ticks[:nb_valid], hit_prob[:nb_valid], event[:nb_valid], hit_pixels[:nb_valid]

@jit
def simulate_probabilistic(params, wfs, unique_pixels):
    """
    Simulates the signal from the drifted electrons and returns probabilistic
    distributions of ADC values and tick times, along with pixel coordinates
    and event numbers.

    Args:
        params: Parameters of the simulation.
        wfs (jnp.ndarray): Waveforms as a JAX array.
        unique_pixels (jnp.ndarray): Unique pixel identifiers for the input waveforms.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - adcs_distrib: Probabilistic distribution of ADC values for each
              pixel/time bin after adding average noise and digitization.
            - pixel_x: X coordinates of the pixels corresponding to the waveforms.
            - pixel_y: Y coordinates of the pixels corresponding to the waveforms.
            - ticks_prob: Tick indices associated with the probabilistic charge/
              ADC distributions.
            - event: Event numbers associated with each pixel.
    """

    # Npix = wfs.shape[0]
    # ROI selection is a discrete, non-differentiable operation; we explicitly stop
    # gradients from flowing through select_split_roi to avoid backpropagating through
    # this indexing/masking logic while still allowing gradients on downstream signals.
    # nb_small_rois, mask_small_rois, roi_start = jax.lax.stop_gradient(select_split_roi(params, wfs))
    # nb_small_rois = int(nb_small_rois)
    # padded_small_nb = pad_size(nb_small_rois, "wfs_roi", 0.1)
    # padded_large_nb = pad_size(Npix - nb_small_rois, "wfs_roi", 0.1)

    # integral, ticks, hit_prob = fee_sim_from_split(params, padded_small_nb, padded_large_nb, wfs[:, 1:], mask_small_rois, roi_start, wfs.shape[1] - 2)
    
    ticks_prob, charge_distrib = get_adc_values_average_noise_vmap(params, wfs)

    adcs_distrib = digitize(params, charge_distrib)
    pixel_x, pixel_y, pixel_plane, event = id2pixel(params, unique_pixels)
    pixel_coords = get_pixel_coordinates(params, pixel_x, pixel_y, pixel_plane)
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]
    
    return adcs_distrib, pixel_x, pixel_y, ticks_prob, event


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
