"""
Utility functions for Taylor approximation validity studies.

Provides:
- setup_params_and_tracks: load detector, LUT, tracks, split by event
- compute_event_all_params: compute J, H, and perturbation scans for all params at once
- save_results: pickle serialization
"""

import numpy as np
import h5py
from numpy.lib import recfunctions as rfn
import pickle

import jax
import jax.numpy as jnp

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import simulate_wfs, simulate_probabilistic, pad_size
from larndsim.fee_jax import get_average_hit_values, digitize, get_adc_values_average_noise_vmap
from optimize.dataio import chop_tracks


def setup_params_and_tracks(
    input_file,
    param_names,
    detector_props='src/larndsim/detector_properties/module0.yaml',
    pixel_layouts='src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml',
    lut_file='src/larndsim/detector_properties/response_44_v2a_full_tick.npz',
    electron_sampling=0.01,
    signal_length=150,
    number_pix_neighbors=4,
):
    """Load detector config, LUT, and tracks split by event.

    Parameters
    ----------
    input_file : str
    param_names : list of str — all parameters to make differentiable

    Returns
    -------
    ref_params : Params (with all param_names as differentiable pytree nodes)
    response : jnp.ndarray
    events : dict[int, jnp.ndarray]
    fields : tuple of str
    """
    Params = build_params_class(param_names)
    ref_params = load_detector_properties(Params, detector_props, pixel_layouts)
    response, ref_params = load_lut(lut_file, ref_params)
    ref_params = ref_params.replace(
        electron_sampling_resolution=electron_sampling,
        number_pix_neighbors=number_pix_neighbors,
        signal_length=signal_length,
        time_window=signal_length,
        diffusion_in_current_sim=True,
    )

    with h5py.File(input_file, 'r') as f:
        tracks_struct = np.array(f['segments'])

    if 't0' not in tracks_struct.dtype.names:
        tracks_struct = rfn.append_fields(
            tracks_struct, 't0', np.zeros(tracks_struct.shape[0]), usemask=False
        )

    fields = tracks_struct.dtype.names
    replace_map = {'event_id': 'eventID', 'traj_id': 'trackID'}
    fields = tuple(replace_map.get(f, f) for f in fields)
    tracks_struct.dtype.names = fields

    for suffix in ('_start', '_end', ''):
        x_col = np.copy(tracks_struct[f'x{suffix}'])
        tracks_struct[f'x{suffix}'] = np.copy(tracks_struct[f'z{suffix}'])
        tracks_struct[f'z{suffix}'] = x_col

    event_ids = np.unique(tracks_struct['eventID'])
    events = {}
    for eid in event_ids:
        mask = tracks_struct['eventID'] == eid
        evt_tracks = tracks_struct[mask]
        evt_np = rfn.structured_to_unstructured(evt_tracks, copy=True, dtype=np.float32)
        evt_chopped = chop_tracks(evt_np, fields, electron_sampling)
        sz = pad_size(evt_chopped.shape[0], 'batch_size', 0.5)
        evt_padded = np.pad(
            evt_chopped, ((0, sz - evt_chopped.shape[0]), (0, 0)),
            mode='constant', constant_values=0,
        )
        events[int(eid)] = jax.device_put(evt_padded)

    return ref_params, response, events, fields


def compute_event_all_params(
    tracks, ref_params, response, fields, param_names, param_ranges, n_points=21,
):
    """Compute nominal, Jacobian, Hessian, and perturbation scan for one event, all params.

    All parameters are differentiable simultaneously (single JAX compilation).
    The Jacobian is computed in one jacfwd call for all parameters.
    The Hessian diagonal is computed in one jacfwd(jacfwd(...)) call.

    Parameters
    ----------
    tracks : jnp.ndarray — padded track array for one event
    ref_params : Params — params with all param_names as differentiable pytree nodes
    response : jnp.ndarray
    fields : tuple of str
    param_names : list of str — parameters to study
    param_ranges : dict[str, float] — half-width of perturbation range per param (fraction)
    n_points : int — number of grid points per parameter

    Returns
    -------
    dict[str, dict] — {param_name: {nominal_value, n_active, H_scalar, mean_true,
                                     mean_jac, mean_hess, rel_deltas}}
    """

    # --- Pipeline wrappers (take full params object) ---
    def pipeline_smooth(params):
        wfs, unique_pixels = simulate_wfs(params, response, tracks, fields)
        log_prob, charge = get_adc_values_average_noise_vmap(params, wfs)
        adcs_distrib = digitize(params, charge)
        ticks_prob = jnp.exp(log_prob)
        _, expected_adcs, _ = get_average_hit_values(ticks_prob, adcs_distrib)
        return expected_adcs[:, 0]

    def pipeline_original(params):
        wfs, unique_pixels = simulate_wfs(params, response, tracks, fields)
        adcs_distrib, _, _, log_tp, _ = simulate_probabilistic(params, wfs, unique_pixels)
        ticks_prob = jnp.exp(log_tp)
        _, expected_adcs, _ = get_average_hit_values(ticks_prob, adcs_distrib)
        return expected_adcs[:, 0]

    def pipeline_scalar(params):
        return jnp.mean(pipeline_smooth(params))

    # --- Nominal ---
    adcs_nom = pipeline_smooth(ref_params)
    mask_active = adcs_nom != 0
    n_active = int(mask_active.sum())

    # --- Jacobian w.r.t. all params (single jacfwd call) ---
    J_pytree = jax.jacfwd(pipeline_smooth)(ref_params)

    # --- Hessian w.r.t. all params (single jacfwd(jacfwd) call) ---
    H_pytree = jax.jacfwd(jax.jacfwd(pipeline_scalar))(ref_params)

    # --- Extract per-parameter results and run perturbation scans ---
    results = {}

    for param_name in param_names:
        nominal_value = float(getattr(ref_params, param_name))
        J_param = getattr(J_pytree, param_name)  # shape (n_pixels,)
        H_param = float(getattr(getattr(H_pytree, param_name), param_name))  # diagonal element

        half_range = param_ranges.get(param_name, 0.50)
        rel_deltas = np.linspace(-half_range, half_range, n_points)
        abs_deltas = rel_deltas * nominal_value

        mean_true = []
        mean_jac = []
        mean_hess = []

        for dParam in abs_deltas:
            params_perturbed = ref_params.replace(**{param_name: nominal_value + dParam})
            adcs_true = pipeline_original(params_perturbed)
            adcs_jac = adcs_nom + J_param * dParam
            adcs_hess = adcs_nom + J_param * dParam + 0.5 * H_param * dParam ** 2

            if mask_active.any():
                mean_true.append(float(adcs_true[mask_active].mean()))
                mean_jac.append(float(adcs_jac[mask_active].mean()))
                mean_hess.append(float(adcs_hess[mask_active].mean()))
            else:
                mean_true.append(0.0)
                mean_jac.append(0.0)
                mean_hess.append(0.0)

        results[param_name] = {
            'param_name': param_name,
            'nominal_value': nominal_value,
            'n_active': n_active,
            'H_scalar': H_param,
            'mean_true': np.array(mean_true),
            'mean_jac': np.array(mean_jac),
            'mean_hess': np.array(mean_hess),
            'rel_deltas': np.array(rel_deltas),
        }

    return results


def save_results(results, path):
    """Save results dict to pickle."""
    with open(path, 'wb') as f:
        pickle.dump(results, f)
