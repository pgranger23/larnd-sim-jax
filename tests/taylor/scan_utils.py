"""
Utility functions for Jacobian/Hessian validity studies.

Provides:
- setup_params_and_tracks: load detector, LUT, tracks, split by event
- compute_event_scan: for one (event, parameter), compute nominal + J + H + scan
- compute_validity_range: find asymmetric perturbation range below a threshold
- save_results / load_results: pickle serialization
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
    detector_props='src/larndsim/detector_properties/module0.yaml',
    pixel_layouts='src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml',
    lut_file='src/larndsim/detector_properties/response_44_v2a_full_tick.npz',
    electron_sampling=0.01,
    signal_length=150,
    number_pix_neighbors=4,
):
    """Load detector config, LUT, and tracks split by event.

    Returns
    -------
    ref_params_base : Params (no differentiable parameters)
    response : jnp.ndarray (LUT response template)
    events : dict[int, jnp.ndarray] (event_id -> padded track array on device)
    fields : tuple of str
    """
    # Base params with no differentiable parameters
    Params = build_params_class([])
    ref_params_base = load_detector_properties(Params, detector_props, pixel_layouts)
    response, ref_params_base = load_lut(lut_file, ref_params_base)
    ref_params_base = ref_params_base.replace(
        electron_sampling_resolution=electron_sampling,
        number_pix_neighbors=number_pix_neighbors,
        signal_length=signal_length,
        time_window=signal_length,
        diffusion_in_current_sim=True,
    )

    # Load and prepare tracks
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

    # Swap x <-> z
    for suffix in ('_start', '_end', ''):
        x_col = np.copy(tracks_struct[f'x{suffix}'])
        tracks_struct[f'x{suffix}'] = np.copy(tracks_struct[f'z{suffix}'])
        tracks_struct[f'z{suffix}'] = x_col

    # Split by event
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

    return ref_params_base, response, events, fields


def compute_event_scan(
    tracks, ref_params_base, response, fields, param_name, rel_deltas,
):
    """Compute nominal, Jacobian, Hessian, and perturbation scan for one event+param.

    Parameters
    ----------
    tracks : jnp.ndarray — padded track array for one event
    ref_params_base : Params — base params (no differentiable parameters)
    response : jnp.ndarray — LUT response template
    fields : tuple of str
    param_name : str — e.g. 'Ab', 'eField', 'lifetime', ...
    rel_deltas : array-like — relative perturbation fractions (e.g. [-0.5, ..., 0.5])

    Returns
    -------
    dict with keys:
        nominal_value, adcs_nom, mask_active, n_active,
        J, H_scalar,
        mean_true, mean_jac, mean_hess (arrays of shape (len(rel_deltas),)),
        rel_deltas
    """
    # Build params with this parameter differentiable
    Params = build_params_class([param_name])
    ref_params = Params(**{
        f.name: getattr(ref_params_base, f.name)
        for f in ref_params_base.__dataclass_fields__.values()
        if f.name in Params.__match_args__
    })

    nominal_value = float(getattr(ref_params, param_name))

    # --- Full pipeline wrapper (smoothed FEE, for derivatives) ---
    def full_pipeline_smooth(param_val):
        params = ref_params.replace(**{param_name: param_val})
        wfs, unique_pixels = simulate_wfs(params, response, tracks, fields)
        log_prob, charge = get_adc_values_average_noise_vmap(params, wfs)
        adcs_distrib = digitize(params, charge)
        ticks_prob = jnp.exp(log_prob)
        _, expected_adcs, _ = get_average_hit_values(ticks_prob, adcs_distrib)
        return expected_adcs[:, 0]

    # --- Full pipeline wrapper (original FEE, for true resimulation) ---
    def full_pipeline_original(param_val):
        params = ref_params.replace(**{param_name: param_val})
        wfs, unique_pixels = simulate_wfs(params, response, tracks, fields)
        adcs_distrib, _, _, log_tp, _ = simulate_probabilistic(params, wfs, unique_pixels)
        ticks_prob = jnp.exp(log_tp)
        _, expected_adcs, _ = get_average_hit_values(ticks_prob, adcs_distrib)
        return expected_adcs[:, 0]

    # --- Scalar wrapper for Hessian ---
    def full_pipeline_scalar(param_val):
        return jnp.mean(full_pipeline_smooth(param_val))

    # --- Nominal ---
    adcs_nom = full_pipeline_smooth(nominal_value)
    mask_active = adcs_nom != 0
    n_active = int(mask_active.sum())

    # --- Jacobian (vector) ---
    J = jax.jacfwd(full_pipeline_smooth)(nominal_value)

    # --- Hessian (scalar) ---
    H_scalar = jax.jacfwd(jax.jacfwd(full_pipeline_scalar))(nominal_value)

    # --- Perturbation scan ---
    abs_deltas = np.array(rel_deltas) * nominal_value
    mean_true = []
    mean_jac = []
    mean_hess = []

    for dParam in abs_deltas:
        param_new = nominal_value + dParam
        adcs_true = full_pipeline_original(param_new)
        adcs_jac = adcs_nom + J * dParam
        adcs_hess = adcs_nom + J * dParam + 0.5 * H_scalar * dParam ** 2

        if mask_active.any():
            mean_true.append(float(adcs_true[mask_active].mean()))
            mean_jac.append(float(adcs_jac[mask_active].mean()))
            mean_hess.append(float(adcs_hess[mask_active].mean()))
        else:
            mean_true.append(0.0)
            mean_jac.append(0.0)
            mean_hess.append(0.0)

    return {
        'param_name': param_name,
        'nominal_value': nominal_value,
        'n_active': n_active,
        'H_scalar': float(H_scalar),
        'mean_true': np.array(mean_true),
        'mean_jac': np.array(mean_jac),
        'mean_hess': np.array(mean_hess),
        'rel_deltas': np.array(rel_deltas),
    }


def save_results(results, path):
    """Save results dict to pickle."""
    with open(path, 'wb') as f:
        pickle.dump(results, f)
