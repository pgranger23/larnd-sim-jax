import jax.numpy as jnp
from jax import jit, vmap
import jax
from jax.nn import softmax
from functools import partial
from larndsim.sim_jax import pad_size, simulate, simulate_parametrized
from larndsim.fee_jax import digitize
from larndsim.detsim_jax import id2pixel, get_pixel_coordinates, get_hit_z

def mse_loss(adcs, pIDs, adcs_ref, pIDs_ref):
    all_pixels = jnp.concatenate([pIDs, pIDs_ref])
    unique_pixels = jnp.sort(jnp.unique(all_pixels))
    nb_pixels = unique_pixels.shape[0]

    nb_pixels = pad_size(nb_pixels, "mse_loss")

    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)

    pix_renumbering_ref = jnp.searchsorted(unique_pixels, pIDs_ref)

    signals = jnp.zeros((nb_pixels, adcs.shape[1]))
    signals = signals.at[pix_renumbering, :].add(adcs)
    signals = signals.at[pix_renumbering_ref, :].add(-adcs_ref)
    adc_loss = jnp.sum(signals**2)
    return adc_loss, dict()

def mse_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref):
    return mse_loss(adcs, pixels, ref, pixels_ref)

def mse_time(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref):
    mask_ref = ref > 0
    mask = adcs > 0
    ticks = ticks.at[~mask].set(0)
    ticks_ref = ticks_ref.at[~mask_ref].set(0)
    return mse_loss(ticks, pixels, ticks_ref, pixels_ref)

def mse_time_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, alpha=0.5):
    loss_adc, _ = mse_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref)
    loss_time, _ = mse_time(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref)
    return alpha * loss_adc + (1 - alpha) * loss_time, dict()

@partial(jit, static_argnames=['match_z'])
def prepare_hits(params, adcs, pixels, ticks, match_z=False):
    pixel_x, pixel_y, pixel_plane, eventID = id2pixel(params, pixels)
    pixel_coords = get_pixel_coordinates(params, pixel_x, pixel_y, pixel_plane)
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]
    if match_z:
        drift = get_hit_z(params, ticks.flatten(), jnp.repeat(pixel_plane, 10))
    else:
        drift = ticks * params.t_sampling

    return pixel_x, pixel_y, drift, adcs, eventID

@jit
def softmin_charge_aware_chamfer(true_positions, sim_positions, true_charges, sim_charges, tau=10.0, lambda_charge=0):
    """
    Computes a softmin-based charge-aware Chamfer distance for smooth optimization.

    Args:
        true_positions (jnp.array): (N, 3) True hit positions.
        true_charges (jnp.array): (N,) True charge intensities.
        sim_positions (jnp.array): (M, 3) Simulated hit positions.
        sim_charges (jnp.array): (M,) Simulated charge intensities.
        tau (float): Temperature parameter for softmin.
        lambda_charge (float): Weighting factor for charge mismatch.

    Returns:
        float: Softmin-based charge-aware Chamfer distance.
    """

    # Normalize charge weights (optional, but helps with stability)
    # true_charges = true_charges / jnp.sum(true_charges)
    # sim_charges = sim_charges / jnp.sum(sim_charges)

    # Compute squared pairwise Euclidean distances (avoids sqrt for efficiency)
    dists_sq = jnp.sum((true_positions[:, None, :] - sim_positions[None, :, :]) ** 2, axis=2)

    # Compute absolute charge differences
    charge_diffs = jnp.abs(true_charges[:, None] - sim_charges[None, :])

    # Softmin using logsumexp for numerical stability
    softmin_true_to_sim = softmax(-tau * dists_sq, axis=1)
    # softmin_sim_to_true = softmax(-tau * dists_sq, axis=0)

    # softmin_true_to_sim = jnp.where(softmin_true_to_sim > 1e4, 0, softmin_true_to_sim)
    # softmin_sim_to_true = jnp.where(softmin_sim_to_true > 1e4, 0, softmin_sim_to_true)


    # Compute total loss efficiently
    loss_true_to_sim = jnp.sum(softmin_true_to_sim*(dists_sq + lambda_charge * charge_diffs), where=dists_sq < 1e4)
    # loss_sim_to_true = jnp.sum(softmin_sim_to_true*(dists_sq.T + lambda_charge * charge_diffs.T), where=dists_sq.T < 1e4)

    return loss_true_to_sim #+ loss_sim_to_true

@jax.jit
def chamfer_distance_3d(pos_a, pos_b, w_a, w_b):
    """
    Compute the Chamfer Distance between two sets of 3D points (x, y, t).
    
    Parameters:
        pos_a: jnp.ndarray of shape (N, 3), positions and times of hits in distribution A.
        pos_b: jnp.ndarray of shape (M, 3), positions and times of hits in distribution B.
        w_a: jnp.ndarray of shape (N,), weights of hits in distribution A.
        w_b: jnp.ndarray of shape (M,), weights of hits in distribution B.
    
    Returns:
        A scalar representing the Chamfer Distance between the two point sets.
    """

    # Calculate pairwise squared distances
    dists_a_to_b = jnp.sum((pos_a[:, None, :] - pos_b[None, :, :])**2, axis=2)
    
    # Find indices of minimum distances
    argmin_dists_a_to_b = jnp.argmin(dists_a_to_b, axis=1)
    argmin_dists_b_to_a = jnp.argmin(dists_a_to_b, axis=0)
    
    # Extract minimum distances
    min_dists_a_to_b = jnp.take_along_axis(dists_a_to_b, argmin_dists_a_to_b[:, None], axis=1).squeeze()
    min_dists_b_to_a = jnp.take_along_axis(dists_a_to_b, argmin_dists_b_to_a[None, :], axis=0).squeeze()
    
    # Extract weights of the closest points
    closest_weights_a_to_b = jnp.take(w_b, argmin_dists_a_to_b)
    closest_weights_b_to_a = jnp.take(w_a, argmin_dists_b_to_a)
    
    # Calculate the weighted Chamfer distance
    chamfer_dist = (
        jnp.sum(min_dists_a_to_b * w_a * closest_weights_a_to_b, where=min_dists_a_to_b < 1e4) +
        jnp.sum(min_dists_b_to_a * w_b * closest_weights_b_to_a, where=min_dists_b_to_a < 1e4)
    )
    return chamfer_dist

def chamfer_3d(params, adcs, pixels, ticks, adcs_ref, pixels_ref, ticks_ref, adc_norm=10., match_z=False):
    pixel_x, pixel_y, drift, adcs, eventID = prepare_hits(params, adcs, pixels, ticks, match_z)
    pixel_x_ref, pixel_y_ref, drift_ref, adcs_ref, eventID_ref = prepare_hits(params, adcs_ref, pixels_ref, ticks_ref, match_z)
    mask = (adcs.flatten() > 0) & (jnp.repeat(eventID, 10) != -1)
    mask_ref = adcs_ref.flatten() > 0

    nb_selected = jnp.count_nonzero(mask)
    nb_selected_ref = jnp.count_nonzero(mask_ref)
    
    padded_size = pad_size(max(nb_selected, nb_selected_ref), "batch_hits")

    eventID_masked = jnp.pad(jnp.repeat(eventID, 10)[mask], (0, padded_size - nb_selected), mode='constant', constant_values=-1e9)
    pixel_x_masked = jnp.pad(jnp.repeat(pixel_x, 10)[mask], (0, padded_size - nb_selected), mode='constant', constant_values=-1e9)
    pixel_y_masked = jnp.pad(jnp.repeat(pixel_y, 10)[mask], (0, padded_size - nb_selected), mode='constant', constant_values=-1e9)
    drift_masked = jnp.pad(drift.flatten()[mask], (0, padded_size - nb_selected), mode='constant', constant_values=-1e9)
    adcs_masked = jnp.pad(adcs.flatten()[mask], (0, padded_size - nb_selected), mode='constant', constant_values=0)/adc_norm

    eventID_masked_ref = jnp.pad(jnp.repeat(eventID_ref, 10)[mask_ref], (0, padded_size - nb_selected_ref), mode='constant', constant_values=-1e9)
    pixel_x_masked_ref = jnp.pad(jnp.repeat(pixel_x_ref, 10)[mask_ref], (0, padded_size - nb_selected_ref), mode='constant', constant_values=-1e9)
    pixel_y_masked_ref = jnp.pad(jnp.repeat(pixel_y_ref, 10)[mask_ref], (0, padded_size - nb_selected_ref), mode='constant', constant_values=-1e9)
    drift_masked_ref = jnp.pad(drift_ref.flatten()[mask_ref], (0, padded_size - nb_selected_ref), mode='constant', constant_values=-1e9)
    adcs_masked_ref = jnp.pad(adcs_ref.flatten()[mask_ref], (0, padded_size - nb_selected_ref), mode='constant', constant_values=0)/adc_norm

    loss = chamfer_distance_3d(jnp.stack((pixel_x_masked + eventID_masked*1e9, pixel_y_masked, drift_masked, adcs_masked), axis=-1), jnp.stack((pixel_x_masked_ref + eventID_masked_ref*1e9, pixel_y_masked_ref, drift_masked_ref, adcs_masked_ref), axis=-1), adcs_masked, adcs_masked_ref)
    # loss = softmin_charge_aware_chamfer(jnp.stack((pixel_x_masked_ref + eventID_masked_ref*1e9, pixel_y_masked_ref, pixel_z_masked_ref, adcs_masked_ref/jnp.sum(adcs_masked_ref)), axis=-1), jnp.stack((pixel_x_masked + eventID_masked*1e9, pixel_y_masked, pixel_z_masked, adcs_masked/jnp.sum(adcs_masked)), axis=-1), adcs_masked_ref, adcs_masked)


    return loss, dict()

def sdtw_loss(adcs, ref, dstw):
    # Assumes pixels are already sorted

    mask = adcs.flatten() > 0
    mask_ref = ref.flatten() > 0
    loss = dstw.pairwise(adcs.flatten()[mask], ref.flatten()[mask_ref])

    return loss, dict()

def sdtw_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw):
    return sdtw_loss(adcs, ref, dstw)

def sdtw_time(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw):
    return sdtw_loss(ticks, ticks_ref, dstw)

def sdtw_time_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw, alpha=0.5):
    loss_adc, _ = sdtw_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw)
    loss_time, _ = sdtw_time(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw)
    return alpha * loss_adc + (1 - alpha) * loss_time, dict()

@jit
@jax.named_scope("get_hits_space_coords")
def get_hits_space_coords(params, pIDs, ticks):
    pixel_x, pixel_y, pixel_plane, eventID = id2pixel(params, pIDs)
    pixel_coords = get_pixel_coordinates(params, pixel_x, pixel_y, pixel_plane)
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]
    pixel_z = get_hit_z(params, ticks.flatten(), jnp.repeat(pixel_plane, 10))

    return pixel_x, pixel_y, pixel_z, eventID

@jit
def cleaning_outputs(params, ref, adcs):
    #Cleaning up baselines to avoid big leaps
    adc_lowest = digitize(params, params.DISCRIMINATION_THRESHOLD)
    ref = jnp.where(ref < adc_lowest, 0, ref - adc_lowest)
    adcs = jnp.where(adcs < adc_lowest, 0, adcs - adc_lowest)
    return ref, adcs

def params_loss(params, response, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0, loss_fn=mse_adc, **loss_kwargs):
    adcs, pixels, ticks, _, _, _ = simulate(params, response, tracks, fields, rngkey)

    ref, adcs = cleaning_outputs(params, ref, adcs)
    
    loss_val, aux = loss_fn(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, **loss_kwargs)
    return loss_val, aux

def params_loss_parametrized(params, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0, loss_fn=mse_adc, diffusion_in_current_sim=False, **loss_kwargs):
    """
    Loss function for parametrized simulation.
    Args:
        params: Parameters of the simulation.
        ref: Reference ADC values.
        pixels_ref: Reference pixel IDs.
        ticks_ref: Reference time ticks.
        tracks: Particle tracks.
        fields: Fields for the simulation.
        rngkey: Random key for simulation.
        loss_fn: Loss function to use.
        diffusion_in_current_sim: Flag for diffusion in current simulation.
        **loss_kwargs: Additional arguments for the loss function.
    Returns:
        loss_val: Loss value.
        aux: Auxiliary values.
    """
    
    adcs, pixels, ticks, _, _, _ = simulate_parametrized(params, tracks, fields, rngkey, diffusion_in_current_sim)

    ref, adcs = cleaning_outputs(params, ref, adcs)
    
    if loss_fn.__name__ in ['sdtw_adc', 'sdtw_time', 'sdtw_time_adc']:
        loss_val, aux = loss_fn(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, loss_kwargs['dstw'])
    else:
        loss_val, aux = loss_fn(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, **loss_kwargs)
    return loss_val, aux

#Code commented below is unused but I still want to keep it for future reference

# def batch_hits(params, adcs, pIDs, ticks):
#     pixel_x, pixel_y, pixel_z, eventID = get_hits_space_coords(params, pIDs, ticks)
#     with jax.named_scope("masking"):
#         mask = adcs.flatten() > 0
#         pixel_x_masked = jnp.repeat(pixel_x, 10)[mask]
#         pixel_y_masked = jnp.repeat(pixel_y, 10)[mask]
#         pixel_z_masked = pixel_z[mask]
#         eventID_masked = jnp.repeat(eventID, 10)[mask]
#         adcs_masked = adcs.flatten()[mask]

#     unique_event_ids, event_start_indices = jnp.unique(eventID_masked, return_index=True)
#     num_events = unique_event_ids.shape[0]
#     # Compute the number of hits per event
#     hits_per_event = jnp.diff(jnp.append(event_start_indices, eventID_masked.shape[0]))
#     # Determine maximum hits per event for padding

#     max_hits_per_event = jnp.max(hits_per_event)
#     padded_size = pad_size(max_hits_per_event, "batch_hits")
#     max_hits_per_event = padded_size
   
#     # Create scatter indices to place hits into batch array
#     with jax.named_scope("batching"):
#         max_range = jnp.arange(max_hits_per_event, dtype=int)
#         event_hit_ranges = max_range[None, :] < hits_per_event[:, None]
#         hit_indices = jnp.where(event_hit_ranges, max_range[None, :], -1).flatten()
#         valid_hit_indices = hit_indices[hit_indices >= 0]
        
#         event_indices = jnp.repeat(jnp.arange(num_events, dtype=int), hits_per_event)
        
#         #Initialize padded array
#         batched_hits = jnp.full((num_events, max_hits_per_event, 4), 0.)
#         # masks = jnp.zeros((num_events, max_hits_per_event), dtype=int)
        
#         batched_hits = batched_hits.at[event_indices, valid_hit_indices, 0].set(pixel_x_masked)
#         batched_hits = batched_hits.at[event_indices, valid_hit_indices, 1].set(pixel_y_masked)
#         batched_hits = batched_hits.at[event_indices, valid_hit_indices, 2].set(pixel_z_masked)
#         batched_hits = batched_hits.at[event_indices, valid_hit_indices, 3].set(adcs_masked)

#     # masks = masks.at[event_indices, valid_hit_indices].set(1)

#     return batched_hits, unique_event_ids

# @jit
# def chamfer_distance_batch(points1, points2, mask1, mask2):
#     """
#     Computes the Chamfer distance for each event independently using masks.
#     Args:
#         points1, points2: Arrays of shape (B, P, D) and (B, Q, D) for batched events.
#         mask1, mask2: Binary masks for valid points (shape (B, P) and (B, Q)).
#     Returns:
#         Chamfer distance for each event.
#     """
#     def chamfer_event(p1, p2, m1, m2):
#         dists = jnp.sum((p1[:, None, :] - p2[None, :, :]) ** 2, axis=-1)

#         valid_dists1 = jnp.where(m2[None, :], dists, 1e10)
#         valid_dists2 = jnp.where(m1[:, None], dists, 1e10)

#         min_dist1 = jnp.min(valid_dists1, axis=1)
#         min_dist2 = jnp.min(valid_dists2, axis=0)

#         mean_dist1 = jnp.sum(min_dist1 * m1)# / jnp.sum(m1)
#         mean_dist2 = jnp.sum(min_dist2 * m2)# / jnp.sum(m2)

#         return mean_dist1 + mean_dist2
#     return vmap(chamfer_event)(points1, points2, mask1, mask2)
