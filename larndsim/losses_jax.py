import jax.numpy as jnp
from jax import jit
from larndsim.sim_jax import simulate, simulate_parametrized
from larndsim.fee_jax import digitize
from larndsim.detsim_jax import id2pixel
from larndsim.consts_jax import get_vdrift
from larndsim.softdtw_jax import SoftDTW


def mse_loss(adcs, pIDs, adcs_ref, pIDs_ref):
    all_pixels = jnp.concatenate([pIDs, pIDs_ref])
    unique_pixels = jnp.sort(jnp.unique(all_pixels))
    nb_pixels = unique_pixels.shape[0]
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
    return mse_loss(ticks, pixels, ticks_ref, pixels_ref)

def mse_time_adc(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, alpha=0.5):
    loss_adc, _ = mse_adc(adcs, pixels, ticks, ref, pixels_ref, ticks_ref)
    loss_time, _ = mse_time(adcs, pixels, ticks, ref, pixels_ref, ticks_ref)
    return alpha * loss_adc + (1 - alpha) * loss_time, dict()

def chamfer_3d(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref):
    pixel_x, pixel_y, pixel_plane, eventID = id2pixel(params, pixels)
    pixel_z = ticks * params.t_sampling/get_vdrift(params)/params.pixel_pitch
    mask = ticks.flatten() > 0
    #TODO: Think about using eventID
    pos_a = jnp.stack([jnp.repeat(pixel_x + eventID*500, 10)[mask], jnp.repeat(pixel_y, 10)[mask], pixel_z.flatten()[mask]], axis=1)
    pixel_x_ref, pixel_y_ref, pixel_plane_ref, eventID_ref = id2pixel(params, pixels_ref)
    pixel_z_ref = ticks_ref * params.t_sampling/get_vdrift(params)/params.pixel_pitch
    mask_ref = ticks_ref.flatten() > 0
    pos_b = jnp.stack([jnp.repeat(pixel_x_ref + eventID_ref*500, 10)[mask_ref], jnp.repeat(pixel_y_ref, 10)[mask_ref], pixel_z_ref.flatten()[mask_ref]], axis=1)
    return chamfer_distance_3d(pos_a, pos_b), dict()

@jit
def chamfer_distance_3d(pos_a, pos_b):
    """
    Compute the Chamfer Distance between two sets of 3D points (x, y, t).
    
    Parameters:
        pos_a: jnp.ndarray of shape (N, 3), positions and times of hits in distribution A.
        pos_b: jnp.ndarray of shape (M, 3), positions and times of hits in distribution B.
    
    Returns:
        A scalar representing the Chamfer Distance between the two point sets.
    """
    # Compute Euclidean distances in 3D between all pairs of points
    dists_a_to_b = jnp.sum((pos_a[:, None, :] - pos_b[None, :, :])**2, axis=2)
    dists_b_to_a = jnp.sum((pos_b[:, None, :] - pos_a[None, :, :])**2, axis=2)
    
    # Find minimum distance from each point in pos_a to pos_b and vice versa
    min_dists_a_to_b = jnp.min(dists_a_to_b, axis=1)
    min_dists_b_to_a = jnp.min(dists_b_to_a, axis=1)
    
    # Calculate Chamfer Distance as the average of these minimum distances
    chamfer_dist = jnp.mean(min_dists_a_to_b) + jnp.mean(min_dists_b_to_a)
    return chamfer_dist

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
    loss_adc, _ = sdtw_adc(adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw)
    loss_time, _ = sdtw_time(adcs, pixels, ticks, ref, pixels_ref, ticks_ref, dstw)
    return alpha * loss_adc + (1 - alpha) * loss_time, dict()

@jit
def cleaning_outputs(params, ref, adcs):
    #Cleaning up baselines to avoid big leaps
    adc_lowest = digitize(params, params.DISCRIMINATION_THRESHOLD)
    ref = jnp.where(ref < adc_lowest, 0, ref - adc_lowest)
    adcs = jnp.where(adcs < adc_lowest, 0, adcs - adc_lowest)
    return ref, adcs

def params_loss(params, response, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0, loss_fn=mse_adc, **loss_kwargs):
    adcs, pixels, ticks = simulate(params, response, tracks, fields, rngkey)

    ref, adcs = cleaning_outputs(params, ref, adcs)
    
    loss_val, aux = loss_fn(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, **loss_kwargs)
    return loss_val, aux

def params_loss_parametrized(params, ref, pixels_ref, ticks_ref, tracks, fields, rngkey=0, loss_fn=mse_adc, **loss_kwargs):
    adcs, pixels, ticks = simulate_parametrized(params, tracks, fields, rngkey)

    ref, adcs = cleaning_outputs(params, ref, adcs)
    
    loss_val, aux = loss_fn(params, adcs, pixels, ticks, ref, pixels_ref, ticks_ref, **loss_kwargs)
    return loss_val, aux