"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import jit, vmap, lax, random, debug

@annotate_function
@jit
def digitize(params, integral_list):
    """
    The function takes as input the integrated charge and returns the digitized
    ADC counts.

    Args:
        integral_list (:obj:`numpy.ndarray`): list of charge collected by each pixel

    Returns:
        numpy.ndarray: list of ADC values for each pixel
    """
    adcs = jnp.minimum((jnp.maximum((integral_list*params.GAIN+params.V_PEDESTAL - params.V_CM), 0) \
                        * params.ADC_COUNTS/(params.V_REF-params.V_CM)), params.ADC_COUNTS)

    return adcs

@annotate_function
@jit
def get_adc_values(params, pixels_signals, noise_rng_key):
    """
    Implementation of self-trigger logic

    Args:
        pixels_signals (:obj:`numpy.ndarray`): list of induced currents for
            each pixel
        time_ticks (:obj:`numpy.ndarray`): list of time ticks for each pixel
        adc_list (:obj:`numpy.ndarray`): list of integrated charges for each
            pixel
        adc_ticks_list (:obj:`numpy.ndarray`): list of the time ticks that
            correspond to each integrated charge.
    """

    #Baseline level of noise on integrated charge

    q_sum_base = random.normal(noise_rng_key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE

    # Charge
    q = pixels_signals*params.t_sampling

    # Collect cumulative charge over all time ticks + add baseline noise
    # q_cumsum = q #Assuming the input is cumulated signal
    q_cumsum = q.cumsum(axis=-1)  # Cumulative sum over time ticks
    q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum


    def find_hit(carry, it):
        key, q_sum, q_cumsum = carry
        # Index of first threshold passing. For nice time axis differentiability: first find index window around threshold.
        selec_func = lambda x: jnp.where((x[1:] >= params.DISCRIMINATION_THRESHOLD) & 
                    (x[:-1] <= params.DISCRIMINATION_THRESHOLD), size=1, fill_value=q_sum.shape[1]-2)
        idx_t, = vmap(selec_func, 0, 0)(q_sum)
        idx_t = idx_t.ravel()
        idx_pix = jnp.arange(0, q_sum.shape[0])
        # Then linearly interpolate for the intersection point.

        dq = jnp.where(q_sum[idx_pix, idx_t + 1] == q_sum[idx_pix, idx_t], q_sum[idx_pix, idx_t + 1] - params.DISCRIMINATION_THRESHOLD, q_sum[idx_pix, idx_t + 1] - q_sum[idx_pix, idx_t])

        idx_val = idx_t + 1 - (q_sum[idx_pix, idx_t+1] - params.DISCRIMINATION_THRESHOLD)/dq
        # debug.print("idx_val: {idx_val}", idx_val=idx_val)

        ic = jnp.zeros((q_sum.shape[0],))
        ic = ic.at[idx_pix].set(idx_val)

        # End point of integration
        interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
        # integrate_end = ic+interval

        #Protect against ic+integrate_end past last index
        #TODO: This is really weird. How is this thing even differentiable?
        #TODO: Hardfixing to something reasonable, clearly not diff
        integrate_end = idx_t + 1 + interval
        integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end]))
        # integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
        # end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end.astype(int)]))

        end2d_idx_next = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), jnp.clip(integrate_end + 1, 0, q_sum.shape[1]-1)]))
        q_vals = q_sum[end2d_idx] + (idx_val - idx_t) *(q_sum[end2d_idx_next] - q_sum[end2d_idx])
        q_vals_no_noise = q_cumsum[end2d_idx] + (idx_val - idx_t) *(q_cumsum[end2d_idx_next] - q_cumsum[end2d_idx])

        # Cumulative => value at end is desired value
        # q_vals = q_sum[end2d_idx] 
        # q_vals_no_noise = q_cumsum[end2d_idx]

        # Uncorrelated noise
        key, = random.split(key, 1)
        extra_noise = random.normal(key, (pixels_signals.shape[0],))  * params.UNCORRELATED_NOISE_CHARGE

        # Only include noise if nonzero
        adc = jnp.where(q_vals_no_noise != 0, q_vals + extra_noise, q_vals_no_noise)

        cond_adc = adc < params.DISCRIMINATION_THRESHOLD
        #TODO: Check if the following selection makes sense
        cond_adc = jnp.logical_or(cond_adc, idx_t == q_sum.shape[1]-2) #Force set to zero if no initial threshold crossing

        # Only include if passes threshold     
        adc = jnp.where(cond_adc, 0, adc)

        # Setup for next loop: baseline noise set to based on adc passing disc. threshold
        key, = random.split(key, 1)
        q_adc_pass = random.normal(key, (pixels_signals.shape[0],)) * params.RESET_NOISE_CHARGE
        key, = random.split(key, 1)
        q_adc_fail = random.normal(key, (pixels_signals.shape[0],)) * params.UNCORRELATED_NOISE_CHARGE
        q_sum_base = jnp.where(cond_adc, q_adc_fail, q_adc_pass)

        # Remove charge already counted
        integrate_end = idx_t + 1 + interval + 1 #Additional +1 present in the original code 
        integrate_end = jnp.where(integrate_end >= q_sum.shape[1], q_sum.shape[1]-1, integrate_end)
        end2d_idx = tuple(jnp.stack([jnp.arange(0, ic.shape[0]).astype(int), integrate_end]))
        q_vals_no_noise = q_cumsum[end2d_idx]
        q_cumsum = q_cumsum - q_vals_no_noise[..., jnp.newaxis]
        q_cumsum = jnp.where(q_cumsum < 0, 0, q_cumsum)
        q_sum = q_sum_base[:, jnp.newaxis] + q_cumsum
        

        return (key, q_sum, q_cumsum), (adc, ic)


    init_loop = (random.split(noise_rng_key, 1)[0], q_sum, q_cumsum)
    
    _, (full_adc, full_ticks) = lax.scan(find_hit, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    # Single iteration to detect NaNs
    # _, (full_adc, full_ticks) = find_hit(init_loop, 0)
    # full_adc = jnp.repeat(full_adc[:, jnp.newaxis], params.MAX_ADC_VALUES, axis=1).T
    # full_ticks = jnp.repeat(full_ticks[:, jnp.newaxis], params.MAX_ADC_VALUES, axis=1).T

    return full_adc.T, full_ticks.T
