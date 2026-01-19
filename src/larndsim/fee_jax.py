"""
Module that simulates the front-end electronics (triggering, ADC)
"""

import jax.numpy as jnp
from jax.profiler import annotate_function
from jax import jit, vmap, lax, random, checkpoint
from jax.scipy.special import erf
from functools import partial

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

@jit
def get_adc_values_average_noise(params, pixels_signals):
    q = pixels_signals*params.t_sampling
    q_sum = q.cumsum(axis=-1)  # Cumulative sum over time ticks

    Nvalues_scaling = params.fee_paths_scaling
    Npix = pixels_signals.shape[0]
    Nvalues = Nvalues_scaling*Npix

    def find_hit(carry, _):
        q_sum_loc, previous_prob, pixid = carry # q_sum_loc[Nvalues, Nticks] ; previous_prob[Nvalues] ; pixid[Nvalues]
        sigma = params.RESET_NOISE_CHARGE #Found out that only considering the reset noise was sufficient
        # sigma = jnp.sqrt(params.RESET_NOISE_CHARGE**2 + params.UNCORRELATED_NOISE_CHARGE**2)
        eps = 1e-10

        _, Nticks = q_sum_loc.shape

        erf_term = erf((q_sum_loc - params.DISCRIMINATION_THRESHOLD)/(jnp.sqrt(2)*sigma)) # erf_term[Nvalues, Nticks]
        # erf_term_signal = jnp.frompyfunc(jnp.maximum, 2, 1).accumulate(erf_term, axis=-1) #erf is increasing so should be faster to make the re-ordering afterwards
        erf_term_signal = lax.cummax(erf_term, axis=1)  # erf_term_signal[Nvalues, Nticks]
        # max_future_signal = jnp.frompyfunc(jnp.maximum, 2, 1).accumulate(q_sum_loc[:, ::-1], axis=-1)[:, ::-1]
        max_future_signal = lax.cummax(q_sum_loc, axis=1, reverse=True)  # max_future_signal[Nvalues, Nticks]
        
        guess = 0.5*(erf_term_signal[..., 1:] - erf_term_signal[..., :-1]) # guess[Nvalues, Nticks - 1]
        # guess = jnp.clip(guess, 0, 1)  # Ensure guess is between 0 and 1, should not be needed but erf got odd behavior for some reason

        interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
        shifted_ticks = jnp.arange(Nticks - 1) + interval + 1 # shifted_ticks[Nticks - 1]
        shifted_ticks = jnp.clip(shifted_ticks, 0, Nticks - 1)

        prob_event = jnp.clip(0.5*(erf_term[..., shifted_ticks] - erf_term[..., :-1]), 0, guess)

        esperance_value = q_sum_loc[..., shifted_ticks] + params.DISCRIMINATION_THRESHOLD - 0.5*(q_sum_loc[..., 1:] + q_sum_loc[..., :-1]) # esperance_value[Nvalues, Nticks - 1]

        norm = jnp.clip(jnp.sum(prob_event, axis=-1) + eps, 0, 1)  # Adding a small epsilon to avoid division by zero ; norm[Nvalues]

        previous_prob_norm = jnp.full(Npix, eps, dtype=jnp.float32)  # Avoid division by zero
        previous_prob_norm = previous_prob_norm.at[pixid].add(previous_prob)

        no_hit_prob_across = jnp.ones(Npix, dtype=jnp.float32)  # Initialize no-hit probability across universes
        no_hit_prob_across = no_hit_prob_across.at[pixid].subtract(norm*previous_prob)  # No-hit probability across universes

        prob_distrib = prob_event*previous_prob[:, None] # prob_distrib[Nvalues, Nticks - 1]

        prob_distrib_across = jnp.zeros((Npix, Nticks - 1), dtype=jnp.float32)  # Initialize probability distribution across universes
        prob_distrib_across = prob_distrib_across.at[pixid].add(prob_distrib)  # prob_distrib_across[Npix, Nticks - 1]

        norm_across = jnp.sum(prob_distrib_across, axis=-1) + eps  # Normalize across universes ; norm_across[Npix]

        tick_avg = jnp.sum(prob_distrib_across/(norm_across[:, None])*jnp.arange(Nticks - 1), axis=-1)  # + 0.5

        charge_avg = jnp.sum(prob_event*esperance_value, axis=-1) # charge_avg[Nvalues] ; average charge for each path

        charge_avg_across = jnp.zeros((Npix,), dtype=jnp.float32)  # Initialize average charge across universes
        charge_avg_across = charge_avg_across.at[pixid].add(charge_avg*previous_prob)/norm_across  # charge_avg_across[Npix]

        future_hit_earliest_end = jnp.clip(shifted_ticks + interval + 1, 0, Nticks - 1) #We shift again
        future_hit_prob = 0.5*(1 + erf((max_future_signal[:, future_hit_earliest_end] - q_sum_loc[:, jnp.clip(shifted_ticks + 1, 0, Nticks - 1)] - params.DISCRIMINATION_THRESHOLD)/(jnp.sqrt(2)*sigma)))
        best_paths = jnp.argsort(prob_distrib*future_hit_prob, axis=None, descending=True)[:Nvalues] # best_paths[Nvalues]
        best_path_universe = best_paths // (Nticks - 1)
        best_path_ticks = best_paths % (Nticks - 1)
        best_path_pixid = pixid[best_path_universe]  # best_path_pixid[Nvalues]

        best_path_next_ticks = jnp.clip(shifted_ticks + 1, 0, Nticks - 1) # best_path_next_ticks[Nticks - 1]
        best_path_esperance = q_sum_loc[best_path_universe, best_path_next_ticks[best_path_ticks]]  # best_path_esperance[Nvalues]
        best_path_wfs = q_sum_loc[best_path_universe, :]
        q_sum_new = best_path_wfs - best_path_esperance[..., None]
        new_prob = prob_distrib[best_path_universe, best_path_ticks] # new_prob[Nvalues]
        return (q_sum_new, new_prob, best_path_pixid), (charge_avg_across, tick_avg, no_hit_prob_across)
    
    q_sum_multi = jnp.zeros((Nvalues, q_sum.shape[1]), dtype=jnp.float32)  # Initialize q_sum for multiple universes
    q_sum_multi = q_sum_multi.at[:q_sum.shape[0], :].set(q_sum)
    previous_prob = jnp.zeros(Nvalues, dtype=jnp.float32)  # Initialize previous probability
    previous_prob = previous_prob.at[:q_sum.shape[0]].set(1.0)  # Set the first universe probability to 1.0
    pixid = jnp.full((Nvalues,), -1, dtype=jnp.int32)  # Initialize pixel IDs for multiple universes
    pixid = pixid.at[:q_sum.shape[0]].set(jnp.arange(q_sum.shape[0]))  # Set pixel IDs for the first universe

    init_loop = (q_sum_multi, previous_prob, pixid)  # Initialize previous_prob
    _, (charge_avg, tick_avg, no_hit_prob) = lax.scan(find_hit, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    # new_loop, (charge_avg, tick_avg, no_hit_prob) = find_hit(init_loop, 0)
    # _, (charge_avg, tick_avg, no_hit_prob) = find_hit(new_loop, 1)

    return (charge_avg.T, tick_avg.T, no_hit_prob.T)

def select_roi(params, wfs):
    roi_threshold = params.roi_threshold
    Npix, Nticks = wfs.shape
    roi_start = jnp.argmax(wfs > roi_threshold, axis=1)
    roi_end = Nticks - jnp.argmax(wfs[:, ::-1] > roi_threshold, axis=1) - 1
    with_roi = roi_start > 0
    largest_roi = jnp.max((roi_end-roi_start)[with_roi])
    wfs_with_roi = wfs.at[jnp.arange(Npix)[:, None], (jnp.arange(largest_roi) + roi_start[:, None])].get(unique_indices=True)
    return wfs_with_roi, roi_start


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
        ic = ic.at[idx_pix].set(idx_t)

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
        q_vals = q_sum[end2d_idx] #+ (idx_val - idx_t) *(q_sum[end2d_idx_next] - q_sum[end2d_idx])
        q_vals_no_noise = q_cumsum[end2d_idx] #+ (idx_val - idx_t) *(q_cumsum[end2d_idx_next] - q_cumsum[end2d_idx])

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
        ic = jnp.where(cond_adc, q_sum.shape[1]-2, ic)

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

# def _find_one_hit_step(q_sum, prev_charges, previous_prob, sigma, threshold, Nvalues):
#     """
#     Calculates a single hit-finding step for one pixel. This function is designed
#     to be vmapped across all pixels.
#     """
#     # This function contains the logic from the previous `_active_branch`.
#     Nticks = q_sum.shape[0]
#     inv_sqrt2_sigma = 1.0 / (jnp.sqrt(2) * sigma)
#     eps = 1e-9 # Using a small internal epsilon
#     interval = 18
#     shifted_ticks = jnp.arange(Nticks - 1) + interval + 1
#     shifted_ticks = jnp.clip(shifted_ticks, 0, Nticks - 1)
#     tick_indices = jnp.arange(Nticks - 1)

#     q_sum_loc = q_sum - prev_charges[..., None]

#     # Steps 1-2: Calculate Event Probabilities
#     erf_term = erf((q_sum_loc - threshold) * inv_sqrt2_sigma)
#     erf_term_signal = lax.cummax(erf_term, axis=1)
#     max_future_signal = lax.cummax(q_sum_loc, axis=1, reverse=True)
#     guess = 0.5 * jnp.diff(erf_term_signal, axis=-1)
#     prob_event = jnp.clip(0.5 * (erf_term[..., shifted_ticks] - erf_term[..., :-1]), 0, guess)
#     esperance_value = q_sum_loc[..., shifted_ticks] + threshold - 0.5 * (q_sum_loc[..., 1:] + q_sum_loc[..., :-1])

#     # Step 3: Aggregate Results
#     prob_distrib = prob_event * previous_prob[:, None]
#     total_hit_prob_per_path = jnp.sum(prob_event, axis=-1) * previous_prob
#     no_hit_prob = 1.0 - jnp.sum(total_hit_prob_per_path)
#     total_distrib_prob_per_tick = jnp.sum(prob_distrib, axis=0)
#     norm_across = jnp.sum(total_hit_prob_per_path)
#     safe_norm_across = jnp.maximum(norm_across, eps)
#     tick_avg = jnp.sum(total_distrib_prob_per_tick * tick_indices) / safe_norm_across
#     charge_avg_paths = jnp.sum(prob_event * esperance_value, axis=-1)
#     charge_avg = jnp.sum(charge_avg_paths * previous_prob) / safe_norm_across
    
#     # Step 4: Optimized Merging & Selection
#     future_hit_earliest_end = jnp.clip(shifted_ticks + interval + 1, 0, Nticks - 1)
#     next_q_sum = q_sum_loc[:, jnp.clip(shifted_ticks + 1, 0, Nticks - 1)]
#     future_hit_prob = 0.5 * (1 + erf((max_future_signal[:, future_hit_earliest_end] - next_q_sum - threshold) * inv_sqrt2_sigma))
#     path_selection_prob = prob_distrib * future_hit_prob
#     total_prob_per_tick = jnp.sum(path_selection_prob, axis=0)
#     _, top_k_ticks = lax.top_k(total_prob_per_tick, k=Nvalues)
    
#     # Step 5: Construct the State for the Next Iteration
#     new_prob = total_distrib_prob_per_tick[top_k_ticks]
#     best_path_next_ticks_indices = jnp.clip(shifted_ticks[top_k_ticks] + 1, 0, Nticks - 1)
#     charges_new = q_sum[best_path_next_ticks_indices]
    
#     # Return the new state and the outputs for this step for this pixel
#     return (charges_new, new_prob), (charge_avg, tick_avg, no_hit_prob, total_distrib_prob_per_tick)
#     # return (charges_new, new_prob), (charge_avg_across, tick_avg, no_hit_prob_across, prob_distrib_across)


def _find_one_hit_step(q_sum, prev_charges, previous_prob, sigma, threshold, interval, Nvalues):
    """
    Calculates a single hit-finding step for one pixel. This function is designed
    to be vmapped across all pixels.
    """
    # This function contains the logic from the previous `_active_branch`.
    Nticks = q_sum.shape[0]
    inv_sqrt2_sigma = 1.0 / (jnp.sqrt(2) * sigma)
    shifted_ticks = jnp.arange(Nticks - 1) + interval + 1
    shifted_ticks = jnp.clip(shifted_ticks, 0, Nticks - 1)
    tick_indices = jnp.arange(Nticks - 1)

    q_sum_loc = q_sum - prev_charges[..., None]

    # Steps 1-2: Calculate Event Probabilities
    erf_term = erf((q_sum_loc - threshold) * inv_sqrt2_sigma)
    erf_term_signal = lax.cummax(erf_term, axis=1)
    max_future_signal = lax.cummax(q_sum_loc, axis=1, reverse=True)
    guess = 0.5 * jnp.diff(erf_term_signal, axis=-1)
    prob_event = jnp.clip(0.5 * (erf_term[..., shifted_ticks] - erf_term[..., :-1]), 0, guess)
    esperance_value = q_sum_loc[..., shifted_ticks] + threshold - 0.5 * (q_sum_loc[..., 1:] + q_sum_loc[..., :-1])

    # Step 3: Aggregate Results
    prob_distrib = prob_event * previous_prob[:, None]
    total_distrib_prob_per_tick = jnp.sum(prob_distrib, axis=0)

    total_hit_prob_per_path = jnp.sum(prob_event, axis=-1) * previous_prob
    no_hit_prob = 1.0 - jnp.sum(total_hit_prob_per_path)
    
    norm_across = jnp.sum(total_hit_prob_per_path)
    safe_norm_across = jnp.maximum(norm_across, 1e-9)
    
    tick_avg = jnp.sum(total_distrib_prob_per_tick * tick_indices) / safe_norm_across
    
    charge_avg_paths = jnp.sum(prob_event * esperance_value, axis=-1)
    charge_avg = jnp.sum(charge_avg_paths * previous_prob) / safe_norm_across
    
    # Step 4: Optimized Merging & Selection
    future_hit_earliest_end = jnp.clip(shifted_ticks + interval + 1, 0, Nticks - 1)
    next_q_sum = q_sum_loc[:, jnp.clip(shifted_ticks + 1, 0, Nticks - 1)]
    future_hit_prob = 0.5 * (1 + erf((max_future_signal[:, future_hit_earliest_end] - next_q_sum - threshold) * inv_sqrt2_sigma))
    path_selection_prob = prob_distrib * future_hit_prob
    total_prob_per_tick = jnp.sum(path_selection_prob, axis=0)
   
    
    # Step 5: Construct the State for the Next Iteration
    _, top_k_ticks = lax.top_k(total_prob_per_tick, k=Nvalues)
    new_prob = total_distrib_prob_per_tick[top_k_ticks]
    best_path_next_ticks_indices = jnp.clip(shifted_ticks[top_k_ticks] + 1, 0, Nticks - 1)
    charges_new = q_sum[best_path_next_ticks_indices]
    
    # Return the new state and the outputs for this step for this pixel
    return (charges_new, new_prob), (charge_avg, tick_avg, no_hit_prob, total_distrib_prob_per_tick)
    # return (charges_new, new_prob), (charge_avg_across, tick_avg, no_hit_prob_across, prob_distrib_across)

@partial(jit, static_argnums=(2))
def get_adc_values_average_noise_vmap(params, wfs, stop_threshold=1e-9):
    """
    A globally-stopped, vmap-based implementation of the beam search. This is a
    highly optimized pattern for parallel hardware.
    """
    Npix, Nticks = wfs.shape
    Nvalues = params.fee_paths_scaling
    interval = round((3 * params.CLOCK_CYCLE + params.ADC_HOLD_DELAY * params.CLOCK_CYCLE) / params.t_sampling)
    
    # --- Vectorize the single-step function ---
    vmapped_step_fun = vmap(
        _find_one_hit_step,
        in_axes=(0, 0, 0, None, None, None, None) # Map over q_sum, charges, probs
    )

    # --- Pre-calculate q_sum for all pixels ---
    q = wfs * params.t_sampling
    q_sum_all = q.cumsum(axis=-1)
    

    def global_scan_fun(carry, _):
        """The main scan function with a global conditional stop."""
        prev_charges_all, prev_probs_all, is_active_global = carry

        # Apply jax.checkpoint to the expensive branch of the computation.
        # This tells JAX not to store intermediate values from this function,
        # saving memory at the cost of some re-computation.
        @checkpoint
        def _active_branch(operand):
            """The expensive vmapped computation, only run when globally active."""
            charges, probs, _ = operand
            # Run one hit-finding step for all pixels in parallel
            (new_charges, new_probs), (charge_avg, tick_avg, no_hit_prob, prob_dist) = vmapped_step_fun(
                q_sum_all, charges, probs, params.RESET_NOISE_CHARGE, params.DISCRIMINATION_THRESHOLD, interval, 
                Nvalues
            )
            # Check if ANY pixel is still active for the next iteration
            new_active_flag = jnp.any(jnp.sum(new_probs, axis=1) > stop_threshold)

            return (new_charges, new_probs, new_active_flag), (charge_avg, tick_avg, no_hit_prob, prob_dist)

        def _inactive_branch(operand):
            """A cheap pass-through, executed when the whole batch is inactive."""
            return operand, (
                jnp.zeros((Npix,), dtype=jnp.float32),
                jnp.zeros((Npix,), dtype=jnp.float32),
                jnp.zeros((Npix,), dtype=jnp.float32),
                jnp.zeros((Npix, Nticks - 1), dtype=jnp.float32)
                )

        # --- Global Conditional Execution ---
        final_carry, final_outputs = lax.cond(
            is_active_global,
            _active_branch,
            _inactive_branch,
            carry
        )
        return final_carry, final_outputs

    # --- Setup and Execute the Global Scan ---
    initial_charges = jnp.zeros((Npix, Nvalues), dtype=jnp.float32)
    initial_probs = jnp.zeros((Npix, Nvalues), dtype=jnp.float32).at[:, 0].set(1.0)
    initial_active = jnp.array(True)
    
    init_loop = (initial_charges, initial_probs, initial_active)

    _, (charge_avg, tick_avg, no_hit_prob, prob_distrib) = lax.scan(global_scan_fun, init_loop, jnp.arange(0, params.MAX_ADC_VALUES))
    # next_loop, (charge_avg_0, tick_avg_0, no_hit_prob_0, prob_distrib_0) = vmapped_step_fun(
    #             q_sum_all, init_loop[0], init_loop[1], params.RESET_NOISE_CHARGE, params.DISCRIMINATION_THRESHOLD, Nvalues
    #         )
    # next_loop, (charge_avg_1, tick_avg_1, no_hit_prob_1, prob_distrib_1) = vmapped_step_fun(
    #             q_sum_all, next_loop[0], next_loop[1], params.RESET_NOISE_CHARGE, params.DISCRIMINATION_THRESHOLD, Nvalues
    #         )

    # charge_avg = jnp.stack([charge_avg_0, charge_avg_1], axis=0)
    # tick_avg = jnp.stack([tick_avg_0, tick_avg_1], axis=0)
    # no_hit_prob = jnp.stack([no_hit_prob_0, no_hit_prob_1], axis=0)
    # prob_distrib = jnp.stack([prob_distrib_0, prob_distrib_1], axis=0)

    # Transpose the outputs to match the desired shape (Npix, num_steps, ...)
    return jnp.moveaxis(charge_avg, 0, 1), jnp.moveaxis(tick_avg, 0, 1), jnp.moveaxis(no_hit_prob, 0, 1), jnp.moveaxis(prob_distrib, 0, 1)