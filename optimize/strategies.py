import jax
import jax.numpy as jnp
from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_parametrized, simulate_probabilistic, simulate_markov, pad_size, parse_output
from larndsim.losses_jax import adc2charge, weighted_wasserstein_1d
from larndsim.detsim_jax import id2pixel, get_pixel_coordinates, get_hit_z
from larndsim.fee_jax import get_average_hit_values, digitize
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@jax.jit
def compute_occurrence_indices(ids):
    """
    Compute occurrence index (0, 1, 2, ...) for each ID in the array.

    For sorted IDs, this counts how many times each ID has appeared so far.
    Example: [100, 100, 100, 200, 200, 300] -> [0, 1, 2, 0, 1, 0]

    Args:
        ids: Array of IDs (should be sorted for meaningful results)

    Returns:
        occurrence_indices: Array where each element is its occurrence count within its ID group
    """
    id_changes = jnp.concatenate([
        jnp.array([True]), # First element is always a new group
        ids[1:] != ids[:-1]  # Compare consecutive elements
    ])

    cumsum = jnp.cumsum(jnp.ones_like(ids, dtype=jnp.int32))
    reset_values = jnp.where(id_changes, cumsum, 0)

    # JAX equivalent of np.maximum.accumulate
    reset_at_boundary = jax.lax.associative_scan(jnp.maximum, reset_values)

    occurrence_indices = cumsum - reset_at_boundary
    return occurrence_indices

def pad_to_closest_multiple(x, dims_to_pad=None, multiple=128, pad_value=0, pad_front=False):
    """
    Efficiently pads array x to the closest multiple of a given value using update-in-place syntax.
    Works with arrays of any number of dimensions.
    
    Args:
        x: Input array to pad
        dims_to_pad: List of dimension indices to pad (default: all dimensions)
        multiple: The multiple to pad to (default: 128)
        pad_value: Value to use for padding (default: 0)
    
    Returns:
        Padded array with shape target_shape
    """

    # Compute target shape by padding each dimension to the closest multiple
    if dims_to_pad is None:
        dims_to_pad = range(x.ndim)
    target_shape = list(x.shape)
    for dim in dims_to_pad:
        target_shape[dim] = ((x.shape[dim] + multiple - 1) // multiple) * multiple
    target_shape = tuple(target_shape)

    logger.info(f"Padding from shape {x.shape} to target shape {target_shape} with pad value {pad_value}")


    # 1. Create a buffer of the target static shape (allocates memory)
    buffer = jnp.full(target_shape, pad_value, dtype=x.dtype)
    
    # 2. Copy 'x' into the start of the buffer
    # Create slice tuple for all dimensions: [:x.shape[0], :x.shape[1], ...]
    if pad_front:
        slices = tuple(slice(target_shape[idim] - dim_size, None) for idim, dim_size in enumerate(x.shape))
    else:
        slices = tuple(slice(0, dim_size) for dim_size in x.shape)
    padded_x = buffer.at[slices].set(x)
    
    return padded_x

class SimulationStrategy:
    def predict(self, params, tracks, fields, rngkey):
        """
        Runs the simulation and returns a dictionary of outputs.
        """
        raise NotImplementedError

class LUTSimulation(SimulationStrategy):
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)
        adcs, x, y, z, ticks, hit_prob, event, hit_pixels = simulate_stochastic(params, wfs, unique_pixels, rngseed=rngkey)
        return {
            'adcs': adcs,
            'pixel_x': x,
            'pixel_y': y,
            'pixel_z': z,
            'ticks': ticks,
            'hit_prob': hit_prob,
            'event': event,
            'hit_pixels': hit_pixels,
            'unique_pixels': unique_pixels,
            'wfs': wfs
        }

class LUTProbabilisticSimulation(SimulationStrategy):
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)

        unique_pixels = pad_to_closest_multiple(unique_pixels, multiple=128, pad_value=-1, pad_front=True)
        wfs = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)

        adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(params, wfs, unique_pixels)
        
        # Extract pixel plane for z-coordinate calculation
        _, _, pixel_plane, _ = id2pixel(params, unique_pixels)
        
        # We return the raw distributions for the ProbabilisticLossStrategy
        return {
            'adcs_distrib': adcs_distrib, # (Npix, Nvalues, Nticks)
            'hit_prob': ticks_prob,     # (Npix, Nvalues, Nticks)
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'pixel_plane': pixel_plane,   # Needed for z-coordinate calculation
            'event': event,
            'unique_pixels': unique_pixels, 
            'hit_pixels': unique_pixels,
            'wfs': wfs
        }


class LUTProbabilisticSamplingSimulation(SimulationStrategy):
    """
    Runs the probabilistic waveform simulation, then draws one stochastic
    realisation from the resulting hit-probability distributions.

    Sampling algorithm per (pixel, hit_slot):
        1. λ = Σ_t exp(hit_prob[pix, slot, t])
        2. has_hit ~ Bernoulli(λ)          [Poisson(λ)→Bernoulli(λ) for λ«1]
        3. if has_hit: t_drawn ~ Categorical(exp(hit_prob) / λ)
        4. if has_hit: q_drawn ~ N(q_expected(t_drawn), UNCORRELATED_NOISE_CHARGE [e-])
        5. adc_drawn = digitize(q_drawn)

    Output format is **identical to LUTSimulation** (flat sorted hit lists),
    so it is fully compatible with all existing LossStrategies.
    """
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        # ── Step 1: waveforms + probabilistic distributions ──────────────────
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)
        unique_pixels = pad_to_closest_multiple(
            unique_pixels, multiple=128, pad_value=-1, pad_front=True)
        wfs = pad_to_closest_multiple(
            wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)

        adcs_distrib, pixel_x_pix, pixel_y_pix, ticks_prob, event_pix = \
            simulate_probabilistic(params, wfs, unique_pixels)
        # ticks_prob:   (Npix, Nhits, Nticks) – log-intensities
        # adcs_distrib: (Npix, Nhits, Nticks) – expected ADC at each tick

        # ── Step 2: sample one realisation ───────────────────────────────────
        Npix, Nhits, Nticks = ticks_prob.shape

        # λ per (pixel, hit_slot): expected number of hits
        lam = jnp.sum(jnp.exp(ticks_prob), axis=-1)          # (Npix, Nhits)

        key = jax.random.key(rngkey if rngkey is not None else 0)
        key_bern, key_tick_base, key_noise = jax.random.split(key, 3)

        # Bernoulli hit existence
        u       = jax.random.uniform(key_bern, shape=(Npix, Nhits))
        has_hit = u < lam                                      # (Npix, Nhits)

        # Categorical tick sampling — one key per (pixel, hit_slot) flat slot
        logits_flat = ticks_prob.reshape(Npix * Nhits, Nticks)
        tick_keys   = jax.random.split(key_tick_base, Npix * Nhits)
        drawn_ticks_flat = jax.vmap(
            lambda logits, k: jax.random.categorical(k, logits)
        )(logits_flat, tick_keys)
        drawn_ticks = drawn_ticks_flat.reshape(Npix, Nhits)    # (Npix, Nhits)

        # Expected charge (in electrons) at drawn tick + Gaussian noise
        pix_idx  = jnp.arange(Npix)[:, None]
        hit_idx  = jnp.arange(Nhits)[None, :]
        exp_adc  = adcs_distrib[pix_idx, hit_idx, drawn_ticks] # (Npix, Nhits)
        # adc2charge returns ke-; multiply by 1000 to get electrons for digitize
        exp_q_e  = adc2charge(exp_adc, params) * 1000.0        # (Npix, Nhits) [e-]
        sigma_e  = params.UNCORRELATED_NOISE_CHARGE             # [e-]
        noise    = jax.random.normal(key_noise, shape=(Npix, Nhits)) * sigma_e
        drawn_q_e = jnp.clip(exp_q_e + noise, 0.0, None)       # no negative charge
        drawn_adc = digitize(params, drawn_q_e)                 # (Npix, Nhits) [ADC]

        # Use Nticks as sentinel for no-hit slots (same convention as simulate_stochastic)
        drawn_ticks = jnp.where(has_hit, drawn_ticks, Nticks)
        drawn_adc   = jnp.where(has_hit, drawn_adc,  jnp.zeros((Npix, Nhits)))

        # ── Step 3: pixel coordinates and z per hit ───────────────────────────
        px, py, pixel_plane, ev = id2pixel(params, unique_pixels)  # (Npix,) each
        pixel_coords = get_pixel_coordinates(params, px, py, pixel_plane)
        pixel_x_flat = pixel_coords[:, 0]
        pixel_y_flat = pixel_coords[:, 1]
        pixel_z_flat = get_hit_z(params, drawn_ticks.flatten(),
                                  jnp.repeat(pixel_plane, Nhits))  # (Npix*Nhits,)

        # ── Step 4: JIT-safe flatten via parse_output ─────────────────────────
        # hit_prob: 1.0 for valid hits, 0.0 for no-hit slots
        hit_prob_2d = jnp.where(drawn_ticks < Nticks - 3, 1.0, 0.0)  # (Npix, Nhits)

        (drawn_adc_out, pixel_x_out, pixel_y_out, pixel_z_out,
         drawn_ticks_out, hit_prob_out, event_out, hit_pixels_out, nb_valid) = \
            parse_output(params, drawn_adc, pixel_x_flat, pixel_y_flat,
                         pixel_z_flat.reshape(Npix, Nhits),
                         drawn_ticks, hit_prob_2d, ev, unique_pixels)

        return {
            'adcs':          drawn_adc_out[:nb_valid],
            'pixel_x':       pixel_x_out[:nb_valid],
            'pixel_y':       pixel_y_out[:nb_valid],
            'pixel_z':       pixel_z_out[:nb_valid],
            'ticks':         drawn_ticks_out[:nb_valid],
            'hit_prob':      hit_prob_out[:nb_valid],
            'event':         event_out[:nb_valid],
            'hit_pixels':    hit_pixels_out[:nb_valid],
            'unique_pixels': unique_pixels,
            'wfs':           wfs,
        }


class LUTMarkovSimulation(SimulationStrategy):
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)
        
        # Pad to handle JIT compilation efficiency for variable sized inputs
        unique_pixels = pad_to_closest_multiple(unique_pixels, multiple=128, pad_value=-1, pad_front=True)
        wfs = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
        
        log_p1, log_T, expected_Q, log_p_none, Q1, log_p_none_at_zero, pixel_x, pixel_y, pixel_plane, event = simulate_markov(params, wfs, unique_pixels)
        return {
            'log_p1': log_p1,
            'log_T': log_T,
            'expected_Q': expected_Q,
            'log_p_none': log_p_none,
            'Q1': Q1,
            'log_p_none_at_zero': log_p_none_at_zero,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'pixel_plane': pixel_plane,
            'event': event,
            'unique_pixels': unique_pixels,
            'hit_pixels': unique_pixels,
            'wfs': wfs
        }

class ParametrizedSimulation(SimulationStrategy):
    def predict(self, params, tracks, fields, rngkey):
        adcs, x, y, z, ticks, hit_prob, event, unique_pixels = simulate_parametrized(params, tracks, fields, rngseed=rngkey)
        return {
            'adcs': adcs,
            'pixel_x': x,
            'pixel_y': y,
            'pixel_z': z,
            'ticks': ticks,
            'hit_prob': hit_prob,
            'event': event,
            'unique_pixels': unique_pixels

        }

class LossStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute(self, params, prediction, target):
        """
        Computes the loss between prediction (dict) and target (dict).
        Target is expected to have keys like 'adcs', 'pixel_x', etc.
        """
        raise NotImplementedError

class GenericLossStrategy(LossStrategy):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute(self, params, prediction, target):
        # We need to adapt the dict output to the function signature of loss_fn
        # Most loss functions in losses_jax.py expect:
        # params, Q, x, y, z, ticks, hit_prob, event, ref_Q, ref_x, ref_y, ref_z, ref_ticks, ref_hit_prob, ref_event
        
        Q = adc2charge(prediction['adcs'], params)
        ref_Q = adc2charge(target['adcs'], params)

        return self.loss_fn(
            params, 
            Q, prediction['pixel_x'], prediction['pixel_y'], prediction['pixel_z'], prediction['ticks'], prediction['hit_prob'], prediction['event'],
            ref_Q, target['pixel_x'], target['pixel_y'], target['pixel_z'], target['ticks'], target['hit_prob'], target['event'],
            **self.kwargs
        )

class CollapsedProbabilisticLossStrategy(LossStrategy):
    def __init__(self, loss_fn, hit_threshold=1e-8, collapsed=True, prob_target=False, **kwargs):
        """
        Collapses probabilistic distributions into expected values and applies a deterministic loss.
        
        For each predicted pixel:
        - Computes λ = Σ_t P(tick|pixel) = expected number of hits
        - If λ > threshold: generates a "pseudo-hit" with expected tick and charge
        - Applies the provided loss_fn as if these were sampled hits
        
        This allows using existing loss functions (MSE, Chamfer, etc.) with probabilistic predictions.
        
        Args:
            loss_fn: A loss function with signature (params, Q, x, y, z, ticks, ..., ref_Q, ref_x, ...)
            hit_threshold: Minimum λ to generate a pseudo-hit (default 1e-8)
        """
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.hit_threshold = hit_threshold
        self.collapsed = collapsed
        self.prob_target = prob_target
    # def _generate_pseudo_hits(self, ticks_prob, adcs_distrib):
    #     Npix, Nhits, Nticks = ticks_prob.shape
    #     expected_ticks_per_hit, expected_adcs_per_hit, lambda_per_hit = get_average_hit_values(ticks_prob, adcs_distrib)
    #     # Filter out hits with negligible probability
    #     has_hit_mask = lambda_per_hit > self.hit_threshold  # (Npix, Nhits)
        
    #     # Flatten to create list of pseudo-hits
    #     # We need to replicate pixel coordinates for each hit
    #     pred_ticks = expected_ticks_per_hit[has_hit_mask]  # (N_total_hits,)
    #     pred_adcs = expected_adcs_per_hit[has_hit_mask]  # (N_total_hits,)
    #     pred_lambda = lambda_per_hit[has_hit_mask]  # (N_total_hits,)
        
    #     # For pixel coordinates, we need to replicate them for each hit
    #     # Create indices for which pixel each hit belongs to
    #     pixel_indices = jnp.arange(Npix)[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)  # (Npix, Nhits)
    #     pred_pixel_idx = pixel_indices[has_hit_mask]  # (N_total_hits,)
        
        
    #     return pred_ticks, pred_adcs, pred_lambda, pred_pixel_idx

    def _generate_distribution_hits(self, params, output):
        # This function can be used to prepare the probabilistic output for loss computation
        # For example, it can compute expected values or filter out low-probability hits
        ticks_prob = output['hit_prob']
        adcs_distrib = output['adcs_distrib']
        pixel_x = output['pixel_x']
        pixel_y = output['pixel_y']

        Npix, Nhits, Nticks = ticks_prob.shape

        mask = ticks_prob > self.hit_threshold

        
        hit_prob = ticks_prob[mask]
        hit_adc = adcs_distrib[mask]
        Q = adc2charge(hit_adc, params)

        all_ticks = jnp.arange(Nticks)[None, None, :]*jnp.ones((Npix, Nhits, Nticks))
        selected_ticks = all_ticks[mask]

        # Get z-coordinates and event IDs from prediction (if available)
        # If not available, compute from drift time or use same default as target
        if 'pixel_z' in output:
            # If prediction has pixel_z (from stochastic simulation), replicate for each hit
            pred_z = output['pixel_z']
        else:
            # For probabilistic predictions without z, compute from drift time
            # z = v_drift * t_drift (same approach as in simulate_stochastic)
            # Get pixel plane for z calculation
            pixel_plane = output.get('pixel_plane')
            selected_planes = (pixel_plane[:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32))[mask]
            pred_z = get_hit_z(params, selected_ticks, selected_planes)

        # Event is per-pixel, replicate for each hit
        pred_event_per_pixel = output['event'][:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32)  # (Npix, Nhits, Nticks)
        pixel_x_per_event = pixel_x[:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32)
        pixel_y_per_event = pixel_y[:, None, None] * jnp.ones((Npix, Nhits, Nticks), dtype=jnp.int32)

        return Q, pixel_x_per_event[mask], pixel_y_per_event[mask], pred_z, selected_ticks, hit_prob, pred_event_per_pixel[mask]

    def _prepare_probabilistic_output(self, params, output):
        # This function can be used to prepare the probabilistic output for loss computation
        # For example, it can compute expected values or filter out low-probability hits
        ticks_prob = output['hit_prob']
        adcs_distrib = output['adcs_distrib']
        pixel_x = output['pixel_x']
        pixel_y = output['pixel_y']
        
        expected_ticks_per_hit, expected_adcs_per_hit, hit_prob = get_average_hit_values(ticks_prob, adcs_distrib)
        Npix, Nhits, Nticks = ticks_prob.shape

        Q = adc2charge(expected_adcs_per_hit, params)

        # Get z-coordinates and event IDs from prediction (if available)
        # If not available, compute from drift time or use same default as target
        if 'pixel_z' in output:
            # If prediction has pixel_z (from stochastic simulation), replicate for each hit
            pred_z = output['pixel_z']
        else:
            # For probabilistic predictions without z, compute from drift time
            # z = v_drift * t_drift (same approach as in simulate_stochastic)
            # Get pixel plane for z calculation
            pixel_plane = output.get('pixel_plane')
            pred_z = get_hit_z(params, expected_ticks_per_hit, pixel_plane[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32))

        # Event is per-pixel, replicate for each hit
        pred_event_per_pixel = output['event'][:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)  # (Npix, Nhits)
        pixel_x_per_event = pixel_x[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)
        pixel_y_per_event = pixel_y[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)

        return Q, pixel_x_per_event, pixel_y_per_event, pred_z, expected_ticks_per_hit, hit_prob, pred_event_per_pixel
        
        

    def compute(self, params, prediction, target):
        """
        Convert probabilistic predictions to pseudo-hits and apply deterministic loss.
        
        Important: ticks_prob and adcs_distrib have shape (Npix, Nhits, Nticks), where:
        - Npix: number of pixels
        - Nhits: maximum number of triggered hits per pixel (different hits, not charge values)
        - Nticks: time ticks
        
        Each (pixel, hit_index) combination should be treated independently.
        """

        if self.prob_target:
            ref_Q, target_x_per_event, target_y_per_event, ref_z, target_ticks, ref_hit_prob, ref_event = self._prepare_probabilistic_output(params, target)
        else:
            ref_Q = adc2charge(target['adcs'], params)
            ref_z = target['pixel_z']
            ref_event = target.get('event', jnp.zeros_like(target['ticks'], dtype=jnp.int32))
            ref_hit_prob = target.get('hit_prob', jnp.ones_like(target['ticks']))
            target_x_per_event = target['pixel_x']
            target_y_per_event = target['pixel_y']
            target_ticks = target['ticks']

        if self.collapsed:
            pred_Q, pixel_x_per_event, pixel_y_per_event, pred_z, pred_ticks, pred_hit_prob, pred_event_per_pixel = self._prepare_probabilistic_output(params, prediction)
        else:
            pred_Q, pixel_x_per_event, pixel_y_per_event, pred_z, pred_ticks, pred_hit_prob, pred_event_per_pixel = self._generate_distribution_hits(params, prediction)

        
        # Apply the deterministic loss function
        loss_val, aux = self.loss_fn(
            params,
            pred_Q.flatten(), pixel_x_per_event.flatten(), pixel_y_per_event.flatten(), pred_z.flatten(), pred_ticks.flatten(), pred_hit_prob.flatten(), pred_event_per_pixel.flatten(),
            ref_Q.flatten(), target_x_per_event.flatten(), target_y_per_event.flatten(), ref_z.flatten(), target_ticks.flatten(), ref_hit_prob.flatten(), ref_event.flatten(),
            **self.kwargs
        )
        
        # Return loss with auxiliary info
        return loss_val, aux


class ProbabilisticLossStrategy(LossStrategy):
    def __init__(self, sigma_charge=500.0, eps=1e-8, time_window=2, sigma_time=1.0, apply_deadtime=False, deadtime_ticks=18, first_hit_only=False, **kwargs):
        """
        Computes negative log-likelihood of observed hits given predicted probability distributions.
        
        Implements a Poisson Point Process (PPP) Negative Log-Likelihood:
        1. Observed hits: -log P(tick, charge | pixel)
        2. False positives: penalty for predicting hits where none observed (expected total hits)
        3. False negatives: penalize missing target hits by log(eps)
        
        Args:
            sigma_charge: Standard deviation for Gaussian charge likelihood (in electrons)
            eps: Background noise rate per bin (pixel*tick). log(eps) sets the minimum probability floor.
                 Default eps=1e-8 gives a penalty of ~18.4 per missing hit.
            time_window: Number of ticks (+/-) to search around the observed target tick.
            sigma_time: Standard deviation for the Gaussian time resolution kernel (in ticks).
            apply_deadtime: Whether to account for electronics deadtime for consecutive hits on the same pixel.
            deadtime_ticks: Duration of the deadtime zone.
            first_hit_only: Whether to only consider the first hit in every pixel for both target and simulation.
        """
        super().__init__(**kwargs)
        self.sigma_charge = sigma_charge
        self.eps = eps
        self.min_log_prob = jnp.log(eps)
        self.time_window = time_window
        self.sigma_time = sigma_time
        self.apply_deadtime = apply_deadtime
        self.deadtime_ticks = deadtime_ticks
        self.first_hit_only = first_hit_only

        if time_window > 0:
            delta = jnp.arange(-time_window, time_window + 1)
            weights = jnp.exp(-0.5 * (delta / sigma_time)**2)
            weights = weights / jnp.sum(weights)
            self.log_time_weights = jnp.log(weights)

    def compute(self, params, prediction, target):
        """
        Compute Poisson Point Process Negative Log-Likelihood (PPP NLL).
        
        NLL = - Σ [log P(tick|pixel) + log P(charge|tick,pixel)]  [matched observed hits]
              - (min_log_prob) * N_unmatched                      [false negative penalty]
              + Σ_pixels Σ_ticks P(tick|pixel)                    [expected total hits / false positive penalty]
        
        Prediction contains:
            - adcs_distrib: (Npix, Nvalues, Nticks) - predicted ADC distributions
            - hit_prob: (Npix, Nvalues, Nticks) - joint log probability log P(value, tick | pixel)
            - unique_pixels: (Npix,) - pixel IDs in sorted order
            
        Target contains:
            - pixel_id: (Nhits,) - pixel ID for each observed hit
            - ticks: (Nhits,) - observed tick for each hit
            - adcs: (Nhits,) - observed ADC for each hit
        """
        # Step 1: Match target hits to predicted pixel distributions
        target_pixel_ids = target.get('pixel_id', target.get('hit_pixels'))
        sim_unique_pixels = prediction['unique_pixels']
        
        # Find indices of target pixels in simulation output (unique_pixels is sorted)
        pixel_indices = jnp.searchsorted(sim_unique_pixels, target_pixel_ids)
        
        # Validate matches (check if pixel was actually simulated)
        pixel_indices_safe = jnp.clip(pixel_indices, 0, sim_unique_pixels.shape[0] - 1)
        pixel_match_valid = (sim_unique_pixels[pixel_indices_safe] == target_pixel_ids) & (target_pixel_ids >= 0)
        
        # Step 2: Extract probability distributions for matched pixels
        ticks_prob = prediction['hit_prob']  # (Npix, Nvalues, Nticks) [log-probabilities]
        adcs_distrib = prediction['adcs_distrib']  # (Npix, Nvalues, Nticks)
        
        # Step 3: For each target hit, extract corresponding simulation probabilities and charges
        target_ticks = target['ticks'].astype(int)
        target_adcs = target['adcs']
        target_charge = adc2charge(target_adcs, params)
        
        trigger_nb = compute_occurrence_indices(target_pixel_ids)
        
        # Identify which target hits are valid and relevant for the current mode
        is_relevant_target_hit = (target_pixel_ids >= 0)
        if self.first_hit_only:
            is_relevant_target_hit = is_relevant_target_hit & (trigger_nb == 0)
        
        # Update match validity to only include relevant hits
        pixel_match_valid = pixel_match_valid & is_relevant_target_hit
        
        Nticks = ticks_prob.shape[-1]
        
        if self.apply_deadtime:
            prev_target_ticks = jnp.roll(target_ticks, shift=1)
            tick_indices = jnp.arange(Nticks)
            
            # Mask out the impossible region for the second hit onwards
            is_valid_mask = jnp.where(
                trigger_nb[:, None] == 0,
                True,
                tick_indices[None, :] > (prev_target_ticks[:, None] + self.deadtime_ticks)
            )
            
            # Renormalize probability mass in the valid zone
            ticks_prob_slice = ticks_prob[pixel_indices_safe, trigger_nb, :]
            log_Z_all = jax.nn.logsumexp(ticks_prob_slice, axis=-1)
            masked_ticks_prob = jnp.where(is_valid_mask, ticks_prob_slice, self.min_log_prob)
            log_Z_valid = jax.nn.logsumexp(masked_ticks_prob, axis=-1)
            log_renorm_boost = log_Z_all - log_Z_valid
            
            # Identify if the target hit itself violates the deadtime rule
            is_valid_target_hit = jnp.where(
                trigger_nb == 0,
                True,
                target_ticks > (prev_target_ticks + self.deadtime_ticks)
            )
        else:
            log_renorm_boost = 0.0
            is_valid_target_hit = jnp.ones_like(target_ticks, dtype=bool)

        if self.time_window > 0:
            # Create a window of ticks around each target hit
            # shape: (Nhits, 2*W + 1)
            delta = jnp.arange(-self.time_window, self.time_window + 1)
            window_ticks = target_ticks[:, None] + delta[None, :]
            window_ticks = jnp.clip(window_ticks, 0, Nticks - 1)
            
            # Extract log-probs for the window
            # shape: (Nhits, 2*W + 1)
            window_log_probs = ticks_prob[pixel_indices_safe[:, None], trigger_nb[:, None], window_ticks]
            
            # Expected charge at each tick in the window
            window_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe[:, None], trigger_nb[:, None], window_ticks], params)
            
            # Compute Gaussian log-likelihood of charge FOR EACH TICK in the window
            charge_diffs = target_charge[:, None] - window_expected_charges
            window_log_charge_intensity = (
                -0.5 * (charge_diffs / (self.sigma_charge/1000)) ** 2 
                - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
            )
            
            # Total joint log probability for each tick in the window
            # log P(t | pixel) + log K(t_obs - t) + log P(q_obs | q_pred(t))
            joint_window_log_probs = window_log_probs + self.log_time_weights[None, :] + window_log_charge_intensity
            
            # Marginalize over the time window using logsumexp
            joint_hit_log_probs = jax.nn.logsumexp(joint_window_log_probs, axis=1)
            
        else:
            # Exact point-wise extraction
            hit_tick_probs = ticks_prob[pixel_indices_safe, trigger_nb, target_ticks]
            hit_expected_charges = adc2charge(adcs_distrib[pixel_indices_safe, trigger_nb, target_ticks], params)
            
            charge_diff = target_charge - hit_expected_charges
            log_charge_intensity = (
                -0.5 * (charge_diff / (self.sigma_charge/1000)) ** 2 
                - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
            )
            joint_hit_log_probs = hit_tick_probs + log_charge_intensity
        
        if self.apply_deadtime:
            # If the target hit is inside the blacklist zone, give it a massive penalty.
            # Otherwise, give it the standard log-likelihood + the renormalization boost.
            joint_hit_log_probs = jnp.where(
                is_valid_target_hit, 
                joint_hit_log_probs + log_renorm_boost, 
                self.min_log_prob
            )

        # Step 5: Compute Poisson Point Process Negative Log-Likelihood (PPP NLL)
        
        # Bound the joint intensity to prevent NaNs or extreme values
        joint_hit_log_probs = jnp.maximum(joint_hit_log_probs, self.min_log_prob)

        # Mask invalid matches for intensity sums
        joint_hit_log_probs = jnp.where(pixel_match_valid, joint_hit_log_probs, 0.0)
        
        # Total log-intensity of observed hits
        ll_hits = jnp.sum(joint_hit_log_probs)

        # Penalty for target hits that are relevant but weren't simulated (False Negatives)
        # We only sum over is_relevant_target_hit to avoid penalizing ignored hits in first_hit_only mode
        no_match_penalty = self.min_log_prob * jnp.sum(is_relevant_target_hit & ~pixel_match_valid)

        # (c) Integral of intensity: Expected total number of hits predicted by simulation
        if self.first_hit_only:
            # Only integrate the intensity of the FIRST hit predicted for each pixel
            expected_total_hits = jnp.sum(jnp.exp(ticks_prob[:, 0, :]))
        else:
            # Integrate over all predicted hits (standard PPP NLL)
            expected_total_hits = jnp.sum(jnp.exp(ticks_prob))
        
        # Combine into final Negative Log-Likelihood
        nll = -ll_hits - no_match_penalty + expected_total_hits
        
        # Auxiliary info for debugging
        aux = {
            "log_likelihood_charge": 0.0,
            "log_likelihood_tick": -jnp.sum(joint_hit_log_probs),
            "no_match_penalty": -no_match_penalty,
            "expected_total_hits": expected_total_hits,
            "matched_hits": jnp.sum(is_relevant_target_hit & pixel_match_valid)
        }
        
        return nll, aux

class DistributionLossStrategy(LossStrategy):
    def __init__(self, feature='charge', threshold=1e-4, num_quantiles=100, **kwargs):
        super().__init__(**kwargs)
        self.feature = feature
        self.threshold = threshold
        self.num_quantiles = num_quantiles

    def _get_samples(self, params, output):
        # Handle both probabilistic (3D) and stochastic/flat (1D) formats
        if output['hit_prob'].ndim == 3:
            prob = jnp.exp(output['hit_prob'])
            mask = prob > self.threshold
            
            if self.feature == 'charge':
                vals = output['adcs_distrib'][mask]
                weights = prob[mask]
            elif self.feature == 'time':
                Nticks = prob.shape[-1]
                ticks_arr = jnp.arange(Nticks)
                tick_broad = jnp.broadcast_to(ticks_arr[None, None, :], prob.shape)
                vals = tick_broad[mask]
                weights = prob[mask]
            elif self.feature == 'x':
                broad_x = jnp.broadcast_to(output['pixel_x'][:, None, None], prob.shape)
                vals = broad_x[mask]
                weights = prob[mask]
            elif self.feature == 'y':
                broad_y = jnp.broadcast_to(output['pixel_y'][:, None, None], prob.shape)
                vals = broad_y[mask]
                weights = prob[mask]
            elif self.feature == 'pix_counts':
                # Sum probability per pixel to get expected hits per pixel
                vals = jnp.sum(prob, axis=(1, 2))
                # Filter out pixels that are unlikely to have fired
                # to match the stochastic behavior (which only reports firing pixels)
                mask_pix = vals > self.threshold
                vals = vals[mask_pix]
                weights = jnp.ones_like(vals)
            else:
                raise ValueError(f"Unknown feature: {self.feature}")
        else:
            # Flat format (e.g. target data)
            # For target data, hit_prob is often 1.0 or already linear (not log)
            # However, in ParamFitter, ref_hit_prob is usually linear probability.
            # We assume it's linear if it comes from the flat target_data.
            prob = output['hit_prob']
            mask = prob > self.threshold
            
            if self.feature == 'charge':
                vals = output['adcs'][mask]
                weights = prob[mask]
            elif self.feature == 'time':
                vals = output['ticks'][mask]
                weights = prob[mask]
            elif self.feature == 'x':
                vals = output['pixel_x'][mask]
                weights = prob[mask]
            elif self.feature == 'y':
                vals = output['pixel_y'][mask]
                weights = prob[mask]
            elif self.feature == 'pix_counts':
                # Sum probability per unique pixel in the flat hit list
                # This is a bit more complex to do differentiably without unique_pixels,
                # but for target data it's usually just counting.
                unique_pids, inverse_indices = jnp.unique(output['hit_pixels'], return_inverse=True, size=output.get('unique_pixels', output['hit_pixels']).shape[0])
                vals = jax.ops.segment_sum(prob, inverse_indices, num_segments=len(unique_pids))
                weights = jnp.ones_like(vals)
            else:
                raise ValueError(f"Unknown feature: {self.feature}")

        return vals, weights

    def compute(self, params, prediction, target):
        pred_vals, pred_weights = self._get_samples(params, prediction)
        target_vals, target_weights = self._get_samples(params, target)
        
        loss_val = weighted_wasserstein_1d(pred_vals, pred_weights, target_vals, target_weights, num_quantiles=self.num_quantiles)
        
        return loss_val, {}


class MarkovLossStrategy(LossStrategy):
    def __init__(self, sigma_charge=500.0, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.sigma_charge = sigma_charge
        self.eps = eps
        self.min_log_prob = jnp.log(eps)

    def compute(self, params, prediction, target):
        log_p1 = prediction['log_p1']
        log_T = prediction['log_T']
        log_p_none = prediction['log_p_none']
        expected_Q = prediction['expected_Q']
        sim_unique_pixels = prediction['unique_pixels']
        Q1 = prediction['Q1']
        log_p_none_zero = prediction['log_p_none_at_zero']

        target_pixel_ids = target.get('pixel_id', target.get('hit_pixels'))
        target_ticks = target['ticks'].astype(int)
        target_adcs = target['adcs']
        target_charge = adc2charge(target_adcs, params)

        pixel_indices = jnp.searchsorted(sim_unique_pixels, target_pixel_ids)
        pixel_indices_safe = jnp.clip(pixel_indices, 0, sim_unique_pixels.shape[0] - 1)
        pixel_match_valid = (sim_unique_pixels[pixel_indices_safe] == target_pixel_ids) & (target_pixel_ids >= 0)

        # We assume target hits are sorted by pixel and time
        trigger_nb = compute_occurrence_indices(target_pixel_ids)

        # 1. Sequence Log-Likelihood
        prev_target_ticks = jnp.roll(target_ticks, shift=1)
        
        log_intensity_tick = jnp.where(
            trigger_nb == 0,
            log_p1[pixel_indices_safe, target_ticks],
            log_T[pixel_indices_safe, prev_target_ticks, target_ticks]
        )
        
        # 2. Charge Log-Likelihood
        expected_Q_hit = jnp.where(
            trigger_nb == 0,
            Q1[pixel_indices_safe, target_ticks],
            expected_Q[pixel_indices_safe, prev_target_ticks, target_ticks]
        )
        
        expected_Q_phys = adc2charge(expected_Q_hit, params)
        charge_diff = target_charge - expected_Q_phys
        log_intensity_charge = (
            -0.5 * (charge_diff / (self.sigma_charge/1000)) ** 2 
            - 0.5 * jnp.log(2 * jnp.pi * (self.sigma_charge/1000)**2)
        )

        log_intensity = jnp.where(pixel_match_valid, log_intensity_tick + log_intensity_charge, 0.0)
        ll_hits = jnp.sum(log_intensity)
        
        # 3. Probability of NO hit after the last observed hit
        # Correctly identify last hits
        is_last_hit = jnp.concatenate([
            target_pixel_ids[:-1] != target_pixel_ids[1:],
            jnp.array([True])
        ])
        
        log_intensity_none = jnp.where(
            pixel_match_valid & is_last_hit,
            log_p_none[pixel_indices_safe, target_ticks],
            0.0
        )
        ll_none = jnp.sum(log_intensity_none)
        
        # 4. Account for pixels that were simulated but had ZERO hits in target
        target_pixel_set = jnp.unique(target_pixel_ids)
        has_hit_in_target = jnp.isin(sim_unique_pixels, target_pixel_set)
        
        log_none_zero_hit_pixels = jnp.where(
            (~has_hit_in_target) & (sim_unique_pixels >= 0),
            log_p_none_zero,
            0.0
        )
        ll_none_zero = jnp.sum(log_none_zero_hit_pixels)

        no_match_penalty = self.min_log_prob * jnp.sum(~pixel_match_valid)

        # 5. Expected Total Hits (Analytical)
        def compute_expected_hits_pixel(lp1, lT):
            p1 = jnp.exp(lp1)
            T = jnp.exp(lT)
            I = jnp.eye(T.shape[0])
            # occ = (I - T)^-T @ p1
            occ = jax.scipy.linalg.solve_triangular(I - T, p1, lower=False, trans='T')
            return jnp.sum(occ)

        expected_hits_per_pixel = jax.vmap(compute_expected_hits_pixel)(log_p1, log_T)
        # Mask out padded pixels in analytical sum
        expected_total_hits = jnp.sum(jnp.where(sim_unique_pixels >= 0, expected_hits_per_pixel, 0.0))
        
        # Combined NLL: Observed sequence LL + Survival probability + FP penalty (Expected hits)
        nll = -(ll_hits + ll_none + ll_none_zero) - no_match_penalty + expected_total_hits
        
        aux = {
            "log_likelihood_charge": -jnp.sum(jnp.where(pixel_match_valid, log_intensity_charge, 0.0)),
            "log_likelihood_tick": -jnp.sum(jnp.where(pixel_match_valid, log_intensity_tick, 0.0)),
            "log_likelihood_none": -(ll_none + ll_none_zero),
            "no_match_penalty": -no_match_penalty,
            "expected_total_hits": expected_total_hits,
            "matched_hits": jnp.sum(pixel_match_valid)
        }
        
        return nll, aux

