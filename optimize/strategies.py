import jax
import jax.numpy as jnp
from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_parametrized, simulate_probabilistic
from larndsim.losses_jax import adc2charge
from larndsim.detsim_jax import id2pixel

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
        adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(params, wfs, unique_pixels)
        
        # Extract pixel plane for z-coordinate calculation
        _, _, pixel_plane, _ = id2pixel(params, unique_pixels)
        
        # We return the raw distributions for the ProbabilisticLossStrategy
        return {
            'adcs_distrib': adcs_distrib, # (Npix, Nvalues, Nticks)
            'ticks_prob': ticks_prob,     # (Npix, Nvalues, Nticks)
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'pixel_plane': pixel_plane,   # Needed for z-coordinate calculation
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
    def __init__(self, loss_fn, hit_threshold=1e-8, **kwargs):
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

    def compute(self, params, prediction, target):
        """
        Convert probabilistic predictions to pseudo-hits and apply deterministic loss.
        
        Important: ticks_prob and adcs_distrib have shape (Npix, Nhits, Nticks), where:
        - Npix: number of pixels
        - Nhits: maximum number of triggered hits per pixel (different hits, not charge values)
        - Nticks: time ticks
        
        Each (pixel, hit_index) combination should be treated independently.
        """
        # Extract probabilistic distributions
        ticks_prob = prediction['ticks_prob']  # (Npix, Nhits, Nticks)
        adcs_distrib = prediction['adcs_distrib']  # (Npix, Nhits, Nticks)
        unique_pixels = prediction['unique_pixels']
        pixel_x = prediction['pixel_x']
        pixel_y = prediction['pixel_y']
        
        Npix, Nhits, Nticks = ticks_prob.shape
        
        # For each (pixel, hit_index), compute λ = Σ_t P(tick | pixel, hit_index)
        # This represents the expected probability of this particular hit existing
        lambda_per_hit = jnp.sum(ticks_prob, axis=2)  # (Npix, Nhits)
        
        # For each (pixel, hit_index), compute expected tick
        # E[tick | pixel, hit_index] = Σ_t t * P(tick | pixel, hit_index) / λ
        tick_range = jnp.arange(Nticks)  # (Nticks,)
        # Broadcast for computation: tick_range shape (1, 1, Nticks)
        expected_ticks_per_hit = jnp.sum(
            tick_range[None, None, :] * ticks_prob, axis=2
        ) / jnp.maximum(lambda_per_hit, 1e-10)  # (Npix, Nhits)
        
        # For each (pixel, hit_index), compute expected ADC
        # E[ADC | pixel, hit_index] = Σ_t ADC(hit_index, tick) * P(tick | pixel, hit_index) / λ
        expected_adcs_per_hit = jnp.sum(
            adcs_distrib * ticks_prob, axis=2
        ) / jnp.maximum(lambda_per_hit, 1e-10)  # (Npix, Nhits)
        
        # Filter out hits with negligible probability
        has_hit_mask = lambda_per_hit > self.hit_threshold  # (Npix, Nhits)
        
        # Flatten to create list of pseudo-hits
        # We need to replicate pixel coordinates for each hit
        pred_ticks = expected_ticks_per_hit[has_hit_mask]  # (N_total_hits,)
        pred_adcs = expected_adcs_per_hit[has_hit_mask]  # (N_total_hits,)
        pred_lambda = lambda_per_hit[has_hit_mask]  # (N_total_hits,)
        
        # For pixel coordinates, we need to replicate them for each hit
        # Create indices for which pixel each hit belongs to
        pixel_indices = jnp.arange(Npix)[:, None] * jnp.ones((Npix, Nhits), dtype=jnp.int32)  # (Npix, Nhits)
        pred_pixel_idx = pixel_indices[has_hit_mask]  # (N_total_hits,)
        
        pred_x = pixel_x[pred_pixel_idx]
        pred_y = pixel_y[pred_pixel_idx]
        
        # Convert ADCs to charge
        pred_Q = adc2charge(pred_adcs, params)
        ref_Q = adc2charge(target['adcs'], params)
        
        # Get z-coordinates and event IDs from prediction (if available)
        # If not available, compute from drift time or use same default as target
        if 'pixel_z' in prediction:
            # If prediction has pixel_z (from stochastic simulation), replicate for each hit
            pred_z = prediction['pixel_z'][pred_pixel_idx]
        else:
            # For probabilistic predictions without z, compute from drift time
            # z = v_drift * t_drift (same approach as in simulate_stochastic)
            from larndsim.detsim_jax import get_hit_z
            # Get pixel plane for z calculation
            pixel_plane = prediction.get('pixel_plane')
            if pixel_plane is not None:
                pred_z = get_hit_z(params, pred_ticks, pixel_plane[pred_pixel_idx])
            else:
                # If we can't compute z, use zeros as fallback
                # This should match what's done for the reference
                pred_z = jnp.zeros_like(pred_y)
        
        # Get reference z-coordinates using same logic
        ref_z = target.get('pixel_z', jnp.zeros_like(target['adcs']))
        
        # Get event IDs from prediction (if available)
        if 'event' in prediction:
            # Event is per-pixel, replicate for each hit
            pred_event_per_pixel = prediction['event']  # (Npix,)
            # Broadcast to (Npix, Nhits) then mask
            pred_event_full = jnp.broadcast_to(pred_event_per_pixel[:, None], (Npix, Nhits))
            pred_event = pred_event_full[has_hit_mask]
        else:
            pred_event = jnp.zeros_like(pred_ticks, dtype=jnp.int32)
        
        # Get reference event IDs
        ref_event = target.get('event', jnp.zeros_like(target['ticks'], dtype=jnp.int32))
        
        pred_hit_prob = pred_lambda 
        ref_hit_prob = target.get('hit_prob', jnp.ones_like(target['ticks']))
        
        # Apply the deterministic loss function
        loss_val, aux = self.loss_fn(
            params,
            pred_Q, pred_x, pred_y, pred_z, pred_ticks, pred_hit_prob, pred_event,
            ref_Q, target['pixel_x'], target['pixel_y'], ref_z, target['ticks'], ref_hit_prob, ref_event,
            **self.kwargs
        )
        
        # Return loss with auxiliary info
        return loss_val, aux


class ProbabilisticLossStrategy(LossStrategy):
    def __init__(self, sigma_charge=500.0, eps=1e-10, **kwargs):
        """
        Computes negative log-likelihood of observed hits given predicted probability distributions.
        
        Implements a complete probabilistic loss that accounts for:
        1. Observed hits: -log P(tick|pixel) - log P(charge|tick,pixel)
        2. False positives: penalty for predicting hits where none observed (Σλ for unobserved pixels)
        
        This ensures the model learns to concentrate probability only at pixels with actual hits.
        
        Args:
            sigma_charge: Standard deviation for Gaussian charge likelihood (in electrons)
            eps: Small constant to avoid log(0)
        """
        super().__init__(**kwargs)
        self.sigma_charge = sigma_charge
        self.eps = eps

    def compute(self, params, prediction, target):
        """
        Compute negative log-likelihood loss with false positive penalty.
        
        Loss = -Σ[log P(tick|pixel) + log P(charge|tick,pixel)]  [observed hits]
               + Σλ(pixel)                                         [false positive penalty]
        
        where λ(pixel) = Σ_t P(tick|pixel) = expected number of hits per pixel
        
        Prediction contains:
            - adcs_distrib: (Npix, Nvalues, Nticks) - predicted ADC distributions
            - ticks_prob: (Npix, Nvalues, Nticks) - joint probability P(value, tick | pixel has hit)
            - unique_pixels: (Npix,) - pixel IDs in sorted order
            - pixel_x, pixel_y: pixel coordinates
            
        Target contains:
            - pixel_id: (Nhits,) - pixel ID for each observed hit
            - ticks: (Nhits,) - observed tick for each hit
            - adcs: (Nhits,) - observed ADC for each hit
        """
        
        # Step 1: Match target hits to predicted pixel distributions
        target_pixel_ids = target['pixel_id']
        sim_unique_pixels = prediction['unique_pixels']
        
        # Find indices of target pixels in simulation output (unique_pixels is sorted)
        pixel_indices = jnp.searchsorted(sim_unique_pixels, target_pixel_ids)
        
        # Validate matches (check if pixel was actually simulated)
        pixel_indices_safe = jnp.clip(pixel_indices, 0, sim_unique_pixels.shape[0] - 1)
        pixel_match_valid = sim_unique_pixels[pixel_indices_safe] == target_pixel_ids
        
        # Step 2: Extract probability distributions for matched pixels
        # ticks_prob shape: (Npix, Nvalues, Nticks)
        # We need P(tick, charge | pixel_id) for the observed (tick, charge) pairs
        
        ticks_prob = prediction['ticks_prob']  # (Npix, Nvalues, Nticks)
        adcs_distrib = prediction['adcs_distrib']  # (Npix, Nvalues, Nticks)
        
        # Compute marginal probability P(tick | pixel) = sum_values P(tick, value | pixel)
        marginal_tick_prob = jnp.sum(ticks_prob, axis=1)  # (Npix, Nticks)
        
        # Step 3: For each target hit, compute likelihood
        target_ticks = target['ticks'].astype(int)
        target_adcs = target['adcs']
        target_charge = adc2charge(target_adcs, params)
        
        # Gather probabilities for the matched pixels at observed ticks
        # For each hit i: marginal_tick_prob[pixel_indices[i], target_ticks[i]]

        hit_tick_probs = marginal_tick_prob[pixel_indices_safe, target_ticks]
        
        # Step 4: Compute expected charge at observed tick for each pixel
        # E[charge | pixel, tick] = sum_values charge(value, tick) * P(value | tick, pixel)
        # where P(value | tick, pixel) = P(value, tick | pixel) / P(tick | pixel)
        
        # Get conditional probability distributions: P(value | tick, pixel)
        safe_marginal = jnp.where(marginal_tick_prob > self.eps, marginal_tick_prob, 1.0)
        conditional_value_prob = ticks_prob / safe_marginal[:, None, :]  # (Npix, Nvalues, Nticks)
        
        # Expected charge at each (pixel, tick)
        expected_charge_adc = jnp.sum(adcs_distrib * conditional_value_prob, axis=1)  # (Npix, Nticks)
        expected_charge = adc2charge(expected_charge_adc, params)
        
        # Gather expected charges for observed hits
        hit_expected_charges = expected_charge[pixel_indices_safe, target_ticks]
        
        # Step 5: Compute log-likelihood components
        
        # (a) Tick likelihood: log P(tick | pixel)
        log_likelihood_tick = jnp.log(hit_tick_probs + self.eps)
        
        # (b) Charge likelihood: log P(charge | tick, pixel) assuming Gaussian
        #     P(charge_obs | charge_expected, sigma) ~ N(charge_expected, sigma^2)
        charge_diff = target_charge - hit_expected_charges
        log_likelihood_charge = (
            -0.5 * (charge_diff / self.sigma_charge) ** 2 
            - 0.5 * jnp.log(2 * jnp.pi * self.sigma_charge**2)
        )
        
        # (c) Cap likelihood instead of masking for very small probabilities
        # When P(tick) is extremely small, cap the penalty instead of setting to 0
        # This ensures bad predictions are still penalized, preventing loss from artificially decreasing
        tick_prob_threshold = 1e-8
        max_negative_ll_tick = -jnp.log(tick_prob_threshold + self.eps)  # ≈ 18.4 for 1e-8
        
        # Cap the tick likelihood: use actual value if reasonable, otherwise use max penalty
        capped_log_likelihood_tick = jnp.where(
            hit_tick_probs > tick_prob_threshold,
            log_likelihood_tick,
            -max_negative_ll_tick  # Large negative value (strong penalty)
        )
        
        # For charge: only compute when tick probability is significant
        # When P(tick) is tiny, the charge term is meaningless, so set to 0
        tick_mask = hit_tick_probs > tick_prob_threshold
        masked_log_likelihood_charge = jnp.where(tick_mask, log_likelihood_charge, 0.0)
        
        # (d) Combined log-likelihood per hit
        log_likelihood_per_hit = capped_log_likelihood_tick + masked_log_likelihood_charge
        
        # Step 6: Handle invalid matches (pixels not in simulation)
        # For invalid matches, assign a very negative log-likelihood (low probability)
        log_likelihood_per_hit = jnp.where(
            pixel_match_valid, 
            log_likelihood_per_hit, 
            jnp.log(self.eps)
        )
        
        # Step 7: Sum log-likelihood over observed hits
        total_log_likelihood_hits = jnp.sum(log_likelihood_per_hit)
        
        # Step 8: Add penalty for false positives (predicted hits where none observed)
        # For each predicted pixel, compute λ = Σ_t P(tick|pixel) = expected number of hits
        lambda_per_pixel = jnp.sum(marginal_tick_prob, axis=1)  # (Npix,)
        
        # Check which predicted pixels have at least one observed hit
        # For each predicted pixel, check if it appears in target_pixel_ids
        pred_pixels = prediction['unique_pixels']
        
        def pixel_has_hit(pred_pixel):
            return jnp.sum(target_pixel_ids == pred_pixel) > 0
        
        pred_pixel_has_hit = jax.vmap(pixel_has_hit)(pred_pixels)  # (Npix,) boolean
        
        # For pixels with no observed hits: penalty = λ (from Poisson P(n=0|λ) = exp(-λ))
        # For pixels with hits: already accounted for in Step 7
        penalty_per_pixel = jnp.where(pred_pixel_has_hit, 0.0, lambda_per_pixel)
        total_false_positive_penalty = jnp.sum(penalty_per_pixel)
        
        # Step 9: Combined loss (negative log-likelihood with false positive penalty)
        nll = -total_log_likelihood_hits + total_false_positive_penalty
        
        # Auxiliary info for debugging
        aux = {
            'n_hits': target_pixel_ids.shape[0],
            'n_valid_matches': jnp.sum(pixel_match_valid),
            'mean_tick_prob': jnp.mean(hit_tick_probs),
            'mean_charge_diff': jnp.mean(jnp.abs(charge_diff)),
            'n_capped_hits': jnp.sum(~tick_mask),  # Renamed: now counts capped hits, not masked
            'false_positive_penalty': total_false_positive_penalty,
            'n_pred_pixels': pred_pixels.shape[0],
            'n_pixels_with_hits': jnp.sum(pred_pixel_has_hit),
        }
        
        return nll, aux 

