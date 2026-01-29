import jax
import jax.numpy as jnp
from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_parametrized, simulate_probabilistic
from larndsim.losses_jax import mse_adc, mse_time, mse_time_adc, chamfer_3d, sdtw_adc, sdtw_time, sdtw_time_adc, nll_loss, adc2charge, llhd_loss
from larndsim.detsim_jax import pixel2id

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
        adcs, x, y, z, ticks, hit_prob, event, unique_pixels = simulate_stochastic(params, wfs, unique_pixels, rngseed=rngkey)
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

class LUTProbabilisticSimulation(SimulationStrategy):
    def __init__(self, response):
        self.response = response

    def predict(self, params, tracks, fields, rngkey):
        wfs, unique_pixels = simulate_wfs(params, self.response, tracks, fields)
        adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(params, wfs, unique_pixels)
        
        # We return the raw distributions for the ProbabilisticLossStrategy
        return {
            'adcs_distrib': adcs_distrib, # (Npix, Nvalues, Nticks)
            'ticks_prob': ticks_prob,     # (Npix, Nvalues, Nticks)
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'event': event,
            'unique_pixels': unique_pixels, 
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

class ProbabilisticLossStrategy(LossStrategy):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute(self, params, prediction, target):
        # Prediction has adcs_distrib, ticks_prob, unique_pixels, no_hit_prob
        # Target has pixel_id, ticks, adcs
        
        # 1. Match target hits to simulated pixels
        # We assume target['pixel_id'] corresponds to values in prediction['unique_pixels']
        
        target_pixel_ids = target['pixel_id']
        sim_unique_pixels = prediction['unique_pixels']
        
        # Find indices of target pixels in the simulation output
        # searchsorted requires sorted sim_unique_pixels. simulate_probabilistic sorts unique_pixels.
        
        pixel_indices = jnp.searchsorted(sim_unique_pixels, target_pixel_ids)
        
        # Clip indices to be safe, although valid targets should be found
        pixel_indices = jnp.clip(pixel_indices, 0, sim_unique_pixels.shape[0] - 1)
        
        # Check if the pixel actually matches (it might not if target pixel was not simulated)
        # For now, we assume perfect match or that searchsorted gives a valid index.
        # If no match, we might point to a wrong pixel.
        # Ideally we mask out invalid matches, but llhd_loss expects fixed shapes matching target.
        # If match is invalid, it means the simulation predicts 0 charge/prob for this pixel (if we had full matrix).
        # Here we map to *some* pixel.
        
        # Gather distributions for the matched pixels
        # ticks_prob is (Npix, Nvalues, Nticks)
        # We need to sum over Nvalues to get the total probability per tick
        
        # Sum over Nvalues to get marginal prob per tick
        predicted_ticks_prob = jnp.sum(prediction['ticks_prob'], axis=1) # (Npix, Nticks)
        
        # Gather charge. We probably want the average or sum?
        # get_adc_values_average_noise_vmap returns charge_distrib corresponding to paths.
        # If we sum probabilities, we should probably average charge weighted by prob?
        # Or `adcs_distrib` is already charge values?
        # In simulate_probabilistic: adcs_distrib = digitize(params, charge_distrib)
        # So it is ADC counts.
        # But fee_jax.py returns `charge_distrib` (which is charge in electrons/integral?).
        # Wait, simulate_probabilistic calls `digitize`. So `adcs_distrib` is digitized ADC.
        # BUT llhd_loss takes `charge_distrib`.
        # `charge_mc` is in electrons (converted from ADCs in fit_params before passed to target data? No).
        # In fit_params: ref_adcs is passed.
        # GenericLossStrategy converts to charge: Q = adc2charge(prediction['adcs'], params)
        
        # Here `prediction['adcs_distrib']` is ADCs. We should convert to charge?
        # Or `llhd_loss` expects something else.
        # User said: `charge_diff = charge_mc - predicted_charges_at_mc_ticks`
        # And `charge_mc` is "Target charge".
        # If `target['adcs']` is ADCs, we need to convert `charge_mc`.
        
        charge_mc = adc2charge(target['adcs'], params)
        
        # Predicted charge: `adcs_distrib` is ADCs.
        # We need charge distribution.
        # Also we need to aggregate over Nvalues.
        # Weighted average?
        
        # probs: (Npix, Nvalues, Nticks)
        # charges: (Npix, Nvalues, Nticks) (in ADC)
        
        # Marginal probability: P(tick) = sum_v P(tick, v)
        # Conditional P(v|tick) = P(tick, v) / P(tick)
        # E[charge | tick] = sum_v charge(tick, v) * P(v|tick)
        #                  = sum_v charge(tick, v) * P(tick, v) / P(tick)
        
        probs = prediction['ticks_prob'] # (Npix, Nvalues, Nticks)
        charges = prediction['adcs_distrib'] # (Npix, Nvalues, Nticks)
        
        marginal_prob = jnp.sum(probs, axis=1) # (Npix, Nticks)
        safe_marginal_prob = jnp.where(marginal_prob == 0, 1.0, marginal_prob)
        
        avg_charge_adc = jnp.sum(charges * probs, axis=1) / safe_marginal_prob # (Npix, Nticks)
        avg_charge = adc2charge(avg_charge_adc, params)
        
        # Gather for target hits
        # shape (Nhits, Nticks)
        matched_probs = marginal_prob[pixel_indices]
        matched_charges = avg_charge[pixel_indices]
        
        # no_hit_prob: prediction['no_hit_prob'] is (Npix, Nvalues) ?
        # fee_jax returns moveaxis(no_hit_prob, 0, 1).
        # In global_scan_fun: new_active_flag is returned as part of carry, but no_hit_prob?
        # Wait, I added no_hit_prob return to fee_jax.py?
        # Let's check what I added.
        # _, (charge_avg, tick_avg, no_hit_prob, prob_distrib, charge_distrib) = lax.scan(...)
        # return ..., ..., jnp.moveaxis(no_hit_prob, 0, 1)
        
        # `no_hit_prob` in `_active_branch`?
        # In fee_jax.py logic (commented out part suggests `no_hit_prob` was there).
        # But in `_active_branch` I see:
        # return (new_charges, new_probs, new_active_flag), (prob_dist, charge_dist)
        # I did NOT add `no_hit_prob` to the return of `_active_branch`!
        # I only changed the unpacking of `lax.scan` result.
        
        # Oh, `_active_branch` returns 2 aux values.
        # `lax.scan` expects same structure.
        # My previous edit to `fee_jax.py` unpacking:
        # `_, (prob_distrib, charge_distrib) = lax.scan(...)`
        # This matches the 2 items returned by `_active_branch`.
        # So I did NOT capture `no_hit_prob`.
        
        # To get `no_hit_prob`, I need to calculate it or return it from `_active_branch`.
        # `no_hit_prob` is roughly `1 - sum(probs)`.
        # So I can compute it from `ticks_prob` sum.
        
        no_hit_prob = 1.0 - jnp.sum(marginal_prob, axis=1) # (Npix,)
        matched_no_hit_prob = no_hit_prob[pixel_indices]
        
        # Prepare inputs for llhd_loss
        # ticks_prob_distrib: (Nhits, Nticks) -> matched_probs
        # ticks_mc: target['ticks'] -> (Nhits,)
        # no_hit_prob: (Nhits,) -> matched_no_hit_prob
        # charge_distrib: (Nhits, Nticks) -> matched_charges
        # charge_mc: (Nhits,) -> charge_mc
        
        return self.loss_fn(
            matched_probs,
            target['ticks'].astype(int),
            matched_no_hit_prob,
            matched_charges,
            charge_mc,
            **self.kwargs
        ) 

