import jax
import jax.numpy as jnp
from larndsim.sim_jax import simulate_wfs, simulate_stochastic, simulate_parametrized, simulate_probabilistic
from larndsim.losses_jax import adc2charge
from larndsim.detsim_jax import id2pixel, get_hit_z
from larndsim.fee_jax import get_average_hit_values
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
    
class Q1dLossStrategy(LossStrategy):
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
            Q, ref_Q, prediction['event'],
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

    def _generate_distribution_hits(self, params, output):
        # This function can be used to prepare the probabilistic output for loss computation
        # For example, it can compute expected values or filter out low-probability hits
        ticks_prob = output['ticks_prob']
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
        ticks_prob = output['ticks_prob']
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
    #eps=1e-10
    def __init__(self, eps=1e-6,
                 w_sobolev_3d_grad=0.01,
                 w_sobolev_3d_grad_local=0.05,
                 w_sobolev_3d_grad_medium=0.01,
                 w_sobolev_3d_grad_global=0.01,
                 target_gaussian_3d_radius_cm=0.3,
                 target_gaussian_3d_sigma_cm=0.1,
                 sobolev_pool_nbin_x_medium=15,
                 sobolev_pool_nbin_z_medium=15,
                 sobolev_pool_nbin_x_global=5,
                 sobolev_pool_nbin_z_global=5,
                 sobolev_pool_layer_balance='running',
                 sobolev_pool_running_decay=0.9,
                 sobolev_pool_weight_local=1.0,
                 sobolev_pool_weight_medium=1.0,
                 sobolev_pool_weight_global=1.0,
                 emit_sobolev_pool_report=True,
                 sobolev_norm_target_source='smeared',
                 nz_local=1999,
                 max_events_per_batch=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.w_sobolev_3d_grad = float(w_sobolev_3d_grad)
        self.w_sobolev_3d_grad_local = float(w_sobolev_3d_grad_local)
        self.w_sobolev_3d_grad_medium = float(w_sobolev_3d_grad_medium)
        self.w_sobolev_3d_grad_global = float(w_sobolev_3d_grad_global)

        # Notebook-style parameters in physical units (cm).
        self.target_gaussian_3d_radius_cm = (
            None if target_gaussian_3d_radius_cm is None else max(float(target_gaussian_3d_radius_cm), 0.0)
        )
        self.target_gaussian_3d_sigma_cm = (
            None if target_gaussian_3d_sigma_cm is None else max(float(target_gaussian_3d_sigma_cm), 1e-6)
        )

        self.sobolev_pool_nbin_x_medium = max(int(sobolev_pool_nbin_x_medium), 1)
        self.sobolev_pool_nbin_z_medium = max(int(sobolev_pool_nbin_z_medium), 1)
        self.sobolev_pool_nbin_y_medium = 2 * self.sobolev_pool_nbin_x_medium
        self.sobolev_pool_medium_bins_xyz = (
            self.sobolev_pool_nbin_x_medium,
            self.sobolev_pool_nbin_y_medium,
            self.sobolev_pool_nbin_z_medium,
        )

        self.sobolev_pool_nbin_x_global = max(int(sobolev_pool_nbin_x_global), 1)
        self.sobolev_pool_nbin_z_global = max(int(sobolev_pool_nbin_z_global), 1)
        self.sobolev_pool_nbin_y_global = 2 * self.sobolev_pool_nbin_x_global
        self.sobolev_pool_global_bins_xyz = (
            self.sobolev_pool_nbin_x_global,
            self.sobolev_pool_nbin_y_global,
            self.sobolev_pool_nbin_z_global,
        )

        self.sobolev_pool_layer_balance = str(sobolev_pool_layer_balance).lower()
        if self.sobolev_pool_layer_balance not in ('running', 'weights', 'none'):
            raise ValueError(
                "sobolev_pool_layer_balance must be one of 'running', 'weights', or 'none'"
            )
        self.sobolev_pool_running_decay = min(max(float(sobolev_pool_running_decay), 0.0), 0.999999)
        self.sobolev_pool_manual_weights = {
            'local': max(float(sobolev_pool_weight_local), 0.0),
            'medium': max(float(sobolev_pool_weight_medium), 0.0),
            'global': max(float(sobolev_pool_weight_global), 0.0),
        }
        self.sobolev_pool_running_scales = {
            'local': 1.0,
            'medium': 1.0,
            'global': 1.0,
        }
        self.auto_reweight_every = 1

        self.emit_sobolev_pool_report = bool(emit_sobolev_pool_report)
        self.sobolev_norm_target_source = str(sobolev_norm_target_source).lower()
        if self.sobolev_norm_target_source == 'unsmeared':
            self.sobolev_norm_target_source = 'non_smeared'
        if self.sobolev_norm_target_source not in ('smeared', 'non_smeared'):
            raise ValueError(
                "sobolev_norm_target_source must be either 'smeared' or 'non_smeared'"
            )

        # Static z-window size for dense Sobolev grids.
        self.nz_local = max(int(nz_local), 1)
        # Fixed event-axis budget for per-event vectorized loss in one batch.
        self.max_events_per_batch = max(int(max_events_per_batch), 1)

    def _gaussian_smear_target(self, target_xyz, radius_cm, sigma_cm,
                               pixel_pitch, z_tick_size):
        """Apply 3D Gaussian smearing on (ny, nx, nz) target field in physical units."""
        import math
        ny, nx, nz = target_xyz.shape
        dx = float(pixel_pitch)
        dy = float(pixel_pitch)
        dz = float(z_tick_size)
        sigma2 = max(float(sigma_cm) ** 2, 1e-12)
        r = max(float(radius_cm), 0.0)
        r2 = r * r
        rx = max(int(math.ceil(r / max(dx, 1e-12))), 0)
        ry = max(int(math.ceil(r / max(dy, 1e-12))), 0)
        rz = max(int(math.ceil(r / max(dz, 1e-12))), 0)

        offsets_weights = []
        for oy in range(-ry, ry + 1):
            for ox in range(-rx, rx + 1):
                for ot in range(-rz, rz + 1):
                    d2 = (oy * dy) ** 2 + (ox * dx) ** 2 + (ot * dz) ** 2
                    if d2 <= r2:
                        offsets_weights.append((oy, ox, ot, math.exp(-0.5 * d2 / sigma2)))

        if not offsets_weights:
            offsets_weights = [(0, 0, 0, 1.0)]
        total_w = max(sum(w for _, _, _, w in offsets_weights), 1e-12)
        offsets_weights = [(oy, ox, ot, w / total_w) for oy, ox, ot, w in offsets_weights]

        padded_ones = jnp.pad(jnp.ones((ny, nx, nz), dtype=jnp.float32),
                              ((ry, ry), (rx, rx), (rz, rz)))
        denom = jnp.zeros((ny, nx, nz), dtype=jnp.float32)
        for oy, ox, ot, w in offsets_weights:
            denom = denom + w * padded_ones[
                ry + oy: ry + oy + ny,
                rx + ox: rx + ox + nx,
                rz + ot: rz + ot + nz,
            ]

        source_scaled = target_xyz / jnp.maximum(denom, 1e-12)
        padded_scaled = jnp.pad(source_scaled, ((ry, ry), (rx, rx), (rz, rz)))
        smeared = jnp.zeros((ny, nx, nz), dtype=jnp.float32)
        for oy, ox, ot, w in offsets_weights:
            smeared = smeared + w * padded_scaled[
                ry + oy: ry + oy + ny,
                rx + ox: rx + ox + nx,
                rz + ot: rz + ot + nz,
            ]
        return smeared

    def _pool_xyz_bins(self, field_xyz, bins_xyz):
        """Pool a dense [ny, nx, nz] field into coarse xyz bins."""
        field_xyz = jnp.asarray(field_xyz, dtype=jnp.float32)
        bin_x = max(int(bins_xyz[0]), 1)
        bin_y = max(int(bins_xyz[1]), 1)
        bin_z = max(int(bins_xyz[2]), 1)
        ny, nx, nz = field_xyz.shape

        x_bin = jnp.minimum((jnp.arange(nx, dtype=jnp.int32) * bin_x) // max(nx, 1), bin_x - 1)
        y_bin = jnp.minimum((jnp.arange(ny, dtype=jnp.int32) * bin_y) // max(ny, 1), bin_y - 1)
        z_bin = jnp.minimum((jnp.arange(nz, dtype=jnp.int32) * bin_z) // max(nz, 1), bin_z - 1)

        flat_y = jnp.repeat(y_bin, nx * nz)
        flat_x = jnp.tile(jnp.repeat(x_bin, nz), ny)
        flat_z = jnp.tile(z_bin, ny * nx)
        flat_idx = ((flat_y * bin_x) + flat_x) * bin_z + flat_z
        flat_vals = field_xyz.reshape(-1)

        pooled_sum = jnp.zeros(bin_y * bin_x * bin_z, dtype=jnp.float32).at[flat_idx].add(flat_vals)
        pooled_count = jnp.zeros(bin_y * bin_x * bin_z, dtype=jnp.float32).at[flat_idx].add(
            jnp.ones_like(flat_vals, dtype=jnp.float32)
        )
        pooled = pooled_sum / jnp.maximum(pooled_count, 1.0)
        return pooled.reshape(bin_y, bin_x, bin_z)

    def _sobolev_layer_metrics(self, pooled_pred, pooled_target, pooled_norm_source,
                               pixel_pitch_cm, z_bin_cm, w_sobolev_3d_grad=None):
        """Notebook-equivalent Sobolev metrics for one pooled layer."""
        eps = self.eps
        residual = pooled_pred - pooled_target
        active_mask = (jnp.abs(pooled_pred) > eps) | (jnp.abs(pooled_target) > eps)
        norm_mask = jnp.abs(pooled_norm_source) > eps
        norm_voxels = jnp.sum(norm_mask.astype(jnp.float32))
        pooled_norm = jnp.maximum(norm_voxels, 1.0)

        value = jnp.sum((residual ** 2) * active_mask.astype(jnp.float32)) / pooled_norm

        grad_x_e = jnp.array(0.0, dtype=jnp.float32)
        if residual.shape[1] > 1:
            dx = residual[:, 1:, :] - residual[:, :-1, :]
            mx = active_mask[:, 1:, :] & active_mask[:, :-1, :]
            grad_x_e = jnp.sum((dx ** 2) * mx.astype(jnp.float32)) / (
                pooled_norm * max(float(pixel_pitch_cm) ** 2, 1e-12)
            )

        grad_y_e = jnp.array(0.0, dtype=jnp.float32)
        if residual.shape[0] > 1:
            dy = residual[1:, :, :] - residual[:-1, :, :]
            my = active_mask[1:, :, :] & active_mask[:-1, :, :]
            grad_y_e = jnp.sum((dy ** 2) * my.astype(jnp.float32)) / (
                pooled_norm * max(float(pixel_pitch_cm) ** 2, 1e-12)
            )

        grad_z_e = jnp.array(0.0, dtype=jnp.float32)
        if residual.shape[2] > 1:
            dz = residual[:, :, 1:] - residual[:, :, :-1]
            mz = active_mask[:, :, 1:] & active_mask[:, :, :-1]
            # Adjust z-gradient contribution by voxel anisotropy (xy pitch over z bin size).
            z_grad_scale = max(float(z_bin_cm) / max(float(pixel_pitch_cm), 1e-12), 1e-12)
            grad_z_e = jnp.sum((dz ** 2) * mz.astype(jnp.float32)) / (
                pooled_norm * max(float(z_bin_cm) ** 2, 1e-12)
            ) * z_grad_scale

        sobolev_3d_grad = (grad_x_e + grad_y_e + grad_z_e) / 3.0
        _w_grad = float(self.w_sobolev_3d_grad) if w_sobolev_3d_grad is None else float(w_sobolev_3d_grad)
        total = value + _w_grad * sobolev_3d_grad
        active_voxels = jnp.sum(active_mask.astype(jnp.float32))

        return {
            'norm_voxels': norm_voxels,
            'active_voxels': active_voxels,
            'value': value,
            'grad_x_e': grad_x_e,
            'grad_y_e': grad_y_e,
            'grad_z_e': grad_z_e,
            'sobolev_3d_grad': sobolev_3d_grad,
            'total': total,
        }

    def _sobolev_pool_layer_weights(self):
        if self.sobolev_pool_layer_balance == 'running':
            raw_weights = jnp.asarray([
                1.0 / max(float(self.sobolev_pool_running_scales['local']), 1e-12),
                1.0 / max(float(self.sobolev_pool_running_scales['medium']), 1e-12),
                1.0 / max(float(self.sobolev_pool_running_scales['global']), 1e-12),
            ], dtype=jnp.float32)
        elif self.sobolev_pool_layer_balance == 'weights':
            raw_weights = jnp.asarray([
                float(self.sobolev_pool_manual_weights['local']),
                float(self.sobolev_pool_manual_weights['medium']),
                float(self.sobolev_pool_manual_weights['global']),
            ], dtype=jnp.float32)
        else:
            raw_weights = jnp.ones(3, dtype=jnp.float32)

        raw_mean = jnp.maximum(jnp.mean(raw_weights), 1e-12)
        return raw_weights / raw_mean

    def auto_reweight(self, aux, total_iter):
        if self.sobolev_pool_layer_balance != 'running':
            return None

        updated = {}
        decay = float(self.sobolev_pool_running_decay)
        for layer_name in ('local', 'medium', 'global'):
            key = f'sobolev_pool_{layer_name}_total'
            if key not in aux:
                continue
            observed = max(abs(float(aux[key])), 1e-12)
            previous = max(float(self.sobolev_pool_running_scales[layer_name]), 1e-12)
            if total_iter <= 0:
                ema = observed
            else:
                ema = decay * previous + (1.0 - decay) * observed
            self.sobolev_pool_running_scales[layer_name] = ema
            updated[f'sobolev_pool_running_scale_{layer_name}'] = ema

        weights = self._sobolev_pool_layer_weights()
        updated['sobolev_pool_weight_local'] = float(weights[0])
        updated['sobolev_pool_weight_medium'] = float(weights[1])
        updated['sobolev_pool_weight_global'] = float(weights[2])
        return updated

    def _compute_three_layer_sobolev_pooling_jax(
        self,
        pred_xyz,
        target_xyz,
        target_norm_xyz,
        pixel_pitch_cm,
        z_bin_cm,
    ):
        """Compute 3-layer Sobolev pooling: local, medium, global."""
        results = {}

        layer_specs = [
            ('local', 'local'),
            ('medium', 'medium_binning'),
            ('global', 'global_binning'),
        ]

        for layer_name, mode in layer_specs:
            if mode == 'local':
                pooled_pred = pred_xyz
                pooled_target = target_xyz
                pooled_norm_source = target_norm_xyz
            elif mode == 'medium_binning':
                pooled_pred = self._pool_xyz_bins(pred_xyz, self.sobolev_pool_medium_bins_xyz)
                pooled_target = self._pool_xyz_bins(target_xyz, self.sobolev_pool_medium_bins_xyz)
                pooled_norm_source = self._pool_xyz_bins(target_norm_xyz, self.sobolev_pool_medium_bins_xyz)
            else:
                pooled_pred = self._pool_xyz_bins(pred_xyz, self.sobolev_pool_global_bins_xyz)
                pooled_target = self._pool_xyz_bins(target_xyz, self.sobolev_pool_global_bins_xyz)
                pooled_norm_source = self._pool_xyz_bins(target_norm_xyz, self.sobolev_pool_global_bins_xyz)

            layer_w_grad = {
                'local': self.w_sobolev_3d_grad_local,
                'medium': self.w_sobolev_3d_grad_medium,
                'global': self.w_sobolev_3d_grad_global,
            }[layer_name]
            results[layer_name] = self._sobolev_layer_metrics(
                pooled_pred,
                pooled_target,
                pooled_norm_source,
                pixel_pitch_cm,
                z_bin_cm,
                w_sobolev_3d_grad=layer_w_grad,
            )

        return results

    def _format_three_layer_sobolev_pooling(self):
        return (
            "3-layer Sobolev pooling report:\n"
            f"  normalization source: {self.sobolev_norm_target_source}\n"
            f"  layer balance: {self.sobolev_pool_layer_balance}\n"
            "  layer 1: local, no pooling\n"
            f"  layer 2: medium xyz binning with bins_xyz={self.sobolev_pool_medium_bins_xyz}\n"
            f"  layer 3: global xyz binning with bins_xyz={self.sobolev_pool_global_bins_xyz}"
        )

    def print_three_layer_sobolev_pooling(self):
        print(self._format_three_layer_sobolev_pooling())

    def sobolev_pooling_bins(self):
        return {
            'medium_bins_xyz': self.sobolev_pool_medium_bins_xyz,
            'global_bins_xyz': self.sobolev_pool_global_bins_xyz,
        }

    def compute(self, params, prediction, target):
        """Compute loss on fixed-shape detector voxels (ny, nx, nz_local).

        nx = number of detector pixels along x.
        ny = number of detector pixels along y.
        nz_local = fixed tick-window size along z (drift/time direction).
        """
        target_pixel_ids = target['pixel_id'].astype(jnp.int64)
        target_ticks_raw = target['ticks'].astype(jnp.int32)
        target_adcs = target['adcs']

        target_x, target_y, _, target_event = id2pixel(params, target_pixel_ids)
        target_x = target_x.astype(jnp.int32)
        target_y = target_y.astype(jnp.int32)
        target_event = target_event.astype(jnp.int32)

        sim_unique_pixels = prediction['unique_pixels'].astype(jnp.int64)
        ticks_prob = prediction['ticks_prob']
        adcs_distrib = prediction['adcs_distrib']

        n_pred_pix = ticks_prob.shape[0]
        n_ticks = ticks_prob.shape[2]
        eps = self.eps
        nx = int(getattr(params, 'n_pixels_x', 0))
        ny = int(getattr(params, 'n_pixels_y', 0))
        if nx <= 0 or ny <= 0:
            raise ValueError(
                "Invalid detector geometry in ProbabilisticLossStrategy: "
                f"n_pixels_x={nx}, n_pixels_y={ny}."
            )
        if n_ticks <= 0:
            raise ValueError(f"Invalid probabilistic prediction shape: n_ticks={n_ticks}.")

        z_tick_size = float(getattr(params, 't_sampling', 1.0)) * float(getattr(params, 'vdrift_static', 1.0))
        pixel_pitch = float(getattr(params, 'pixel_pitch', 1.0))

        pred_pixel_x, pred_pixel_y, _, pred_event = id2pixel(params, sim_unique_pixels)
        pred_pixel_x = pred_pixel_x.astype(jnp.int32)
        pred_pixel_y = pred_pixel_y.astype(jnp.int32)
        pred_event = pred_event.astype(jnp.int32)
        valid_pred_pix = (
            (sim_unique_pixels >= 0)
            & (pred_pixel_x >= 0) & (pred_pixel_x < nx)
            & (pred_pixel_y >= 0) & (pred_pixel_y < ny)
            & (pred_event >= 0)
        )
        pred_px_safe = jnp.where(valid_pred_pix, jnp.clip(pred_pixel_x, 0, nx - 1), 0)
        pred_py_safe = jnp.where(valid_pred_pix, jnp.clip(pred_pixel_y, 0, ny - 1), 0)

        pred_hit_prob = jnp.clip(ticks_prob, 0.0, 1.0)
        pred_charge_raw = adc2charge(adcs_distrib, params)
        pred_expected_local = jnp.sum(pred_hit_prob * pred_charge_raw, axis=1)
        pred_occ_local = jnp.clip(1.0 - jnp.prod(1.0 - pred_hit_prob, axis=1), 0.0, 1.0)
        valid_2d = valid_pred_pix[:, None]
        pred_occ = jnp.where(valid_2d, pred_occ_local, 0.0)
        pred_expected = jnp.where(valid_2d, pred_expected_local, 0.0)

        nz_local = min(int(self.nz_local), n_ticks)
        z_bin_size_sparse = float(z_tick_size)

        target_charge = adc2charge(target_adcs, params)
        _tgt_abs = jnp.where(
            (target_ticks_raw >= 0) & (target_ticks_raw < n_ticks),
            jnp.abs(target_charge),
            0.0,
        )
        t_center = jnp.sum(target_ticks_raw.astype(jnp.float32) * _tgt_abs) / jnp.maximum(jnp.sum(_tgt_abs), 1.0)
        t_start = jnp.clip(
            jnp.round(t_center).astype(jnp.int32) - nz_local // 2,
            0,
            max(n_ticks - nz_local, 0),
        )

        pred_expected_win = jax.lax.dynamic_slice(pred_expected, (0, t_start), (n_pred_pix, nz_local))
        pred_occ_win = jax.lax.dynamic_slice(pred_occ, (0, t_start), (n_pred_pix, nz_local))

        total_target_charge = jnp.maximum(jnp.sum(jnp.abs(target_charge)), 1.0)
        target_pix_valid = (
            (target_pixel_ids >= 0)
            & (target_x >= 0) & (target_x < nx)
            & (target_y >= 0) & (target_y < ny)
            & (target_event >= 0)
        )
        target_tick_valid = (target_ticks_raw >= 0) & (target_ticks_raw < n_ticks)
        target_valid = target_pix_valid & target_tick_valid
        target_x_safe = jnp.clip(target_x, 0, nx - 1)
        target_y_safe = jnp.clip(target_y, 0, ny - 1)
        overflow_target_hits = jnp.sum(
            ((target_event >= self.max_events_per_batch) & target_valid).astype(jnp.float32)
        )
        overflow_pred_pixels = jnp.sum(
            ((pred_event >= self.max_events_per_batch) & valid_pred_pix).astype(jnp.float32)
        )

        layer_weights = self._sobolev_pool_layer_weights()
        event_ids = jnp.arange(self.max_events_per_batch, dtype=jnp.int32)
        ns_flat = ny * nx * nz_local

        zero_metrics = {
            'sobolev_3d_value': jnp.array(0.0, dtype=jnp.float32),
            'sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'sobolev_3d': jnp.array(0.0, dtype=jnp.float32),
            'local_value': jnp.array(0.0, dtype=jnp.float32),
            'local_grad_x_e': jnp.array(0.0, dtype=jnp.float32),
            'local_grad_y_e': jnp.array(0.0, dtype=jnp.float32),
            'local_grad_z_e': jnp.array(0.0, dtype=jnp.float32),
            'local_sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'local_norm_voxels': jnp.array(0.0, dtype=jnp.float32),
            'local_active_voxels': jnp.array(0.0, dtype=jnp.float32),
            'local_total': jnp.array(0.0, dtype=jnp.float32),
            'medium_value': jnp.array(0.0, dtype=jnp.float32),
            'medium_grad_x_e': jnp.array(0.0, dtype=jnp.float32),
            'medium_grad_y_e': jnp.array(0.0, dtype=jnp.float32),
            'medium_grad_z_e': jnp.array(0.0, dtype=jnp.float32),
            'medium_sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'medium_norm_voxels': jnp.array(0.0, dtype=jnp.float32),
            'medium_active_voxels': jnp.array(0.0, dtype=jnp.float32),
            'medium_total': jnp.array(0.0, dtype=jnp.float32),
            'global_value': jnp.array(0.0, dtype=jnp.float32),
            'global_grad_x_e': jnp.array(0.0, dtype=jnp.float32),
            'global_grad_y_e': jnp.array(0.0, dtype=jnp.float32),
            'global_grad_z_e': jnp.array(0.0, dtype=jnp.float32),
            'global_sobolev_3d_grad': jnp.array(0.0, dtype=jnp.float32),
            'global_norm_voxels': jnp.array(0.0, dtype=jnp.float32),
            'global_active_voxels': jnp.array(0.0, dtype=jnp.float32),
            'global_total': jnp.array(0.0, dtype=jnp.float32),
            'mean_pred_occupancy': jnp.array(0.0, dtype=jnp.float32),
            'mean_target_occupancy': jnp.array(0.0, dtype=jnp.float32),
            'residual_mean_abs': jnp.array(0.0, dtype=jnp.float32),
            'pred_field_mean': jnp.array(0.0, dtype=jnp.float32),
            'target_field_mean': jnp.array(0.0, dtype=jnp.float32),
            'z_win_start_tick': jnp.array(0.0, dtype=jnp.float32),
            'event_weight': jnp.array(0.0, dtype=jnp.float32),
            'event_target_charge': jnp.array(0.0, dtype=jnp.float32),
            'active_event': jnp.array(0.0, dtype=jnp.float32),
        }

        def _event_metrics(evt_id):
            evt_tgt = target_valid & (target_event == evt_id)
            evt_pred = valid_pred_pix & (pred_event == evt_id)
            has_evt = jnp.any(evt_tgt) | jnp.any(evt_pred)

            def _compute_for_event(_):
                evt_abs_charge = jnp.where(evt_tgt, jnp.abs(target_charge), 0.0)
                evt_target_charge = jnp.sum(evt_abs_charge)
                evt_weight = jnp.maximum(evt_target_charge, 1.0)

                t_center_evt = jnp.sum(target_ticks_raw.astype(jnp.float32) * evt_abs_charge) / jnp.maximum(evt_target_charge, 1.0)
                t_start_evt = jnp.clip(
                    jnp.round(t_center_evt).astype(jnp.int32) - nz_local // 2,
                    0,
                    max(n_ticks - nz_local, 0),
                )

                pred_expected_evt = jnp.where(evt_pred[:, None], pred_expected, 0.0)
                pred_occ_evt = jnp.where(evt_pred[:, None], pred_occ, 0.0)
                pred_expected_win_evt = jax.lax.dynamic_slice(pred_expected_evt, (0, t_start_evt), (n_pred_pix, nz_local))
                pred_occ_win_evt = jax.lax.dynamic_slice(pred_occ_evt, (0, t_start_evt), (n_pred_pix, nz_local))

                target_tick_win_evt = (target_ticks_raw - t_start_evt).astype(jnp.int32)
                target_tick_win_safe_evt = jnp.clip(target_tick_win_evt, 0, nz_local - 1)
                target_in_win_evt = evt_tgt & (target_tick_win_evt >= 0) & (target_tick_win_evt < nz_local)

                _flat_tgt_evt = jnp.where(
                    target_in_win_evt,
                    target_y_safe * (nx * nz_local) + target_x_safe * nz_local + target_tick_win_safe_evt,
                    ns_flat,
                )
                _ns_evt = ns_flat + 1
                target_occ_xyz_evt = jnp.clip(
                    (jnp.zeros(_ns_evt, dtype=jnp.float32)
                     .at[_flat_tgt_evt].add(jnp.ones(target_pixel_ids.shape[0], dtype=jnp.float32))
                     )[:ns_flat].reshape(ny, nx, nz_local),
                    0.0,
                    1.0,
                )
                target_expected_xyz_unsmeared_evt = (
                    jnp.zeros(_ns_evt, dtype=jnp.float32)
                    .at[_flat_tgt_evt].add(jnp.where(target_in_win_evt, target_charge, 0.0))
                )[:ns_flat].reshape(ny, nx, nz_local)

                _flat_pred_evt = (
                    pred_py_safe[:, None] * (nx * nz_local)
                    + pred_px_safe[:, None] * nz_local
                    + jnp.arange(nz_local, dtype=jnp.int32)[None, :]
                ).reshape(-1)
                pred_expected_xyz_evt = (
                    jnp.zeros(ns_flat, dtype=jnp.float32)
                    .at[_flat_pred_evt].add(jnp.where(evt_pred[:, None], pred_expected_win_evt, 0.0).reshape(-1))
                ).reshape(ny, nx, nz_local)
                pred_occ_xyz_evt = jnp.clip(
                    (jnp.zeros(ns_flat, dtype=jnp.float32)
                     .at[_flat_pred_evt].add(jnp.where(evt_pred[:, None], pred_occ_win_evt, 0.0).reshape(-1))
                     ).reshape(ny, nx, nz_local),
                    0.0,
                    1.0,
                )

                radius_cm_eff_evt = 0.0 if self.target_gaussian_3d_radius_cm is None else float(self.target_gaussian_3d_radius_cm)
                if self.target_gaussian_3d_sigma_cm is not None:
                    sigma_cm_eff_evt = float(self.target_gaussian_3d_sigma_cm)
                else:
                    sigma_cm_eff_evt = max(float(radius_cm_eff_evt) / 2.0, float(pixel_pitch))
                sigma_cm_eff_evt = max(sigma_cm_eff_evt, 1e-6)

                if radius_cm_eff_evt > 0.0:
                    target_expected_xyz_evt = self._gaussian_smear_target(
                        target_expected_xyz_unsmeared_evt,
                        radius_cm_eff_evt,
                        sigma_cm_eff_evt,
                        pixel_pitch,
                        z_bin_size_sparse,
                    )
                else:
                    target_expected_xyz_evt = target_expected_xyz_unsmeared_evt

                target_norm_xyz_evt = (
                    target_expected_xyz_evt if self.sobolev_norm_target_source == 'smeared'
                    else target_expected_xyz_unsmeared_evt
                )

                sobolev_pool_layers_evt = self._compute_three_layer_sobolev_pooling_jax(
                    pred_expected_xyz_evt,
                    target_expected_xyz_evt,
                    target_norm_xyz_evt,
                    pixel_pitch,
                    z_bin_size_sparse,
                )
                sobolev_pool_local_evt = sobolev_pool_layers_evt['local']
                sobolev_pool_medium_evt = sobolev_pool_layers_evt['medium']
                sobolev_pool_global_evt = sobolev_pool_layers_evt['global']

                sobolev_3d_value_evt = (
                    layer_weights[0] * sobolev_pool_local_evt['value']
                    + layer_weights[1] * sobolev_pool_medium_evt['value']
                    + layer_weights[2] * sobolev_pool_global_evt['value']
                ) / 3.0
                sobolev_3d_grad_evt = (
                    layer_weights[0] * sobolev_pool_local_evt['sobolev_3d_grad']
                    + layer_weights[1] * sobolev_pool_medium_evt['sobolev_3d_grad']
                    + layer_weights[2] * sobolev_pool_global_evt['sobolev_3d_grad']
                ) / 3.0
                sobolev_3d_evt = (
                    layer_weights[0] * sobolev_pool_local_evt['total']
                    + layer_weights[1] * sobolev_pool_medium_evt['total']
                    + layer_weights[2] * sobolev_pool_global_evt['total']
                ) / 3.0

                residual_xyz_evt = pred_expected_xyz_evt - target_expected_xyz_evt
                mean_pred_occupancy_evt = jnp.sum(pred_occ_xyz_evt) / float(ny * nx * nz_local)
                mean_target_occupancy_evt = jnp.sum(target_occ_xyz_evt) / float(ny * nx * nz_local)

                return {
                    'sobolev_3d_value': sobolev_3d_value_evt,
                    'sobolev_3d_grad': sobolev_3d_grad_evt,
                    'sobolev_3d': sobolev_3d_evt,
                    'local_value': sobolev_pool_local_evt['value'],
                    'local_grad_x_e': sobolev_pool_local_evt['grad_x_e'],
                    'local_grad_y_e': sobolev_pool_local_evt['grad_y_e'],
                    'local_grad_z_e': sobolev_pool_local_evt['grad_z_e'],
                    'local_sobolev_3d_grad': sobolev_pool_local_evt['sobolev_3d_grad'],
                    'local_norm_voxels': sobolev_pool_local_evt['norm_voxels'],
                    'local_active_voxels': sobolev_pool_local_evt['active_voxels'],
                    'local_total': sobolev_pool_local_evt['total'],
                    'medium_value': sobolev_pool_medium_evt['value'],
                    'medium_grad_x_e': sobolev_pool_medium_evt['grad_x_e'],
                    'medium_grad_y_e': sobolev_pool_medium_evt['grad_y_e'],
                    'medium_grad_z_e': sobolev_pool_medium_evt['grad_z_e'],
                    'medium_sobolev_3d_grad': sobolev_pool_medium_evt['sobolev_3d_grad'],
                    'medium_norm_voxels': sobolev_pool_medium_evt['norm_voxels'],
                    'medium_active_voxels': sobolev_pool_medium_evt['active_voxels'],
                    'medium_total': sobolev_pool_medium_evt['total'],
                    'global_value': sobolev_pool_global_evt['value'],
                    'global_grad_x_e': sobolev_pool_global_evt['grad_x_e'],
                    'global_grad_y_e': sobolev_pool_global_evt['grad_y_e'],
                    'global_grad_z_e': sobolev_pool_global_evt['grad_z_e'],
                    'global_sobolev_3d_grad': sobolev_pool_global_evt['sobolev_3d_grad'],
                    'global_norm_voxels': sobolev_pool_global_evt['norm_voxels'],
                    'global_active_voxels': sobolev_pool_global_evt['active_voxels'],
                    'global_total': sobolev_pool_global_evt['total'],
                    'mean_pred_occupancy': mean_pred_occupancy_evt,
                    'mean_target_occupancy': mean_target_occupancy_evt,
                    'residual_mean_abs': jnp.mean(jnp.abs(residual_xyz_evt)),
                    'pred_field_mean': jnp.mean(pred_expected_xyz_evt),
                    'target_field_mean': jnp.mean(target_expected_xyz_evt),
                    'z_win_start_tick': t_start_evt.astype(jnp.float32),
                    'event_weight': evt_weight,
                    'event_target_charge': evt_target_charge,
                    'active_event': jnp.array(1.0, dtype=jnp.float32),
                }

            return jax.lax.cond(has_evt, _compute_for_event, lambda _: zero_metrics, operand=None)

        def _scan_body(carry, evt_id):
            evt_metrics = _event_metrics(evt_id)
            w = evt_metrics['event_weight']
            for key in zero_metrics.keys():
                carry[key] = carry[key] + w * evt_metrics[key]
            carry['sum_weights'] = carry['sum_weights'] + w
            carry['sum_active_events'] = carry['sum_active_events'] + evt_metrics['active_event']
            carry['sum_event_target_charge'] = carry['sum_event_target_charge'] + evt_metrics['event_target_charge']
            return carry, 0

        weighted_sums = {k: jnp.array(0.0, dtype=jnp.float32) for k in zero_metrics.keys()}
        weighted_sums['sum_weights'] = jnp.array(0.0, dtype=jnp.float32)
        weighted_sums['sum_active_events'] = jnp.array(0.0, dtype=jnp.float32)
        weighted_sums['sum_event_target_charge'] = jnp.array(0.0, dtype=jnp.float32)
        weighted_sums, _ = jax.lax.scan(_scan_body, weighted_sums, event_ids)

        denom_weights = jnp.maximum(weighted_sums['sum_weights'], 1e-12)
        sobolev_pool_reports = {
            'local_norm_voxels': weighted_sums['local_norm_voxels'] / denom_weights,
            'local_active_voxels': weighted_sums['local_active_voxels'] / denom_weights,
            'local_value': weighted_sums['local_value'] / denom_weights,
            'local_grad_x_e': weighted_sums['local_grad_x_e'] / denom_weights,
            'local_grad_y_e': weighted_sums['local_grad_y_e'] / denom_weights,
            'local_grad_z_e': weighted_sums['local_grad_z_e'] / denom_weights,
            'local_sobolev_3d_grad': weighted_sums['local_sobolev_3d_grad'] / denom_weights,
            'local_total': weighted_sums['local_total'] / denom_weights,
            'medium_norm_voxels': weighted_sums['medium_norm_voxels'] / denom_weights,
            'medium_active_voxels': weighted_sums['medium_active_voxels'] / denom_weights,
            'medium_value': weighted_sums['medium_value'] / denom_weights,
            'medium_grad_x_e': weighted_sums['medium_grad_x_e'] / denom_weights,
            'medium_grad_y_e': weighted_sums['medium_grad_y_e'] / denom_weights,
            'medium_grad_z_e': weighted_sums['medium_grad_z_e'] / denom_weights,
            'medium_sobolev_3d_grad': weighted_sums['medium_sobolev_3d_grad'] / denom_weights,
            'medium_total': weighted_sums['medium_total'] / denom_weights,
            'global_norm_voxels': weighted_sums['global_norm_voxels'] / denom_weights,
            'global_active_voxels': weighted_sums['global_active_voxels'] / denom_weights,
            'global_value': weighted_sums['global_value'] / denom_weights,
            'global_grad_x_e': weighted_sums['global_grad_x_e'] / denom_weights,
            'global_grad_y_e': weighted_sums['global_grad_y_e'] / denom_weights,
            'global_grad_z_e': weighted_sums['global_grad_z_e'] / denom_weights,
            'global_sobolev_3d_grad': weighted_sums['global_sobolev_3d_grad'] / denom_weights,
            'global_total': weighted_sums['global_total'] / denom_weights,
        }

        target_gaussian_radius_cm_eff = 0.0 if self.target_gaussian_3d_radius_cm is None else float(self.target_gaussian_3d_radius_cm)
        if self.target_gaussian_3d_sigma_cm is not None:
            target_gaussian_sigma_cm_eff = float(self.target_gaussian_3d_sigma_cm)
        else:
            target_gaussian_sigma_cm_eff = max(float(target_gaussian_radius_cm_eff) / 2.0, float(pixel_pitch))
        target_gaussian_sigma_cm_eff = max(target_gaussian_sigma_cm_eff, 1e-6)

        sobolev_pool_reports['layer_weight_local'] = layer_weights[0]
        sobolev_pool_reports['layer_weight_medium'] = layer_weights[1]
        sobolev_pool_reports['layer_weight_global'] = layer_weights[2]
        sobolev_pool_reports['running_scale_local'] = jnp.array(
            float(self.sobolev_pool_running_scales['local']), dtype=jnp.float32
        )
        sobolev_pool_reports['running_scale_medium'] = jnp.array(
            float(self.sobolev_pool_running_scales['medium']), dtype=jnp.float32
        )
        sobolev_pool_reports['running_scale_global'] = jnp.array(
            float(self.sobolev_pool_running_scales['global']), dtype=jnp.float32
        )

        if self.emit_sobolev_pool_report:
            print(self._format_three_layer_sobolev_pooling())
            jax.debug.print(
                (
                    "3-layer Sobolev metrics\n"
                    "  local : norm_vox={ln:.1f}, active_vox={la:.1f}, value={lv:.6e}, grad_x={lx:.6e}, grad_y={ly:.6e}, grad_z={lz:.6e}, grad_3d={l3:.6e}, total={lt:.6e}\n"
                    "  medium: norm_vox={mn:.1f}, active_vox={ma:.1f}, value={mv:.6e}, grad_x={mx:.6e}, grad_y={my:.6e}, grad_z={mz:.6e}, grad_3d={m3:.6e}, total={mt:.6e}\n"
                    "  global: norm_vox={gn:.1f}, active_vox={ga:.1f}, value={gv:.6e}, grad_x={gx:.6e}, grad_y={gy:.6e}, grad_z={gz:.6e}, grad_3d={g3:.6e}, total={gt:.6e}"
                ),
                ln=sobolev_pool_reports['local_norm_voxels'],
                la=sobolev_pool_reports['local_active_voxels'],
                lv=sobolev_pool_reports['local_value'],
                lx=sobolev_pool_reports['local_grad_x_e'],
                ly=sobolev_pool_reports['local_grad_y_e'],
                lz=sobolev_pool_reports['local_grad_z_e'],
                l3=sobolev_pool_reports['local_sobolev_3d_grad'],
                lt=sobolev_pool_reports['local_total'],
                mn=sobolev_pool_reports['medium_norm_voxels'],
                ma=sobolev_pool_reports['medium_active_voxels'],
                mv=sobolev_pool_reports['medium_value'],
                mx=sobolev_pool_reports['medium_grad_x_e'],
                my=sobolev_pool_reports['medium_grad_y_e'],
                mz=sobolev_pool_reports['medium_grad_z_e'],
                m3=sobolev_pool_reports['medium_sobolev_3d_grad'],
                mt=sobolev_pool_reports['medium_total'],
                gn=sobolev_pool_reports['global_norm_voxels'],
                ga=sobolev_pool_reports['global_active_voxels'],
                gv=sobolev_pool_reports['global_value'],
                gx=sobolev_pool_reports['global_grad_x_e'],
                gy=sobolev_pool_reports['global_grad_y_e'],
                gz=sobolev_pool_reports['global_grad_z_e'],
                g3=sobolev_pool_reports['global_sobolev_3d_grad'],
                gt=sobolev_pool_reports['global_total'],
            )

        sobolev_3d_value = weighted_sums['sobolev_3d_value'] / denom_weights
        sobolev_3d_grad = weighted_sums['sobolev_3d_grad'] / denom_weights
        sobolev_3d = weighted_sums['sobolev_3d'] / denom_weights

        total_loss = sobolev_3d
        mean_pred_occupancy = weighted_sums['mean_pred_occupancy'] / denom_weights
        mean_target_occupancy = weighted_sums['mean_target_occupancy'] / denom_weights

        aux = {
            'total_target_charge': total_target_charge,
            'sobolev_integrated_field_mean_pred': weighted_sums['pred_field_mean'] / denom_weights,
            'sobolev_integrated_field_mean_target': weighted_sums['target_field_mean'] / denom_weights,
            'sobolev_3d_value': sobolev_3d_value,
            'sobolev_3d_grad': sobolev_3d_grad,
            'sobolev_3d': sobolev_3d,
            'sobolev_pool_local_value': sobolev_pool_reports['local_value'],
            'sobolev_pool_local_grad_x_e': sobolev_pool_reports['local_grad_x_e'],
            'sobolev_pool_local_grad_y_e': sobolev_pool_reports['local_grad_y_e'],
            'sobolev_pool_local_grad_z_e': sobolev_pool_reports['local_grad_z_e'],
            'sobolev_pool_local_sobolev_3d_grad': sobolev_pool_reports['local_sobolev_3d_grad'],
            'sobolev_pool_local_norm_voxels': sobolev_pool_reports['local_norm_voxels'],
            'sobolev_pool_local_active_voxels': sobolev_pool_reports['local_active_voxels'],
            'sobolev_pool_local_total': sobolev_pool_reports['local_total'],
            'sobolev_pool_medium_value': sobolev_pool_reports['medium_value'],
            'sobolev_pool_medium_grad_x_e': sobolev_pool_reports['medium_grad_x_e'],
            'sobolev_pool_medium_grad_y_e': sobolev_pool_reports['medium_grad_y_e'],
            'sobolev_pool_medium_grad_z_e': sobolev_pool_reports['medium_grad_z_e'],
            'sobolev_pool_medium_sobolev_3d_grad': sobolev_pool_reports['medium_sobolev_3d_grad'],
            'sobolev_pool_medium_norm_voxels': sobolev_pool_reports['medium_norm_voxels'],
            'sobolev_pool_medium_active_voxels': sobolev_pool_reports['medium_active_voxels'],
            'sobolev_pool_medium_total': sobolev_pool_reports['medium_total'],
            'sobolev_pool_global_value': sobolev_pool_reports['global_value'],
            'sobolev_pool_global_grad_x_e': sobolev_pool_reports['global_grad_x_e'],
            'sobolev_pool_global_grad_y_e': sobolev_pool_reports['global_grad_y_e'],
            'sobolev_pool_global_grad_z_e': sobolev_pool_reports['global_grad_z_e'],
            'sobolev_pool_global_sobolev_3d_grad': sobolev_pool_reports['global_sobolev_3d_grad'],
            'sobolev_pool_global_norm_voxels': sobolev_pool_reports['global_norm_voxels'],
            'sobolev_pool_global_active_voxels': sobolev_pool_reports['global_active_voxels'],
            'sobolev_pool_global_total': sobolev_pool_reports['global_total'],
            'sobolev_pool_layer_balance_mode': self.sobolev_pool_layer_balance,
            'sobolev_pool_layer_weight_local': sobolev_pool_reports['layer_weight_local'],
            'sobolev_pool_layer_weight_medium': sobolev_pool_reports['layer_weight_medium'],
            'sobolev_pool_layer_weight_global': sobolev_pool_reports['layer_weight_global'],
            'sobolev_pool_running_scale_local': sobolev_pool_reports['running_scale_local'],
            'sobolev_pool_running_scale_medium': sobolev_pool_reports['running_scale_medium'],
            'sobolev_pool_running_scale_global': sobolev_pool_reports['running_scale_global'],
            'sobolev_pool_running_decay': jnp.array(self.sobolev_pool_running_decay, dtype=jnp.float32),
            'target_gaussian_3d_radius_cm': jnp.array(target_gaussian_radius_cm_eff, dtype=jnp.float32),
            'target_gaussian_3d_sigma_cm': jnp.array(target_gaussian_sigma_cm_eff, dtype=jnp.float32),
            'z_tick_size': jnp.array(z_tick_size, dtype=jnp.float32),
            'w_sobolev_3d_grad': self.w_sobolev_3d_grad,
            'mean_pred_occupancy': mean_pred_occupancy,
            'mean_target_occupancy': mean_target_occupancy,
            'z_win_start_tick': weighted_sums['z_win_start_tick'] / denom_weights,
            'nz_local': jnp.array(nz_local, dtype=jnp.float32),
            'z_bin_size_sparse': jnp.array(z_bin_size_sparse, dtype=jnp.float32),
            'residual_mean_abs': weighted_sums['residual_mean_abs'] / denom_weights,
            'event_weighted_mean_denom': denom_weights,
            'event_active_count': weighted_sums['sum_active_events'],
            'event_target_charge_sum': weighted_sums['sum_event_target_charge'],
            'max_events_per_batch': jnp.array(self.max_events_per_batch, dtype=jnp.float32),
            'event_overflow_target_hits': overflow_target_hits,
            'event_overflow_pred_pixels': overflow_pred_pixels,
        }

        return total_loss, aux
