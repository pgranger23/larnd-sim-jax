"""
Coordinate transformation utilities for SIREN surrogate integration.

Provides functions to convert physical quantities (diffusion, position)
to SIREN input coordinates, and to convert SIREN output to simulation values.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Any

from flax.core.frozen_dict import freeze
from src.siren.core import create_siren
from src.siren.training.checkpointing import load_final_model


# ==============================================================================
# Input Coordinate Transformations
# ==============================================================================

@jax.jit
def long_diff_to_siren_diff(
    long_diff_ticks: jnp.ndarray,
    diff_min: float = 0.001,
    diff_max: float = 10.0,
    n_diff: int = 100,
) -> jnp.ndarray:
    """
    Convert longitudinal diffusion (in ticks) to SIREN diffusion coordinate.

    The LUT uses `long_diff_template = linspace(0.001, 10, 100)` which maps
    to indices 0-99. We linearly interpolate to get continuous values.

    Args:
        long_diff_ticks: Diffusion in time ticks (computed as
            long_diff_cm / v_drift / t_sampling).
        diff_min: Minimum diffusion in template (0.001 ticks).
        diff_max: Maximum diffusion in template (10.0 ticks).
        n_diff: Number of diffusion values in template (100).

    Returns:
        SIREN diffusion coordinate in [0, n_diff-1] range.
    """
    # Linear interpolation: diff_ticks -> index
    # Template: [0.001, 0.102, 0.203, ..., 10.0] at indices [0, 1, 2, ..., 99]
    diff_siren = (long_diff_ticks - diff_min) / (diff_max - diff_min) * (n_diff - 1)

    # Clamp to valid range
    return jnp.clip(diff_siren, 0.0, n_diff - 1)


@jax.jit
def position_to_siren_xy(
    x_dist: jnp.ndarray,
    y_dist: jnp.ndarray,
    response_bin_size: float = 0.04434,
    n_bins: int = 45,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert position distances to fractional SIREN bin indices.

    The LUT discretizes position as `i = (x_dist / response_bin_size).astype(int)`.
    The surrogate uses continuous fractional indices for better accuracy.

    Args:
        x_dist: Distance from electron to pixel center in x (cm).
        y_dist: Distance from electron to pixel center in y (cm).
        response_bin_size: LUT bin size (0.04434 cm).
        n_bins: Number of bins per axis (45, for indices 0-44).

    Returns:
        Tuple of (x_frac, y_frac) fractional bin indices in [0, n_bins-1].
    """
    x_frac = x_dist / response_bin_size
    y_frac = y_dist / response_bin_size

    # Clamp to valid range (LUT has 45 bins: indices 0-44)
    x_frac = jnp.clip(x_frac, 0.0, n_bins - 1)
    y_frac = jnp.clip(y_frac, 0.0, n_bins - 1)

    return x_frac, y_frac


@jax.jit
def normalize_siren_inputs(
    diff: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    diff_range: Tuple[float, float] = (0.0, 99.0),
    x_range: Tuple[float, float] = (0.0, 44.0),
    y_range: Tuple[float, float] = (0.0, 44.0),
    t_range: Tuple[float, float] = (0.0, 1949.0),
) -> jnp.ndarray:
    """
    Normalize SIREN input coordinates to [-1, 1] range.

    Args:
        diff: Diffusion index (0-99 continuous).
        x: X bin index (0-44 continuous).
        y: Y bin index (0-44 continuous).
        t: Time tick (0-1949 discrete).
        *_range: Min/max values for each coordinate.

    Returns:
        Normalized coordinates of shape (..., 4) with values in [-1, 1].
    """
    def _normalize(val, lo, hi):
        return 2.0 * (val - lo) / (hi - lo) - 1.0

    diff_norm = _normalize(diff, diff_range[0], diff_range[1])
    x_norm = _normalize(x, x_range[0], x_range[1])
    y_norm = _normalize(y, y_range[0], y_range[1])
    t_norm = _normalize(t, t_range[0], t_range[1])

    # Stack into (N, 4) array
    return jnp.stack([diff_norm, x_norm, y_norm, t_norm], axis=-1)


# ==============================================================================
# Output Transformations
# ==============================================================================

@jax.jit
def siren_cdf_to_q_cumsum(
    siren_output: jnp.ndarray,
    n_electrons: jnp.ndarray,
    t_sampling: float,
    output_scale: float = 10.0,
) -> jnp.ndarray:
    """
    Convert SIREN CDF output to charge cumsum for FEE processing.

    The SIREN is trained on CDF/10 values, where CDF = cumsum(response) along time.
    The FEE simulation expects q_cumsum = wfs.cumsum(axis=-1) * t_sampling.

    Since the SIREN outputs CDF values directly (main pixel ends at ~1.0 after /10),
    we scale back up and multiply by the number of electrons.

    Formula: q_cumsum = siren_output * 10 * n_electrons * t_sampling

    Args:
        siren_output: Raw SIREN output (CDF/10 values).
        n_electrons: Number of electrons per charge deposit.
        t_sampling: Time sampling in microseconds.
        output_scale: Scale factor (10.0, since we trained on CDF/10).

    Returns:
        Charge cumsum values compatible with FEE simulation.
    """
    # SIREN output is CDF/10, scale back up
    # Then multiply by n_electrons and t_sampling for charge units
    return siren_output * output_scale * n_electrons * t_sampling


@jax.jit
def denormalize_siren_output(
    output_normalized: jnp.ndarray,
    output_min: float,
    output_max: float,
) -> jnp.ndarray:
    """
    Denormalize SIREN output from [0, 1] to original scale.

    Only needed if normalize_outputs=True was used during training.

    Args:
        output_normalized: Normalized SIREN output.
        output_min: Minimum value in training data.
        output_max: Maximum value in training data.

    Returns:
        Denormalized output values.
    """
    return output_normalized * (output_max - output_min) + output_min


# ==============================================================================
# Model Loading
# ==============================================================================

def load_surrogate_model(model_path: str) -> Tuple[Any, Any, Dict, Dict]:
    """
    Load a trained SIREN surrogate model for simulation.

    Args:
        model_path: Path to the saved model file (.npz).

    Returns:
        Tuple of (model, params, model_config, norm_params) where:
            - model: SIREN model instance
            - params: Frozen parameter dictionary
            - model_config: Model architecture config
            - norm_params: Normalization parameters for inputs/outputs
    """
    params, model_config, norm_params, dataset_stats, metadata = load_final_model(model_path)

    # Create model instance
    model = create_siren(**model_config)

    print(f"Loaded surrogate model from {model_path}")
    print(f"  Architecture: {model_config['hidden_features']} features x {model_config['hidden_layers']} layers")
    print(f"  Trained for {metadata['final_step']} steps")
    print(f"  Input ranges: diff=[0,99], x=[0,44], y=[0,44], t=[0,1949]")

    return model, params, model_config, norm_params


# ==============================================================================
# Batched Evaluation Utilities
# ==============================================================================

def create_siren_apply_fn(model, params):
    """
    Create a JIT-compiled function for SIREN evaluation.

    Args:
        model: SIREN model instance.
        params: Model parameters.

    Returns:
        JIT-compiled function that takes normalized coordinates and returns output.
    """
    @jax.jit
    def apply_fn(coords_normalized):
        return model.apply(params, coords_normalized)

    return apply_fn


@partial(jax.jit, static_argnums=(0,))
def batch_evaluate_siren(
    apply_fn,
    diff: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
    norm_params: Dict,
) -> jnp.ndarray:
    """
    Evaluate SIREN at a batch of coordinates with proper normalization.

    Args:
        apply_fn: JIT-compiled SIREN apply function.
        diff: Diffusion values (0-99 continuous).
        x: X bin indices (0-44 continuous).
        y: Y bin indices (0-44 continuous).
        t: Time ticks (0-1949 discrete).
        norm_params: Normalization parameters dict.

    Returns:
        SIREN output values (CDF/10 if trained with use_cdf=True).
    """
    # Normalize inputs
    coords_norm = normalize_siren_inputs(
        diff, x, y, t,
        diff_range=tuple(norm_params['diff_range']),
        x_range=tuple(norm_params['x_range']),
        y_range=tuple(norm_params['y_range']),
        t_range=tuple(norm_params['t_range']),
    )

    # Evaluate model
    output = apply_fn(coords_norm)

    # Denormalize output if needed
    if norm_params.get('normalize_outputs', True):
        output = denormalize_siren_output(
            output,
            norm_params['output_min'],
            norm_params['output_max'],
        )

    return output.squeeze(-1)


# ==============================================================================
# Pixel Grid Generation
# ==============================================================================

def generate_neighbor_offsets(n_neighbors_per_side: int = 2) -> jnp.ndarray:
    """
    Generate relative pixel offsets for the main pixel and its neighbors.

    For a 5x5 neighborhood (n_neighbors_per_side=2), returns offsets for
    25 pixels centered on (0, 0).

    LUT convention: pixel (0, 0) in the offset is the main pixel.
    Offsets are in units of pixels, converted to LUT bins later.

    Args:
        n_neighbors_per_side: Number of neighbor pixels on each side (2 = 5x5 grid).

    Returns:
        Array of shape (n_pixels, 2) with (dx, dy) offsets in pixels.
    """
    offsets = []
    for dx in range(-n_neighbors_per_side, n_neighbors_per_side + 1):
        for dy in range(-n_neighbors_per_side, n_neighbors_per_side + 1):
            offsets.append([dx, dy])

    return jnp.array(offsets, dtype=jnp.float32)


def pixel_offset_to_lut_offset(
    pixel_offset: jnp.ndarray,
    bins_per_pixel: int = 9,
) -> jnp.ndarray:
    """
    Convert pixel offsets to LUT bin offsets.

    One pixel spacing corresponds to 9 LUT bins (since pixel_pitch / response_bin_size ≈ 9).
    The LUT is centered at bin 22 (center of 45 bins) for the main pixel.

    Args:
        pixel_offset: Pixel offsets of shape (n_pixels, 2).
        bins_per_pixel: Number of LUT bins per pixel (9).

    Returns:
        LUT bin offsets from center (bin 22).
    """
    return pixel_offset * bins_per_pixel
