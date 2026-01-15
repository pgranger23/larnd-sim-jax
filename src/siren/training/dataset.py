"""
Dataset handling for SIREN surrogate training.

Loads the response_template LUT and provides batched sampling for training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class NormalizationParams:
    """Parameters for input/output normalization."""
    # Input ranges (for normalizing to [-1, 1])
    diff_range: Tuple[float, float] = (0.0, 99.0)
    x_range: Tuple[float, float] = (0.0, 44.0)
    y_range: Tuple[float, float] = (0.0, 44.0)
    t_range: Tuple[float, float] = (0.0, 1949.0)

    # Output normalization
    output_min: float = 0.0
    output_max: float = 1.0
    normalize_inputs: bool = True
    normalize_outputs: bool = True

    def to_dict(self) -> dict:
        return {
            'diff_range': self.diff_range,
            'x_range': self.x_range,
            'y_range': self.y_range,
            't_range': self.t_range,
            'output_min': self.output_min,
            'output_max': self.output_max,
            'normalize_inputs': self.normalize_inputs,
            'normalize_outputs': self.normalize_outputs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'NormalizationParams':
        return cls(**d)


class ResponseTemplateDataset:
    """
    Dataset for training SIREN surrogate on the response_template LUT.

    The response_template has shape (N_diff, Nx, Ny, Nt) = (100, 45, 45, 1950).
    We flatten this to create coordinate-value pairs for training.
    """

    def __init__(
        self,
        lut_path: str,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
        seed: int = 42,
        precomputed_template: Optional[np.ndarray] = None,
        use_cdf: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            lut_path: Path to response NPZ file.
            normalize_inputs: Whether to normalize inputs to [-1, 1].
            normalize_outputs: Whether to normalize outputs.
            seed: Random seed for sampling.
            precomputed_template: If provided, use this instead of loading from file.
            use_cdf: If True, train on cumulative distribution (CDF/10) instead of raw response.
        """
        self.lut_path = lut_path
        self.seed = seed
        self.use_cdf = use_cdf

        # Load or use provided template
        if precomputed_template is not None:
            self.response_template = precomputed_template
        else:
            self.response_template = self._load_and_build_template(lut_path)

        # Compute CDF if needed (cumsum along time axis, normalized by 10)
        if use_cdf:
            self.cdf_template = np.cumsum(self.response_template, axis=-1) / 10.0
            print(f"CDF template computed: shape {self.cdf_template.shape}")
            print(f"CDF range: [{self.cdf_template.min():.4f}, {self.cdf_template.max():.4f}]")
            # Verify: main pixel (0,0) should end at ~1.0, neighbors should end at ~0
            print(f"CDF final values - Main pixel (0,0,0): {self.cdf_template[0, 0, 0, -1]:.4f}")
            print(f"CDF final values - Neighbor (0,9,9): {self.cdf_template[0, 9, 9, -1]:.4f}")

        # Get dimensions
        self.n_diff, self.n_x, self.n_y, self.n_t = self.response_template.shape
        self.total_points = self.n_diff * self.n_x * self.n_y * self.n_t

        print(f"Response template shape: {self.response_template.shape}")
        print(f"Total data points: {self.total_points:,}")

        # Create coordinate grids
        self._create_coordinate_grids()

        # Setup normalization
        # For CDF mode, we use the CDF values which are in [0, ~1] for main pixel
        if use_cdf:
            output_min = float(self.cdf_template.min())
            output_max = float(self.cdf_template.max())
        else:
            output_min = float(self.response_template.min())
            output_max = float(self.response_template.max())

        self.norm_params = NormalizationParams(
            diff_range=(0.0, float(self.n_diff - 1)),
            x_range=(0.0, float(self.n_x - 1)),
            y_range=(0.0, float(self.n_y - 1)),
            t_range=(0.0, float(self.n_t - 1)),
            output_min=output_min,
            output_max=output_max,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
        )

        print(f"Output range: [{self.norm_params.output_min:.4f}, {self.norm_params.output_max:.4f}]")

    def _load_and_build_template(self, lut_path: str) -> np.ndarray:
        """Load raw LUT and build response_template with diffusion convolution."""
        print(f"Loading LUT from {lut_path}")

        # Load raw response
        data = np.load(lut_path)
        if 'response' in data:
            response = data['response']
        else:
            response = data

        response = response.astype(np.float32)
        print(f"Raw response shape: {response.shape}")

        # Build response template with diffusion convolution
        # Following the logic in consts_jax.py
        long_diff_template = np.linspace(0.001, 10, 100)
        long_diff_extent = 20

        # Create Gaussian kernels for each diffusion value
        t_range = np.arange(-long_diff_extent, long_diff_extent + 1, 1)
        gaus = norm.pdf(t_range, scale=long_diff_template[:, None])
        gaus = gaus / gaus.sum(axis=1, keepdims=True)  # Normalize
        gaus = gaus.astype(np.float32)

        print(f"Building response_template with {len(long_diff_template)} diffusion values...")
        print("  Using scipy convolution (CPU)...")

        # Use scipy for convolution (CPU-based, avoids CUDA initialization issues)
        from scipy.ndimage import convolve1d

        n_x, n_y, n_t = response.shape
        n_diff = len(long_diff_template)

        # Preallocate output
        response_template = np.zeros((n_diff, n_x, n_y, n_t), dtype=np.float32)

        # Convolve along time axis for each diffusion kernel
        for i, kernel in enumerate(gaus):
            # convolve1d applies along axis=-1 (time) by default
            response_template[i] = convolve1d(response, kernel, axis=-1, mode='constant')

        # Set index 0 to raw response (no diffusion)
        response_template[0] = response

        print(f"Response template built: {response_template.shape}")
        return response_template

    def _create_coordinate_grids(self):
        """
        Setup for lazy coordinate computation.

        We don't precompute all 395M coordinates - instead we compute them
        on-the-fly during sampling using index arithmetic.
        """
        # Store flattened values for fast indexing
        # In CDF mode, primary values are CDF; also store response for derivative loss
        if self.use_cdf:
            self.values = self.cdf_template.ravel().astype(np.float32)
            self.response_values = self.response_template.ravel().astype(np.float32)
        else:
            self.values = self.response_template.ravel().astype(np.float32)
            self.response_values = None  # Not needed in non-CDF mode

        # Precompute strides for index-to-coordinate conversion
        # Flat index = d * (n_x * n_y * n_t) + x * (n_y * n_t) + y * n_t + t
        self._stride_d = self.n_x * self.n_y * self.n_t
        self._stride_x = self.n_y * self.n_t
        self._stride_y = self.n_t

    def _indices_to_coords(self, flat_indices: np.ndarray) -> np.ndarray:
        """Convert flat indices to 4D coordinates."""
        d = flat_indices // self._stride_d
        remainder = flat_indices % self._stride_d
        x = remainder // self._stride_x
        remainder = remainder % self._stride_x
        y = remainder // self._stride_y
        t = remainder % self._stride_y

        return np.stack([d, x, y, t], axis=1).astype(np.float32)

    def normalize_inputs(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Normalize coordinates to [-1, 1] range."""
        if not self.norm_params.normalize_inputs:
            return coords

        # Normalize each dimension
        normalized = jnp.zeros_like(coords)
        ranges = [
            self.norm_params.diff_range,
            self.norm_params.x_range,
            self.norm_params.y_range,
            self.norm_params.t_range,
        ]

        for i, (lo, hi) in enumerate(ranges):
            normalized = normalized.at[:, i].set(
                2.0 * (coords[:, i] - lo) / (hi - lo) - 1.0
            )

        return normalized

    def denormalize_inputs(self, normalized: jnp.ndarray) -> jnp.ndarray:
        """Convert normalized coordinates back to original range."""
        if not self.norm_params.normalize_inputs:
            return normalized

        coords = jnp.zeros_like(normalized)
        ranges = [
            self.norm_params.diff_range,
            self.norm_params.x_range,
            self.norm_params.y_range,
            self.norm_params.t_range,
        ]

        for i, (lo, hi) in enumerate(ranges):
            coords = coords.at[:, i].set(
                (normalized[:, i] + 1.0) * (hi - lo) / 2.0 + lo
            )

        return coords

    def normalize_outputs(self, values: jnp.ndarray) -> jnp.ndarray:
        """Normalize output values to [0, 1] range."""
        if not self.norm_params.normalize_outputs:
            return values

        lo = self.norm_params.output_min
        hi = self.norm_params.output_max
        return (values - lo) / (hi - lo)

    def denormalize_outputs(self, normalized: jnp.ndarray) -> jnp.ndarray:
        """Convert normalized outputs back to original range."""
        if not self.norm_params.normalize_outputs:
            return normalized

        lo = self.norm_params.output_min
        hi = self.norm_params.output_max
        return normalized * (hi - lo) + lo

    def sample_batch(
        self,
        rng_key: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample a random batch from the dataset.

        Args:
            rng_key: JAX random key.
            batch_size: Number of samples.

        Returns:
            Tuple of (normalized_coords, normalized_values).
        """
        # Sample random indices from all data points
        batch_indices = jax.random.choice(
            rng_key, self.total_points, shape=(batch_size,), replace=True
        )
        batch_indices = np.array(batch_indices)

        # Compute coordinates from flat indices (lazy evaluation)
        coords = self._indices_to_coords(batch_indices)
        values = self.values[batch_indices]

        # Convert to JAX arrays and normalize
        coords_jax = jnp.array(coords)
        values_jax = jnp.array(values)

        coords_norm = self.normalize_inputs(coords_jax)
        values_norm = self.normalize_outputs(values_jax)

        return coords_norm, values_norm.reshape(-1, 1)

    def sample_batch_cdf(
        self,
        rng_key: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample a random batch for CDF training with derivative loss.

        Returns both CDF targets and response targets (for derivative loss).

        Args:
            rng_key: JAX random key.
            batch_size: Number of samples.

        Returns:
            Tuple of (normalized_coords, cdf_values, response_values).
            - cdf_values: CDF/10 values (training target for SIREN)
            - response_values: Raw response values (training target for derivative)
        """
        if not self.use_cdf:
            raise ValueError("sample_batch_cdf requires use_cdf=True")

        # Sample random indices from all data points
        batch_indices = jax.random.choice(
            rng_key, self.total_points, shape=(batch_size,), replace=True
        )
        batch_indices = np.array(batch_indices)

        # Compute coordinates from flat indices (lazy evaluation)
        coords = self._indices_to_coords(batch_indices)

        # Get both CDF and response values
        cdf_values = self.values[batch_indices]  # Already CDF/10 in CDF mode
        response_values = self.response_values[batch_indices]

        # Convert to JAX arrays
        coords_jax = jnp.array(coords)
        cdf_jax = jnp.array(cdf_values)
        response_jax = jnp.array(response_values)

        # Normalize coordinates
        coords_norm = self.normalize_inputs(coords_jax)

        # Note: We don't normalize CDF or response values for the derivative loss
        # CDF is already in [0, ~1] range, response is used as-is for derivative target

        return coords_norm, cdf_jax.reshape(-1, 1), response_jax.reshape(-1, 1)

    def get_slice(
        self,
        diff_idx: Optional[int] = None,
        x_idx: Optional[int] = None,
        y_idx: Optional[int] = None,
        t_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a slice of the response_template for visualization.

        Fix one or more indices to get lower-dimensional slices.

        Returns:
            Tuple of (coordinates, values) for the slice.
        """
        # Get the slice directly from response_template
        if diff_idx is not None and x_idx is not None and y_idx is not None:
            # 1D slice over time
            values = self.response_template[diff_idx, x_idx, y_idx, :]
            t_vals = np.arange(self.n_t)
            coords = np.stack([
                np.full(self.n_t, diff_idx),
                np.full(self.n_t, x_idx),
                np.full(self.n_t, y_idx),
                t_vals,
            ], axis=1).astype(np.float32)
            return coords, values

        elif diff_idx is not None and t_idx is not None:
            # 2D slice over x, y
            values = self.response_template[diff_idx, :, :, t_idx].ravel()
            x_grid, y_grid = np.meshgrid(np.arange(self.n_x), np.arange(self.n_y), indexing='ij')
            coords = np.stack([
                np.full(self.n_x * self.n_y, diff_idx),
                x_grid.ravel(),
                y_grid.ravel(),
                np.full(self.n_x * self.n_y, t_idx),
            ], axis=1).astype(np.float32)
            return coords, values

        else:
            raise ValueError("Must fix at least (diff, x, y) or (diff, t) for slice")

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'shape': self.response_template.shape,
            'total_points': self.total_points,
            'value_min': float(self.norm_params.output_min),
            'value_max': float(self.norm_params.output_max),
            'value_mean': float(self.values.mean()),
            'value_std': float(self.values.std()),
        }
