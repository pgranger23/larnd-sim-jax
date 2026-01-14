"""
Inference utilities for SIREN surrogate.

Provides a production-ready predictor wrapper for using trained SIREN models.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, Any
from flax.core.frozen_dict import freeze

from .core import SIREN, create_siren
from .training.checkpointing import load_final_model


class SurrogatePredictor:
    """
    Production-ready wrapper for SIREN surrogate inference.

    Handles model loading, input normalization, output denormalization,
    and provides convenient prediction interfaces.

    Example usage:
        predictor = SurrogatePredictor('siren_training/final_model.npz')

        # Single prediction
        response = predictor.predict(diff=50, x=22, y=22, t=1000)

        # Batch prediction
        coords = np.array([[50, 22, 22, 1000], [50, 22, 22, 1001]])
        responses = predictor.predict_batch(coords)

        # Get full waveform for a position
        waveform = predictor.get_waveform(diff=50, x=22, y=22)
    """

    def __init__(self, model_path: str):
        """
        Initialize predictor from saved model.

        Args:
            model_path: Path to saved model (.npz file).
        """
        self.model_path = model_path

        # Load model
        params, model_config, norm_params, dataset_stats, metadata = load_final_model(model_path)

        self.params = params
        self.model_config = model_config
        self.norm_params = norm_params
        self.dataset_stats = dataset_stats
        self.metadata = metadata

        # Create model instance
        self.model = create_siren(**model_config)

        # Store normalization ranges
        self.diff_range = tuple(norm_params['diff_range'])
        self.x_range = tuple(norm_params['x_range'])
        self.y_range = tuple(norm_params['y_range'])
        self.t_range = tuple(norm_params['t_range'])
        self.output_min = norm_params['output_min']
        self.output_max = norm_params['output_max']
        self.normalize_inputs = norm_params['normalize_inputs']
        self.normalize_outputs = norm_params['normalize_outputs']

        # JIT compile prediction function
        self._predict_fn = jax.jit(lambda params, x: self.model.apply(params, x))

        print(f"Loaded SIREN surrogate from {model_path}")
        print(f"  Model: {model_config['hidden_features']} features × {model_config['hidden_layers']} layers")
        print(f"  Trained for {metadata['final_step']} steps")

    def _normalize_coords(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Normalize coordinates to [-1, 1]."""
        if not self.normalize_inputs:
            return coords

        normalized = jnp.zeros_like(coords)
        ranges = [self.diff_range, self.x_range, self.y_range, self.t_range]

        for i, (lo, hi) in enumerate(ranges):
            normalized = normalized.at[..., i].set(
                2.0 * (coords[..., i] - lo) / (hi - lo) - 1.0
            )

        return normalized

    def _denormalize_output(self, output: jnp.ndarray) -> jnp.ndarray:
        """Denormalize output to original scale."""
        if not self.normalize_outputs:
            return output

        return output * (self.output_max - self.output_min) + self.output_min

    def predict(
        self,
        diff: float,
        x: float,
        y: float,
        t: float,
    ) -> float:
        """
        Predict response value at a single point.

        Args:
            diff: Diffusion index (0-99).
            x: X bin index (0-44).
            y: Y bin index (0-44).
            t: Time tick (0-1949).

        Returns:
            Predicted response value.
        """
        coords = jnp.array([[diff, x, y, t]], dtype=jnp.float32)
        coords_norm = self._normalize_coords(coords)
        pred_norm = self._predict_fn(self.params, coords_norm)
        pred = self._denormalize_output(pred_norm)
        return float(pred[0, 0])

    def predict_batch(
        self,
        coords: Union[np.ndarray, jnp.ndarray],
    ) -> np.ndarray:
        """
        Predict response values for a batch of coordinates.

        Args:
            coords: Array of shape (N, 4) with columns [diff, x, y, t].

        Returns:
            Array of predicted response values of shape (N,).
        """
        coords = jnp.array(coords, dtype=jnp.float32)
        coords_norm = self._normalize_coords(coords)
        pred_norm = self._predict_fn(self.params, coords_norm)
        pred = self._denormalize_output(pred_norm)
        return np.array(pred).ravel()

    def get_waveform(
        self,
        diff: float,
        x: float,
        y: float,
        t_start: int = 0,
        t_end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get full waveform for a fixed position.

        Args:
            diff: Diffusion index.
            x: X bin index.
            y: Y bin index.
            t_start: Starting time tick.
            t_end: Ending time tick (defaults to max).

        Returns:
            Array of response values over time.
        """
        if t_end is None:
            t_end = int(self.t_range[1]) + 1

        t_vals = np.arange(t_start, t_end, dtype=np.float32)
        n_t = len(t_vals)

        coords = np.stack([
            np.full(n_t, diff, dtype=np.float32),
            np.full(n_t, x, dtype=np.float32),
            np.full(n_t, y, dtype=np.float32),
            t_vals,
        ], axis=1)

        return self.predict_batch(coords)

    def get_spatial_slice(
        self,
        diff: float,
        t: float,
    ) -> np.ndarray:
        """
        Get 2D spatial slice at fixed diffusion and time.

        Args:
            diff: Diffusion index.
            t: Time tick.

        Returns:
            Array of shape (n_x, n_y) with response values.
        """
        n_x = int(self.x_range[1]) + 1
        n_y = int(self.y_range[1]) + 1

        x_vals, y_vals = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing='ij')

        coords = np.stack([
            np.full(n_x * n_y, diff, dtype=np.float32),
            x_vals.ravel().astype(np.float32),
            y_vals.ravel().astype(np.float32),
            np.full(n_x * n_y, t, dtype=np.float32),
        ], axis=1)

        preds = self.predict_batch(coords)
        return preds.reshape(n_x, n_y)

    def get_diffusion_slice(
        self,
        x: float,
        y: float,
        t: float,
    ) -> np.ndarray:
        """
        Get response as function of diffusion at fixed position.

        Args:
            x: X bin index.
            y: Y bin index.
            t: Time tick.

        Returns:
            Array of response values for each diffusion index.
        """
        n_diff = int(self.diff_range[1]) + 1
        diff_vals = np.arange(n_diff, dtype=np.float32)

        coords = np.stack([
            diff_vals,
            np.full(n_diff, x, dtype=np.float32),
            np.full(n_diff, y, dtype=np.float32),
            np.full(n_diff, t, dtype=np.float32),
        ], axis=1)

        return self.predict_batch(coords)

    def get_info(self) -> Dict[str, Any]:
        """Get model and training information."""
        return {
            'model_path': self.model_path,
            'model_config': self.model_config,
            'normalization': {
                'diff_range': self.diff_range,
                'x_range': self.x_range,
                'y_range': self.y_range,
                't_range': self.t_range,
                'output_range': (self.output_min, self.output_max),
            },
            'training': self.metadata,
            'dataset_stats': self.dataset_stats,
        }

    def __repr__(self) -> str:
        return (
            f"SurrogatePredictor("
            f"model={self.model_config['hidden_features']}×{self.model_config['hidden_layers']}, "
            f"trained={self.metadata['final_step']} steps)"
        )


def load_surrogate(model_path: str) -> SurrogatePredictor:
    """
    Convenience function to load a SIREN surrogate.

    Args:
        model_path: Path to saved model.

    Returns:
        SurrogatePredictor instance.
    """
    return SurrogatePredictor(model_path)
