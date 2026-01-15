"""
SIREN (Sinusoidal Representation Network) model for LUT surrogate.

Based on "Implicit Neural Representations with Periodic Activation Functions"
by Sitzmann et al. (2020).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Sequence, Optional


class SineLayer(nn.Module):
    """
    Sine activation layer with SIREN-specific initialization.

    Attributes:
        features: Number of output features.
        is_first: Whether this is the first layer (affects initialization).
        omega_0: Frequency multiplier for the sine activation.
    """
    features: int
    is_first: bool = False
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_dim = inputs.shape[-1]

        # Initialize weights following SIREN paper
        if self.is_first:
            # First layer: uniform in [-1/input_dim, 1/input_dim]
            weight_init = nn.initializers.uniform(scale=1.0 / input_dim)
        else:
            # Hidden layers: uniform in [-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
            scale = np.sqrt(6.0 / input_dim) / self.omega_0
            weight_init = nn.initializers.uniform(scale=scale)

        x = nn.Dense(
            features=self.features,
            kernel_init=weight_init,
            bias_init=nn.initializers.uniform(scale=1.0)
        )(inputs)

        return jnp.sin(self.omega_0 * x)


class SIREN(nn.Module):
    """
    SIREN network for learning implicit representations.

    Attributes:
        hidden_features: Number of features in hidden layers.
        hidden_layers: Number of hidden sine layers.
        out_features: Number of output features (typically 1).
        outermost_linear: If True, use linear output layer instead of sine.
        w0: Frequency parameter omega_0 for all layers.
    """
    hidden_features: int = 256
    hidden_layers: int = 4
    out_features: int = 1
    outermost_linear: bool = True
    w0: float = 30.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the SIREN network.

        Args:
            inputs: Input coordinates of shape [..., input_dim].

        Returns:
            Output values of shape [..., out_features].
        """
        # First sine layer
        x = SineLayer(
            features=self.hidden_features,
            is_first=True,
            omega_0=self.w0,
            name='sine_0'
        )(inputs)

        # Hidden sine layers
        for i in range(self.hidden_layers):
            x = SineLayer(
                features=self.hidden_features,
                is_first=False,
                omega_0=self.w0,
                name=f'sine_{i + 1}'
            )(x)

        # Output layer
        if self.outermost_linear:
            # Linear output layer with appropriate initialization
            scale = np.sqrt(6.0 / self.hidden_features) / self.w0
            output_init = nn.initializers.uniform(scale=scale)
            x = nn.Dense(
                features=self.out_features,
                kernel_init=output_init,
                bias_init=nn.initializers.uniform(scale=1.0),
                name='output'
            )(x)
        else:
            x = SineLayer(
                features=self.out_features,
                is_first=False,
                omega_0=self.w0,
                name='sine_output'
            )(x)

        return x


def create_siren(
    hidden_features: int = 256,
    hidden_layers: int = 4,
    out_features: int = 1,
    w0: float = 30.0,
    outermost_linear: bool = True,
) -> SIREN:
    """
    Factory function to create a SIREN model.

    Args:
        hidden_features: Width of hidden layers.
        hidden_layers: Number of hidden layers.
        out_features: Output dimension.
        w0: Frequency parameter.
        outermost_linear: Whether to use linear output layer.

    Returns:
        SIREN model instance.
    """
    return SIREN(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=outermost_linear,
        w0=w0,
    )


def init_siren(
    model: SIREN,
    input_dim: int,
    rng_key: jax.random.PRNGKey,
) -> dict:
    """
    Initialize SIREN model parameters.

    Args:
        model: SIREN model instance.
        input_dim: Dimension of input coordinates.
        rng_key: JAX random key.

    Returns:
        Initialized parameter dictionary.
    """
    dummy_input = jnp.ones((1, input_dim))
    variables = model.init(rng_key, dummy_input)
    return variables['params']


def count_parameters(params: dict) -> int:
    """Count total number of trainable parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
