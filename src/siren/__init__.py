"""
SIREN Surrogate for LUT in larnd-sim-jax.

This package provides a SIREN (Sinusoidal Representation Network) surrogate
to replace the Look-Up Table used in detector simulation.
"""

from .core import SIREN, SineLayer
from .inference import SurrogatePredictor

__all__ = ['SIREN', 'SineLayer', 'SurrogatePredictor']
