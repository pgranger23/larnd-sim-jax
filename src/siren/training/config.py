"""
Training configuration for SIREN surrogate.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import json
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for SIREN surrogate training.

    Attributes:
        # Model architecture
        hidden_features: Width of hidden layers.
        hidden_layers: Number of hidden layers.
        w0: SIREN frequency parameter.
        outermost_linear: Use linear output layer.

        # Training
        batch_size: Training batch size.
        num_steps: Total training steps.
        learning_rate: Initial learning rate.
        weight_decay: AdamW weight decay.

        # Learning rate scheduling
        lr_scheduler: Type of LR scheduler.
        lr_decay_rate: Decay rate for exponential scheduler.
        lr_patience: Steps without improvement before reducing LR.
        lr_factor: Factor to reduce LR by.
        lr_min: Minimum learning rate.

        # Regularization
        gradient_clip_norm: Max gradient norm (None to disable).

        # Logging and checkpointing
        log_every: Steps between logging.
        val_every: Steps between validation.
        checkpoint_every: Steps between checkpoints.

        # Data
        val_fraction: Fraction of data for validation.
        normalize_inputs: Whether to normalize inputs to [-1, 1].
        normalize_outputs: Whether to normalize outputs.

        # Paths
        lut_path: Path to response_template NPZ file.
        output_dir: Directory for outputs.

        # Reproducibility
        seed: Random seed.
    """
    # Model architecture
    hidden_features: int = 256
    hidden_layers: int = 4
    w0: float = 30.0
    outermost_linear: bool = True

    # Training
    batch_size: int = 65536
    num_steps: int = 50000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Learning rate scheduling
    lr_scheduler: Literal['constant', 'exponential', 'patience'] = 'exponential'
    lr_decay_rate: float = 0.9999
    lr_patience: int = 1000
    lr_factor: float = 0.5
    lr_min: float = 1e-7

    # Regularization
    gradient_clip_norm: Optional[float] = 1.0

    # Logging and checkpointing
    log_every: int = 100
    val_every: int = 500
    checkpoint_every: int = 5000

    # Data
    val_fraction: float = 0.1
    normalize_inputs: bool = True
    normalize_outputs: bool = True

    # Paths
    lut_path: str = 'src/larndsim/detector_properties/response_44_v2a_full_tick.npz'
    output_dir: str = 'siren_training'

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save config to file (JSON or YAML based on extension)."""
        path = Path(path)
        data = self.to_dict()

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from file (JSON or YAML)."""
        path = Path(path)

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                data = json.load(f)

        return cls(**data)

    @classmethod
    def from_args(cls, args) -> 'TrainingConfig':
        """Create config from argparse namespace."""
        # Start with defaults
        config = cls()

        # Override with provided args
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainingConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
