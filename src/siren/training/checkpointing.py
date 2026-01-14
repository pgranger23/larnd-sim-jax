"""
Checkpointing utilities for SIREN surrogate training.

Handles saving and loading of model parameters, optimizer state, and training history.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from flax.core.frozen_dict import freeze, unfreeze
import optax


def _to_numpy(obj: Any) -> Any:
    """Recursively convert JAX arrays to numpy for saving."""
    if isinstance(obj, jnp.ndarray):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy(x) for x in obj)
    else:
        return obj


def _to_jax(obj: Any) -> Any:
    """Recursively convert numpy arrays to JAX arrays."""
    if isinstance(obj, np.ndarray):
        return jnp.array(obj)
    elif isinstance(obj, dict):
        return {k: _to_jax(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_jax(x) for x in obj)
    else:
        return obj


def save_checkpoint(
    path: str,
    params: Dict,
    opt_state: optax.OptState,
    step: int,
    config: Dict,
    history: Dict,
    normalization_params: Dict,
    dataset_stats: Dict,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Path to save checkpoint.
        params: Model parameters.
        opt_state: Optimizer state.
        step: Current training step.
        config: Training configuration.
        history: Training history (losses, etc.).
        normalization_params: Data normalization parameters.
        dataset_stats: Dataset statistics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy for saving
    params_np = _to_numpy(unfreeze(params) if hasattr(params, 'unfreeze') else params)
    opt_state_np = _to_numpy(opt_state)

    # Save as npz
    np.savez(
        path,
        params=params_np,
        opt_state=opt_state_np,
        step=step,
        config=config,
        history=history,
        normalization_params=normalization_params,
        dataset_stats=dataset_stats,
    )

    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
) -> Tuple[Dict, Any, int, Dict, Dict, Dict, Dict]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file.

    Returns:
        Tuple of (params, opt_state, step, config, history, normalization_params, dataset_stats).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = np.load(path, allow_pickle=True)

    params = freeze({'params': _to_jax(data['params'].item())})
    opt_state = data['opt_state'].item()  # Will be recreated by trainer
    step = int(data['step'])
    config = data['config'].item()
    history = data['history'].item()
    normalization_params = data['normalization_params'].item()
    dataset_stats = data['dataset_stats'].item()

    print(f"Loaded checkpoint from {path} at step {step}")

    return params, opt_state, step, config, history, normalization_params, dataset_stats


def save_final_model(
    path: str,
    params: Dict,
    config: Dict,
    normalization_params: Dict,
    dataset_stats: Dict,
    final_step: int,
    final_train_loss: Optional[float] = None,
    final_val_loss: Optional[float] = None,
) -> None:
    """
    Save final trained model (without optimizer state).

    Args:
        path: Path to save model.
        params: Model parameters.
        config: Model configuration.
        normalization_params: Data normalization parameters.
        dataset_stats: Dataset statistics.
        final_step: Final training step.
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract just the params (not the full frozen dict)
    if isinstance(params, dict) and 'params' in params:
        params_to_save = params['params']
    else:
        params_to_save = params

    params_np = _to_numpy(params_to_save)

    # Model config for reconstruction
    model_config = {
        'hidden_features': config.get('hidden_features', 256),
        'hidden_layers': config.get('hidden_layers', 4),
        'out_features': 1,
        'outermost_linear': config.get('outermost_linear', True),
        'w0': config.get('w0', 30.0),
    }

    np.savez(
        path,
        params=params_np,
        model_config=model_config,
        normalization_params=normalization_params,
        dataset_stats=dataset_stats,
        final_step=final_step,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
    )

    print(f"Saved final model to {path}")


def load_final_model(path: str) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Load a final trained model.

    Args:
        path: Path to model file.

    Returns:
        Tuple of (params, model_config, normalization_params, dataset_stats, metadata).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    data = np.load(path, allow_pickle=True)

    params = freeze({'params': _to_jax(data['params'].item())})
    model_config = data['model_config'].item()
    normalization_params = data['normalization_params'].item()
    dataset_stats = data['dataset_stats'].item()

    metadata = {
        'final_step': int(data['final_step']),
        'final_train_loss': float(data['final_train_loss']) if data['final_train_loss'] is not None else None,
        'final_val_loss': float(data['final_val_loss']) if data['final_val_loss'] is not None else None,
    }

    print(f"Loaded model from {path}")
    print(f"  Final step: {metadata['final_step']}")
    if metadata['final_train_loss'] is not None:
        print(f"  Final train loss: {metadata['final_train_loss']:.6f}")
    if metadata['final_val_loss'] is not None:
        print(f"  Final val loss: {metadata['final_val_loss']:.6f}")

    return params, model_config, normalization_params, dataset_stats, metadata


def save_history(path: str, history: Dict) -> None:
    """Save training history to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy/jax arrays to lists
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            history_serializable[key] = value.tolist()
        elif isinstance(value, list):
            history_serializable[key] = [
                float(v) if isinstance(v, (np.floating, jnp.floating)) else v
                for v in value
            ]
        else:
            history_serializable[key] = value

    with open(path, 'w') as f:
        json.dump(history_serializable, f, indent=2)


def load_history(path: str) -> Dict:
    """Load training history from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in an output directory.

    Args:
        output_dir: Directory to search.

    Returns:
        Path to latest checkpoint, or None if none found.
    """
    output_dir = Path(output_dir)

    # Look for checkpoint files
    checkpoints = list(output_dir.glob('checkpoint_step_*.npz'))

    if not checkpoints:
        # Try checkpoint_latest.npz
        latest = output_dir / 'checkpoint_latest.npz'
        if latest.exists():
            return str(latest)
        return None

    # Sort by step number
    def get_step(p):
        try:
            return int(p.stem.split('_')[-1])
        except ValueError:
            return -1

    checkpoints.sort(key=get_step, reverse=True)
    return str(checkpoints[0])
