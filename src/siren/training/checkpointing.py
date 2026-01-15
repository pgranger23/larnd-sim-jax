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
    """Recursively convert JAX arrays to numpy for saving using tree_map."""
    def convert_leaf(x):
        if isinstance(x, jnp.ndarray):
            return np.array(x)
        return x
    return jax.tree_util.tree_map(convert_leaf, obj)


def _to_jax(obj: Any) -> Any:
    """Recursively convert numpy arrays to JAX arrays using tree_map."""
    def convert_leaf(x):
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x
    return jax.tree_util.tree_map(convert_leaf, obj)


def _serialize_opt_state(opt_state: Any) -> Dict:
    """Serialize optimizer state to a dict for saving.

    Saves only the array leaves - the structure will be recreated from
    a fresh optimizer init, then values copied over.
    """
    # Flatten the pytree to get leaves only
    leaves, _ = jax.tree_util.tree_flatten(opt_state)

    # Convert leaves to numpy
    leaves_np = [np.array(leaf) if hasattr(leaf, '__array__') else leaf for leaf in leaves]

    return {'leaves': leaves_np}


def _restore_opt_state(saved_data: Dict, template_opt_state: Any) -> Any:
    """Restore optimizer state by copying saved values into a template structure.

    Args:
        saved_data: Dict with 'leaves' key containing saved array values.
        template_opt_state: Fresh optimizer state with correct structure.

    Returns:
        Optimizer state with values from saved_data but structure from template.
    """
    saved_leaves = saved_data['leaves']

    # Convert saved leaves to JAX arrays
    saved_leaves_jax = [jnp.array(leaf) if isinstance(leaf, np.ndarray) else leaf
                        for leaf in saved_leaves]

    # Get template structure
    _, treedef = jax.tree_util.tree_flatten(template_opt_state)

    # Reconstruct with saved values but correct structure
    return jax.tree_util.tree_unflatten(treedef, saved_leaves_jax)


def save_checkpoint(
    path: str,
    params: Dict,
    step: int,
    config: Dict,
    history: Dict,
    normalization_params: Dict,
    dataset_stats: Dict,
    opt_state: Optional[Any] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Path to save checkpoint.
        params: Model parameters.
        step: Current training step.
        config: Training configuration.
        history: Training history (losses, etc.).
        normalization_params: Data normalization parameters.
        dataset_stats: Dataset statistics.
        opt_state: Optimizer state (for proper resume).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert params to numpy for saving
    params_np = _to_numpy(unfreeze(params) if hasattr(params, 'unfreeze') else params)

    # Serialize optimizer state (handles complex optax structures)
    opt_state_serialized = _serialize_opt_state(opt_state) if opt_state is not None else None

    # Save as npz
    np.savez(
        path,
        params=params_np,
        step=step,
        config=config,
        history=history,
        normalization_params=normalization_params,
        dataset_stats=dataset_stats,
        opt_state=opt_state_serialized,
    )

    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
) -> Tuple[Dict, int, Dict, Dict, Dict, Dict, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file.

    Returns:
        Tuple of (params, step, config, history, normalization_params, dataset_stats, opt_state).
        opt_state may be None if not saved in checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = np.load(path, allow_pickle=True)

    # Params are already saved with {'params': ...} structure, don't double-wrap
    params = freeze(_to_jax(data['params'].item()))
    step = int(data['step'])
    config = data['config'].item()
    history = data['history'].item()
    normalization_params = data['normalization_params'].item()
    dataset_stats = data['dataset_stats'].item()

    # Load optimizer state data if available (for proper resume)
    # Note: This returns raw saved data, use _restore_opt_state() with a template to restore
    opt_state_data = None
    if 'opt_state' in data.files and data['opt_state'].item() is not None:
        opt_state_data = data['opt_state'].item()

    print(f"Loaded checkpoint from {path} at step {step}")
    if opt_state_data is not None:
        print(f"  Optimizer state loaded for proper resume")

    return params, step, config, history, normalization_params, dataset_stats, opt_state_data


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

    # Final model saves just the inner params (not wrapped), so wrap here
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
