"""
Trainer class for SIREN surrogate.

Handles training loop, optimization, and logging.
"""

import jax
import jax.numpy as jnp
import optax
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
from functools import partial

# Use non-interactive backend for matplotlib (no X11 needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..core import SIREN, create_siren, init_siren
from .config import TrainingConfig
from .dataset import ResponseTemplateDataset
from .checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_final_model,
    save_history,
    find_latest_checkpoint,
)


class SurrogateTrainer:
    """
    Trainer for SIREN surrogate model.

    Handles:
    - Model initialization
    - Training loop with JIT-compiled steps
    - Learning rate scheduling
    - Checkpointing and resumption
    - Logging
    """

    def __init__(
        self,
        config: TrainingConfig,
        dataset: ResponseTemplateDataset,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration.
            dataset: Dataset instance.
        """
        self.config = config
        self.dataset = dataset

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(self.output_dir / 'config.yaml')

        # Handle square_output: keep linear output layer, square the result
        # This matches LUCiD implementation
        if config.square_output:
            print("Square output enabled: linear output layer, then squaring")

        # Initialize model
        self.model = create_siren(
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            out_features=1,
            w0=config.w0,
            outermost_linear=config.outermost_linear,  # Keep default (True)
        )
        self.square_output = config.square_output

        # Initialize parameters
        self.rng_key = jax.random.PRNGKey(config.seed)
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.params = {'params': init_siren(self.model, input_dim=4, rng_key=init_key)}

        # Count parameters
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"Model parameters: {n_params:,}")

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Initialize training state
        self.step = 0
        self.history = {
            'train_loss': [],
            'learning_rate': [],
            'step': [],
        }

        # Compile training and evaluation functions
        self._compile_functions()

    def _setup_optimizer(self) -> None:
        """Setup optimizer with learning rate schedule."""
        config = self.config

        # Build learning rate schedule
        if config.lr_scheduler == 'constant':
            lr_schedule = config.learning_rate
        elif config.lr_scheduler == 'exponential':
            lr_schedule = optax.exponential_decay(
                init_value=config.learning_rate,
                transition_steps=1,
                decay_rate=config.lr_decay_rate,
                end_value=config.lr_min,
            )
        else:
            # Patience-based: use constant LR, we'll manually adjust
            lr_schedule = config.learning_rate

        # Build optimizer chain
        optimizer_parts = []

        # Gradient clipping
        if config.gradient_clip_norm is not None:
            optimizer_parts.append(optax.clip_by_global_norm(config.gradient_clip_norm))

        # Adam or AdamW
        if config.weight_decay > 0:
            optimizer_parts.append(optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=config.weight_decay,
            ))
        else:
            optimizer_parts.append(optax.adam(learning_rate=lr_schedule))

        self.optimizer = optax.chain(*optimizer_parts)
        self.opt_state = self.optimizer.init(self.params)

        # Store schedule for logging
        self.lr_schedule = lr_schedule

    def _apply_model(self, params, coords):
        """Apply model with optional squaring for [0,1] output."""
        preds = self.model.apply(params, coords)
        if self.square_output:
            preds = preds ** 2
        return preds

    def _compile_functions(self) -> None:
        """JIT-compile training and evaluation functions."""
        square_output = self.square_output

        @jax.jit
        def train_step(params, opt_state, coords, targets):
            """Single training step."""
            def loss_fn(params):
                preds = self.model.apply(params, coords)
                if square_output:
                    preds = preds ** 2
                return jnp.mean((preds - targets) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, loss

        @jax.jit
        def eval_step(params, coords, targets):
            """Evaluation step."""
            preds = self.model.apply(params, coords)
            if square_output:
                preds = preds ** 2
            return jnp.mean((preds - targets) ** 2)

        self._train_step = train_step
        self._eval_step = eval_step

        # Compile CDF-specific functions if needed
        if self.config.use_cdf:
            self._compile_cdf_functions()

    def _compile_cdf_functions(self) -> None:
        """JIT-compile CDF training functions with derivative loss."""
        lambda_deriv = self.config.lambda_deriv
        loss_type = self.config.loss_type
        square_output = self.square_output

        # Get normalization scale for time coordinate
        # Time is normalized to [-1, 1], so we need to scale the derivative
        t_range = self.dataset.norm_params.t_range
        t_scale = (t_range[1] - t_range[0]) / 2.0  # d(t_norm)/dt = 2/(t_max - t_min)

        def compute_loss(pred, target, loss_type):
            """Compute loss based on loss_type."""
            if loss_type == 'mse':
                return jnp.mean((pred - target) ** 2)
            elif loss_type == 'relative_mse':
                # Relative MSE: better for small values
                # Add eps to avoid division by zero
                eps = 1e-6
                rel_error = (pred - target) / (jnp.abs(target) + eps)
                return jnp.mean(rel_error ** 2)
            elif loss_type == 'log_cosh':
                # Log-cosh loss: robust to outliers, smooth
                diff = pred - target
                return jnp.mean(jnp.log(jnp.cosh(diff * 10.0)))  # Scale for sensitivity
            else:
                return jnp.mean((pred - target) ** 2)

        @jax.jit
        def train_step_cdf(params, opt_state, coords, cdf_targets, response_targets):
            """Training step with CDF + optional derivative loss."""

            def loss_fn(params):
                # CDF loss
                cdf_pred = self.model.apply(params, coords)
                if square_output:
                    cdf_pred = cdf_pred ** 2
                cdf_loss = compute_loss(cdf_pred, cdf_targets, loss_type)

                if lambda_deriv > 0:
                    # Derivative loss using JAX autodiff
                    # coords shape: (batch, 4) where coords[:, 3] is normalized time

                    def siren_scalar(single_coord):
                        """SIREN output for a single coordinate."""
                        out = self.model.apply(params, single_coord[None, :])[0, 0]
                        if square_output:
                            out = out ** 2
                        return out

                    # vmap over batch to get per-sample gradients w.r.t. inputs
                    grad_fn = jax.vmap(jax.grad(siren_scalar))
                    grads = grad_fn(coords)  # Shape: (batch, 4)
                    dcdf_dt_norm = grads[:, 3]  # Gradient w.r.t. normalized time

                    # Scale derivative: d(CDF)/dt = d(CDF)/dt_norm * dt_norm/dt
                    # CDF is scaled by 1/10, so actual response = 10 * d(CDF/10)/dt
                    # dt_norm/dt = 2/(t_max - t_min)
                    dcdf_dt = dcdf_dt_norm * (2.0 / t_scale) * 10.0

                    deriv_loss = compute_loss(dcdf_dt, response_targets.ravel(), loss_type)
                    return cdf_loss + lambda_deriv * deriv_loss

                return cdf_loss

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, loss

        @jax.jit
        def eval_step_cdf(params, coords, cdf_targets):
            """CDF evaluation step (CDF loss only)."""
            cdf_pred = self.model.apply(params, coords)
            if square_output:
                cdf_pred = cdf_pred ** 2
            return compute_loss(cdf_pred, cdf_targets, loss_type)

        self._train_step_cdf = train_step_cdf
        self._eval_step_cdf = eval_step_cdf

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if callable(self.lr_schedule):
            return float(self.lr_schedule(self.step))
        return float(self.lr_schedule)

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run training loop.

        Args:
            resume_from: Path to checkpoint to resume from.
        """
        config = self.config

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)

        print(f"\nStarting training from step {self.step}")
        print(f"Target steps: {config.num_steps}")
        print(f"Batch size: {config.batch_size}")
        print(f"Initial LR: {self.get_current_lr():.2e}")
        if config.use_cdf:
            print(f"Mode: CDF training (lambda_deriv={config.lambda_deriv})")
        print()

        start_time = time.time()
        last_log_time = start_time

        while self.step < config.num_steps:
            # Sample batch
            self.rng_key, batch_key = jax.random.split(self.rng_key)

            if config.use_cdf:
                # CDF mode: get both CDF and response targets
                coords, cdf_targets, response_targets = self.dataset.sample_batch_cdf(
                    batch_key, config.batch_size
                )
                # Training step with CDF loss
                self.params, self.opt_state, train_loss = self._train_step_cdf(
                    self.params, self.opt_state, coords, cdf_targets, response_targets
                )
            else:
                # Standard mode
                coords, targets = self.dataset.sample_batch(
                    batch_key, config.batch_size
                )
                # Training step
                self.params, self.opt_state, train_loss = self._train_step(
                    self.params, self.opt_state, coords, targets
                )

            self.step += 1

            # Logging
            if self.step % config.log_every == 0:
                elapsed = time.time() - last_log_time
                steps_per_sec = config.log_every / elapsed
                lr = self.get_current_lr()

                self.history['train_loss'].append(float(train_loss))
                self.history['learning_rate'].append(lr)
                self.history['step'].append(self.step)

                print(f"Step {self.step:6d} | Loss: {train_loss:.6f} | "
                      f"LR: {lr:.2e} | {steps_per_sec:.1f} steps/s")

                last_log_time = time.time()

            # Generate plots at regular intervals
            if self.step % config.plot_every == 0:
                # Save progress plot
                self._save_progress_plot()

                # Save prediction comparison plot (CDF mode only)
                self._save_prediction_plot()

            # Checkpointing
            if self.step % config.checkpoint_every == 0:
                self._save_checkpoint()

        # Final checkpoint and model save
        self._save_checkpoint()
        self._save_final_model()

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        print(f"Final train loss: {self.history['train_loss'][-1]:.6f}")

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        path = self.output_dir / f'checkpoint_step_{self.step}.npz'

        save_checkpoint(
            path=str(path),
            params=self.params,
            step=self.step,
            config=self.config.to_dict(),
            history=self.history,
            normalization_params=self.dataset.norm_params.to_dict(),
            dataset_stats=self.dataset.get_stats(),
        )

        # Also save as latest
        latest_path = self.output_dir / 'checkpoint_latest.npz'
        save_checkpoint(
            path=str(latest_path),
            params=self.params,
            step=self.step,
            config=self.config.to_dict(),
            history=self.history,
            normalization_params=self.dataset.norm_params.to_dict(),
            dataset_stats=self.dataset.get_stats(),
        )

        # Save history as JSON
        save_history(self.output_dir / 'history.json', self.history)

    def _save_final_model(self) -> None:
        """Save final trained model."""
        path = self.output_dir / 'final_model.npz'

        final_train_loss = self.history['train_loss'][-1] if self.history['train_loss'] else None

        save_final_model(
            path=str(path),
            params=self.params,
            config=self.config.to_dict(),
            normalization_params=self.dataset.norm_params.to_dict(),
            dataset_stats=self.dataset.get_stats(),
            final_step=self.step,
            final_train_loss=final_train_loss,
        )

    def _save_progress_plot(self) -> None:
        """Save training progress plot with 2 panels: loss and learning rate."""
        if not self.history['train_loss']:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        steps = self.history['step']
        train_loss = self.history['train_loss']

        # Plot 1: Training loss
        ax = axes[0]
        ax.plot(steps, train_loss, 'b-', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Plot 2: Learning rate
        ax = axes[1]
        if self.history['learning_rate']:
            ax.plot(steps, self.history['learning_rate'], 'g-', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'SIREN Training Progress - Step {self.step}', fontsize=12)
        plt.tight_layout()

        plot_path = self.output_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _save_prediction_plot(self) -> None:
        """Save LUT vs SIREN prediction comparison plot for CDF mode.

        Shows 3x3 grid of pixels (main + 8 neighbors) with CDF and derivative.
        """
        if not self.config.use_cdf:
            return

        import numpy as np

        # Fixed parameters
        diff_idx = 49
        n_t = self.dataset.n_t

        # 3x3 pixel grid: LUT indices for main pixel and neighbors
        # Main pixel = (0,0), neighbors at 1 pixel spacing = 9 LUT bins
        pixel_grid = [
            [(9, 9, "(-1,-1)"), (9, 0, "(-1,0)"), (9, 9, "(-1,+1)")],
            [(0, 9, "(0,-1)"),  (0, 0, "Main"),   (0, 9, "(0,+1)")],
            [(9, 9, "(+1,-1)"), (9, 0, "(+1,0)"), (9, 9, "(+1,+1)")],
        ]

        # Compute scale factor for derivative
        t_range = self.dataset.norm_params.t_range
        scale_factor = (2.0 / (t_range[1] - t_range[0])) * 10.0

        # Create figure with 3x3 subplots for CDF
        fig_cdf, axes_cdf = plt.subplots(3, 3, figsize=(15, 12))
        fig_deriv, axes_deriv = plt.subplots(3, 3, figsize=(15, 12))

        t_vals = jnp.arange(n_t, dtype=jnp.float32)
        t_vals_np = np.array(t_vals)

        for row in range(3):
            for col in range(3):
                lut_x, lut_y, label = pixel_grid[row][col]

                # Build coordinates
                coords = jnp.stack([
                    jnp.full(n_t, diff_idx, dtype=jnp.float32),
                    jnp.full(n_t, lut_x, dtype=jnp.float32),
                    jnp.full(n_t, lut_y, dtype=jnp.float32),
                    t_vals,
                ], axis=1)
                coords_norm = self.dataset.normalize_inputs(coords)

                # Get LUT values
                lut_response = self.dataset.response_template[diff_idx, lut_x, lut_y, :]
                lut_cdf = self.dataset.cdf_template[diff_idx, lut_x, lut_y, :]

                # Get SIREN predictions (CDF)
                siren_cdf_raw = self.model.apply(self.params, coords_norm).ravel()
                if self.square_output:
                    siren_cdf = np.array(siren_cdf_raw ** 2)
                else:
                    siren_cdf = np.array(siren_cdf_raw)

                # Compute derivative via autodiff
                def siren_scalar(coord):
                    out = self.model.apply(self.params, coord[None, :])[0, 0]
                    if self.square_output:
                        out = out ** 2
                    return out
                grad_fn = jax.vmap(jax.grad(siren_scalar))
                grads = grad_fn(coords_norm)
                siren_deriv = np.array(grads[:, 3]) * scale_factor

                # Compute MAE for this pixel
                cdf_mae = np.mean(np.abs(siren_cdf - lut_cdf))
                deriv_mae = np.mean(np.abs(siren_deriv - lut_response))

                # Plot CDF
                ax = axes_cdf[row, col]
                ax.plot(t_vals_np, lut_cdf, 'b-', alpha=0.7, linewidth=1, label='LUT')
                ax.plot(t_vals_np, siren_cdf, 'r--', alpha=0.7, linewidth=1, label='SIREN')
                ax.set_title(f'{label} (MAE={cdf_mae:.4f})', fontsize=10)
                if row == 2:
                    ax.set_xlabel('Time')
                if col == 0:
                    ax.set_ylabel('CDF')
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 2:
                    ax.legend(loc='upper left', fontsize=8)

                # Plot derivative
                ax = axes_deriv[row, col]
                ax.plot(t_vals_np, lut_response, 'b-', alpha=0.7, linewidth=1, label='LUT')
                ax.plot(t_vals_np, siren_deriv, 'r--', alpha=0.7, linewidth=1, label='SIREN')
                ax.set_title(f'{label} (MAE={deriv_mae:.4f})', fontsize=10)
                if row == 2:
                    ax.set_xlabel('Time')
                if col == 0:
                    ax.set_ylabel('Response')
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 2:
                    ax.legend(loc='upper right', fontsize=8)

        fig_cdf.suptitle(f'CDF: LUT vs SIREN - Step {self.step} (diff={diff_idx})', fontsize=14)
        fig_cdf.tight_layout()
        fig_cdf.savefig(self.output_dir / 'prediction_cdf.png', dpi=100, bbox_inches='tight')
        plt.close(fig_cdf)

        fig_deriv.suptitle(f'Derivative: LUT vs SIREN - Step {self.step} (diff={diff_idx})', fontsize=14)
        fig_deriv.tight_layout()
        fig_deriv.savefig(self.output_dir / 'prediction_deriv.png', dpi=100, bbox_inches='tight')
        plt.close(fig_deriv)

    def _resume_from_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        params, step, config, history, norm_params, dataset_stats = load_checkpoint(path)

        self.params = params
        self.step = step
        self.history = history

        # Reinitialize optimizer
        self._setup_optimizer()
        self.opt_state = self.optimizer.init(self.params)

        print(f"Resumed from step {step}")
