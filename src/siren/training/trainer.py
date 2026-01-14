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

        # Initialize model
        self.model = create_siren(
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            out_features=1,
            w0=config.w0,
            outermost_linear=config.outermost_linear,
        )

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
            'val_loss': [],
            'learning_rate': [],
            'step': [],
        }

        # Best validation loss for patience-based scheduling
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0

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

    def _compile_functions(self) -> None:
        """JIT-compile training and evaluation functions."""

        @jax.jit
        def train_step(params, opt_state, coords, targets):
            """Single training step."""
            def loss_fn(params):
                preds = self.model.apply(params, coords)
                return jnp.mean((preds - targets) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, loss

        @jax.jit
        def eval_step(params, coords, targets):
            """Evaluation step."""
            preds = self.model.apply(params, coords)
            return jnp.mean((preds - targets) ** 2)

        self._train_step = train_step
        self._eval_step = eval_step

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
        print()

        start_time = time.time()
        last_log_time = start_time

        while self.step < config.num_steps:
            # Sample batch
            self.rng_key, batch_key = jax.random.split(self.rng_key)
            coords, targets = self.dataset.sample_batch(
                batch_key, config.batch_size, split='train'
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

            # Validation
            if self.step % config.val_every == 0:
                val_loss = self._validate()
                self.history['val_loss'].append(float(val_loss))

                print(f"  Validation loss: {val_loss:.6f}")

                # Patience-based LR scheduling
                if config.lr_scheduler == 'patience':
                    self._update_lr_patience(val_loss)

                # Save progress plot
                self._save_progress_plot()

            # Checkpointing
            if self.step % config.checkpoint_every == 0:
                self._save_checkpoint()

        # Final checkpoint and model save
        self._save_checkpoint()
        self._save_final_model()

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        print(f"Final train loss: {self.history['train_loss'][-1]:.6f}")
        if self.history['val_loss']:
            print(f"Final val loss: {self.history['val_loss'][-1]:.6f}")

    def _validate(self) -> float:
        """Run validation and return mean loss."""
        # Sample several batches for validation
        total_loss = 0.0
        n_batches = 10

        for _ in range(n_batches):
            self.rng_key, batch_key = jax.random.split(self.rng_key)
            coords, targets = self.dataset.sample_batch(
                batch_key, self.config.batch_size, split='val'
            )
            loss = self._eval_step(self.params, coords, targets)
            total_loss += float(loss)

        return total_loss / n_batches

    def _update_lr_patience(self, val_loss: float) -> None:
        """Update learning rate based on patience."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += self.config.val_every

            if self.steps_since_improvement >= self.config.lr_patience:
                # Reduce learning rate
                current_lr = self.get_current_lr()
                new_lr = max(current_lr * self.config.lr_factor, self.config.lr_min)

                if new_lr < current_lr:
                    print(f"  Reducing LR: {current_lr:.2e} -> {new_lr:.2e}")
                    self.lr_schedule = new_lr
                    # Reinitialize optimizer with new LR
                    self._setup_optimizer()
                    self.opt_state = self.optimizer.init(self.params)

                self.steps_since_improvement = 0

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        path = self.output_dir / f'checkpoint_step_{self.step}.npz'

        save_checkpoint(
            path=str(path),
            params=self.params,
            opt_state=self.opt_state,
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
            opt_state=self.opt_state,
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
        final_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else None

        save_final_model(
            path=str(path),
            params=self.params,
            config=self.config.to_dict(),
            normalization_params=self.dataset.norm_params.to_dict(),
            dataset_stats=self.dataset.get_stats(),
            final_step=self.step,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
        )

    def _save_progress_plot(self) -> None:
        """Save training progress plot."""
        if not self.history['train_loss']:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        steps = self.history['step']
        train_loss = self.history['train_loss']

        # Plot 1: Training loss
        ax = axes[0]
        ax.plot(steps, train_loss, 'b-', alpha=0.7, label='Train')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Validation loss
        ax = axes[1]
        if self.history['val_loss']:
            # Validation is less frequent
            val_steps_count = len(self.history['val_loss'])
            val_steps = [steps[i * (len(steps) // val_steps_count)]
                         for i in range(val_steps_count)] if val_steps_count > 0 else []
            if len(val_steps) == len(self.history['val_loss']):
                ax.plot(val_steps, self.history['val_loss'], 'r-o', alpha=0.7, label='Validation')
            else:
                ax.plot(self.history['val_loss'], 'r-o', alpha=0.7, label='Validation')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('Validation Loss')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No validation data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Loss')

        # Plot 3: Learning rate
        ax = axes[2]
        if self.history['learning_rate']:
            ax.plot(steps, self.history['learning_rate'], 'g-', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate')

        plt.suptitle(f'SIREN Training Progress - Step {self.step}', fontsize=12)
        plt.tight_layout()

        plot_path = self.output_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _resume_from_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        params, opt_state, step, config, history, norm_params, dataset_stats = load_checkpoint(path)

        self.params = params
        self.step = step
        self.history = history

        # Reinitialize optimizer (don't restore opt_state, it may be incompatible)
        self._setup_optimizer()
        self.opt_state = self.optimizer.init(self.params)

        print(f"Resumed from step {step}")
