#!/usr/bin/env python3
"""
Train SIREN surrogate for LUT in larnd-sim-jax.

Usage:
    python -m src.siren.train_surrogate --output_dir siren_training --num_steps 50000

    # Resume from checkpoint
    python -m src.siren.train_surrogate --resume siren_training/checkpoint_latest.npz
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.siren.training.config import TrainingConfig
from src.siren.training.dataset import ResponseTemplateDataset
from src.siren.training.trainer import SurrogateTrainer
from src.siren.training.checkpointing import find_latest_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SIREN surrogate for LUT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Width of hidden layers')
    parser.add_argument('--hidden_layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--w0', type=float, default=30.0,
                        help='SIREN frequency parameter')

    # Training
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Training batch size')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for AdamW')

    # Learning rate scheduling
    parser.add_argument('--lr_scheduler', type=str, default='exponential',
                        choices=['constant', 'exponential', 'patience'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9999,
                        help='Decay rate for exponential scheduler')

    # Logging and checkpointing
    parser.add_argument('--log_every', type=int, default=100,
                        help='Steps between logging')
    parser.add_argument('--val_every', type=int, default=500,
                        help='Steps between validation')
    parser.add_argument('--checkpoint_every', type=int, default=5000,
                        help='Steps between checkpoints')

    # Data
    parser.add_argument('--lut_path', type=str,
                        default='src/larndsim/detector_properties/response_44_v2a_full_tick.npz',
                        help='Path to LUT file')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of data for validation')

    # Output
    parser.add_argument('--output_dir', type=str, default='siren_training',
                        help='Output directory for checkpoints and logs')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Automatically resume from latest checkpoint in output_dir')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML/JSON config file (overrides CLI args)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("SIREN Surrogate Training for LUT")
    print("=" * 60)

    # Load config from file or create from args
    if args.config:
        print(f"Loading config from {args.config}")
        config = TrainingConfig.load(args.config)
        # Override with any CLI args that were explicitly set
        for key, value in vars(args).items():
            if key not in ['config', 'resume', 'auto_resume'] and value is not None:
                if hasattr(config, key):
                    setattr(config, key, value)
    else:
        config = TrainingConfig.from_args(args)

    print("\nConfiguration:")
    print(config)
    print()

    # Check for auto-resume
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        resume_path = find_latest_checkpoint(config.output_dir)
        if resume_path:
            print(f"Auto-resuming from {resume_path}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = ResponseTemplateDataset(
        lut_path=config.lut_path,
        val_fraction=config.val_fraction,
        normalize_inputs=config.normalize_inputs,
        normalize_outputs=config.normalize_outputs,
        seed=config.seed,
    )

    print("\nDataset statistics:")
    for key, value in dataset.get_stats().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SurrogateTrainer(config=config, dataset=dataset)

    # Train
    print("\nStarting training...")
    trainer.train(resume_from=resume_path)

    print("\nTraining complete!")
    print(f"Final model saved to: {config.output_dir}/final_model.npz")
    print(f"Training history saved to: {config.output_dir}/history.json")


if __name__ == '__main__':
    main()
