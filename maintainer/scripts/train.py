#!/usr/bin/env python3
"""
Train MaintainerNet via behavior cloning.

Usage:
    python -m maintainer.scripts.train --data data/train.pkl --output checkpoints/model.pt
"""

import argparse
import os
import pickle
from pathlib import Path
import torch

# Load .env file for wandb API key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from maintainer.core.model import MaintainerNet, ModelConfig
from maintainer.training.dataset import BCDataset
from maintainer.training.trainer import BCTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train MaintainerNet via behavior cloning"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training data pickle file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Validation data pickle file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'latest' for checkpoint_latest.pt)",
    )
    # Wandb options
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable wandb logging (default: enabled)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="maintainer-v1",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name",
    )
    
    args = parser.parse_args()
    
    print("Loading training data...")
    with open(args.data, 'rb') as f:
        train_bundle = pickle.load(f)
    
    all_data = train_bundle['data']
    print(f"  Loaded {len(all_data)} total instances")
    
    # Load separate validation data if provided, otherwise split
    val_data = None
    if args.val_data:
        with open(args.val_data, 'rb') as f:
            val_bundle = pickle.load(f)
        val_data = val_bundle['data']
        train_data = all_data
        print(f"  Using separate validation set: {len(val_data)} instances")
    else:
        # Auto train/val split (80/20)
        import random
        random.seed(42)  # Reproducible split
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        split_idx = int(len(all_data) * 0.8)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        print(f"  Auto split: {len(train_data)} train, {len(val_data)} val (80/20)")
    
    # Create datasets
    train_dataset = BCDataset(train_data, normalize_rewards=True)
    val_dataset = BCDataset(val_data, normalize_rewards=True) if val_data else None
    
    print(f"  {train_dataset.get_stats()['n_samples']} training samples")
    
    # Find max aircraft and max sequence length for model config
    max_aircraft = max(inst.n_aircraft for inst, _ in train_data)
    
    # Compute max tasks from trajectories (each trajectory step adds a task)
    max_tasks = max(len(traj) for _, traj in train_data)
    # Add buffer for safety
    max_tasks = int(max_tasks * 1.2)
    # max_seq_len = 1 (global) + n_aircraft + max_tasks
    max_seq_len = 1 + max_aircraft + max_tasks + 100  # buffer
    
    print(f"  Max aircraft: {max_aircraft}")
    print(f"  Max tasks: {max_tasks}")
    print(f"  Max sequence length: {max_seq_len}")
    
    # Create model
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=8,
        max_aircraft=max(50, max_aircraft),
        max_tasks=max_tasks,
        max_seq_len=max_seq_len,
        n_aircraft=max_aircraft,
    )
    
    model = MaintainerNet(model_config)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Handle resume path
    resume_from = None
    if args.resume:
        if args.resume == 'latest':
            resume_from = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
        else:
            resume_from = args.resume
        if os.path.exists(resume_from):
            print(f"\nWill resume from: {resume_from}")
        else:
            print(f"\nWarning: Checkpoint not found: {resume_from}, starting fresh")
            resume_from = None
    
    # Create trainer
    use_wandb = args.wandb and not args.no_wandb
    train_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=resume_from,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    if use_wandb:
        print("\nWandB logging enabled")
    
    trainer = BCTrainer(model, train_config, model_config=model_config)
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(train_dataset, val_dataset, verbose=True)
    
    # Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save_checkpoint(str(output_path))
    print(f"\nSaved checkpoint to: {output_path}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Policy accuracy: {history['policy_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()

