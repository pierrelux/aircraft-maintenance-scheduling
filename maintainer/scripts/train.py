#!/usr/bin/env python3
"""
Train MaintainerNet via behavior cloning.

Usage:
    python -m maintainer.scripts.train --data data/train.pkl --output checkpoints/model.pt
"""

import argparse
import pickle
from pathlib import Path
import torch

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
    
    args = parser.parse_args()
    
    print("Loading training data...")
    with open(args.data, 'rb') as f:
        train_bundle = pickle.load(f)
    
    train_data = train_bundle['data']
    print(f"  Loaded {len(train_data)} training instances")
    
    # Load validation data if provided
    val_data = None
    if args.val_data:
        with open(args.val_data, 'rb') as f:
            val_bundle = pickle.load(f)
        val_data = val_bundle['data']
        print(f"  Loaded {len(val_data)} validation instances")
    
    # Create datasets
    train_dataset = BCDataset(train_data, normalize_rewards=True)
    val_dataset = BCDataset(val_data, normalize_rewards=True) if val_data else None
    
    print(f"  {train_dataset.get_stats()['n_samples']} training samples")
    
    # Find max aircraft for model config
    max_aircraft = max(inst.n_aircraft for inst, _ in train_data)
    
    # Create model
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=8,
        max_aircraft=max(50, max_aircraft),
        n_aircraft=max_aircraft,
    )
    
    model = MaintainerNet(model_config)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    train_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=args.device,
    )
    
    trainer = BCTrainer(model, train_config)
    
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

