#!/usr/bin/env python3
"""
Evaluate Maintainer against Deng baseline.

Usage:
    python -m maintainer.scripts.evaluate --checkpoint checkpoints/model.pt --n-instances 50
"""

import argparse
import pickle
from pathlib import Path
import torch

from maintainer.core.model import MaintainerNet, ModelConfig
from maintainer.core.mcts import MCTSConfig
from maintainer.training.evaluation import (
    evaluate_on_instances,
    evaluate_policy_accuracy,
    batch_evaluate_decoder_loss,
)
from maintainer.training.data_gen import DataGenConfig, generate_training_data
from maintainer.amcs.instance import AMCSInstance


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Maintainer vs Deng baseline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=50,
        help="Number of instances to evaluate",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=100,
        help="MCTS simulations per action",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="MCTS exploration constant",
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Use greedy policy only (no MCTS)",
    )
    parser.add_argument(
        "--decoder-diagnostic",
        action="store_true",
        help="Run decoder loss diagnostic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for instance generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Reconstruct model config
    # Try to load from checkpoint, otherwise use defaults
    model_config = ModelConfig(
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_aircraft=50,
        n_aircraft=10,
    )
    
    model = MaintainerNet(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    print(f"  Loaded from: {args.checkpoint}")
    
    # Generate evaluation instances
    print(f"\nGenerating {args.n_instances} evaluation instances...")
    config = DataGenConfig(
        n_instances=args.n_instances,
        seed=args.seed,
    )
    data = generate_training_data(config, use_deng=False, verbose=True)
    instances = [inst for inst, _ in data]
    
    # Decoder diagnostic
    if args.decoder_diagnostic:
        print("\n" + "=" * 50)
        print("Decoder Loss Diagnostic")
        print("=" * 50)
        
        decoder_stats = batch_evaluate_decoder_loss(instances, verbose=True)
        
        print("\nResults:")
        for key, value in decoder_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Policy accuracy on trajectories
    print("\n" + "=" * 50)
    print("Policy Accuracy on Expert Trajectories")
    print("=" * 50)
    
    trajectories = [traj for _, traj in data]
    policy_metrics = evaluate_policy_accuracy(model, trajectories, args.device)
    
    print("\nResults:")
    for key, value in policy_metrics.items():
        if key == 'n_samples':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Full evaluation vs Deng
    print("\n" + "=" * 50)
    print("Maintainer+MCTS vs Deng")
    print("=" * 50)
    
    if args.policy_only:
        print("Mode: Greedy policy only (no MCTS)")
        mcts_config = MCTSConfig(n_simulations=0)
    else:
        print(f"Mode: MCTS with {args.n_simulations} simulations")
        mcts_config = MCTSConfig(
            n_simulations=args.n_simulations,
            c_puct=args.c_puct,
        )
    
    results = evaluate_on_instances(
        instances,
        model,
        mcts_config,
        args.device,
        verbose=True,
    )
    
    print("\n" + results.summary())


if __name__ == "__main__":
    main()

