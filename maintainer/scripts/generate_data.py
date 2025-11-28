#!/usr/bin/env python3
"""
Generate behavior cloning training data for Maintainer.

Usage:
    python -m maintainer.scripts.generate_data --output data/train.pkl --n-instances 1000
"""

import argparse
import pickle
from pathlib import Path

from maintainer.training.data_gen import (
    generate_training_data,
    generate_synthetic_data,
    compute_dataset_statistics,
    DataGenConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BC training data for Maintainer"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--n-instances", "-n",
        type=int,
        default=100,
        help="Number of instances to generate",
    )
    parser.add_argument(
        "--n-aircraft-min",
        type=int,
        default=5,
        help="Minimum number of aircraft",
    )
    parser.add_argument(
        "--n-aircraft-max",
        type=int,
        default=10,
        help="Maximum number of aircraft",
    )
    parser.add_argument(
        "--horizon-min",
        type=int,
        default=180,
        help="Minimum horizon (days)",
    )
    parser.add_argument(
        "--horizon-max",
        type=int,
        default=365,
        help="Maximum horizon (days)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no Deng solver required)",
    )
    
    args = parser.parse_args()
    
    print("Generating training data...")
    print(f"  Instances: {args.n_instances}")
    print(f"  Aircraft: {args.n_aircraft_min}-{args.n_aircraft_max}")
    print(f"  Horizon: {args.horizon_min}-{args.horizon_max} days")
    
    if args.synthetic:
        print("  Mode: Synthetic (no Deng)")
        data = generate_synthetic_data(
            n_instances=args.n_instances,
            seed=args.seed,
            verbose=True,
        )
    else:
        config = DataGenConfig(
            n_instances=args.n_instances,
            n_aircraft_range=(args.n_aircraft_min, args.n_aircraft_max),
            horizon_range=(args.horizon_min, args.horizon_max),
            seed=args.seed,
        )
        data = generate_training_data(config, use_deng=True, verbose=True)
    
    # Compute statistics
    stats = compute_dataset_statistics(data)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': data,
            'stats': stats,
            'config': vars(args),
        }, f)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

