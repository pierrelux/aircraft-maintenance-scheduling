#!/usr/bin/env python3
"""
Generate behavior cloning training data for Maintainer.

Usage:
    # From real Deng Excel data (recommended):
    python -m maintainer.scripts.generate_data --output data/train.pkl --excel data/Sep\ 25\ 2017\ \(Case\ Study\ 2018-2021\).xlsx

    # From synthetic data (for testing only):
    python -m maintainer.scripts.generate_data --output data/train.pkl --synthetic --n-instances 100
"""

import argparse
import pickle
from pathlib import Path

from maintainer.training.data_gen import (
    generate_training_data,
    generate_training_data_from_excel,
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
        "--excel",
        type=str,
        default=None,
        help="Path to Deng Excel file (e.g., 'Sep 25 2017 (Case Study 2018-2021).xlsx')",
    )
    parser.add_argument(
        "--n-instances", "-n",
        type=int,
        default=100,
        help="Number of instances to generate",
    )
    parser.add_argument(
        "--max-aircraft",
        type=int,
        default=None,
        help="Limit number of aircraft (for testing)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=365 * 4,
        help="Planning horizon in days (default: 4 years)",
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
        help="Use synthetic data (for testing only, not recommended for training)",
    )
    
    args = parser.parse_args()
    
    print("Generating training data...")
    print(f"  Instances: {args.n_instances}")
    
    if args.synthetic:
        print("  Mode: SYNTHETIC (for testing only!)")
        print("  WARNING: Synthetic data uses a simple heuristic, not the Deng solver.")
        data = generate_synthetic_data(
            n_instances=args.n_instances,
            seed=args.seed,
            verbose=True,
        )
    elif args.excel:
        print(f"  Mode: Real Deng data")
        print(f"  Excel: {args.excel}")
        print(f"  Horizon: {args.horizon_days} days")
        if args.max_aircraft:
            print(f"  Max aircraft: {args.max_aircraft}")
        
        data = generate_training_data_from_excel(
            excel_path=args.excel,
            n_instances=args.n_instances,
            horizon_days=args.horizon_days,
            max_aircraft=args.max_aircraft,
            seed=args.seed,
            verbose=True,
        )
    else:
        raise ValueError(
            "Must specify either --excel <path> for real data or --synthetic for test data. "
            "Real data is required for proper training."
        )
    
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

