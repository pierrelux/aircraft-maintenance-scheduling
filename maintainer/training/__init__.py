"""
Training module for Maintainer.

Contains data generation, dataset handling, and training loop.
"""

from .data_gen import generate_training_data, generate_synthetic_data
from .dataset import BCDataset, collate_trajectories
from .trainer import BCTrainer, TrainingConfig

__all__ = [
    "generate_training_data",
    "generate_synthetic_data",
    "BCDataset",
    "collate_trajectories",
    "BCTrainer",
    "TrainingConfig",
]

