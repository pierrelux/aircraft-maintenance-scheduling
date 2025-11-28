"""
PyTorch Dataset for behavior cloning training.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from maintainer.amcs.instance import AMCSInstance
from maintainer.amcs.extraction import Trajectory, TrajectoryStep
from maintainer.core.state import CPDState, Action, ActionType
from maintainer.core.actions import action_to_index, get_action_mask


@dataclass
class BCBatch:
    """A batch of behavior cloning data."""
    states: List[CPDState]          # List of states
    actions: torch.Tensor           # [batch] action indices
    rewards: torch.Tensor           # [batch] rewards
    action_masks: torch.Tensor      # [batch, n_actions] legal action masks
    
    def to(self, device: torch.device) -> "BCBatch":
        """Move tensors to device."""
        return BCBatch(
            states=self.states,
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            action_masks=self.action_masks.to(device),
        )


class BCDataset(Dataset):
    """
    PyTorch Dataset for behavior cloning.
    
    Flattens trajectories into individual (state, action, reward) samples.
    """
    
    def __init__(
        self,
        data: List[Tuple[AMCSInstance, Trajectory]],
        normalize_rewards: bool = True,
    ):
        """
        Args:
            data: List of (instance, trajectory) pairs
            normalize_rewards: Whether to normalize rewards to [-1, 1]
        """
        self.samples: List[TrajectoryStep] = []
        self.instance_map: List[AMCSInstance] = []
        
        # Flatten trajectories
        for instance, trajectory in data:
            for step in trajectory:
                self.samples.append(step)
                self.instance_map.append(instance)
        
        # Compute normalization statistics
        if normalize_rewards and self.samples:
            rewards = [s.reward for s in self.samples]
            self.reward_mean = sum(rewards) / len(rewards)
            self.reward_std = (sum((r - self.reward_mean)**2 for r in rewards) / len(rewards)) ** 0.5
            if self.reward_std < 1e-8:
                self.reward_std = 1.0
        else:
            self.reward_mean = 0.0
            self.reward_std = 1.0
        
        self.normalize_rewards = normalize_rewards
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - state: CPDState
            - action: Action
            - action_idx: int (flat action index)
            - reward: float (normalized)
            - action_mask: list (flat action mask)
        """
        step = self.samples[idx]
        instance = self.instance_map[idx]
        
        # Compute action index
        n_aircraft = instance.n_aircraft
        action_idx = action_to_index(step.action, n_aircraft)
        
        # Normalize reward
        reward = step.reward
        if self.normalize_rewards:
            reward = (reward - self.reward_mean) / self.reward_std
            reward = max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
        
        # Get action mask
        from maintainer.core.actions import get_flat_action_mask
        mask = get_flat_action_mask(step.state)
        
        return {
            'state': step.state,
            'action': step.action,
            'action_idx': action_idx,
            'reward': reward,
            'action_mask': mask.tolist(),
            'n_aircraft': n_aircraft,
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            'n_samples': len(self.samples),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
        }


def collate_trajectories(batch: List[Dict]) -> BCBatch:
    """
    Collate function for DataLoader.
    
    Groups samples into a BCBatch.
    """
    states = [b['state'] for b in batch]
    actions = torch.tensor([b['action_idx'] for b in batch], dtype=torch.long)
    rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32)
    
    # Pad action masks to max size in batch
    max_actions = max(len(b['action_mask']) for b in batch)
    action_masks = torch.zeros(len(batch), max_actions, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask = b['action_mask']
        action_masks[i, :len(mask)] = torch.tensor(mask, dtype=torch.bool)
    
    return BCBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        action_masks=action_masks,
    )


class TrajectoryDataset(Dataset):
    """
    Dataset that keeps trajectories intact (for sequence-level operations).
    """
    
    def __init__(
        self,
        data: List[Tuple[AMCSInstance, Trajectory]],
    ):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[AMCSInstance, Trajectory]:
        return self.data[idx]

