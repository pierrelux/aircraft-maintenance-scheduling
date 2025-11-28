"""
Behavior Cloning Trainer for Maintainer.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from maintainer.core.model import MaintainerNet, ModelConfig, PolicyLogits
from maintainer.core.state import CPDState, Action, ActionType
from maintainer.core.actions import get_action_mask, action_to_index
from .dataset import BCDataset, BCBatch, collate_trajectories


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    n_epochs: int = 100
    grad_clip: float = 1.0
    
    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 0.5
    
    # Evaluation
    eval_interval: int = 10
    
    # Logging
    log_interval: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BCTrainer:
    """
    Behavior Cloning trainer for MaintainerNet.
    
    Trains the network to imitate Deng's actions (policy) and
    predict final schedule costs (value).
    """
    
    def __init__(
        self,
        model: MaintainerNet,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Will be set in train()
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'policy_accuracy': [],
        }
    
    def compute_loss(
        self,
        batch: BCBatch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BC loss for a batch.
        
        Loss = policy_weight * CE(policy, expert_action) + value_weight * MSE(value, reward)
        
        Returns:
            (total_loss, metrics_dict)
        """
        states = batch.states
        target_actions = batch.actions.to(self.device)
        target_rewards = batch.rewards.to(self.device)
        action_masks = batch.action_masks.to(self.device)
        
        batch_size = len(states)
        
        # Forward pass for each state
        # (Could be batched more efficiently, but this is clearer)
        all_policy_losses = []
        all_value_losses = []
        all_correct = []
        
        for i, state in enumerate(states):
            mask = get_action_mask(state)
            policy, value = self.model(state, mask, self.device)
            
            # Policy loss: cross-entropy
            n_aircraft = state.instance.n_aircraft
            target_action_idx = target_actions[i].item()
            
            # Compute full action logits
            # Actions: [AddTask(i,A) for i] + [AddTask(i,C) for i] + [Stop]
            all_logits = torch.cat([
                policy.aircraft,  # A-check logits (one per aircraft)
                policy.aircraft,  # C-check logits (reuse, but should be separate in practice)
                policy.type[2:3],  # Stop logit
            ])
            
            # Actually, let's do this properly with the factored representation
            # Policy gives: aircraft logits [n_aircraft] and type logits [3]
            # Action is encoded as: idx < n_aircraft => AddTask(idx, A)
            #                       idx < 2*n_aircraft => AddTask(idx-n_aircraft, C)
            #                       idx == 2*n_aircraft => Stop
            
            # For cross-entropy, we need to compute log p(action)
            # p(AddTask(i, A)) = p(type=A) * p(aircraft=i)
            # p(AddTask(i, C)) = p(type=C) * p(aircraft=i)
            # p(Stop) = p(type=Stop)
            
            type_log_probs = F.log_softmax(policy.type, dim=-1)
            aircraft_log_probs = F.log_softmax(policy.aircraft, dim=-1)
            
            if target_action_idx < n_aircraft:
                # AddTask(i, A)
                ac_id = target_action_idx
                log_prob = type_log_probs[0] + aircraft_log_probs[ac_id]
            elif target_action_idx < 2 * n_aircraft:
                # AddTask(i, C)
                ac_id = target_action_idx - n_aircraft
                log_prob = type_log_probs[1] + aircraft_log_probs[ac_id]
            else:
                # Stop
                log_prob = type_log_probs[2]
            
            policy_loss = -log_prob
            all_policy_losses.append(policy_loss)
            
            # Value loss: MSE
            value_loss = (value - target_rewards[i]) ** 2
            all_value_losses.append(value_loss)
            
            # Accuracy: check if greedy action matches target
            pred_type = policy.type.argmax().item()
            pred_aircraft = policy.aircraft.argmax().item()
            
            if target_action_idx == 2 * n_aircraft:
                # Target is Stop
                correct = (pred_type == 2)
            elif target_action_idx < n_aircraft:
                # Target is AddTask(i, A)
                correct = (pred_type == 0 and pred_aircraft == target_action_idx)
            else:
                # Target is AddTask(i, C)
                correct = (pred_type == 1 and pred_aircraft == target_action_idx - n_aircraft)
            
            all_correct.append(float(correct))
        
        # Aggregate losses
        policy_loss = torch.stack(all_policy_losses).mean()
        value_loss = torch.stack(all_value_losses).mean()
        
        total_loss = (
            self.config.policy_weight * policy_loss +
            self.config.value_weight * value_loss
        )
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': sum(all_correct) / len(all_correct),
        }
        
        return total_loss, metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward and loss
            loss, metrics = self.compute_loss(batch)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += metrics['total_loss']
            total_policy_loss += metrics['policy_loss']
            total_value_loss += metrics['value_loss']
            total_accuracy += metrics['accuracy']
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'acc': f"{metrics['accuracy']:.2%}",
            })
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'accuracy': total_accuracy / n_batches,
        }
        
        return avg_metrics
    
    def train(
        self,
        train_data: BCDataset,
        val_data: Optional[BCDataset] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            verbose: Print progress
        
        Returns:
            Training history
        """
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_trajectories,
        )
        
        val_loader = None
        if val_data is not None:
            val_loader = DataLoader(
                val_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_trajectories,
            )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.n_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )
        
        for epoch in range(1, self.config.n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['policy_loss'].append(train_metrics['policy_loss'])
            self.history['value_loss'].append(train_metrics['value_loss'])
            self.history['policy_accuracy'].append(train_metrics['accuracy'])
            
            if verbose:
                print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                      f"policy_loss={train_metrics['policy_loss']:.4f}, "
                      f"value_loss={train_metrics['value_loss']:.4f}, "
                      f"accuracy={train_metrics['accuracy']:.2%}")
            
            # Evaluate
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                if verbose:
                    print(f"  Val: loss={val_metrics['loss']:.4f}, "
                          f"accuracy={val_metrics['accuracy']:.2%}")
        
        return self.history
    
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                _, metrics = self.compute_loss(batch)
                
                total_loss += metrics['total_loss']
                total_accuracy += metrics['accuracy']
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_accuracy / n_batches,
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

