"""
Behavior Cloning Trainer for Maintainer.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from maintainer.core.model import MaintainerNet, ModelConfig, PolicyLogits
from maintainer.core.state import CPDState, Action, ActionType
from maintainer.core.actions import get_action_mask, action_to_index
from .dataset import BCDataset, BCBatch, collate_trajectories


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.1  # Increased for regularization
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
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 1  # Save every N epochs
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # Early stopping
    early_stopping_patience: int = 5  # Stop if val loss doesn't improve for N epochs
    early_stopping_min_delta: float = 0.001  # Minimum improvement to count as progress
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "maintainer-v1"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


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
        model_config: Optional[ModelConfig] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.wandb_run = None
        
        self.model.to(self.device)
        
        # Initialize wandb if available and enabled
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb(model_config)
        
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
    
    def _init_wandb(self, model_config: Optional[ModelConfig] = None):
        """Initialize Weights & Biases logging."""
        # Load .env if exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Build config dict for wandb
        wandb_config = {
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "batch_size": self.config.batch_size,
            "n_epochs": self.config.n_epochs,
            "grad_clip": self.config.grad_clip,
            "policy_weight": self.config.policy_weight,
            "value_weight": self.config.value_weight,
            "device": self.config.device,
        }
        
        if model_config:
            wandb_config.update({
                "d_model": model_config.d_model,
                "n_layers": model_config.n_layers,
                "n_heads": model_config.n_heads,
                "max_aircraft": model_config.max_aircraft,
            })
        
        # Count model parameters
        wandb_config["n_parameters"] = sum(p.numel() for p in self.model.parameters())
        
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            entity=self.config.wandb_entity,
            config=wandb_config,
            reinit=True,
        )
        
        # Watch model for gradient logging
        wandb.watch(self.model, log="gradients", log_freq=100)
    
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
            
            # Clamp logits for numerical stability before log_softmax
            type_logits_clamped = policy.type.clamp(-50, 50)
            aircraft_logits_clamped = policy.aircraft.clamp(-50, 50)
            
            type_log_probs = F.log_softmax(type_logits_clamped, dim=-1)
            aircraft_log_probs = F.log_softmax(aircraft_logits_clamped, dim=-1)
            
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
            
            # Clamp policy loss to prevent inf (max ~10 corresponds to 1/20000 probability)
            policy_loss = torch.clamp(-log_prob, max=20.0)
            all_policy_losses.append(policy_loss)
            
            # Value loss: MSE with clamping for stability
            value_loss = (value - target_rewards[i]) ** 2
            value_loss = torch.clamp(value_loss, max=1000.0)  # Prevent explosion
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
        global_step = (epoch - 1) * len(dataloader)
        
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
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'acc': f"{metrics['accuracy']:.2%}",
            })
            
            # Log batch metrics to wandb
            if self.wandb_run and n_batches % self.config.log_interval == 0:
                wandb.log({
                    "batch/loss": metrics['total_loss'],
                    "batch/policy_loss": metrics['policy_loss'],
                    "batch/value_loss": metrics['value_loss'],
                    "batch/accuracy": metrics['accuracy'],
                    "batch/lr": self.optimizer.param_groups[0]['lr'],
                }, step=global_step)
            
            # Mid-epoch checkpoint (every 500 batches)
            if self.config.checkpoint_dir and n_batches % 500 == 0:
                self._save_mid_epoch_checkpoint(epoch, n_batches)
        
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
        
        # Resume from checkpoint if specified
        start_epoch = 1
        if self.config.resume_from:
            start_epoch = self._load_checkpoint_for_resume(self.config.resume_from)
            if verbose:
                print(f"Resuming from epoch {start_epoch}")
        
        # Early stopping state
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config.n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['policy_loss'].append(train_metrics['policy_loss'])
            self.history['value_loss'].append(train_metrics['value_loss'])
            self.history['policy_accuracy'].append(train_metrics['accuracy'])
            
            # Log epoch metrics to wandb
            if self.wandb_run:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics['loss'],
                    "train/policy_loss": train_metrics['policy_loss'],
                    "train/value_loss": train_metrics['value_loss'],
                    "train/accuracy": train_metrics['accuracy'],
                })
            
            if verbose:
                print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                      f"policy_loss={train_metrics['policy_loss']:.4f}, "
                      f"value_loss={train_metrics['value_loss']:.4f}, "
                      f"accuracy={train_metrics['accuracy']:.2%}")
            
            # Save checkpoint
            if self.config.checkpoint_dir and epoch % self.config.checkpoint_interval == 0:
                self._save_epoch_checkpoint(epoch)
            
            # Evaluate on validation set every epoch (important for early stopping)
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                # Log validation metrics to wandb
                if self.wandb_run:
                    wandb.log({
                        "epoch": epoch,
                        "val/loss": val_metrics['loss'],
                        "val/accuracy": val_metrics['accuracy'],
                    })
                
                if verbose:
                    print(f"  Val: loss={val_metrics['loss']:.4f}, "
                          f"accuracy={val_metrics['accuracy']:.2%}")
                
                # Early stopping check
                if val_metrics['loss'] < best_val_loss - self.config.early_stopping_min_delta:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    if self.config.checkpoint_dir:
                        self._save_best_checkpoint(epoch, val_metrics['loss'])
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"  Early stopping: {patience_counter}/{self.config.early_stopping_patience}")
                    
                    if patience_counter >= self.config.early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                        break
        
        # Finish wandb run
        if self.wandb_run:
            wandb.finish()
        
        return self.history
    
    def _save_epoch_checkpoint(self, epoch: int):
        """Save checkpoint for this epoch."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch:03d}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
        }, checkpoint_path)
        
        # Also save as 'latest' for easy resume
        latest_path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
        }, latest_path)
        
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    def _save_mid_epoch_checkpoint(self, epoch: int, batch: int):
        """Save mid-epoch checkpoint for crash recovery."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Save as 'latest' for easy resume
        latest_path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pt")
        torch.save({
            'epoch': epoch,
            'batch': batch,  # Track which batch we're at
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
        }, latest_path)
        
        print(f"  [Mid-epoch checkpoint: epoch {epoch}, batch {batch}]")
    
    def _save_best_checkpoint(self, epoch: int, val_loss: float):
        """Save the best model based on validation loss."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        best_path = os.path.join(self.config.checkpoint_dir, "checkpoint_best.pt")
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
        }, best_path)
        
        print(f"  [New best model: val_loss={val_loss:.4f}]")
    
    def _load_checkpoint_for_resume(self, checkpoint_path: str) -> int:
        """Load checkpoint for resuming training. Returns next epoch to train."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint['epoch'] + 1  # Return next epoch
    
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

