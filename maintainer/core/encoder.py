"""
Token Encoder for Maintainer.

Converts CPDState to a sequence of token embeddings for the transformer.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .state import CPDState, TaskNode


@dataclass
class EncoderConfig:
    """Configuration for the token encoder."""
    d_model: int = 256          # Embedding dimension
    max_aircraft: int = 50      # Maximum number of aircraft
    max_tasks: int = 500        # Maximum tasks in a CPD
    max_seq_len: int = 600      # Maximum sequence length (1 + max_aircraft + max_tasks)
    
    # Feature dimensions for MLPs
    global_feat_dim: int = 4    # horizon, capacity, n_aircraft, progress
    aircraft_feat_dim: int = 8  # intervals, durations, tolerances, counts
    task_feat_dim: int = 4      # duration, earliest, latest, flexibility


class TokenEncoder(nn.Module):
    """
    Encode CPDState to token embeddings.
    
    Produces a sequence: [GLOBAL] [AC_0] [AC_1] ... [AC_N] [TASK_0] ... [TASK_M]
    
    Each token is embedded as:
    - Learned type embedding (global/aircraft/task)
    - Feature MLP output
    - Positional encoding
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        
        # Type embeddings
        self.global_embed = nn.Parameter(torch.randn(d) * 0.02)
        self.aircraft_embed = nn.Embedding(config.max_aircraft, d)
        self.check_type_embed = nn.Embedding(2, d)  # 0 = A, 1 = C
        self.seq_index_embed = nn.Embedding(20, d)  # Max 20 checks per aircraft per type
        
        # Feature MLPs
        self.global_mlp = nn.Sequential(
            nn.Linear(config.global_feat_dim, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d),
        )
        
        self.aircraft_mlp = nn.Sequential(
            nn.Linear(config.aircraft_feat_dim, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d),
        )
        
        self.task_mlp = nn.Sequential(
            nn.Linear(config.task_feat_dim, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d),
        )
        
        # Positional encoding
        self.pos_embed = nn.Embedding(config.max_seq_len, d)
        
        # Layer norm for final embeddings
        self.layer_norm = nn.LayerNorm(d)
    
    def encode_global(
        self,
        instance,
        state: CPDState,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode the global context token."""
        if device is None:
            device = next(self.parameters()).device
            
        # Compute progress (fraction of expected tasks added)
        expected_tasks = sum(
            instance.get_aircraft(i).max_A_checks(instance.horizon) +
            instance.get_aircraft(i).max_C_checks(instance.horizon)
            for i in range(instance.n_aircraft)
        )
        progress = len(state.nodes) / max(1, expected_tasks)
        
        features = torch.tensor([
            instance.horizon / 365.0,  # Normalize to ~1
            instance.capacity / 5.0,   # Normalize to ~1
            instance.n_aircraft / 20.0,  # Normalize to ~1
            progress,
        ], dtype=torch.float32, device=device)
        
        embed = self.global_embed + self.global_mlp(features)
        return embed
    
    def encode_aircraft(
        self,
        instance,
        state: CPDState,
        aircraft_id: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode an aircraft context token."""
        if device is None:
            device = next(self.parameters()).device
            
        ac = instance.get_aircraft(aircraft_id)
        
        # Normalize features
        features = torch.tensor([
            ac.A_interval / 50.0,
            ac.C_interval / 500.0,
            ac.A_duration / 5.0,
            ac.C_duration / 30.0,
            ac.A_tolerance / 10.0,
            ac.C_tolerance / 30.0,
            state.get_task_count(aircraft_id) / 20.0,
            instance.max_tasks_for_aircraft(aircraft_id) / 50.0,
        ], dtype=torch.float32, device=device)
        
        embed = (
            self.aircraft_embed(torch.tensor(aircraft_id, device=device)) +
            self.aircraft_mlp(features)
        )
        return embed
    
    def encode_task(
        self,
        instance,
        task: TaskNode,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode a task token."""
        if device is None:
            device = next(self.parameters()).device
            
        horizon = instance.horizon
        
        features = torch.tensor([
            task.duration / 30.0,
            task.earliest_day / horizon,
            task.latest_day / horizon,
            (task.latest_day - task.earliest_day) / horizon,  # Flexibility
        ], dtype=torch.float32, device=device)
        
        check_type_idx = 0 if task.check_type == 'A' else 1
        seq_idx = min(task.sequence_index, 19)  # Clamp to embedding size
        
        embed = (
            self.aircraft_embed(torch.tensor(task.aircraft_id, device=device)) +
            self.check_type_embed(torch.tensor(check_type_idx, device=device)) +
            self.seq_index_embed(torch.tensor(seq_idx, device=device)) +
            self.task_mlp(features)
        )
        return embed
    
    def forward(
        self,
        state: CPDState,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a CPDState to token embeddings.
        
        Args:
            state: The CPD state to encode
            device: Target device
        
        Returns:
            (embeddings, attention_mask)
            embeddings: [seq_len, d_model]
            attention_mask: [seq_len] boolean mask (True = valid token)
        """
        if device is None:
            device = next(self.parameters()).device
        
        instance = state.instance
        n_aircraft = instance.n_aircraft
        n_tasks = len(state.nodes)
        
        # Sequence: [GLOBAL] + [AC_i for i in range(n_aircraft)] + [TASK_j for j in tasks]
        seq_len = 1 + n_aircraft + n_tasks
        
        embeddings = []
        
        # Global token
        embeddings.append(self.encode_global(instance, state, device))
        
        # Aircraft tokens
        for i in range(n_aircraft):
            embeddings.append(self.encode_aircraft(instance, state, i, device))
        
        # Task tokens
        for task in state.nodes:
            embeddings.append(self.encode_task(instance, task, device))
        
        # Stack embeddings
        embeddings = torch.stack(embeddings, dim=0).to(device)  # [seq_len, d_model]
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=device)
        embeddings = embeddings + self.pos_embed(positions)
        
        # Layer norm
        embeddings = self.layer_norm(embeddings)
        
        # Attention mask (all valid)
        attention_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        
        return embeddings, attention_mask
    
    def encode_batch(
        self,
        states: list,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of states with padding.
        
        Args:
            states: List of CPDState objects
            device: Target device
        
        Returns:
            (embeddings, attention_mask)
            embeddings: [batch, max_seq_len, d_model]
            attention_mask: [batch, max_seq_len]
        """
        if device is None:
            device = next(self.parameters()).device
        
        batch_embeddings = []
        batch_masks = []
        
        for state in states:
            emb, mask = self.forward(state, device)
            batch_embeddings.append(emb)
            batch_masks.append(mask)
        
        # Pad to max length in batch
        max_len = max(e.shape[0] for e in batch_embeddings)
        d_model = self.config.d_model
        
        padded_embeddings = torch.zeros(len(states), max_len, d_model, device=device)
        padded_masks = torch.zeros(len(states), max_len, dtype=torch.bool, device=device)
        
        for i, (emb, mask) in enumerate(zip(batch_embeddings, batch_masks)):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len] = emb
            padded_masks[i, :seq_len] = mask
        
        return padded_embeddings, padded_masks

